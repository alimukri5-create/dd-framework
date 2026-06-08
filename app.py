"""DD Framework - Institutional Due Diligence Streamlit app."""

from __future__ import annotations

import hashlib
import html
import json
import os
import re
import tempfile
from dataclasses import asdict, dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import streamlit as st
import streamlit.components.v1 as components
import requests
import yfinance as yf
from openai import APIConnectionError, APIError, APITimeoutError, AuthenticationError, OpenAI, RateLimitError


APP_TITLE = "DD Framework - Institutional Due Diligence"
DEFAULT_MODEL = "gpt-4-turbo"
DEFAULT_TIMEOUT_SECONDS = 180
SCORE_NAMES = [
    "Momentum",
    "Exhaustion Risk",
    "Fundamental Trend",
    "Valuation Stretch",
    "Event Risk",
    "Dilution Risk",
    "Ownership Signal",
    "Narrative Heat",
    "Squeeze Risk",
    "Asymmetry",
]

SYSTEM_PROMPT = """
You are an institutional trading analyst producing fast decision support, not a generic company profile.
Prioritize what moves the stock, what matters now, what would change the view, and whether the setup is
actionable. Be concise, evidence-led, and explicit about uncertainty. Cite sources wherever available, never
invent precise figures, and flag missing or stale data. Avoid narrative filler.

At the end of every answer, include a compact section named DD_FRAMEWORK_SIGNAL with scores,
probabilities, catalysts, invalidation levels, or recommendation evidence that can support a dashboard.
""".strip()


@dataclass
class DDResult:
    ticker: str
    company_name: str | None
    cached_at: str
    model: str
    step_1: str
    step_2: str
    step_3: str
    dashboard_scores: dict[str, int]
    recommendation: str
    recommendation_reason: str
    decision_card: dict[str, str]
    market_snapshot: dict[str, str]
    trade_verdict: dict[str, str]
    evidence_flags: list[str]
    earnings_intel: dict[str, str]
    financial_trends: dict[str, str]
    ownership_intel: dict[str, str]
    narrative_heat: dict[str, str]
    expectations_baseline: dict[str, str]
    payoff_distribution: dict[str, str]
    unconventional_signals: dict[str, str]
    asymmetry_assessment: dict[str, str]


class UserFacingError(Exception):
    """Exception type for errors that should be shown without a stack trace."""


def configure_page() -> None:
    st.set_page_config(page_title=APP_TITLE, page_icon="DD", layout="wide")
    st.markdown(
        """
        <style>
        :root {
            color-scheme: dark;
        }
        .stApp {
            background: #090b10;
            color: #eef2f8;
        }
        [data-testid="stHeader"] {
            background: rgba(9, 11, 16, 0.85);
        }
        section[data-testid="stSidebar"] {
            background: #0e121a;
        }
        .block-container {
            padding-top: 2rem;
            max-width: 1220px;
        }
        div[data-testid="stMetric"] {
            background: #111722;
            border: 1px solid #263244;
            border-radius: 8px;
            padding: 1rem;
        }
        div[data-testid="stAlert"] {
            border-radius: 8px;
        }
        .dd-shell {
            border: 1px solid #263244;
            border-radius: 8px;
            padding: 1rem 1.1rem;
            background: #0f141d;
        }
        .copy-button {
            background: #2563eb;
            color: white;
            border: 0;
            border-radius: 6px;
            padding: 0.55rem 0.8rem;
            font-weight: 700;
            cursor: pointer;
        }
        .copy-button:hover {
            background: #1d4ed8;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def normalize_ticker(raw_ticker: str) -> str:
    ticker = raw_ticker.strip().upper()
    if not re.fullmatch(r"[A-Z][A-Z0-9.\-]{0,9}", ticker):
        raise UserFacingError("Enter a valid ticker symbol, for example RR, ACHR, UUUU, BRK.B, or RIVN.")
    return ticker


def get_secret(name: str, default: Any = None) -> Any:
    try:
        return st.secrets.get(name, default)
    except FileNotFoundError:
        return default


def get_openai_client() -> OpenAI:
    api_key = get_secret("OPENAI_API_KEY")
    if not api_key or str(api_key).strip() in {"", "replace_me"}:
        raise UserFacingError(
            "OpenAI API key is missing. Add OPENAI_API_KEY to .streamlit/secrets.toml, then rerun the app."
        )

    timeout = int(get_secret("OPENAI_TIMEOUT_SECONDS", DEFAULT_TIMEOUT_SECONDS))
    return OpenAI(api_key=str(api_key), timeout=timeout)


def get_model() -> str:
    return str(get_secret("OPENAI_MODEL", DEFAULT_MODEL)).strip() or DEFAULT_MODEL


def fetch_market_data(ticker: str) -> dict[str, Any]:
    """Fetch a compact market data pack to ground the model before analysis."""
    data: dict[str, Any] = {
        "ticker": ticker,
        "source": "Yahoo Finance via yfinance",
        "fetched_at": datetime.now(timezone.utc).isoformat(),
        "is_stale": False,
    }
    try:
        clear_proxy_env_for_market_data()
        yf_cache_dir = Path(tempfile.gettempdir()) / "dd-framework-yfinance-cache"
        yf_cache_dir.mkdir(parents=True, exist_ok=True)
        yf.set_tz_cache_location(str(yf_cache_dir))
        asset = yf.Ticker(ticker)
        info = asset.get_info() or {}
        history = asset.history(period="6mo", interval="1d", auto_adjust=True)
    except Exception as exc:
        fallback = fetch_chart_fallback(ticker, str(exc))
        if fallback and not fallback.get("error"):
            save_market_data_cache(ticker, fallback)
            return fallback
        cached = load_market_data_cache(ticker)
        if cached:
            cached["is_stale"] = True
            cached["stale_reason"] = f"Fresh market data fetch failed: {exc}"
            cached["source"] = f"{cached.get('source', 'Cached market data')} (stale fallback)"
            return cached
        data["error"] = f"Market data fetch failed: {exc}"
        return data

    data.update(
        {
            "name": info.get("shortName") or info.get("longName"),
            "business_summary": info.get("longBusinessSummary"),
            "sector": info.get("sector"),
            "industry": info.get("industry"),
            "website": info.get("website"),
            "market_cap": info.get("marketCap"),
            "enterprise_value": info.get("enterpriseValue"),
            "current_price": info.get("currentPrice") or info.get("regularMarketPrice"),
            "fifty_two_week_high": info.get("fiftyTwoWeekHigh"),
            "fifty_two_week_low": info.get("fiftyTwoWeekLow"),
            "average_volume": info.get("averageVolume"),
            "float_shares": info.get("floatShares"),
            "short_percent_float": info.get("shortPercentOfFloat"),
            "total_revenue": info.get("totalRevenue"),
            "total_cash": info.get("totalCash"),
            "total_debt": info.get("totalDebt"),
            "operating_cashflow": info.get("operatingCashflow"),
            "free_cashflow": info.get("freeCashflow"),
            "shares_outstanding": info.get("sharesOutstanding"),
            "implied_shares_outstanding": info.get("impliedSharesOutstanding"),
            "book_value": info.get("bookValue"),
            "revenue_growth": info.get("revenueGrowth"),
            "gross_margins": info.get("grossMargins"),
            "operating_margins": info.get("operatingMargins"),
            "profit_margins": info.get("profitMargins"),
            "ebitda_margins": info.get("ebitdaMargins"),
            "price_to_sales": info.get("priceToSalesTrailing12Months"),
            "trailing_pe": info.get("trailingPE"),
            "forward_pe": info.get("forwardPE"),
            "beta": info.get("beta"),
            "target_mean_price": info.get("targetMeanPrice"),
            "target_high_price": info.get("targetHighPrice"),
            "target_low_price": info.get("targetLowPrice"),
            "recommendation_mean": info.get("recommendationMean"),
            "recommendation_key": info.get("recommendationKey"),
            "number_of_analyst_opinions": info.get("numberOfAnalystOpinions"),
        }
    )

    if not history.empty and "Close" in history:
        close = history["Close"].dropna()
        volume = history["Volume"].dropna() if "Volume" in history else None
        data.update(
            {
                "last_close": float(close.iloc[-1]) if len(close) else None,
                "return_5d": pct_return(close, 5),
                "return_1m": pct_return(close, 21),
                "return_3m": pct_return(close, 63),
                "return_6m": pct_return(close, 126),
                "avg_volume_20d": float(volume.tail(20).mean()) if volume is not None and len(volume) else None,
            }
        )
    save_market_data_cache(ticker, data)
    return data


def fetch_chart_fallback(ticker: str, original_error: str) -> dict[str, Any] | None:
    """Fetch lightweight Yahoo chart data when yfinance's richer endpoints are rate-limited."""
    try:
        clear_proxy_env_for_market_data()
        url = f"https://query1.finance.yahoo.com/v8/finance/chart/{ticker}"
        response = requests.get(
            url,
            params={"range": "6mo", "interval": "1d"},
            headers={"User-Agent": "Mozilla/5.0"},
            timeout=15,
        )
        response.raise_for_status()
        payload = response.json()
        result = payload.get("chart", {}).get("result", [{}])[0]
        meta = result.get("meta", {})
        quote = (result.get("indicators", {}).get("quote") or [{}])[0]
        close_values = [value for value in result.get("indicators", {}).get("adjclose", [{}])[0].get("adjclose", []) if value]
        if not close_values:
            close_values = [value for value in quote.get("close", []) if value]
        volume_values = [value for value in quote.get("volume", []) if value]
        data: dict[str, Any] = {
            "ticker": ticker,
            "source": "Yahoo Finance chart endpoint fallback",
            "fetched_at": datetime.now(timezone.utc).isoformat(),
            "is_stale": False,
            "fallback_reason": original_error,
            "current_price": meta.get("regularMarketPrice"),
            "fifty_two_week_high": meta.get("fiftyTwoWeekHigh"),
            "fifty_two_week_low": meta.get("fiftyTwoWeekLow"),
            "average_volume": meta.get("averageDailyVolume3Month"),
        }
        if close_values:
            data.update(
                {
                    "last_close": float(close_values[-1]),
                    "return_5d": pct_return_list(close_values, 5),
                    "return_1m": pct_return_list(close_values, 21),
                    "return_3m": pct_return_list(close_values, 63),
                    "return_6m": pct_return_list(close_values, 126),
                }
            )
        if volume_values:
            data["avg_volume_20d"] = sum(volume_values[-20:]) / min(len(volume_values), 20)
        return data
    except Exception:
        return None


def fetch_v21_data(ticker: str) -> dict[str, Any]:
    """Fetch richer DD layers that can fail independently from the core market pack."""
    clear_proxy_env_for_market_data()
    data: dict[str, Any] = {
        "earnings_intel": {},
        "financial_trends": {},
        "ownership_intel": {},
        "narrative_heat": {},
    }
    try:
        yf_cache_dir = Path(tempfile.gettempdir()) / "dd-framework-yfinance-cache"
        yf_cache_dir.mkdir(parents=True, exist_ok=True)
        yf.set_tz_cache_location(str(yf_cache_dir))
        asset = yf.Ticker(ticker)
    except Exception as exc:
        data["error"] = f"V2.1 data init failed: {exc}"
        return data

    data["earnings_intel"] = extract_earnings_intel(asset)
    data["financial_trends"] = extract_financial_trends(asset)
    data["ownership_intel"] = extract_ownership_intel(asset)
    data["narrative_heat"] = extract_narrative_heat(asset)
    return data


def extract_earnings_intel(asset: Any) -> dict[str, str]:
    intel: dict[str, str] = {}
    try:
        calendar = asset.calendar or {}
        earnings_date = calendar.get("Earnings Date")
        if isinstance(earnings_date, list) and earnings_date:
            earnings_date = earnings_date[0]
        if earnings_date:
            intel["Next Earnings"] = str(earnings_date)
            intel["Days To Earnings"] = days_until_text(earnings_date)
        for key in ["Earnings Average", "Earnings High", "Earnings Low", "Revenue Average"]:
            if calendar.get(key) is not None:
                intel[key] = format_market_value(calendar.get(key))
    except Exception as exc:
        intel["Calendar Status"] = f"Unavailable: {exc}"

    try:
        dates = asset.earnings_dates
        if dates is not None and not dates.empty:
            last_reported = dates[dates["Reported EPS"].notna()].head(1)
            if not last_reported.empty:
                row = last_reported.iloc[0]
                intel["Last Earnings Date"] = str(last_reported.index[0])
                if row.get("EPS Estimate") is not None:
                    intel["Last EPS Estimate"] = format_market_value(row.get("EPS Estimate"))
                if row.get("Reported EPS") is not None:
                    intel["Last Reported EPS"] = format_market_value(row.get("Reported EPS"))
                if row.get("Surprise(%)") is not None:
                    intel["Last EPS Surprise"] = f"{float(row.get('Surprise(%)')):,.1f}%"
    except Exception as exc:
        intel["Earnings History Status"] = f"Unavailable: {exc}"
    return intel


def extract_financial_trends(asset: Any) -> dict[str, str]:
    trends: dict[str, str] = {}
    try:
        financials = asset.quarterly_financials
        trends.update(
            {
                "Quarterly Revenue Trend": trend_from_statement(financials, ["Total Revenue"]),
                "Quarterly Gross Profit Trend": trend_from_statement(financials, ["Gross Profit"]),
                "Quarterly Operating Income Trend": trend_from_statement(
                    financials, ["Operating Income", "Operating Revenue"]
                ),
                "Quarterly EBITDA Trend": trend_from_statement(financials, ["Normalized EBITDA", "EBITDA"]),
            }
        )
    except Exception as exc:
        trends["Financial Statement Status"] = f"Unavailable: {exc}"

    try:
        cashflow = asset.quarterly_cashflow
        trends.update(
            {
                "Quarterly FCF Trend": trend_from_statement(cashflow, ["Free Cash Flow"]),
                "Recent Capital Stock Issuance": latest_statement_value(cashflow, ["Issuance Of Capital Stock"]),
                "Recent End Cash": latest_statement_value(cashflow, ["End Cash Position"]),
            }
        )
    except Exception as exc:
        trends["Cash Flow Status"] = f"Unavailable: {exc}"

    try:
        balance = asset.quarterly_balance_sheet
        trends.update(
            {
                "Share Count Trend": trend_from_statement(balance, ["Ordinary Shares Number", "Share Issued"]),
                "Recent Total Debt": latest_statement_value(balance, ["Total Debt"]),
                "Recent Working Capital": latest_statement_value(balance, ["Working Capital"]),
            }
        )
    except Exception as exc:
        trends["Balance Sheet Status"] = f"Unavailable: {exc}"
    return {key: value for key, value in trends.items() if value not in {"N/A", None, ""}}


def extract_ownership_intel(asset: Any) -> dict[str, str]:
    intel: dict[str, str] = {}
    try:
        holders = asset.major_holders
        if holders is not None and not holders.empty:
            values = holders["Value"].to_dict()
            intel["Insiders Held"] = format_percent_ratio(values.get("insidersPercentHeld"))
            intel["Institutions Held"] = format_percent_ratio(values.get("institutionsPercentHeld"))
            intel["Institutions Float Held"] = format_percent_ratio(values.get("institutionsFloatPercentHeld"))
            if values.get("institutionsCount") is not None:
                intel["Institution Count"] = format_market_value(values.get("institutionsCount"))
    except Exception as exc:
        intel["Major Holders Status"] = f"Unavailable: {exc}"

    try:
        institutional = asset.institutional_holders
        if institutional is not None and not institutional.empty:
            top = institutional.head(3)
            intel["Top Institutions"] = "; ".join(
                f"{row.get('Holder')}: {format_percent_ratio(row.get('pctHeld'))}"
                for _, row in top.iterrows()
                if row.get("Holder") is not None
            )
    except Exception as exc:
        intel["Institutional Holders Status"] = f"Unavailable: {exc}"

    try:
        insiders = asset.insider_transactions
        if insiders is not None and not insiders.empty:
            recent = insiders.head(10)
            shares = numeric_series(recent.get("Shares")).sum()
            value = numeric_series(recent.get("Value")).sum()
            intel["Recent Insider Activity"] = f"{len(recent)} rows; shares {shares:,.0f}; value {value:,.0f}"
            sample_text = "; ".join(str(item) for item in recent.get("Transaction", []).head(3).tolist() if item)
            if sample_text:
                intel["Recent Insider Transaction Types"] = sample_text
    except Exception as exc:
        intel["Insider Transactions Status"] = f"Unavailable: {exc}"

    try:
        recs = asset.recommendations
        if recs is not None and not recs.empty:
            row = recs.iloc[0]
            intel["Analyst Recommendation Mix"] = (
                f"strongBuy {row.get('strongBuy', 0)}, buy {row.get('buy', 0)}, "
                f"hold {row.get('hold', 0)}, sell {row.get('sell', 0)}, strongSell {row.get('strongSell', 0)}"
            )
    except Exception as exc:
        intel["Recommendations Status"] = f"Unavailable: {exc}"
    return intel


def extract_narrative_heat(asset: Any) -> dict[str, str]:
    intel: dict[str, str] = {}
    try:
        news = asset.news or []
        titles: list[str] = []
        summaries: list[str] = []
        for item in news[:10]:
            content = item.get("content", {}) if isinstance(item, dict) else {}
            title = content.get("title") or item.get("title")
            summary = content.get("summary") or content.get("description") or item.get("summary")
            if title:
                titles.append(str(title))
            if summary:
                summaries.append(str(summary))
        joined = " ".join(titles + summaries).lower()
        intel["Headline Count"] = str(len(titles))
        if titles:
            intel["Top Headlines"] = " | ".join(titles[:3])
        themes = detect_narrative_themes(joined)
        intel["Narrative Themes"] = ", ".join(themes) if themes else "None obvious"
        intel["Narrative Heat"] = narrative_heat_label(joined, len(titles), themes)
    except Exception as exc:
        intel["News Status"] = f"Unavailable: {exc}"
    return intel


def days_until_text(value: Any) -> str:
    try:
        if hasattr(value, "date"):
            target = value.date()
        else:
            target = datetime.fromisoformat(str(value)).date()
        delta = (target - datetime.now(timezone.utc).date()).days
        return str(delta)
    except Exception:
        return "Unknown"


def trend_from_statement(frame: Any, labels: list[str]) -> str:
    if frame is None or frame.empty:
        return "N/A"
    series = statement_series(frame, labels)
    if series is None or len(series) < 2:
        return "N/A"
    latest = float(series.iloc[0])
    prior = float(series.iloc[1])
    delta = latest - prior
    pct = (delta / abs(prior) * 100) if prior else None
    direction = "improving" if delta > 0 else "deteriorating" if delta < 0 else "flat"
    pct_text = f", {pct:,.1f}% QoQ" if pct is not None else ""
    return f"{format_market_value(latest)} vs {format_market_value(prior)} prior ({direction}{pct_text})"


def latest_statement_value(frame: Any, labels: list[str]) -> str:
    if frame is None or frame.empty:
        return "N/A"
    series = statement_series(frame, labels)
    if series is None or len(series) == 0:
        return "N/A"
    return format_market_value(float(series.iloc[0]))


def statement_series(frame: Any, labels: list[str]) -> Any | None:
    for label in labels:
        if label in frame.index:
            return frame.loc[label].dropna()
    return None


def numeric_series(values: Any) -> Any:
    try:
        import pandas as pd

        return pd.to_numeric(values, errors="coerce").fillna(0)
    except Exception:
        return []


def detect_narrative_themes(text: str) -> list[str]:
    theme_terms = {
        "AI": [" ai ", "artificial intelligence", "physical ai"],
        "Defense": ["defense", "aerospace", "military"],
        "Critical Minerals": ["critical minerals", "rare earth", "neodymium", "dysprosium"],
        "Autonomy": ["autonomous", "lidar", "robotaxi", "sensor"],
        "Nuclear/Energy": ["nuclear", "uranium", "energy"],
        "Momentum Coverage": ["surge", "rally", "soar", "upside", "too late"],
    }
    return [theme for theme, terms in theme_terms.items() if any(term in text for term in terms)]


def narrative_heat_label(text: str, count: int, themes: list[str]) -> str:
    heat = 0
    heat += min(3, count // 3)
    if themes:
        heat += min(3, len(themes))
    if any(word in text for word in ["surge", "rally", "soar", "too late", "momentum"]):
        heat += 2
    if heat >= 6:
        return "High"
    if heat >= 3:
        return "Medium"
    return "Low"


def pct_return_list(values: list[float], periods: int) -> float | None:
    if len(values) <= periods:
        return None
    start = float(values[-periods - 1])
    end = float(values[-1])
    if start == 0:
        return None
    return (end / start - 1) * 100


def clear_proxy_env_for_market_data() -> None:
    """Avoid inherited local-blocking proxy settings when fetching public market data."""
    for name in [
        "HTTP_PROXY",
        "HTTPS_PROXY",
        "ALL_PROXY",
        "http_proxy",
        "https_proxy",
        "all_proxy",
    ]:
        os.environ.pop(name, None)


def pct_return(close: Any, periods: int) -> float | None:
    if len(close) <= periods:
        return None
    start = float(close.iloc[-periods - 1])
    end = float(close.iloc[-1])
    if start == 0:
        return None
    return (end / start - 1) * 100


def format_market_data(data: dict[str, Any]) -> str:
    if data.get("error"):
        return f"MARKET DATA SOURCE: {data['source']}\nFETCH STATUS: {data['error']}"
    labels = {
        "name": "Company",
        "business_summary": "Business Summary",
        "sector": "Sector",
        "industry": "Industry",
        "website": "Website",
        "market_cap": "Market Cap",
        "enterprise_value": "Enterprise Value",
        "current_price": "Current Price",
        "last_close": "Last Close",
        "fifty_two_week_high": "52W High",
        "fifty_two_week_low": "52W Low",
        "return_5d": "5D Return %",
        "return_1m": "1M Return %",
        "return_3m": "3M Return %",
        "return_6m": "6M Return %",
        "average_volume": "Average Volume",
        "avg_volume_20d": "20D Avg Volume",
        "float_shares": "Float Shares",
        "short_percent_float": "Short % Float",
        "total_revenue": "Total Revenue",
        "total_cash": "Total Cash",
        "total_debt": "Total Debt",
        "operating_cashflow": "Operating Cash Flow",
        "free_cashflow": "Free Cash Flow",
        "shares_outstanding": "Shares Outstanding",
        "implied_shares_outstanding": "Implied Shares Outstanding",
        "book_value": "Book Value",
        "revenue_growth": "Revenue Growth",
        "gross_margins": "Gross Margin",
        "operating_margins": "Operating Margin",
        "profit_margins": "Profit Margin",
        "ebitda_margins": "EBITDA Margin",
        "price_to_sales": "Price/Sales",
        "trailing_pe": "Trailing P/E",
        "forward_pe": "Forward P/E",
        "beta": "Beta",
        "target_mean_price": "Analyst Target Mean",
        "target_high_price": "Analyst Target High",
        "target_low_price": "Analyst Target Low",
        "recommendation_mean": "Recommendation Mean",
        "recommendation_key": "Recommendation Key",
        "number_of_analyst_opinions": "Analyst Opinion Count",
    }
    lines = [
        f"MARKET DATA SOURCE: {data.get('source')}",
        f"FETCHED AT: {data.get('fetched_at')}",
    ]
    for key, label in labels.items():
        value = data.get(key)
        if value is not None:
            lines.append(f"{label}: {format_market_value(value)}")
    return "\n".join(lines)


def build_market_snapshot(data: dict[str, Any]) -> dict[str, str]:
    if data.get("error"):
        return {"Data": data["error"]}

    snapshot: dict[str, str] = {}
    if data.get("is_stale"):
        snapshot["Data Status"] = "STALE FALLBACK"
    elif data.get("fallback_reason"):
        snapshot["Data Status"] = "LIGHTWEIGHT FALLBACK"
    else:
        snapshot["Data Status"] = "FRESH"
    snapshot["Price"] = format_market_value(data.get("current_price") or data.get("last_close") or "N/A")
    snapshot["1M / 3M"] = (
        f"{format_percent(data.get('return_1m'))} / {format_percent(data.get('return_3m'))}"
    )
    snapshot["Valuation"] = f"P/S {format_market_value(data.get('price_to_sales') or 'N/A')}"
    snapshot["Margins"] = (
        f"Gross {format_percent_ratio(data.get('gross_margins'))}, "
        f"Op {format_percent_ratio(data.get('operating_margins'))}, "
        f"Net {format_percent_ratio(data.get('profit_margins'))}"
    )
    snapshot["Cash / Debt"] = (
        f"{format_market_value(data.get('total_cash') or 'N/A')} / "
        f"{format_market_value(data.get('total_debt') or 'N/A')}"
    )
    snapshot["FCF"] = format_market_value(data.get("free_cashflow") or "N/A")
    snapshot["Volatility"] = f"Beta {format_market_value(data.get('beta') or 'N/A')}"
    snapshot["Quick Read"] = build_quick_read(data)
    return snapshot


def build_v3_edge_layer(
    ticker: str,
    data: dict[str, Any],
    v21_data: dict[str, Any],
    scores: dict[str, int],
) -> tuple[dict[str, str], dict[str, str], dict[str, str], dict[str, str]]:
    """Build honest expectations/payoff diagnostics before claiming asymmetry."""
    expectations = build_expectations_baseline(ticker, data, v21_data)
    payoff = build_payoff_distribution(data, expectations)
    unconventional = build_unconventional_signals(ticker, data, v21_data)
    asymmetry = build_true_asymmetry_assessment(scores, expectations, payoff, unconventional)
    return expectations, payoff, unconventional, asymmetry


def build_expectations_baseline(ticker: str, data: dict[str, Any], v21_data: dict[str, Any]) -> dict[str, str]:
    baseline: dict[str, str] = {
        "Baseline Rule": "No asymmetry claim without market-implied expectations and payoff math.",
        "Price Baseline": format_market_value(data.get("current_price") or data.get("last_close") or "N/A"),
    }

    price = number_or_none(data.get("current_price") or data.get("last_close"))
    target = number_or_none(data.get("target_mean_price"))
    target_high = number_or_none(data.get("target_high_price"))
    target_low = number_or_none(data.get("target_low_price"))
    if price and target:
        baseline["Sell-Side Mean Target Gap"] = f"{percent_change(target, price):,.1f}%"
    if price and target_high and target_low:
        baseline["Sell-Side Target Range"] = (
            f"{format_market_value(target_low)} to {format_market_value(target_high)} "
            f"({percent_change(target_low, price):,.1f}% / {percent_change(target_high, price):,.1f}%)"
        )
    if data.get("recommendation_key"):
        baseline["Consensus Recommendation"] = str(data.get("recommendation_key")).upper()
    if data.get("number_of_analyst_opinions") is not None:
        baseline["Analyst Count"] = format_market_value(data.get("number_of_analyst_opinions"))

    options = fetch_options_expectations(ticker, price, v21_data)
    baseline.update(options)

    if not any(key in baseline for key in ["Options Implied Move", "Sell-Side Mean Target Gap"]):
        baseline["Expectations Status"] = "Weak: no options-implied move or analyst target baseline available."
    elif "Options Implied Move" not in baseline:
        baseline["Expectations Status"] = "Partial: analyst baseline available, options-implied move missing."
    elif "Sell-Side Mean Target Gap" not in baseline:
        baseline["Expectations Status"] = "Partial: options baseline available, sell-side target missing."
    else:
        baseline["Expectations Status"] = "Usable: options and sell-side baselines available."
    return baseline


def fetch_options_expectations(ticker: str, price: float | None, v21_data: dict[str, Any]) -> dict[str, str]:
    if not price or price <= 0:
        return {"Options Status": "Unavailable: current price missing."}
    try:
        clear_proxy_env_for_market_data()
        asset = yf.Ticker(ticker)
        expiries = list(asset.options or [])
        if not expiries:
            return {"Options Status": "Unavailable: no listed options found via yfinance."}
        expiry = choose_options_expiry(expiries, v21_data)
        chain = asset.option_chain(expiry)
        call = closest_strike_row(chain.calls, price)
        put = closest_strike_row(chain.puts, price)
        if call is None or put is None:
            return {"Options Status": "Unavailable: no ATM call/put pair found."}
        call_price = option_mid_or_last(call)
        put_price = option_mid_or_last(put)
        if call_price is None or put_price is None:
            return {"Options Status": "Unavailable: ATM option prices missing."}
        implied_move = (call_price + put_price) / price * 100
        strike = call.get("strike") if call.get("strike") is not None else price
        return {
            "Options Expiry Used": str(expiry),
            "ATM Strike Used": format_market_value(strike),
            "ATM Straddle Price": format_market_value(call_price + put_price),
            "Options Implied Move": f"+/- {implied_move:,.1f}%",
            "Options Status": "Usable: nearest relevant ATM straddle used as market-implied event/risk baseline.",
        }
    except Exception as exc:
        return {"Options Status": f"Unavailable: {exc}"}


def choose_options_expiry(expiries: list[str], v21_data: dict[str, Any]) -> str:
    days = int_or_none(v21_data.get("earnings_intel", {}).get("Days To Earnings"))
    if days is None or days < 0:
        return expiries[0]
    today = datetime.now(timezone.utc).date()
    target_date = today + timedelta(days=days)
    for expiry in expiries:
        try:
            if datetime.fromisoformat(expiry).date() >= target_date:
                return expiry
        except ValueError:
            continue
    return expiries[0]


def closest_strike_row(frame: Any, price: float) -> dict[str, Any] | None:
    try:
        if frame is None or frame.empty or "strike" not in frame:
            return None
        closest_index = (frame["strike"] - price).abs().idxmin()
        return frame.loc[closest_index].to_dict()
    except Exception:
        return None


def option_mid_or_last(row: dict[str, Any]) -> float | None:
    bid = number_or_none(row.get("bid"))
    ask = number_or_none(row.get("ask"))
    last = number_or_none(row.get("lastPrice"))
    if bid is not None and ask is not None and bid > 0 and ask > 0 and ask >= bid:
        return (bid + ask) / 2
    if last is not None and last > 0:
        return last
    return None


def build_payoff_distribution(data: dict[str, Any], expectations: dict[str, str]) -> dict[str, str]:
    price = number_or_none(data.get("current_price") or data.get("last_close"))
    if not price or price <= 0:
        return {"Payoff Status": "Unavailable: current price missing."}

    target = number_or_none(data.get("target_mean_price"))
    target_high = number_or_none(data.get("target_high_price"))
    target_low = number_or_none(data.get("target_low_price"))
    high_52w = number_or_none(data.get("fifty_two_week_high"))
    low_52w = number_or_none(data.get("fifty_two_week_low"))
    cash = number_or_none(data.get("total_cash"))
    shares = number_or_none(data.get("shares_outstanding") or data.get("implied_shares_outstanding"))
    cash_per_share = cash / shares if cash and shares else None
    implied_move = parse_percent_from_text(expectations.get("Options Implied Move"))

    bull_price = first_valid_number(
        [
            target_high if target_high and target_high > price else None,
            high_52w if high_52w and high_52w > price else None,
            price * (1 + (implied_move or 20) / 100),
        ]
    )
    base_price = first_valid_number([target, price])
    bear_floor = first_valid_number(
        [
            target_low if target_low and target_low < price else None,
            low_52w if low_52w and low_52w < price else None,
            cash_per_share if cash_per_share and cash_per_share < price else None,
            price * 0.8,
        ]
    )
    bear_price = max(cash_per_share or 0, bear_floor) if bear_floor else price * 0.8

    upside = percent_change(bull_price, price) if bull_price else None
    base = percent_change(base_price, price) if base_price else None
    downside = percent_change(bear_price, price) if bear_price else None
    payoff: dict[str, str] = {
        "Current Price": format_market_value(price),
        "Bull Case Price": format_market_value(bull_price) if bull_price else "N/A",
        "Base Case Price": format_market_value(base_price) if base_price else "N/A",
        "Bear Case Floor": format_market_value(bear_price),
        "Bull / Base / Bear Move": (
            f"{format_signed_percent(upside)} / {format_signed_percent(base)} / {format_signed_percent(downside)}"
        ),
    }
    if cash_per_share:
        payoff["Cash Per Share Floor"] = format_market_value(cash_per_share)
    if upside is not None and downside is not None and downside < 0:
        payoff["Upside/Downside Ratio"] = f"{upside / abs(downside):,.2f}x"
    else:
        payoff["Upside/Downside Ratio"] = "N/A"
    payoff["Payoff Status"] = "Heuristic: uses listed options, sell-side targets, 52-week range, and cash/share when available."
    return payoff


def build_unconventional_signals(ticker: str, data: dict[str, Any], v21_data: dict[str, Any]) -> dict[str, str]:
    signals = fetch_sec_filing_signals(ticker)
    insider_text = v21_data.get("ownership_intel", {}).get("Recent Insider Transaction Types", "")
    if insider_text:
        signals["Insider Cluster Signal"] = classify_insider_activity(insider_text)
    narrative = v21_data.get("narrative_heat", {})
    if narrative.get("Narrative Heat"):
        signals["Narrative Acceleration Caveat"] = (
            "Yahoo headline heat only; not true alt-data. Treat as crowding/attention, not edge."
        )
    if not signals:
        signals["Unconventional Status"] = "None found in the free-data layer."
    return signals


def fetch_sec_filing_signals(ticker: str) -> dict[str, str]:
    try:
        clear_proxy_env_for_market_data()
        headers = {"User-Agent": "DDFramework/1.0 research-app@example.com"}
        mapping_response = requests.get("https://www.sec.gov/files/company_tickers.json", headers=headers, timeout=15)
        mapping_response.raise_for_status()
        mapping = mapping_response.json()
        match = next(
            (item for item in mapping.values() if str(item.get("ticker", "")).upper() == ticker.upper()),
            None,
        )
        if not match:
            return {"SEC Status": "No SEC company match found for ticker."}
        cik = str(match["cik_str"]).zfill(10)
        submissions = requests.get(f"https://data.sec.gov/submissions/CIK{cik}.json", headers=headers, timeout=15)
        submissions.raise_for_status()
        recent = submissions.json().get("filings", {}).get("recent", {})
        forms = recent.get("form", [])[:40]
        dates = recent.get("filingDate", [])[:40]
        docs = recent.get("primaryDocument", [])[:40]
        rows = list(zip(forms, dates, docs))
        risky_forms = [row for row in rows if row[0] in {"S-3", "S-3ASR", "424B5", "424B3", "FWP"}]
        eight_ks = [row for row in rows if row[0] == "8-K"]
        signals: dict[str, str] = {
            "SEC CIK": cik,
            "Recent SEC Filings": "; ".join(f"{form} {date}" for form, date, _ in rows[:6]) or "N/A",
        }
        if risky_forms:
            signals["Shelf/Offering Overhang"] = "; ".join(f"{form} {date}" for form, date, _ in risky_forms[:5])
        else:
            signals["Shelf/Offering Overhang"] = "No S-3/424B/FWP forms found in latest 40 SEC filings."
        if eight_ks:
            signals["Recent 8-K Activity"] = f"{len(eight_ks)} recent 8-K filings in latest 40 filings."
        signals["SEC Source"] = "SEC submissions JSON"
        return signals
    except Exception as exc:
        return {"SEC Status": f"Unavailable: {exc}"}


def classify_insider_activity(text: str) -> str:
    lower = text.lower()
    sell_count = sum(lower.count(term) for term in ["sale", "sell", "disposition"])
    buy_count = sum(lower.count(term) for term in ["purchase", "buy", "acquisition"])
    if buy_count >= 2 and buy_count > sell_count:
        return "Potential insider-buying cluster; verify transaction codes manually."
    if sell_count >= 2 and sell_count > buy_count:
        return "Insider-selling cluster risk; verify if sales are planned/10b5-1 before penalizing."
    return "Mixed/unclear insider activity; not a standalone edge."


def build_true_asymmetry_assessment(
    scores: dict[str, int],
    expectations: dict[str, str],
    payoff: dict[str, str],
    unconventional: dict[str, str],
) -> dict[str, str]:
    ratio = parse_first_number(payoff.get("Upside/Downside Ratio"))
    has_expectations = "Options Implied Move" in expectations or "Sell-Side Mean Target Gap" in expectations
    has_payoff = ratio is not None
    shelf_text = unconventional.get("Shelf/Offering Overhang", "")
    has_offering_overhang = bool(shelf_text and not shelf_text.lower().startswith("no "))
    insider_signal = unconventional.get("Insider Cluster Signal", "")
    has_positive_non_consensus = "Potential insider-buying cluster" in insider_signal
    has_material_filing_activity = "Recent 8-K Activity" in unconventional
    if not has_expectations:
        status = "NOT ESTABLISHED"
        reason = "No market-expectations baseline found."
    elif not has_payoff:
        status = "NOT ESTABLISHED"
        reason = "Payoff distribution is incomplete."
    elif ratio < 1.5:
        status = "UNFAVORABLE"
        reason = f"Upside/downside ratio is only {ratio:,.2f}x."
    elif has_offering_overhang:
        status = "TACTICAL ONLY"
        reason = "Payoff skew is visible, but shelf/offering overhang blocks a clean asymmetric claim."
    elif not (has_positive_non_consensus or has_material_filing_activity):
        status = "UNPROVEN"
        reason = "Payoff is visible, but no non-consensus/free-data signal was found."
    elif scores.get("Dilution Risk", 5) >= 8 or scores.get("Exhaustion Risk", 5) >= 8:
        status = "TACTICAL ONLY"
        reason = "Payoff exists, but dilution/exhaustion risk blocks a clean asymmetric claim."
    else:
        status = "POTENTIALLY ASYMMETRIC"
        reason = "Expectations baseline, payoff skew, and at least one non-consensus signal are present."
    return {
        "True Asymmetry Status": status,
        "Reason": reason,
        "Required Standard": "Market baseline + payoff distribution + non-consensus evidence.",
        "Calibration Status": "Uncalibrated: use as a checklist until historical forward-return validation exists.",
    }


def apply_true_asymmetry_guard(verdict: dict[str, str], asymmetry: dict[str, str]) -> dict[str, str]:
    guarded = dict(verdict)
    status = asymmetry.get("True Asymmetry Status", "NOT ESTABLISHED")
    guarded["True Asymmetry"] = status
    if guarded.get("Action") == "PROCEED" and status not in {"POTENTIALLY ASYMMETRIC"}:
        guarded["Verdict"] = "DISCIPLINE PASS / EDGE UNPROVEN"
        guarded["Action"] = "HOLD"
        guarded["Bias"] = "WATCHLIST"
        guarded["Trade Type"] = "DISCIPLINE PASS, NO PROVEN EDGE"
        guarded["Why"] = f"{guarded.get('Why', '')}; true asymmetry {status.lower()}: {asymmetry.get('Reason', '')}".strip("; ")
    return guarded


def build_v2_evidence_engine(
    data: dict[str, Any], v21_data: dict[str, Any] | None = None
) -> tuple[dict[str, int], dict[str, str], list[str]]:
    """Score the setup using deterministic trade-decision heuristics."""
    v21_data = v21_data or {}
    if data.get("error"):
        scores = {name: 5 for name in SCORE_NAMES}
        verdict = {
            "Verdict": "DATA LIMITED",
            "Action": "HOLD",
            "Bias": "WATCHLIST",
            "Trade Type": "NO EDGE",
            "Why": data["error"],
            "Confirm": "Restore market data before acting.",
            "Invalidate": "N/A",
            "Asymmetry": "Cannot assess without market data.",
        }
        return scores, verdict, [data["error"]]

    scores = {
        "Momentum": score_momentum(data),
        "Exhaustion Risk": score_exhaustion_risk(data),
        "Fundamental Trend": score_fundamental_validation(data, v21_data),
        "Valuation Stretch": score_valuation_stretch(data),
        "Event Risk": score_catalyst_proximity(data, v21_data),
        "Dilution Risk": score_dilution_risk(data, v21_data),
        "Ownership Signal": score_ownership_signal(v21_data),
        "Narrative Heat": score_narrative_heat(v21_data),
        "Squeeze Risk": score_squeeze_risk(data),
    }
    scores["Asymmetry"] = score_asymmetry(scores, data)
    flags = build_evidence_flags(scores, data, v21_data)
    return scores, build_trade_verdict(scores, data, flags), flags


def score_momentum(data: dict[str, Any]) -> int:
    score = 5
    r5 = number_or_none(data.get("return_5d")) or 0
    r1 = number_or_none(data.get("return_1m")) or 0
    r3 = number_or_none(data.get("return_3m")) or 0
    volume_ratio = volume_surge_ratio(data)
    if r5 > 8:
        score += 1
    if r1 > 20:
        score += 1
    if r1 > 50:
        score += 1
    if r3 > 50:
        score += 1
    if volume_ratio and volume_ratio > 1.5:
        score += 1
    if r1 < -20:
        score -= 2
    if r3 < -30:
        score -= 2
    return clamp_score(score)


def score_exhaustion_risk(data: dict[str, Any]) -> int:
    score = 3
    r1 = number_or_none(data.get("return_1m")) or 0
    r3 = number_or_none(data.get("return_3m")) or 0
    price = number_or_none(data.get("current_price") or data.get("last_close"))
    high = number_or_none(data.get("fifty_two_week_high"))
    if r1 > 40:
        score += 2
    if r3 > 80:
        score += 2
    if r3 > 150:
        score += 1
    if price and high and high > 0 and price / high > 0.9:
        score += 1
    if number_or_none(data.get("beta")) and number_or_none(data.get("beta")) >= 2:
        score += 1
    return clamp_score(score)


def score_fundamental_validation(data: dict[str, Any], v21_data: dict[str, Any]) -> int:
    score = 5
    revenue_growth = number_or_none(data.get("revenue_growth"))
    gross = number_or_none(data.get("gross_margins"))
    op = number_or_none(data.get("operating_margins"))
    net = number_or_none(data.get("profit_margins"))
    if revenue_growth is not None and revenue_growth > 0.25:
        score += 2
    elif revenue_growth is not None and revenue_growth < 0:
        score -= 2
    if gross is not None and gross > 0.4:
        score += 1
    if op is not None and op < -0.2:
        score -= 2
    elif op is not None and op > 0:
        score += 2
    if net is not None and net < -0.2:
        score -= 1
    elif net is not None and net > 0:
        score += 1
    trends = v21_data.get("financial_trends", {})
    trend_text = " ".join(trends.values()).lower()
    if "quarterly revenue trend" in " ".join(trends.keys()).lower() and "improving" in trends.get("Quarterly Revenue Trend", "").lower():
        score += 1
    if "Quarterly Operating Income Trend" in trends and "deteriorating" in trends["Quarterly Operating Income Trend"].lower():
        score -= 1
    if "Quarterly FCF Trend" in trends and "improving" in trends["Quarterly FCF Trend"].lower():
        score += 1
    if "Quarterly FCF Trend" in trends and "deteriorating" in trends["Quarterly FCF Trend"].lower():
        score -= 1
    if "Share Count Trend" in trends and "deteriorating" in trends["Share Count Trend"].lower():
        score -= 1
    if "Last EPS Surprise" in v21_data.get("earnings_intel", {}) and "-" in v21_data["earnings_intel"]["Last EPS Surprise"]:
        score -= 1
    return clamp_score(score)


def score_valuation_stretch(data: dict[str, Any]) -> int:
    score = 3
    ps = number_or_none(data.get("price_to_sales"))
    if ps is None:
        return 5
    if ps > 5:
        score += 2
    if ps > 10:
        score += 2
    if ps > 20:
        score += 1
    revenue_growth = number_or_none(data.get("revenue_growth"))
    if revenue_growth is not None and revenue_growth > 0.4:
        score -= 1
    if number_or_none(data.get("operating_margins")) and number_or_none(data.get("operating_margins")) < 0:
        score += 1
    return clamp_score(score)


def score_catalyst_proximity(data: dict[str, Any], v21_data: dict[str, Any]) -> int:
    score = 5
    if abs(number_or_none(data.get("return_5d")) or 0) > 8:
        score += 1
    if volume_surge_ratio(data) and volume_surge_ratio(data) > 1.5:
        score += 1
    if number_or_none(data.get("revenue_growth")) and number_or_none(data.get("revenue_growth")) > 0.25:
        score += 1
    if number_or_none(data.get("operating_margins")) and number_or_none(data.get("operating_margins")) < -0.3:
        score += 1
    earnings = v21_data.get("earnings_intel", {})
    days = int_or_none(earnings.get("Days To Earnings"))
    if days is not None:
        if 0 <= days <= 30:
            score += 3
        elif 31 <= days <= 75:
            score += 1
        elif days < 0:
            score -= 1
    if earnings.get("Last EPS Surprise") and "-" in earnings.get("Last EPS Surprise", ""):
        score += 1
    narrative = v21_data.get("narrative_heat", {})
    if narrative.get("Narrative Heat") == "High":
        score += 1
    return clamp_score(score)


def score_dilution_risk(data: dict[str, Any], v21_data: dict[str, Any]) -> int:
    score = 3
    op_margin = number_or_none(data.get("operating_margins"))
    profit_margin = number_or_none(data.get("profit_margins"))
    ps = number_or_none(data.get("price_to_sales"))
    r3 = number_or_none(data.get("return_3m"))
    cash = number_or_none(data.get("total_cash"))
    fcf = number_or_none(data.get("free_cashflow"))
    annual_burn = abs(fcf) if fcf is not None and fcf < 0 else None

    if op_margin is not None and op_margin < -0.2:
        score += 2
    if profit_margin is not None and profit_margin < -0.2:
        score += 1
    if ps is not None and ps > 10:
        score += 1
    if r3 is not None and r3 > 75:
        score += 1
    if cash is not None and annual_burn is not None:
        runway_years = cash / annual_burn if annual_burn else None
        if runway_years and runway_years > 3:
            score -= 2
        elif runway_years and runway_years < 1:
            score += 2
    trends = v21_data.get("financial_trends", {})
    issuance = parse_first_number(trends.get("Recent Capital Stock Issuance"))
    share_trend = trends.get("Share Count Trend", "")
    if issuance and issuance > 10_000_000:
        score += 2
    if "Share Count Trend" in trends and ("improving" in share_trend.lower() or "deteriorating" in share_trend.lower()):
        pct = parse_percent_from_text(share_trend)
        if pct and abs(pct) > 10:
            score += 2
    return clamp_score(score)


def score_ownership_signal(v21_data: dict[str, Any]) -> int:
    ownership = v21_data.get("ownership_intel", {})
    if not ownership:
        return 5

    score = 5
    institutions = parse_percent_from_text(ownership.get("Institutions Held"))
    insiders = parse_percent_from_text(ownership.get("Insiders Held"))
    rec_mix = ownership.get("Analyst Recommendation Mix", "")

    if institutions is not None:
        if institutions >= 40:
            score += 1
        if institutions >= 60:
            score += 1
        if institutions < 15:
            score -= 1
    if insiders is not None:
        if insiders >= 5:
            score += 1
        if insiders >= 15:
            score += 1

    buy_count = parse_named_count(rec_mix, "strongBuy") + parse_named_count(rec_mix, "buy")
    bearish_count = parse_named_count(rec_mix, "sell") + parse_named_count(rec_mix, "strongSell")
    hold_count = parse_named_count(rec_mix, "hold")
    if buy_count > hold_count + bearish_count and buy_count > 0:
        score += 1
    if bearish_count > buy_count and bearish_count > 0:
        score -= 1

    insider_activity = ownership.get("Recent Insider Transaction Types", "").lower()
    if any(term in insider_activity for term in ["sale", "sell", "disposition"]):
        score -= 1
    if any(term in insider_activity for term in ["purchase", "buy", "acquisition"]):
        score += 1
    return clamp_score(score)


def score_narrative_heat(v21_data: dict[str, Any]) -> int:
    narrative = v21_data.get("narrative_heat", {})
    heat = narrative.get("Narrative Heat")
    if heat == "High":
        score = 9
    elif heat == "Medium":
        score = 6
    elif heat == "Low":
        score = 3
    else:
        score = 5
    themes = narrative.get("Narrative Themes", "")
    if "Momentum Coverage" in themes:
        score += 1
    return clamp_score(score)


def score_squeeze_risk(data: dict[str, Any]) -> int:
    score = 3
    short_float = number_or_none(data.get("short_percent_float"))
    if short_float is not None:
        if short_float > 0.1:
            score += 2
        if short_float > 0.2:
            score += 2
    if number_or_none(data.get("return_5d")) and number_or_none(data.get("return_5d")) > 8:
        score += 1
    if volume_surge_ratio(data) and volume_surge_ratio(data) > 2:
        score += 1
    return clamp_score(score)


def score_asymmetry(scores: dict[str, int], data: dict[str, Any]) -> int:
    score = 5
    if scores["Fundamental Trend"] >= 7:
        score += 2
    if scores["Event Risk"] >= 7:
        score += 1
    if scores["Ownership Signal"] >= 7:
        score += 1
    if scores["Narrative Heat"] >= 7 and scores["Momentum"] >= 6:
        score += 1
    if scores["Momentum"] >= 7 and scores["Exhaustion Risk"] <= 6:
        score += 1
    if scores["Valuation Stretch"] >= 7:
        score -= 2
    if scores["Dilution Risk"] >= 7:
        score -= 1
    if scores["Exhaustion Risk"] >= 8:
        score -= 2
    if scores["Narrative Heat"] >= 8 and scores["Exhaustion Risk"] >= 7:
        score -= 1
    return clamp_score(score)


def build_evidence_flags(scores: dict[str, int], data: dict[str, Any], v21_data: dict[str, Any]) -> list[str]:
    flags: list[str] = []
    r1 = number_or_none(data.get("return_1m"))
    r3 = number_or_none(data.get("return_3m"))
    if r1 is not None:
        flags.append(f"1M return {r1:,.1f}%")
    if r3 is not None:
        flags.append(f"3M return {r3:,.1f}%")
    if scores["Momentum"] >= 8:
        flags.append("strong tape momentum")
    if scores["Exhaustion Risk"] >= 8:
        flags.append("move is statistically extended")
    if scores["Valuation Stretch"] >= 7:
        flags.append("valuation already prices strong execution")
    if scores["Fundamental Trend"] <= 4:
        flags.append("fundamentals do not yet validate the tape")
    if scores["Event Risk"] >= 7:
        flags.append("near-term event risk can reprice the setup")
    if scores["Dilution Risk"] >= 7:
        flags.append("losses plus rally create dilution/financing risk")
    if scores["Ownership Signal"] >= 7:
        flags.append("ownership/analyst signal is supportive")
    elif scores["Ownership Signal"] <= 4:
        flags.append("ownership/analyst signal is weak or conflicted")
    if scores["Narrative Heat"] >= 8:
        flags.append("narrative heat is high; watch story-stock crowding")
    runway = cash_runway_years(data)
    if runway is not None and runway > 3:
        flags.append(f"cash runway appears strong at roughly {runway:,.1f} years of current FCF burn")
    elif runway is not None and runway < 1:
        flags.append(f"cash runway appears tight at roughly {runway:,.1f} years of current FCF burn")
    if scores["Squeeze Risk"] >= 7:
        flags.append("short-interest setup can amplify moves")
    if scores["Asymmetry"] <= 4:
        flags.append("discipline skew unfavorable until next confirming data point")
    elif scores["Asymmetry"] >= 7:
        flags.append("discipline skew favorable if catalyst confirms")

    earnings = v21_data.get("earnings_intel", {})
    days = earnings.get("Days To Earnings")
    if days not in {None, "", "Unknown"}:
        flags.append(f"earnings timing: {days} days")
    if earnings.get("Last EPS Surprise"):
        flags.append(f"last EPS surprise: {earnings['Last EPS Surprise']}")

    trends = v21_data.get("financial_trends", {})
    for key in ["Quarterly Revenue Trend", "Quarterly Operating Income Trend", "Quarterly FCF Trend", "Share Count Trend"]:
        if trends.get(key):
            flags.append(f"{key.lower()}: {trends[key]}")

    ownership = v21_data.get("ownership_intel", {})
    if ownership.get("Institutions Held"):
        flags.append(f"institutional ownership: {ownership['Institutions Held']}")
    if ownership.get("Recent Insider Transaction Types"):
        flags.append(f"recent insider actions: {ownership['Recent Insider Transaction Types']}")

    narrative = v21_data.get("narrative_heat", {})
    if narrative.get("Narrative Heat"):
        flags.append(f"narrative heat: {narrative['Narrative Heat']} ({narrative.get('Narrative Themes', 'themes unclear')})")
    return flags


def cash_runway_years(data: dict[str, Any]) -> float | None:
    cash = number_or_none(data.get("total_cash"))
    fcf = number_or_none(data.get("free_cashflow"))
    if cash is None or fcf is None or fcf >= 0:
        return None
    burn = abs(fcf)
    if burn == 0:
        return None
    return cash / burn


def build_trade_verdict(scores: dict[str, int], data: dict[str, Any], flags: list[str]) -> dict[str, str]:
    if (
        scores["Asymmetry"] >= 7
        and scores["Exhaustion Risk"] <= 6
        and scores["Valuation Stretch"] <= 6
        and scores["Fundamental Trend"] >= 6
    ):
        verdict = "CHASEABLE"
        action = "PROCEED"
        bias = "LONG"
        trade_type = "MOMENTUM WITH VALIDATION"
    elif scores["Exhaustion Risk"] >= 8 and scores["Momentum"] >= 7:
        verdict = "WAIT FOR PULLBACK"
        action = "HOLD"
        bias = "WATCHLIST"
        trade_type = "EXTENDED MOMENTUM"
    elif scores["Valuation Stretch"] >= 8 and scores["Fundamental Trend"] <= 5:
        verdict = "AVOID CHASING"
        action = "AVOID"
        bias = "AVOID"
        trade_type = "PRICED FOR PERFECTION"
    elif scores["Event Risk"] >= 7 and scores["Asymmetry"] >= 4:
        verdict = "EVENT WATCH"
        action = "HOLD"
        bias = "WATCHLIST"
        trade_type = "WAIT FOR CATALYST"
    elif scores["Narrative Heat"] >= 8 and scores["Fundamental Trend"] <= 6:
        verdict = "NARRATIVE WATCH"
        action = "HOLD"
        bias = "WATCHLIST"
        trade_type = "STORY STOCK WITHOUT FULL VALIDATION"
    else:
        verdict = "WATCHLIST"
        action = "HOLD"
        bias = "WATCHLIST"
        trade_type = "NO CLEAR EDGE"

    return {
        "Verdict": verdict,
        "Action": action,
        "Bias": bias,
        "Trade Type": trade_type,
        "Why": "; ".join(flags[:3]) if flags else "No decisive deterministic edge.",
        "Confirm": build_confirm_trigger(data),
        "Invalidate": build_invalidation_trigger(data),
        "Asymmetry": describe_asymmetry(scores),
    }


def build_confirm_trigger(data: dict[str, Any]) -> str:
    if number_or_none(data.get("operating_margins")) and number_or_none(data.get("operating_margins")) < 0:
        return "Revenue growth continues while operating margin loss narrows."
    if number_or_none(data.get("return_1m")) and number_or_none(data.get("return_1m")) > 30:
        return "Pullback holds above prior breakout and volume stays elevated."
    return "Next catalyst confirms revenue, margin, or guidance improvement."


def build_invalidation_trigger(data: dict[str, Any]) -> str:
    if number_or_none(data.get("price_to_sales")) and number_or_none(data.get("price_to_sales")) > 10:
        return "Growth or margin data disappoints while valuation remains premium."
    if number_or_none(data.get("return_1m")) and number_or_none(data.get("return_1m")) > 30:
        return "Momentum breaks and volume fades after an extended move."
    return "Key KPI deteriorates at the next update."


def describe_asymmetry(scores: dict[str, int]) -> str:
    if scores["Asymmetry"] >= 7:
        return "Discipline skew positive if the next catalyst confirms fundamentals before valuation stretches further."
    if scores["Asymmetry"] <= 4:
        return "Discipline skew negative: price/expectations look ahead of confirmed fundamentals."
    return "Mixed discipline skew: catalyst can move the stock, but confirmation is needed."


def volume_surge_ratio(data: dict[str, Any]) -> float | None:
    avg = number_or_none(data.get("average_volume"))
    avg20 = number_or_none(data.get("avg_volume_20d"))
    if not avg or not avg20:
        return None
    return avg20 / avg


def format_engine_evidence(
    scores: dict[str, int],
    verdict: dict[str, str],
    flags: list[str],
    v21_data: dict[str, Any] | None = None,
    expectations_baseline: dict[str, str] | None = None,
    payoff_distribution: dict[str, str] | None = None,
    unconventional_signals: dict[str, str] | None = None,
    asymmetry_assessment: dict[str, str] | None = None,
) -> str:
    score_lines = "\n".join(f"{name}: {score}/10" for name, score in scores.items())
    verdict_lines = "\n".join(f"{name}: {value}" for name, value in verdict.items())
    flag_lines = "\n".join(f"- {flag}" for flag in flags) if flags else "- No decisive deterministic flags."
    v21_lines: list[str] = []
    if v21_data:
        sections = {
            "EARNINGS INTELLIGENCE": v21_data.get("earnings_intel", {}),
            "QUARTERLY FINANCIAL TRENDS": v21_data.get("financial_trends", {}),
            "OWNERSHIP / INSIDER SIGNALS": v21_data.get("ownership_intel", {}),
            "NEWS / NARRATIVE HEAT": v21_data.get("narrative_heat", {}),
        }
        for title, values in sections.items():
            if values:
                v21_lines.append(title)
                v21_lines.extend(f"{key}: {value}" for key, value in values.items())
                v21_lines.append("")
    v3_sections = {
        "MARKET EXPECTATIONS BASELINE": expectations_baseline or {},
        "PAYOFF DISTRIBUTION": payoff_distribution or {},
        "UNCONVENTIONAL / LESS-COMMON SIGNALS": unconventional_signals or {},
        "TRUE ASYMMETRY ASSESSMENT": asymmetry_assessment or {},
    }
    v3_lines: list[str] = []
    for title, values in v3_sections.items():
        if values:
            v3_lines.append(title)
            v3_lines.extend(f"{key}: {value}" for key, value in values.items())
            v3_lines.append("")
    return f"""DETERMINISTIC ENGINE SCORES
{score_lines}

DETERMINISTIC TRADE VERDICT
{verdict_lines}

EVIDENCE FLAGS
{flag_lines}

V2.1 DATA COMPLETION LAYER
{chr(10).join(v21_lines).strip() or "No additional V2.1 data available."}

V3 EDGE / ASYMMETRY LAYER
{chr(10).join(v3_lines).strip() or "No V3 edge diagnostics available."}"""


def build_quick_read(data: dict[str, Any]) -> str:
    flags: list[str] = []
    if number_or_none(data.get("return_1m")) and number_or_none(data.get("return_1m")) >= 50:
        flags.append("extended 1M move")
    if number_or_none(data.get("return_3m")) and number_or_none(data.get("return_3m")) >= 100:
        flags.append("major 3M momentum")
    if number_or_none(data.get("price_to_sales")) and number_or_none(data.get("price_to_sales")) >= 10:
        flags.append("premium sales multiple")
    if number_or_none(data.get("operating_margins")) and number_or_none(data.get("operating_margins")) < 0:
        flags.append("operating losses")
    if number_or_none(data.get("beta")) and number_or_none(data.get("beta")) >= 2:
        flags.append("high beta")
    if not flags:
        return "No obvious market-data red flags from the available snapshot."
    return ", ".join(flags)


def format_percent(value: Any) -> str:
    parsed = number_or_none(value)
    return "N/A" if parsed is None else f"{parsed:,.1f}%"


def format_percent_ratio(value: Any) -> str:
    parsed = number_or_none(value)
    return "N/A" if parsed is None else f"{parsed * 100:,.1f}%"


def number_or_none(value: Any) -> float | None:
    try:
        if value is None:
            return None
        if isinstance(value, str) and value.strip() in {"", "N/A", "Unknown"}:
            return None
        parsed = float(value)
        if parsed != parsed:
            return None
        return parsed
    except (TypeError, ValueError):
        return None


def first_valid_number(values: list[Any]) -> float | None:
    for value in values:
        parsed = number_or_none(value)
        if parsed is not None:
            return parsed
    return None


def percent_change(target: float, base: float) -> float:
    if base == 0:
        return 0.0
    return (target / base - 1) * 100


def format_signed_percent(value: float | None) -> str:
    if value is None:
        return "N/A"
    return f"{value:+,.1f}%"


def int_or_none(value: Any) -> int | None:
    try:
        if value in {None, "", "Unknown", "N/A"}:
            return None
        return int(float(str(value).replace(",", "").replace("%", "").strip()))
    except (TypeError, ValueError):
        return None


def parse_first_number(text: str | None) -> float | None:
    if not text:
        return None
    match = re.search(r"-?\d[\d,]*(?:\.\d+)?", str(text))
    if not match:
        return None
    return float(match.group(0).replace(",", ""))


def parse_percent_from_text(text: str | None) -> float | None:
    if not text:
        return None
    match = re.search(r"(-?\d+(?:\.\d+)?)\s*%", str(text))
    if not match:
        return None
    return float(match.group(1))


def parse_named_count(text: str, name: str) -> int:
    match = re.search(rf"\b{re.escape(name)}\s+(\d+)", text, flags=re.IGNORECASE)
    return int(match.group(1)) if match else 0


def format_market_value(value: Any) -> str:
    if isinstance(value, float):
        return f"{value:,.2f}"
    if isinstance(value, int):
        return f"{value:,}"
    return str(value)


def get_cache_dir() -> Path:
    configured = str(get_secret("CACHE_DIR", "cache")).strip() or "cache"
    cache_dir = Path(configured)
    if not cache_dir.is_absolute():
        cache_dir = Path(__file__).resolve().parent / cache_dir
    cache_dir.mkdir(parents=True, exist_ok=True)
    return cache_dir


def get_market_cache_dir() -> Path:
    cache_dir = get_cache_dir() / "market_data"
    cache_dir.mkdir(parents=True, exist_ok=True)
    return cache_dir


def market_cache_path_for_ticker(ticker: str) -> Path:
    return get_market_cache_dir() / f"{ticker}.json"


def load_market_data_cache(ticker: str) -> dict[str, Any] | None:
    path = market_cache_path_for_ticker(ticker)
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return None


def save_market_data_cache(ticker: str, data: dict[str, Any]) -> None:
    if data.get("error"):
        return
    market_cache_path_for_ticker(ticker).write_text(
        json.dumps(data, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )


def cache_path_for_ticker(ticker: str) -> Path:
    return get_cache_dir() / f"{ticker}.json"


def load_cache(ticker: str) -> DDResult | None:
    path = cache_path_for_ticker(ticker)
    if not path.exists():
        return None
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        if "decision_card" not in data:
            recommendation = data.get("recommendation", "HOLD")
            reason = data.get("recommendation_reason", "Legacy cached result.")
            data["decision_card"] = extract_decision_card(data.get("step_3", ""), recommendation, reason)
        if "market_snapshot" not in data:
            data["market_snapshot"] = {}
        if "trade_verdict" not in data:
            data["trade_verdict"] = {
                "Verdict": data.get("recommendation", "HOLD"),
                "Action": data.get("recommendation", "HOLD"),
                "Bias": "WATCHLIST",
                "Trade Type": "LEGACY CACHE",
                "Why": data.get("recommendation_reason", "Legacy cached result."),
                "Confirm": "Refresh analysis for V2 evidence engine.",
                "Invalidate": "Refresh analysis for V2 evidence engine.",
                "Asymmetry": "Legacy cached result.",
            }
        if "evidence_flags" not in data:
            data["evidence_flags"] = []
        for key in [
            "earnings_intel",
            "financial_trends",
            "ownership_intel",
            "narrative_heat",
            "expectations_baseline",
            "payoff_distribution",
            "unconventional_signals",
            "asymmetry_assessment",
        ]:
            data.setdefault(key, {})
        return DDResult(**data)
    except (json.JSONDecodeError, TypeError, OSError):
        return None


def save_cache(result: DDResult) -> None:
    cache_path_for_ticker(result.ticker).write_text(
        json.dumps(asdict(result), indent=2, ensure_ascii=False),
        encoding="utf-8",
    )


def cached_on_label(cached_at: str) -> str:
    try:
        parsed = datetime.fromisoformat(cached_at.replace("Z", "+00:00"))
        return parsed.strftime("%b %d, %Y %H:%M UTC")
    except ValueError:
        return cached_at


def call_openai(client: OpenAI, prompt: str, model: str, *, step_name: str) -> str:
    try:
        response = client.responses.create(
            model=model,
            instructions=SYSTEM_PROMPT,
            input=prompt,
            max_output_tokens=2600,
        )
        text = getattr(response, "output_text", None)
        if text:
            return clean_model_text(text)
        return clean_model_text(extract_responses_text(response))
    except AttributeError:
        return call_openai_chat_completion(client, prompt, model)
    except RateLimitError as exc:
        raise UserFacingError(
            f"{step_name} hit an OpenAI rate limit. Wait a moment, then rerun or use the cached result if available."
        ) from exc
    except AuthenticationError as exc:
        raise UserFacingError("OpenAI authentication failed. Check OPENAI_API_KEY in .streamlit/secrets.toml.") from exc
    except (APITimeoutError, APIConnectionError) as exc:
        raise UserFacingError(
            f"{step_name} could not complete because the OpenAI request timed out or lost connection."
        ) from exc
    except APIError as exc:
        raise UserFacingError(f"{step_name} failed because OpenAI returned an API error: {exc.message}") from exc


def call_openai_chat_completion(client: OpenAI, prompt: str, model: str) -> str:
    completion = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
        ],
        max_tokens=2600,
    )
    return clean_model_text(completion.choices[0].message.content)


def clean_model_text(text: str | None) -> str:
    """Repair common mojibake sequences that can appear in exported model text."""
    if not text:
        return ""
    if any(marker in text for marker in ["Ã", "Â", "â"]):
        try:
            repaired = text.encode("latin1").decode("utf-8")
            if repaired.count("\ufffd") <= text.count("\ufffd"):
                text = repaired
        except UnicodeError:
            pass
    replacements = {
        "\u00e2\u20ac\u2122": "'",
        "\u00e2\u20ac\u02dc": "'",
        "\u00e2\u20ac\u0153": '"',
        "\u00e2\u20ac\ufffd": '"',
        "\u00e2\u20ac\u009d": '"',
        "\u00e2\u20ac\u201c": "-",
        "\u00e2\u20ac\u201d": "-",
        "\u00e2\u20ac\u00a6": "...",
        "\u00e2\u201a\u00ac": "EUR",
        "\u00c2\u00a3": "GBP",
        "\u00c2\u00a5": "JPY",
        "\u00c2\u00ae": "(R)",
        "\u00c2\u00a9": "(C)",
        "\u00c2": "",
    }
    cleaned = text
    for broken, fixed in replacements.items():
        cleaned = cleaned.replace(broken, fixed)
    cleaned = apply_known_fact_guards(cleaned)
    return cleaned.strip()


def apply_known_fact_guards(text: str) -> str:
    lower = text.lower()
    if "ouster" in lower and "velodyne" in lower and "ouster completed its merger with velodyne" not in lower:
        text += (
            "\n\nData quality note: Ouster completed its merger with Velodyne in 2023, "
            "so Velodyne should not be treated as a current standalone competitor."
        )
    return text


def extract_responses_text(response: Any) -> str:
    chunks: list[str] = []
    for item in getattr(response, "output", []) or []:
        for content in getattr(item, "content", []) or []:
            text = getattr(content, "text", None)
            if text:
                chunks.append(text)
    return "\n".join(chunks)


def build_step_1_prompt(ticker: str, market_data: str, engine_evidence: str) -> str:
    return f"""
Research {ticker} for a fast trading decision.

Use this market data pack as your factual starting point. If a value is missing, say it is missing.
Do not replace missing values with generic guesses.

{market_data}

Use this deterministic evidence engine as the trading frame. Treat the V2.1 scores as discipline checks,
not calibrated alpha. Treat the V3 edge/asymmetry layer as the only place where asymmetry can be discussed.
Do not ignore it or replace it with generic company commentary.

{engine_evidence}

Return a concise decision briefing, not a company story:
1. WHAT MOVES THE STOCK: top 3 drivers in the next 1-90 days.
2. CURRENT SETUP: recent revenue/growth/profitability, valuation context, liquidity/volatility if known.
3. BUSINESS ONLY IF TRADE-RELEVANT: revenue mix, installed base/contract model, recurring vs one-time revenue.
4. COMPETITION: top 3 competitors and the one competitive fact most likely to matter to the stock.
   Do not list acquired, merged, delisted, or former companies as current competitors. If current competitor
   status is uncertain, say "competitor list needs verification" instead of guessing.
5. SOURCE QUALITY: cite the market data source above and list missing data that would change the decision.
6. CONFIDENCE: score each section 1-10, but explicitly say whether confidence is data-grounded or heuristic.

Format: 350-500 words. Use bullets. No background filler.
""".strip()


def build_step_2_prompt(step_1: str, engine_evidence: str) -> str:
    return f"""
Based on Step 1: {step_1}

Keep this deterministic evidence engine in view:
{engine_evidence}

Convert the company facts into investable/tradable economics:
1. ECONOMIC ENGINE: what KPI actually drives the stock for this sector? Examples: orders/bookings, ARPU,
   churn, gross margin, utilization, commodity price exposure, deliveries, backlog, licensing, FCF.
2. CUSTOMER/UNIT ECONOMICS: switching cost, replacement/upgrade cycle, payback, recurring revenue,
   margin structure, or sector-equivalent economics. If a metric is irrelevant for the sector, say N/A.
3. CATALYST MAP: next 3 likely catalysts, expected timing, and whether each is bullish/bearish/ambiguous.
4. STALL MAP: what would make the thesis fail fast?
5. EXPECTATIONS GAP: what does the options/target/payoff layer imply the market already prices?
6. FAST CHECK: what single data point should a trader verify before acting?

Format: 350-500 words. Use bullets/table style. Show confidence. No repeated company overview.
""".strip()


def build_step_3_prompt(step_1: str, step_2: str, engine_evidence: str) -> str:
    return f"""
Based on Steps 1-2: {step_1}

{step_2}

The deterministic evidence engine is the primary decision layer. Do not invent asymmetry if the V3
True Asymmetry Status is NOT ESTABLISHED, UNPROVEN, UNFAVORABLE, or TACTICAL ONLY:
{engine_evidence}

Produce a decision-first trading dashboard:
1. DECISION: Use the deterministic trade verdict exactly. Distinguish the verdict from action:
   for example, WAIT FOR PULLBACK can map to Action: HOLD and Bias: WATCHLIST.
2. WHY NOW: one sentence. If no near-term edge, say so.
3. BULL CASE: 3 drivers, what goes right, probability (%), trigger to confirm.
4. BEAR CASE: 3 risks, what breaks, probability (%), trigger to invalidate.
5. ASYMMETRY: use the V3 True Asymmetry Status. If it is not established, say exactly why.
6. TRADE CARD: horizon, top catalyst, invalidation trigger, key metric to monitor, confidence 1-10.

End with this exact block:
TRADING_DECISION_CARD
Engine Verdict: [use deterministic Verdict exactly]
Trade Type: [use deterministic Trade Type exactly]
Action: PROCEED/HOLD/AVOID
Bias: LONG/WATCHLIST/AVOID
Horizon: [timeframe]
Why Now: [one sentence]
Top Catalyst: [one sentence]
Invalidation: [one sentence]
Key Metric: [one sentence]
Bull Probability: [%]
Bear Probability: [%]
Confidence: [1-10]

Format: 450-650 words. No generic background. Bull/bear probabilities are heuristic unless calibrated data is present.
""".strip()


def run_analysis(ticker: str, force_refresh: bool = False) -> DDResult:
    cached = load_cache(ticker)
    if cached and not force_refresh:
        return cached

    client = get_openai_client()
    model = get_model()
    market_data = fetch_market_data(ticker)
    v21_data = fetch_v21_data(ticker) if not market_data.get("error") else {}
    market_data_text = format_market_data(market_data)
    market_snapshot = build_market_snapshot(market_data)
    engine_scores, trade_verdict, evidence_flags = build_v2_evidence_engine(market_data, v21_data)
    expectations_baseline, payoff_distribution, unconventional_signals, asymmetry_assessment = build_v3_edge_layer(
        ticker,
        market_data,
        v21_data,
        engine_scores,
    )
    trade_verdict = apply_true_asymmetry_guard(trade_verdict, asymmetry_assessment)
    engine_evidence = format_engine_evidence(
        engine_scores,
        trade_verdict,
        evidence_flags,
        v21_data,
        expectations_baseline,
        payoff_distribution,
        unconventional_signals,
        asymmetry_assessment,
    )
    if market_data.get("error"):
        st.warning(market_data["error"])
        raise UserFacingError(
            "Market data is temporarily unavailable and no cached fallback exists for this ticker. "
            "Wait a few minutes, then rerun with Refresh cache enabled."
        )
    elif market_data.get("is_stale"):
        st.warning(f"Using stale market data fallback. {market_data.get('stale_reason', '')}")
    elif market_data.get("fallback_reason"):
        st.warning("Using lightweight market-data fallback because the richer Yahoo Finance endpoint was unavailable.")
    else:
        st.success(f"Loaded market data from {market_data['source']}.")

    step_1_box = st.empty()
    step_2_box = st.empty()
    step_3_box = st.empty()

    with step_1_box.container():
        with st.spinner("Step 1: building fast setup and stock-mover map..."):
            step_1 = call_openai(
                client,
                build_step_1_prompt(ticker, market_data_text, engine_evidence),
                model,
                step_name="Step 1",
            )
        render_step("Step 1 - Fast Setup", step_1)

    with step_2_box.container():
        with st.spinner("Step 2: mapping economic engine, catalysts, and stalls..."):
            step_2 = call_openai(client, build_step_2_prompt(step_1, engine_evidence), model, step_name="Step 2")
        render_step("Step 2 - Economic Engine", step_2)

    with step_3_box.container():
        with st.spinner("Step 3: translating evidence into decision card..."):
            step_3 = call_openai(
                client,
                build_step_3_prompt(step_1, step_2, engine_evidence),
                model,
                step_name="Step 3",
            )
        render_step("Step 3 - Decision Support", step_3)

    recommendation = trade_verdict["Action"]
    reason = trade_verdict["Why"]
    decision_card = extract_decision_card(step_3, recommendation, reason)
    decision_card["Engine Verdict"] = trade_verdict["Verdict"]
    decision_card["Trade Type"] = trade_verdict["Trade Type"]
    decision_card["Action"] = trade_verdict["Action"]
    decision_card["Bias"] = trade_verdict["Bias"]
    result = DDResult(
        ticker=ticker,
        company_name=extract_company_name(step_1),
        cached_at=datetime.now(timezone.utc).isoformat(),
        model=model,
        step_1=step_1,
        step_2=step_2,
        step_3=step_3,
        dashboard_scores=engine_scores,
        recommendation=recommendation,
        recommendation_reason=reason,
        decision_card=decision_card,
        market_snapshot=market_snapshot,
        trade_verdict=trade_verdict,
        evidence_flags=evidence_flags,
        earnings_intel=v21_data.get("earnings_intel", {}),
        financial_trends=v21_data.get("financial_trends", {}),
        ownership_intel=v21_data.get("ownership_intel", {}),
        narrative_heat=v21_data.get("narrative_heat", {}),
        expectations_baseline=expectations_baseline,
        payoff_distribution=payoff_distribution,
        unconventional_signals=unconventional_signals,
        asymmetry_assessment=asymmetry_assessment,
    )
    save_cache(result)
    return result


def extract_company_name(step_1: str) -> str | None:
    patterns = [
        r"(?:Company|Issuer|Business)\s*:\s*([A-Z][^\n|]{2,80})",
        r"^#?\s*([A-Z][A-Za-z0-9&.,'\- ]{2,80})\s+\(",
    ]
    for pattern in patterns:
        match = re.search(pattern, step_1, flags=re.MULTILINE)
        if match:
            return match.group(1).strip(" .")
    return None


def extract_dashboard_scores(step_1: str, step_2: str, step_3: str) -> dict[str, int]:
    combined = "\n".join([step_1, step_2, step_3])
    scores = {
        "Setup Quality": score_setup_quality(combined),
        "Catalyst Clarity": score_catalyst_clarity(combined),
        "Risk/Reward": score_risk_reward(combined, step_3),
        "Decision Confidence": find_score(
            combined,
            [r"confidence\D{0,40}(\d{1,2})\s*/?\s*10", r"confidence\D{0,40}(\d{1,2})\b"],
            default=6,
        ),
    }
    return {name: clamp_score(score) for name, score in scores.items()}


def find_score(text: str, patterns: list[str], default: int) -> int:
    for pattern in patterns:
        match = re.search(pattern, text, flags=re.IGNORECASE)
        if match:
            return clamp_score(int(match.group(1)))
    return default


def score_setup_quality(text: str) -> int:
    score = 6
    positive_terms = ["setup", "valuation", "liquidity", "volatility", "growth", "margin", "revenue", "profitability"]
    uncertainty_terms = ["unknown", "not disclosed", "insufficient", "missing", "limited data"]
    score += min(2, sum(1 for term in positive_terms if term in text.lower()) // 2)
    score -= min(3, sum(1 for term in uncertainty_terms if term in text.lower()))
    return clamp_score(score)


def score_catalyst_clarity(text: str) -> int:
    score = 6
    lower = text.lower()
    positives = ["catalyst", "timing", "earnings", "guidance", "approval", "orders", "backlog", "launch"]
    negatives = ["no near-term", "unclear", "not enough", "missing", "stale"]
    score += min(3, sum(1 for term in positives if term in lower) // 2)
    score -= min(3, sum(1 for term in negatives if term in lower))
    return clamp_score(score)


def score_risk_reward(text: str, step_3: str) -> int:
    bull = extract_probability(text, "bull")
    bear = extract_probability(text, "bear")
    base = find_score(step_3, [r"risk/reward\D{0,40}(\d{1,2})\s*/?\s*10"], default=6)
    if bull is None or bear is None:
        return base
    spread = bull - bear
    if spread >= 25:
        return clamp_score(base + 2)
    if spread >= 10:
        return clamp_score(base + 1)
    if spread <= -20:
        return clamp_score(base - 3)
    if spread <= -5:
        return clamp_score(base - 1)
    return base


def extract_decision_card(step_3: str, recommendation: str, reason: str) -> dict[str, str]:
    return {
        "Action": extract_card_field(step_3, "Action") or recommendation,
        "Bias": extract_card_field(step_3, "Bias") or bias_from_recommendation(recommendation),
        "Horizon": extract_card_field(step_3, "Horizon") or "Not specified",
        "Why Now": extract_card_field(step_3, "Why Now") or reason,
        "Top Catalyst": extract_card_field(step_3, "Top Catalyst") or "Not specified",
        "Invalidation": extract_card_field(step_3, "Invalidation") or "Not specified",
        "Key Metric": extract_card_field(step_3, "Key Metric") or "Not specified",
        "Bull Probability": extract_card_field(step_3, "Bull Probability") or format_probability(extract_probability(step_3, "bull")),
        "Bear Probability": extract_card_field(step_3, "Bear Probability") or format_probability(extract_probability(step_3, "bear")),
        "Confidence": extract_card_field(step_3, "Confidence") or "Not specified",
    }


def extract_card_field(text: str, field: str) -> str | None:
    match = re.search(rf"^\s*{re.escape(field)}\s*:\s*(.+?)\s*$", text, flags=re.IGNORECASE | re.MULTILINE)
    if not match:
        return None
    return match.group(1).strip()


def bias_from_recommendation(recommendation: str) -> str:
    if recommendation == "PROCEED":
        return "LONG"
    if recommendation == "AVOID":
        return "AVOID"
    return "WATCHLIST"


def format_probability(value: int | None) -> str:
    return f"{value}%" if value is not None else "Not specified"


def extract_probability(text: str, label: str) -> int | None:
    section = extract_labeled_section(text, label)
    if not section:
        return None
    match = re.search(r"(?:probability|prob\.)\D{0,60}(\d{1,3})\s*%", section, flags=re.IGNORECASE)
    if not match:
        match = re.search(r"(\d{1,3})\s*%", section)
    if not match:
        return None
    return max(0, min(100, int(match.group(1))))


def extract_labeled_section(text: str, label: str) -> str | None:
    label_match = re.search(rf"\b{label}\b", text, flags=re.IGNORECASE)
    if not label_match:
        return None
    section = text[label_match.start() :]
    opposite = "bear" if label.lower() == "bull" else "bull"
    next_match = re.search(rf"\n\s*(?:#+\s*)?\**{opposite}\b", section[1:], flags=re.IGNORECASE)
    if next_match:
        section = section[: next_match.start() + 1]
    return section


def clamp_score(value: int) -> int:
    return max(1, min(10, int(value)))


def build_recommendation(scores: dict[str, int], step_3: str) -> tuple[str, str]:
    average_score = sum(scores.values()) / len(scores)
    bull = extract_probability(step_3, "bull")
    bear = extract_probability(step_3, "bear")

    if bull is not None and bear is not None:
        if average_score >= 7.25 and bull >= bear + 15:
            return "PROCEED", f"Score average {average_score:.1f}/10 with bull probability {bull}% vs bear {bear}%."
        if average_score <= 4.75 or bear >= bull + 15:
            return "AVOID", f"Score average {average_score:.1f}/10 with bear probability {bear}% vs bull {bull}%."
        return "HOLD", f"Score average {average_score:.1f}/10 with a mixed bull/bear spread ({bull}% vs {bear}%)."

    if average_score >= 7.5:
        return "PROCEED", f"Score average {average_score:.1f}/10; probability parsing was inconclusive."
    if average_score <= 4.75:
        return "AVOID", f"Score average {average_score:.1f}/10; probability parsing was inconclusive."
    return "HOLD", f"Score average {average_score:.1f}/10; probability parsing was inconclusive."


def export_json(result: DDResult) -> str:
    return json.dumps(asdict(result), indent=2, ensure_ascii=False)


def export_markdown(result: DDResult) -> str:
    scores = "\n".join(
        f"- **{display_score_name(name)}:** {score}/10" for name, score in result.dashboard_scores.items()
    )
    decision = "\n".join(f"- **{name}:** {value}" for name, value in result.decision_card.items())
    market = "\n".join(f"- **{name}:** {value}" for name, value in result.market_snapshot.items())
    verdict = "\n".join(f"- **{name}:** {value}" for name, value in result.trade_verdict.items())
    flags = "\n".join(f"- {flag}" for flag in result.evidence_flags)
    earnings = markdown_dict(result.earnings_intel)
    trends = markdown_dict(result.financial_trends)
    ownership = markdown_dict(result.ownership_intel)
    narrative = markdown_dict(result.narrative_heat)
    expectations = markdown_dict(result.expectations_baseline)
    payoff = markdown_dict(result.payoff_distribution)
    unconventional = markdown_dict(result.unconventional_signals)
    true_asymmetry = markdown_dict(result.asymmetry_assessment)
    company = f"\n**Company:** {result.company_name}" if result.company_name else ""
    return f"""# {APP_TITLE}

**Ticker:** {result.ticker}{company}
**Model:** {result.model}
**Cached on:** {cached_on_label(result.cached_at)}
**Recommendation:** {result.recommendation}
**Reason:** {result.recommendation_reason}

## Dashboard

{scores}

## Market Snapshot

{market}

## V2.1 Trade Verdict

{verdict}

## Evidence Flags

{flags}

## V3 Market Expectations Baseline

{expectations}

## V3 Payoff Distribution

{payoff}

## V3 Unconventional Signals

{unconventional}

## V3 True Asymmetry Assessment

{true_asymmetry}

## Earnings Intelligence

{earnings}

## Quarterly Financial Trends

{trends}

## Ownership / Insider Signals

{ownership}

## News / Narrative Heat

{narrative}

## Decision Card

{decision}

## Step 1 - Fast Setup

{result.step_1}

## Step 2 - Economic Engine

{result.step_2}

## Step 3 - Decision Support

{result.step_3}
"""


def markdown_dict(values: dict[str, str]) -> str:
    if not values:
        return "- No data available."
    return "\n".join(f"- **{name}:** {value}" for name, value in values.items())


def render_dashboard(result: DDResult) -> None:
    st.subheader("V3 Trade Verdict")
    verdict = result.trade_verdict
    verdict_cols = st.columns([1.2, 1, 1.2, 1.6])
    verdict_cols[0].metric("Verdict", verdict.get("Verdict", result.recommendation))
    verdict_cols[1].metric("Action", verdict.get("Action", result.recommendation))
    verdict_cols[2].metric("Bias", verdict.get("Bias", "WATCHLIST"))
    verdict_cols[3].metric("Trade Type", verdict.get("Trade Type", "N/A"))
    with st.container(border=True):
        st.markdown(f"**Why:** {verdict.get('Why', result.recommendation_reason)}")
        st.markdown(f"**Confirm:** {verdict.get('Confirm', 'Not specified')}")
        st.markdown(f"**Invalidate:** {verdict.get('Invalidate', 'Not specified')}")
        st.markdown(f"**Discipline skew:** {verdict.get('Asymmetry', 'Not specified')}")
        st.markdown(f"**True asymmetry:** {verdict.get('True Asymmetry', 'Not assessed')}")
        if result.evidence_flags:
            st.markdown("**Evidence flags:** " + " | ".join(result.evidence_flags))

    if result.market_snapshot:
        st.subheader("Market Snapshot")
        snap_cols = st.columns(min(4, max(1, len(result.market_snapshot))))
        for index, (label, value) in enumerate(result.market_snapshot.items()):
            snap_cols[index % len(snap_cols)].metric(label, value)

    render_info_grid("Earnings Intelligence", result.earnings_intel)
    render_info_grid("Quarterly Financial Trends", result.financial_trends)
    render_info_grid("Ownership / Insider Signals", result.ownership_intel)
    render_info_grid("News / Narrative Heat", result.narrative_heat)
    render_info_grid("V3 Market Expectations Baseline", result.expectations_baseline)
    render_info_grid("V3 Payoff Distribution", result.payoff_distribution)
    render_info_grid("V3 Unconventional Signals", result.unconventional_signals)
    render_info_grid("V3 True Asymmetry Assessment", result.asymmetry_assessment)

    st.subheader("Decision Card")
    card = result.decision_card
    top_cols = st.columns([1, 1, 2, 2])
    top_cols[0].metric("Action", card.get("Action", result.recommendation))
    top_cols[1].metric("Bias", card.get("Bias", "WATCHLIST"))
    top_cols[2].metric("Horizon", card.get("Horizon", "Not specified"))
    top_cols[3].metric("Confidence", card.get("Confidence", "Not specified"))

    with st.container(border=True):
        st.markdown(f"**Why now:** {card.get('Why Now', result.recommendation_reason)}")
        st.markdown(f"**Top catalyst:** {card.get('Top Catalyst', 'Not specified')}")
        st.markdown(f"**Invalidation:** {card.get('Invalidation', 'Not specified')}")
        st.markdown(f"**Key metric:** {card.get('Key Metric', 'Not specified')}")
        st.markdown(
            f"**Bull/Bear probability:** {card.get('Bull Probability', 'Not specified')} / "
            f"{card.get('Bear Probability', 'Not specified')}"
        )

    st.subheader("V3 Discipline Scores")
    cols = st.columns(4)
    for index, name in enumerate(SCORE_NAMES):
        cols[index % len(cols)].metric(display_score_name(name), f"{result.dashboard_scores.get(name, 0)}/10")

    rec = result.recommendation
    if rec == "PROCEED":
        st.success(f"Recommendation: {rec} - {result.recommendation_reason}")
    elif rec == "AVOID":
        st.error(f"Recommendation: {rec} - {result.recommendation_reason}")
    else:
        st.warning(f"Recommendation: {rec} - {result.recommendation_reason}")


def render_info_grid(title: str, values: dict[str, str]) -> None:
    if not values:
        return
    st.subheader(title)
    with st.container(border=True):
        for label, value in values.items():
            st.markdown(f"**{label}:** {value}")


def display_score_name(name: str) -> str:
    return "Discipline Skew" if name == "Asymmetry" else name


def render_step(title: str, content: str) -> None:
    with st.expander(title, expanded=False):
        st.markdown(content)


def render_exports(result: DDResult) -> None:
    st.subheader("Export")
    json_payload = export_json(result)
    markdown_payload = export_markdown(result)
    col1, col2, col3 = st.columns([1, 1, 1])
    col1.download_button(
        "Download JSON",
        data=json_payload,
        file_name=f"{result.ticker}_dd_framework.json",
        mime="application/json",
        use_container_width=True,
    )
    col2.download_button(
        "Download Markdown",
        data=markdown_payload,
        file_name=f"{result.ticker}_dd_framework.md",
        mime="text/markdown",
        use_container_width=True,
    )
    with col3:
        render_copy_button(markdown_payload)


def render_copy_button(text: str) -> None:
    button_id = "copy_" + hashlib.sha256(text.encode("utf-8")).hexdigest()[:10]
    escaped = html.escape(text)
    components.html(
        f"""
        <textarea id="{button_id}_payload" style="position:absolute;left:-9999px;">{escaped}</textarea>
        <button class="copy-button" onclick="
          const payload = document.getElementById('{button_id}_payload').value;
          navigator.clipboard.writeText(payload).then(() => {{
            const button = document.getElementById('{button_id}');
            button.innerText = 'Copied';
            setTimeout(() => button.innerText = 'Copy Markdown', 1400);
          }});
        " id="{button_id}">Copy Markdown</button>
        <style>
        .copy-button {{
            width: 100%;
            background: #2563eb;
            color: white;
            border: 0;
            border-radius: 6px;
            padding: 0.58rem 0.8rem;
            font-weight: 700;
            cursor: pointer;
            font-family: sans-serif;
        }}
        .copy-button:hover {{ background: #1d4ed8; }}
        </style>
        """,
        height=48,
    )


def render_cached_notice(result: DDResult) -> None:
    st.info(f"Cached on {cached_on_label(result.cached_at)} using `{result.model}`.")


def main() -> None:
    configure_page()
    st.title(APP_TITLE)
    st.caption("Ticker-in, decision-first trading insight: catalyst, invalidation, risk/reward, confidence, exports.")

    with st.sidebar:
        st.header("Analysis Setup")
        raw_ticker = st.text_input("Ticker", placeholder="RR, ACHR, UUUU", max_chars=10)
        force_refresh = st.toggle("Refresh cache", value=False)
        st.caption(f"Default model: `{get_model()}`")
        run_button = st.button("Run Due Diligence", type="primary", use_container_width=True)

    if not run_button:
        st.markdown(
            """
            <div class="dd-shell">
            Enter a ticker to generate a concise decision card. Cached analyses return instantly unless refresh is enabled.
            </div>
            """,
            unsafe_allow_html=True,
        )
        return

    try:
        ticker = normalize_ticker(raw_ticker)
        cached = load_cache(ticker)
        if cached and not force_refresh:
            result = cached
            render_cached_notice(result)
        else:
            result = run_analysis(ticker, force_refresh=force_refresh)
            render_cached_notice(result)

        render_dashboard(result)
        render_step("Step 1 - Fast Setup", result.step_1)
        render_step("Step 2 - Economic Engine", result.step_2)
        render_step("Step 3 - Decision Support", result.step_3)
        render_exports(result)
    except UserFacingError as exc:
        st.error(str(exc))
    except Exception as exc:  # pragma: no cover - defensive Streamlit UX guard
        st.error("Something unexpected happened while running the DD workflow.")
        with st.expander("Technical detail"):
            st.exception(exc)


if __name__ == "__main__":
    main()
