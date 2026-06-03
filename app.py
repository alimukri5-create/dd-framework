"""DD Framework - Institutional Due Diligence Streamlit app."""

from __future__ import annotations

import hashlib
import html
import json
import re
import tempfile
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import streamlit as st
import streamlit.components.v1 as components
import yfinance as yf
from openai import APIConnectionError, APIError, APITimeoutError, AuthenticationError, OpenAI, RateLimitError


APP_TITLE = "DD Framework - Institutional Due Diligence"
DEFAULT_MODEL = "gpt-4-turbo"
DEFAULT_TIMEOUT_SECONDS = 180
SCORE_NAMES = [
    "Setup Quality",
    "Catalyst Clarity",
    "Risk/Reward",
    "Decision Confidence",
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
    }
    try:
        yf_cache_dir = Path(tempfile.gettempdir()) / "dd-framework-yfinance-cache"
        yf_cache_dir.mkdir(parents=True, exist_ok=True)
        yf.set_tz_cache_location(str(yf_cache_dir))
        asset = yf.Ticker(ticker)
        info = asset.get_info() or {}
        history = asset.history(period="6mo", interval="1d", auto_adjust=True)
    except Exception as exc:
        data["error"] = f"Market data fetch failed: {exc}"
        return data

    data.update(
        {
            "name": info.get("shortName") or info.get("longName"),
            "sector": info.get("sector"),
            "industry": info.get("industry"),
            "market_cap": info.get("marketCap"),
            "enterprise_value": info.get("enterpriseValue"),
            "current_price": info.get("currentPrice") or info.get("regularMarketPrice"),
            "fifty_two_week_high": info.get("fiftyTwoWeekHigh"),
            "fifty_two_week_low": info.get("fiftyTwoWeekLow"),
            "average_volume": info.get("averageVolume"),
            "float_shares": info.get("floatShares"),
            "short_percent_float": info.get("shortPercentOfFloat"),
            "total_revenue": info.get("totalRevenue"),
            "revenue_growth": info.get("revenueGrowth"),
            "gross_margins": info.get("grossMargins"),
            "operating_margins": info.get("operatingMargins"),
            "profit_margins": info.get("profitMargins"),
            "ebitda_margins": info.get("ebitdaMargins"),
            "price_to_sales": info.get("priceToSalesTrailing12Months"),
            "trailing_pe": info.get("trailingPE"),
            "forward_pe": info.get("forwardPE"),
            "beta": info.get("beta"),
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
    return data


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
        "sector": "Sector",
        "industry": "Industry",
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
        "revenue_growth": "Revenue Growth",
        "gross_margins": "Gross Margin",
        "operating_margins": "Operating Margin",
        "profit_margins": "Profit Margin",
        "ebitda_margins": "EBITDA Margin",
        "price_to_sales": "Price/Sales",
        "trailing_pe": "Trailing P/E",
        "forward_pe": "Forward P/E",
        "beta": "Beta",
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
    return cleaned.strip()


def extract_responses_text(response: Any) -> str:
    chunks: list[str] = []
    for item in getattr(response, "output", []) or []:
        for content in getattr(item, "content", []) or []:
            text = getattr(content, "text", None)
            if text:
                chunks.append(text)
    return "\n".join(chunks)


def build_step_1_prompt(ticker: str, market_data: str) -> str:
    return f"""
Research {ticker} for a fast trading decision.

Use this market data pack as your factual starting point. If a value is missing, say it is missing.
Do not replace missing values with generic guesses.

{market_data}

Return a concise decision briefing, not a company story:
1. WHAT MOVES THE STOCK: top 3 drivers in the next 1-90 days.
2. CURRENT SETUP: recent revenue/growth/profitability, valuation context, liquidity/volatility if known.
3. BUSINESS ONLY IF TRADE-RELEVANT: revenue mix, installed base/contract model, recurring vs one-time revenue.
4. COMPETITION: top 3 competitors and the one competitive fact most likely to matter to the stock.
5. SOURCE QUALITY: cite the market data source above and list missing data that would change the decision.
6. CONFIDENCE: score each section 1-10.

Format: 350-500 words. Use bullets. No background filler.
""".strip()


def build_step_2_prompt(step_1: str) -> str:
    return f"""
Based on Step 1: {step_1}

Convert the company facts into investable/tradable economics:
1. ECONOMIC ENGINE: what KPI actually drives the stock for this sector? Examples: orders/bookings, ARPU,
   churn, gross margin, utilization, commodity price, reimbursement, deliveries, backlog, licensing, FCF.
2. CUSTOMER/UNIT ECONOMICS: switching cost, replacement/upgrade cycle, payback, recurring revenue,
   margin structure, or sector-equivalent economics. If hospital/reimbursement is irrelevant, say N/A.
3. CATALYST MAP: next 3 likely catalysts, expected timing, and whether each is bullish/bearish/ambiguous.
4. STALL MAP: what would make the thesis fail fast?
5. FAST CHECK: what single data point should a trader verify before acting?

Format: 350-500 words. Use bullets/table style. Show confidence. No repeated company overview.
""".strip()


def build_step_3_prompt(step_1: str, step_2: str) -> str:
    return f"""
Based on Steps 1-2: {step_1}

{step_2}

Produce a decision-first trading dashboard:
1. DECISION: PROCEED / HOLD / AVOID. Also give bias: LONG / WATCHLIST / AVOID.
2. WHY NOW: one sentence. If no near-term edge, say so.
3. BULL CASE: 3 drivers, what goes right, probability (%), trigger to confirm.
4. BEAR CASE: 3 risks, what breaks, probability (%), trigger to invalidate.
5. ASYMMETRY: upside/downside balance, what the market may be pricing in vs missing.
6. TRADE CARD: horizon, top catalyst, invalidation trigger, key metric to monitor, confidence 1-10.

End with this exact block:
TRADING_DECISION_CARD
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

Format: 450-650 words. No generic background.
""".strip()


def run_analysis(ticker: str, force_refresh: bool = False) -> DDResult:
    cached = load_cache(ticker)
    if cached and not force_refresh:
        return cached

    client = get_openai_client()
    model = get_model()
    market_data = fetch_market_data(ticker)
    market_data_text = format_market_data(market_data)
    if market_data.get("error"):
        st.warning(market_data["error"])
    else:
        st.success(f"Loaded market data from {market_data['source']}.")

    step_1_box = st.empty()
    step_2_box = st.empty()
    step_3_box = st.empty()

    with step_1_box.container():
        with st.spinner("Step 1: researching business model, moat, metrics, and confidence..."):
            step_1 = call_openai(client, build_step_1_prompt(ticker, market_data_text), model, step_name="Step 1")
        render_step("Step 1 - Business, Moat, Metrics", step_1)

    with step_2_box.container():
        with st.spinner("Step 2: analyzing hospital economics, reimbursement, growth, and unit economics..."):
            step_2 = call_openai(client, build_step_2_prompt(step_1), model, step_name="Step 2")
        render_step("Step 2 - Economics, Reimbursement, Growth", step_2)

    with step_3_box.container():
        with st.spinner("Step 3: building bull/bear assessment and moat outlook..."):
            step_3 = call_openai(client, build_step_3_prompt(step_1, step_2), model, step_name="Step 3")
        render_step("Step 3 - Bull/Bear Assessment", step_3)

    scores = extract_dashboard_scores(step_1, step_2, step_3)
    recommendation, reason = build_recommendation(scores, step_3)
    decision_card = extract_decision_card(step_3, recommendation, reason)
    result = DDResult(
        ticker=ticker,
        company_name=extract_company_name(step_1),
        cached_at=datetime.now(timezone.utc).isoformat(),
        model=model,
        step_1=step_1,
        step_2=step_2,
        step_3=step_3,
        dashboard_scores=scores,
        recommendation=recommendation,
        recommendation_reason=reason,
        decision_card=decision_card,
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
    scores = "\n".join(f"- **{name}:** {score}/10" for name, score in result.dashboard_scores.items())
    decision = "\n".join(f"- **{name}:** {value}" for name, value in result.decision_card.items())
    company = f"\n**Company:** {result.company_name}" if result.company_name else ""
    return f"""# {APP_TITLE}

**Ticker:** {result.ticker}{company}
**Model:** {result.model}
**Cached on:** {cached_on_label(result.cached_at)}
**Recommendation:** {result.recommendation}
**Reason:** {result.recommendation_reason}

## Dashboard

{scores}

## Decision Card

{decision}

## Step 1 - Business, Moat, Metrics

{result.step_1}

## Step 2 - Economics, Reimbursement, Growth

{result.step_2}

## Step 3 - Bull/Bear Assessment

{result.step_3}
"""


def render_dashboard(result: DDResult) -> None:
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

    st.subheader("Setup Scores")
    cols = st.columns(4)
    for col, name in zip(cols, SCORE_NAMES):
        col.metric(name, f"{result.dashboard_scores.get(name, 0)}/10")

    rec = result.recommendation
    if rec == "PROCEED":
        st.success(f"Recommendation: {rec} - {result.recommendation_reason}")
    elif rec == "AVOID":
        st.error(f"Recommendation: {rec} - {result.recommendation_reason}")
    else:
        st.warning(f"Recommendation: {rec} - {result.recommendation_reason}")


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
        render_step("Step 1 - Business, Moat, Metrics", result.step_1)
        render_step("Step 2 - Economics, Reimbursement, Growth", result.step_2)
        render_step("Step 3 - Bull/Bear Assessment", result.step_3)
        render_exports(result)
    except UserFacingError as exc:
        st.error(str(exc))
    except Exception as exc:  # pragma: no cover - defensive Streamlit UX guard
        st.error("Something unexpected happened while running the DD workflow.")
        with st.expander("Technical detail"):
            st.exception(exc)


if __name__ == "__main__":
    main()
