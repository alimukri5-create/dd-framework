"""Velocity Scanner v1: fast pre-trade hazard and context scan."""

from __future__ import annotations

import argparse
import math
import os
import tempfile
from dataclasses import dataclass, field
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import pandas as pd
import requests
import yfinance as yf

import vs_cache


PRICE_TTL_SECONDS = 12 * 60 * 60
EDGAR_TTL_SECONDS = 12 * 60 * 60
REGIME_TTL_SECONDS = 24 * 60 * 60
INFO_TTL_SECONDS = 60 * 60
CACHE_VERSION = "v2"

TODAY = date.today
SEC_USER_AGENT = (
    os.getenv("VELOCITY_SCANNER_SEC_USER_AGENT")
    or os.getenv("VELOCITY_SCANNER_EMAIL")
    or "VelocityScanner/1.0 alimukri5@gmail.com"
)
SEC_HEADERS = {"User-Agent": SEC_USER_AGENT, "Accept-Encoding": "gzip, deflate"}
VERDICT_RANK = {"GO": 0, "CHECK": 1, "NO-GO": 2}
PROXY_ENV_KEYS = ("HTTP_PROXY", "HTTPS_PROXY", "ALL_PROXY", "http_proxy", "https_proxy", "all_proxy")


@dataclass(frozen=True)
class Signal:
    check: str
    status: str
    severity: str
    message: str
    confidence: float
    data: dict[str, Any] = field(default_factory=dict)


@dataclass
class ScanResult:
    ticker: str
    verdict: str
    confidence: float
    regime: str
    signals: list[Signal]
    cache_notes: list[str] = field(default_factory=list)
    thin_data: bool = False


@dataclass
class DataProvider:
    cache_notes: list[str] = field(default_factory=list)
    _history_memory: dict[str, pd.DataFrame | None] = field(default_factory=dict)
    _info_memory: dict[str, dict[str, Any]] = field(default_factory=dict)
    _edgar_memory: dict[str, dict[str, Any]] = field(default_factory=dict)
    _regime_memory: dict[str, Any] = field(default_factory=dict)

    def get_info(self, ticker: str) -> dict[str, Any]:
        symbol = ticker.upper()
        if symbol in self._info_memory:
            return self._info_memory[symbol]
        cache_key = f"{CACHE_VERSION}:info:{symbol}"
        cached = vs_cache.get(cache_key, INFO_TTL_SECONDS)
        if cached is not None and isinstance(cached.value, dict):
            self.cache_notes.append(f"{symbol} info {vs_cache.format_age(cached.age_seconds)}")
            self._info_memory[symbol] = cached.value
            return cached.value

        clear_proxy_env_for_market_data()
        set_yfinance_cache_location()
        asset = yf.Ticker(symbol)
        info: dict[str, Any] = {}
        try:
            info = asset.get_info() or {}
        except Exception as exc:
            info = {"error": f"Ticker info failed: {exc}"}
        info["earnings_candidates"] = _extract_earnings_candidates(asset)
        if "error" not in info:
            vs_cache.set(cache_key, info)
        self._info_memory[symbol] = info
        return info

    def get_history(self, ticker: str, period: str = "3y") -> pd.DataFrame | None:
        symbol = ticker.upper()
        memory_key = f"{symbol}:{period}"
        if memory_key in self._history_memory:
            return self._history_memory[memory_key]
        cache_key = f"{CACHE_VERSION}:history:{symbol}:{period}"
        cached = vs_cache.get(cache_key, PRICE_TTL_SECONDS)
        if cached is not None:
            history = _history_from_cache(cached.value)
            self.cache_notes.append(f"{symbol} prices {vs_cache.format_age(cached.age_seconds)}")
            self._history_memory[memory_key] = history
            return history

        history = fetch_price_history(symbol, period=period)
        if history is not None:
            vs_cache.set(cache_key, _history_to_cache(history))
        self._history_memory[memory_key] = history
        return history

    def get_edgar(self, ticker: str) -> dict[str, Any]:
        symbol = ticker.upper()
        if symbol in self._edgar_memory:
            return self._edgar_memory[symbol]
        cache_key = f"{CACHE_VERSION}:edgar:{symbol}"
        cached = vs_cache.get(cache_key, EDGAR_TTL_SECONDS)
        if cached is not None and isinstance(cached.value, dict):
            self.cache_notes.append(f"{symbol} EDGAR {vs_cache.format_age(cached.age_seconds)}")
            self._edgar_memory[symbol] = cached.value
            return cached.value

        filings = fetch_recent_edgar_filings(symbol)
        if "error" not in filings:
            vs_cache.set(cache_key, filings)
        self._edgar_memory[symbol] = filings
        return filings

    def get_regime(self) -> dict[str, Any]:
        if "regime" in self._regime_memory:
            return self._regime_memory["regime"]
        today_key = TODAY().isoformat()
        cache_key = f"{CACHE_VERSION}:regime:{today_key}"
        cached = vs_cache.get(cache_key, REGIME_TTL_SECONDS)
        if cached is not None and isinstance(cached.value, dict):
            self.cache_notes.append(f"regime {vs_cache.format_age(cached.age_seconds)}")
            self._regime_memory["regime"] = cached.value
            return cached.value

        regime = compute_regime(self.get_history("SPY", period="2y"))
        if regime.get("confidence", 0) > 0:
            vs_cache.set(cache_key, regime)
        self._regime_memory["regime"] = regime
        return regime


def _extract_earnings_candidates(asset: Any) -> list[str]:
    candidates: list[str] = []
    try:
        calendar = asset.calendar
        if isinstance(calendar, dict):
            raw = calendar.get("Earnings Date") or calendar.get("EarningsDate")
            candidates.extend(_coerce_dates(raw))
        elif isinstance(calendar, pd.DataFrame) and not calendar.empty:
            for value in calendar.to_numpy().flatten().tolist():
                candidates.extend(_coerce_dates(value))
    except Exception:
        pass
    try:
        earnings_dates = asset.get_earnings_dates(limit=4)
        if isinstance(earnings_dates, pd.DataFrame) and not earnings_dates.empty:
            candidates.extend(item.date().isoformat() for item in pd.to_datetime(earnings_dates.index).to_pydatetime())
    except Exception:
        pass
    return sorted(set(candidates))


def _coerce_dates(raw: Any) -> list[str]:
    if raw is None:
        return []
    items = raw if isinstance(raw, (list, tuple, set)) else [raw]
    dates: list[str] = []
    for item in items:
        try:
            parsed = pd.to_datetime(item)
            if pd.isna(parsed):
                continue
            dates.append(parsed.date().isoformat())
        except Exception:
            continue
    return dates


def fetch_price_history(ticker: str, period: str = "3y") -> pd.DataFrame | None:
    try:
        clear_proxy_env_for_market_data()
        set_yfinance_cache_location()
        asset = yf.Ticker(ticker)
        history = asset.history(period=period, interval="1d", auto_adjust=True)
        if history is not None and not history.empty and "Close" in history:
            return history.dropna(subset=["Close"])
    except Exception:
        pass
    return fetch_chart_fallback(ticker, period)


def fetch_chart_fallback(ticker: str, period: str) -> pd.DataFrame | None:
    range_map = {"6mo": "6mo", "1y": "1y", "2y": "2y", "3y": "3y"}
    yahoo_range = range_map.get(period, "3y")
    url = f"https://query1.finance.yahoo.com/v8/finance/chart/{ticker}"
    try:
        clear_proxy_env_for_market_data()
        response = requests.get(url, params={"range": yahoo_range, "interval": "1d"}, timeout=8)
        response.raise_for_status()
        result = response.json()["chart"]["result"][0]
        timestamps = result.get("timestamp") or []
        quote = result["indicators"]["quote"][0]
        close = quote.get("close") or []
        volume = quote.get("volume") or []
        if not timestamps or not close:
            return None
        frame = pd.DataFrame(
            {
                "Date": pd.to_datetime(timestamps, unit="s", utc=True),
                "Close": close,
                "Volume": volume,
            }
        ).dropna(subset=["Close"])
        return frame.set_index("Date")
    except Exception:
        return None


def _history_to_cache(history: pd.DataFrame) -> dict[str, list[Any]]:
    frame = history.copy()
    frame = frame[[column for column in ["Close", "Volume"] if column in frame.columns]].dropna(subset=["Close"])
    return {
        "index": [pd.Timestamp(index).isoformat() for index in frame.index],
        "close": [float(value) for value in frame["Close"].tolist()],
        "volume": [float(value) if pd.notna(value) else None for value in frame.get("Volume", pd.Series(index=frame.index)).tolist()],
    }


def _history_from_cache(value: Any) -> pd.DataFrame | None:
    if not isinstance(value, dict):
        return None
    try:
        frame = pd.DataFrame(
            {
                "Close": value["close"],
                "Volume": value.get("volume", [None] * len(value["close"])),
            },
            index=pd.to_datetime(value["index"]),
        )
        return frame.dropna(subset=["Close"])
    except Exception:
        return None


def fetch_recent_edgar_filings(ticker: str) -> dict[str, Any]:
    try:
        clear_proxy_env_for_market_data()
        cik = lookup_cik(ticker)
        if cik is None:
            return {"ticker": ticker, "error": "CIK not found", "filings": []}
        url = f"https://data.sec.gov/submissions/CIK{cik:010d}.json"
        response = requests.get(url, headers=SEC_HEADERS, timeout=8)
        response.raise_for_status()
        recent = response.json().get("filings", {}).get("recent", {})
        forms = recent.get("form", [])
        dates = recent.get("filingDate", [])
        accessions = recent.get("accessionNumber", [])
        documents = recent.get("primaryDocument", [])
        filings = []
        for index, form in enumerate(forms):
            filings.append(
                {
                    "form": str(form),
                    "filing_date": dates[index] if index < len(dates) else None,
                    "accession": accessions[index] if index < len(accessions) else None,
                    "document": documents[index] if index < len(documents) else None,
                }
            )
        return {"ticker": ticker, "cik": cik, "filings": filings}
    except Exception as exc:
        return {"ticker": ticker, "error": f"EDGAR fetch failed: {exc}", "filings": []}


def lookup_cik(ticker: str) -> int | None:
    cache_key = f"{CACHE_VERSION}:edgar:company_tickers"
    cached = vs_cache.get(cache_key, EDGAR_TTL_SECONDS)
    if cached is not None and isinstance(cached.value, dict):
        mapping = cached.value
    else:
        clear_proxy_env_for_market_data()
        response = requests.get("https://www.sec.gov/files/company_tickers.json", headers=SEC_HEADERS, timeout=8)
        response.raise_for_status()
        raw = response.json()
        mapping = {
            str(item.get("ticker", "")).upper(): int(item["cik_str"])
            for item in raw.values()
            if isinstance(item, dict) and item.get("ticker") and item.get("cik_str")
        }
        vs_cache.set(cache_key, mapping)
    return mapping.get(ticker.upper())


def clear_proxy_env_for_market_data() -> None:
    for key in PROXY_ENV_KEYS:
        os.environ.pop(key, None)


def set_yfinance_cache_location() -> None:
    cache_dir = Path(tempfile.gettempdir()) / "dd-framework-yfinance-cache"
    cache_dir.mkdir(parents=True, exist_ok=True)
    try:
        yf.set_tz_cache_location(str(cache_dir))
    except Exception:
        pass


def check_shariah(info: dict[str, Any]) -> Signal:
    sector = _text(info.get("sector"))
    industry = _text(info.get("industry"))
    debt = _number(info.get("totalDebt") if "totalDebt" in info else info.get("total_debt"))
    market_cap = _number(info.get("marketCap") if "marketCap" in info else info.get("market_cap"))
    available = sum(value is not None and value != "" for value in [sector, industry, debt, market_cap])
    confidence = available / 4
    combined = f"{sector} {industry}".lower()
    exclusions = [
        "alcohol",
        "gambling",
        "casino",
        "weapon",
        "defense",
        "aerospace & defense",
        "bank",
        "mortgage",
        "credit services",
        "insurance",
        "tobacco",
        "adult",
        "porn",
    ]
    for keyword in exclusions:
        if keyword in combined:
            return Signal("Shariah", "FAIL", "no-go", f"Shariah exclusion: {sector or industry}", confidence, {"keyword": keyword})
    if debt is None or market_cap is None or market_cap <= 0:
        return Signal("Shariah", "FLAG", "check", "Shariah unverifiable: missing debt or market cap", confidence, {})
    ratio = debt / market_cap
    if ratio >= 0.33:
        return Signal("Shariah", "FAIL", "no-go", f"Shariah debt/mcap {ratio:.0%} >= 33%", confidence, {"ratio": ratio})
    return Signal("Shariah", "PASS", "pass", f"Shariah: pass (debt/mcap {ratio:.0%})", confidence, {"ratio": ratio})


def check_offering_overhang(edgar: dict[str, Any], as_of: date | None = None) -> Signal:
    as_of = as_of or TODAY()
    if isinstance(edgar, dict) and edgar.get("error"):
        return Signal("Offering overhang", "FLAG", "check", f"Offering overhang unverifiable: {edgar['error']}", 0.0, {})
    filings = edgar.get("filings") if isinstance(edgar, dict) else None
    if not isinstance(filings, list):
        return Signal("Offering overhang", "FLAG", "check", "Offering overhang unverifiable: EDGAR unavailable", 0.0, {})
    offering_forms = []
    for filing in filings:
        form = str(filing.get("form", "")).upper().replace(" ", "")
        filed = _parse_date(filing.get("filing_date"))
        if filed is None or not _is_offering_form(form):
            continue
        age = (as_of - filed).days
        if age < 0:
            continue
        offering_forms.append((age, form, filed))
    if not offering_forms:
        return Signal("Offering overhang", "PASS", "pass", "No shelf/prospectus filings in last 120d", 1.0, {})
    age, form, filed = min(offering_forms, key=lambda item: item[0])
    if age <= 45:
        return Signal("Offering overhang", "FAIL", "no-go", f"Shelf {form} filed {filed.isoformat()} (overhang, {age}d ago)", 1.0, {"age_days": age, "form": form})
    if age <= 120:
        return Signal("Offering overhang", "FLAG", "check", f"Shelf {form} filed {filed.isoformat()} ({age}d ago)", 1.0, {"age_days": age, "form": form})
    return Signal("Offering overhang", "PASS", "pass", "No shelf/prospectus filings in last 120d", 1.0, {})


def check_insider_cluster(edgar: dict[str, Any], as_of: date | None = None) -> Signal:
    as_of = as_of or TODAY()
    if isinstance(edgar, dict) and edgar.get("error"):
        return Signal("Insider cluster", "FLAG", "check", "Form 144 unverifiable: EDGAR unavailable", 0.0, {})
    filings = edgar.get("filings") if isinstance(edgar, dict) else None
    if not isinstance(filings, list):
        return Signal("Insider cluster", "FLAG", "check", "Form 144 unverifiable: EDGAR unavailable", 0.0, {})
    count = 0
    for filing in filings:
        form = str(filing.get("form", "")).upper().strip()
        filed = _parse_date(filing.get("filing_date"))
        if form == "144" and filed is not None and 0 <= (as_of - filed).days <= 120:
            count += 1
    if count >= 2:
        return Signal("Insider cluster", "FLAG", "check", f"{count}x Form 144 insider-sale notices in recent filings", 1.0, {"count": count})
    return Signal("Insider cluster", "PASS", "pass", "Insider Form 144 cluster: none", 1.0, {"count": count})


def check_earnings(info: dict[str, Any], as_of: date | None = None) -> Signal:
    as_of = as_of or TODAY()
    candidates = [_parse_date(item) for item in info.get("earnings_candidates", [])]
    future_dates = sorted(item for item in candidates if item is not None and item >= as_of)
    if not future_dates:
        return Signal("Earnings proximity", "FLAG", "check", "Earnings date UNKNOWN", 0.0, {})
    next_date = future_dates[0]
    days = (next_date - as_of).days
    if days <= 5:
        return Signal("Earnings proximity", "FLAG", "check", f"Earnings in {days} days ({next_date.isoformat()})", 1.0, {"days": days})
    return Signal("Earnings proximity", "PASS", "pass", f"Earnings not inside 5d window ({next_date.isoformat()})", 1.0, {"days": days})


def check_liquidity(history: pd.DataFrame | None) -> Signal:
    if history is None or history.empty or "Close" not in history or "Volume" not in history:
        return Signal("Liquidity", "FLAG", "check", "Liquidity unverifiable: missing price/volume history", 0.0, {})
    recent = history.dropna(subset=["Close", "Volume"]).tail(20)
    if len(recent) < 20:
        return Signal("Liquidity", "FLAG", "check", "Liquidity unverifiable: fewer than 20 trading days", len(recent) / 20, {})
    avg_dollar_volume = float((recent["Close"] * recent["Volume"]).mean())
    if avg_dollar_volume < 5_000_000:
        return Signal("Liquidity", "FLAG", "check", f"Liquidity: ${avg_dollar_volume / 1_000_000:.1f}M avg daily $ (exit risk)", 1.0, {"avg_dollar_volume": avg_dollar_volume})
    return Signal("Liquidity", "PASS", "pass", f"Liquidity: ${avg_dollar_volume / 1_000_000:.0f}M avg daily $", 1.0, {"avg_dollar_volume": avg_dollar_volume})


def check_relative_strength(
    ticker_history: pd.DataFrame | None,
    spy_history: pd.DataFrame | None,
    sector_history: pd.DataFrame | None,
    sector_etf: str,
) -> Signal:
    if ticker_history is None or spy_history is None or sector_history is None:
        return Signal("Relative strength", "FLAG", "check", "RS unavailable: missing benchmark or ticker history", 0.0, {"sector_etf": sector_etf})
    if len(ticker_history) < 80:
        return Signal("Relative strength", "FLAG", "check", "RS unavailable: not enough ticker history", min(len(ticker_history) / 80, 1), {"sector_etf": sector_etf})
    stock, spy_returns, sector_returns = align_return_series(
        ticker_history["Close"].pct_change().dropna() * 100,
        spy_history["Close"].pct_change().dropna() * 100,
        sector_history["Close"].pct_change().dropna() * 100,
    )
    if len(stock) < 60:
        return Signal("Relative strength", "FLAG", "check", "RS unavailable: not enough aligned observations", len(stock) / 60, {"sector_etf": sector_etf})
    beta_spy = beta(stock[-126:], spy_returns[-126:])
    beta_sector = beta(stock[-126:], sector_returns[-126:])
    raw_20 = compounded_return(stock[-20:])
    spy_20 = compounded_return(spy_returns[-20:])
    sector_20 = compounded_return(sector_returns[-20:])
    spy_adj = raw_20 - beta_spy * spy_20 if beta_spy is not None else None
    sector_adj = raw_20 - beta_sector * sector_20 if beta_sector is not None else None
    confidence = sum(value is not None for value in [raw_20, spy_adj, sector_adj]) / 3
    message = f"RS: 20D {_signed_pct(raw_20)} raw | {_signed_pct(spy_adj)} SPY-adj | {_signed_pct(sector_adj)} sector-adj ({sector_etf})"
    if spy_adj is None or sector_adj is None:
        return Signal("Relative strength", "FLAG", "check", f"{message}; beta adjustment incomplete", confidence, {"sector_etf": sector_etf})
    if spy_adj < 0 or sector_adj < 0:
        return Signal("Relative strength", "FLAG", "check", f"{message} (weak vs market)", confidence, {"sector_etf": sector_etf, "spy_adj": spy_adj, "sector_adj": sector_adj})
    return Signal("Relative strength", "PASS", "pass", message, confidence, {"sector_etf": sector_etf, "spy_adj": spy_adj, "sector_adj": sector_adj})


def check_regime(regime: dict[str, Any]) -> Signal:
    value = str(regime.get("regime", "UNKNOWN"))
    confidence = float(regime.get("confidence", 0.0))
    if value == "RISK-OFF":
        return Signal("Regime", "FLAG", "check", "Regime RISK-OFF: broad tape adds hazard", confidence, regime)
    return Signal("Regime", "PASS", "pass", f"Regime {value}", confidence, regime)


def compute_regime(spy_history: pd.DataFrame | None) -> dict[str, Any]:
    if spy_history is None or spy_history.empty or "Close" not in spy_history or len(spy_history) < 220:
        return {"regime": "UNKNOWN", "confidence": 0.0}
    close = spy_history["Close"].dropna()
    returns = close.pct_change().dropna()
    if len(close) < 220 or len(returns) < 252:
        confidence = min(len(close) / 220, 1.0) * 0.75
    else:
        confidence = 1.0
    last_close = float(close.iloc[-1])
    sma_200 = float(close.tail(200).mean())
    realized_vol = float(returns.tail(20).std() * math.sqrt(252))
    rolling_vol = returns.rolling(20).std().dropna().tail(252) * math.sqrt(252)
    median_vol = float(rolling_vol.median()) if not rolling_vol.empty else None
    if median_vol is None or math.isnan(median_vol):
        return {"regime": "UNKNOWN", "confidence": confidence * 0.5}
    if last_close > sma_200 and realized_vol <= median_vol:
        label = "RISK-ON"
    elif last_close < sma_200 and realized_vol > median_vol:
        label = "RISK-OFF"
    else:
        label = "NEUTRAL"
    return {
        "regime": label,
        "confidence": confidence,
        "spy_close": last_close,
        "spy_200dma": sma_200,
        "vol_20d": realized_vol,
        "vol_1y_median": median_vol,
    }


def select_sector_etf(info: dict[str, Any]) -> str:
    text = f"{_text(info.get('sector'))} {_text(info.get('industry'))}".lower()
    if "semiconductor" in text or "semis" in text:
        return "SMH"
    if "space" in text:
        return "ARKX"
    if "defense" in text or "aerospace" in text:
        return "ITA"
    if "uranium" in text or "nuclear" in text:
        return "URA"
    if "biotech" in text or "biotechnology" in text:
        return "XBI"
    if "software" in text:
        return "IGV"
    return "IWM"


def scan_ticker(ticker: str, provider: DataProvider | None = None) -> ScanResult:
    provider = provider or DataProvider()
    symbol = ticker.upper()
    start_note_count = len(provider.cache_notes)
    signals: list[Signal] = []
    info = provider.get_info(symbol)
    regime = provider.get_regime()
    regime_label = str(regime.get("regime", "UNKNOWN"))

    shariah = check_shariah(info)
    signals.append(shariah)
    if shariah.severity == "no-go":
        return finalize_result(symbol, "NO-GO", regime_label, signals, provider.cache_notes[start_note_count:])

    edgar = provider.get_edgar(symbol)
    offering = check_offering_overhang(edgar)
    signals.append(offering)
    if offering.severity == "no-go":
        return finalize_result(symbol, "NO-GO", regime_label, signals, provider.cache_notes[start_note_count:])

    signals.append(check_insider_cluster(edgar))
    signals.append(check_earnings(info))
    history = provider.get_history(symbol, period="3y")
    sector_etf = select_sector_etf(info)
    signals.append(check_liquidity(history))
    signals.append(
        check_relative_strength(
            history,
            provider.get_history("SPY", period="3y"),
            provider.get_history(sector_etf, period="3y"),
            sector_etf,
        )
    )
    signals.append(check_regime(regime))
    verdict = "CHECK" if any(signal.severity == "check" for signal in signals) else "GO"
    return finalize_result(symbol, verdict, regime_label, signals, provider.cache_notes[start_note_count:])


def finalize_result(ticker: str, verdict: str, regime: str, signals: list[Signal], cache_notes: list[str]) -> ScanResult:
    confidence = overall_confidence(signals)
    thin_data = confidence < 0.6
    if thin_data and verdict == "GO":
        verdict = "CHECK"
        signals = signals + [Signal("Data quality", "FLAG", "check", "Thin data: confidence below 60%", confidence, {})]
    elif thin_data and verdict == "CHECK":
        signals = signals + [Signal("Data quality", "FLAG", "check", "Thin data: confidence below 60%", confidence, {})]
    return ScanResult(ticker, verdict, confidence, regime, signals, cache_notes, thin_data)


def overall_confidence(signals: Iterable[Signal]) -> float:
    values = [max(0.0, min(1.0, signal.confidence)) for signal in signals]
    if not values:
        return 0.0
    return sum(values) / len(values)


def order_evidence(signals: list[Signal], verdict: str) -> list[Signal]:
    if verdict == "NO-GO":
        hard = [signal for signal in signals if signal.severity == "no-go"]
        return hard[:1]
    flagged = [signal for signal in signals if signal.severity == "check"]
    passed = [signal for signal in signals if signal.severity == "pass"]
    return flagged + passed


def format_result(result: ScanResult) -> str:
    header = f"{result.ticker} - {result.verdict} (confidence {result.confidence:.0%}, regime: {result.regime})"
    if result.cache_notes:
        header += f" [cache: {', '.join(result.cache_notes)}]"
    lines = [header]
    for signal in order_evidence(result.signals, result.verdict)[:6]:
        marker = "x" if signal.severity == "no-go" else "!" if signal.severity == "check" else "+"
        lines.append(f"  {marker} {signal.message}")
    return "\n".join(lines)


def format_batch(results: list[ScanResult]) -> str:
    sorted_results = sorted(results, key=lambda item: (VERDICT_RANK[item.verdict], item.ticker))
    rows = ["Ticker  Verdict  Conf  Regime    Evidence"]
    for result in sorted_results:
        evidence = "; ".join(signal.message for signal in order_evidence(result.signals, result.verdict)[:2])
        rows.append(f"{result.ticker:<7} {result.verdict:<7} {result.confidence:>4.0%}  {result.regime:<8} {evidence}")
    return "\n".join(rows)


def align_return_series(*series_items: Any) -> tuple[list[float], ...]:
    try:
        normalized = []
        for series in series_items:
            item = series.copy()
            item.index = pd.to_datetime(item.index).date
            normalized.append(item)
        frame = pd.concat(normalized, axis=1, join="inner").dropna()
        return tuple(frame.iloc[:, index].astype(float).tolist() for index in range(frame.shape[1]))
    except Exception:
        return tuple([] for _ in series_items)


def beta(stock: list[float], benchmark: list[float]) -> float | None:
    if len(stock) < 20 or len(stock) != len(benchmark):
        return None
    mean_stock = sum(stock) / len(stock)
    mean_benchmark = sum(benchmark) / len(benchmark)
    variance = sum((item - mean_benchmark) ** 2 for item in benchmark)
    if variance == 0:
        return None
    covariance = sum((stock[i] - mean_stock) * (benchmark[i] - mean_benchmark) for i in range(len(stock)))
    return covariance / variance


def compounded_return(values: list[float]) -> float:
    total = 1.0
    for value in values:
        total *= 1 + value / 100
    return (total - 1) * 100


def _is_offering_form(form: str) -> bool:
    normalized = form.upper().replace(" ", "")
    return normalized.startswith("S-1") or normalized.startswith("S-3") or normalized == "S-3A" or normalized.startswith("424B")


def _parse_date(value: Any) -> date | None:
    try:
        return pd.to_datetime(value).date()
    except Exception:
        return None


def _number(value: Any) -> float | None:
    try:
        if value is None or value == "":
            return None
        parsed = float(value)
        if math.isnan(parsed):
            return None
        return parsed
    except (TypeError, ValueError):
        return None


def _text(value: Any) -> str:
    return str(value or "").strip()


def _signed_pct(value: float | None) -> str:
    if value is None:
        return "N/A"
    return f"{value:+.0f}%"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Velocity Scanner v1")
    parser.add_argument("tickers", nargs="+", help="Ticker(s) to scan")
    args = parser.parse_args(argv)
    provider = DataProvider()
    results = [scan_ticker(ticker, provider) for ticker in args.tickers]
    if len(results) == 1:
        print(format_result(results[0]))
    else:
        print(format_batch(results))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
