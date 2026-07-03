"""Streamlit wrapper for Velocity Scanner v1."""

from __future__ import annotations

import streamlit as st

from velocity_scanner import DataProvider, ScanResult, order_evidence, scan_ticker


VERDICT_STYLE = {
    "GO": ("#0f7b3d", "#e8f6ee"),
    "CHECK": ("#9a6200", "#fff5dd"),
    "NO-GO": ("#a32929", "#fdecec"),
}


def split_tickers(raw: str) -> list[str]:
    tickers = [item.strip().upper() for item in raw.replace(",", " ").split()]
    return list(dict.fromkeys(item for item in tickers if item))


def render_result(result: ScanResult) -> None:
    color, background = VERDICT_STYLE.get(result.verdict, ("#333333", "#f4f4f4"))
    st.markdown(
        f"""
        <div style="border:1px solid {color}; background:{background}; padding:0.85rem 1rem; border-radius:6px; margin:0.75rem 0;">
          <div style="display:flex; justify-content:space-between; gap:1rem; align-items:center;">
            <strong style="font-size:1.15rem;">{result.ticker}</strong>
            <span style="color:{color}; font-weight:700;">{result.verdict}</span>
          </div>
          <div style="font-size:0.9rem; color:#444; margin-top:0.25rem;">
            Confidence {result.confidence:.0%} | Regime {result.regime}
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    for signal in order_evidence(result.signals, result.verdict)[:6]:
        if signal.severity == "no-go":
            st.error(signal.message, icon="X")
        elif signal.severity == "check":
            st.warning(signal.message, icon="!")
        else:
            st.success(signal.message, icon="+")
    if result.cache_notes:
        st.caption("Cache: " + ", ".join(result.cache_notes))


def main() -> None:
    st.set_page_config(page_title="Velocity Scanner", page_icon="VS", layout="centered")
    st.title("Velocity Scanner")

    raw_tickers = st.text_input("Ticker(s)", value="USAR ASTS RKLB", placeholder="ASTS RKLB USAR")
    run = st.button("Scan", type="primary", use_container_width=True)

    if not run:
        return

    tickers = split_tickers(raw_tickers)
    if not tickers:
        st.warning("Enter at least one ticker.")
        return

    provider = DataProvider()
    with st.spinner("Scanning..."):
        results = [scan_ticker(ticker, provider) for ticker in tickers]

    results.sort(key=lambda item: {"GO": 0, "CHECK": 1, "NO-GO": 2}.get(item.verdict, 9))
    for result in results:
        render_result(result)


if __name__ == "__main__":
    main()
