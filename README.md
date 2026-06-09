# DD Framework - Institutional Due Diligence

Production-grade Streamlit app for decision-first equity due diligence and trading insight.

The V3 engine computes deterministic market, financial, ownership, earnings, narrative, expectations, and payoff
evidence first, then uses OpenAI to explain the setup tersely.
The goal is not a company story; it is a fast read on whether a ticker is chaseable, waitlist-only, event-driven,
or avoid.

## Features

- Ticker input and validation
- Deterministic V3 discipline and edge engine
- Earnings date/surprise intelligence
- Quarterly financial trend extraction
- Ownership, insider, analyst recommendation, and narrative heat checks
- Options-implied move and sell-side target expectations baseline
- Bull/base/stress scenario map with unweighted magnitude ratio
- SEC filing scan for shelf/offering overhang and recent 8-K activity
- True-asymmetry gate that refuses to claim edge without expectations + payoff + non-consensus evidence
- Three sequential GPT synthesis steps
- Cached ticker reports with a cached-on timestamp
- Dashboard scores from 1-10 for momentum, exhaustion, fundamental trend, valuation stretch, event risk,
  dilution, ownership signal, narrative heat, squeeze risk, and asymmetry
- Recommendation: PROCEED, HOLD, or AVOID
- Trade verdict: CHASEABLE, WAIT FOR PULLBACK, EVENT WATCH, WATCHLIST, or AVOID CHASING
- JSON, Markdown, and copy-to-clipboard exports
- Friendly handling for missing API keys, invalid tickers, rate limits, timeouts, and API errors

## V3 Logic

The app separates discipline from edge.

The discipline layer pulls public market intelligence using yfinance/Yahoo Finance, then computes:

- Momentum
- Exhaustion Risk
- Fundamental Trend
- Valuation Stretch
- Event Risk
- Dilution Risk
- Ownership Signal
- Narrative Heat
- Squeeze Risk
- Discipline Skew

The edge layer then asks whether true asymmetry can be established:

- What is already priced in through options implied move and sell-side targets?
- What is the bull/base/stress scenario map?
- Is there any non-consensus signal from filings, insider activity, or less-common data?
- Is the result calibrated or still heuristic?

OpenAI receives those computed facts and explains the trade setup, catalyst, invalidation trigger, and key metric.
The model is synthesis, not the primary scoring engine.

## Decision Philosophy

The framework is built for quick trading decisions:

- Avoid chasing hot stories when fundamentals, dilution, or valuation do not confirm the move.
- Separate real setup quality from narrative heat.
- Reward catalysts only when timing and evidence are visible.
- Treat missing data as a risk signal instead of filling gaps with confident prose.
- Surface the one or two datapoints that would change the view.
- No market expectations baseline means no true asymmetry claim.
- No calibrated probability-weighted payoff means no true asymmetry claim.
- No non-consensus evidence means no edge claim.

## Setup

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

Create `.streamlit/secrets.toml`:

```toml
OPENAI_API_KEY = "sk-your-key"
OPENAI_MODEL = "gpt-4.1"
OPENAI_TIMEOUT_SECONDS = 180
CACHE_DIR = "cache"
ALLOW_MODEL_OVERRIDE = false
ALLOW_LEGACY_TURBO = false
```

## Run

```powershell
streamlit run app.py
```

## Model Notes

The app defaults to `gpt-4.1` for stronger instruction following than the original `gpt-4-turbo` workflow. A legacy `OPENAI_MODEL = "gpt-4-turbo"` secret is automatically upgraded to the default unless `ALLOW_LEGACY_TURBO = true`. For stronger current citations, set `OPENAI_MODEL` to a search-enabled or newer OpenAI model available to your account.
