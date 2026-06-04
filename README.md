# DD Framework - Institutional Due Diligence

Production-grade Streamlit app for decision-first equity due diligence and trading insight.

The V2 engine computes deterministic market evidence first, then uses OpenAI to explain the setup tersely.
The goal is not a company story; it is a fast read on whether a ticker is chaseable, waitlist-only, event-driven,
or avoid.

## Features

- Ticker input and validation
- Deterministic V2 evidence engine
- Three sequential GPT synthesis steps
- Cached ticker reports with a cached-on timestamp
- Dashboard scores from 1-10 for momentum, exhaustion, fundamentals, valuation, catalyst, dilution, squeeze, and asymmetry
- Recommendation: PROCEED, HOLD, or AVOID
- Trade verdict: CHASEABLE, WAIT FOR PULLBACK, EVENT WATCH, WATCHLIST, or AVOID CHASING
- JSON, Markdown, and copy-to-clipboard exports
- Friendly handling for missing API keys, invalid tickers, rate limits, timeouts, and API errors

## V2 Logic

The app pulls a market data pack using yfinance, then computes:

- Momentum
- Exhaustion Risk
- Fundamental Validation
- Valuation Stretch
- Catalyst Proximity
- Dilution Risk
- Squeeze Risk
- Asymmetry

OpenAI receives those computed facts and explains the trade setup, catalyst, invalidation trigger, and key metric.

## Setup

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

Create `.streamlit/secrets.toml`:

```toml
OPENAI_API_KEY = "sk-your-key"
OPENAI_MODEL = "gpt-4-turbo"
OPENAI_TIMEOUT_SECONDS = 180
CACHE_DIR = "cache"
ALLOW_MODEL_OVERRIDE = false
```

## Run

```powershell
streamlit run app.py
```

## Model Notes

The app defaults to `gpt-4-turbo` to match the original workflow. For stronger current citations, set `OPENAI_MODEL` to a search-enabled or newer OpenAI model available to your account.
