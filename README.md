# DD Framework - Institutional Due Diligence

Production-grade Streamlit app for a three-step OpenAI-powered equity due diligence workflow.

## Features

- Ticker input and validation
- Three sequential GPT analysis steps
- Cached ticker reports with a cached-on timestamp
- Dashboard scores from 1-10
- Recommendation: PROCEED, HOLD, or AVOID
- JSON, Markdown, and copy-to-clipboard exports
- Friendly handling for missing API keys, invalid tickers, rate limits, timeouts, and API errors

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
