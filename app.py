"""DD Framework - Institutional Due Diligence Streamlit app."""

from __future__ import annotations

import hashlib
import html
import json
import re
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import streamlit as st
import streamlit.components.v1 as components
from openai import APIConnectionError, APIError, APITimeoutError, AuthenticationError, OpenAI, RateLimitError


APP_TITLE = "DD Framework - Institutional Due Diligence"
DEFAULT_MODEL = "gpt-4-turbo"
DEFAULT_TIMEOUT_SECONDS = 180
SCORE_NAMES = [
    "Moat Score",
    "Adoption Clarity",
    "Competitive Position",
    "Thesis Strength",
]

SYSTEM_PROMPT = """
You are an institutional equity research analyst preparing an investment committee due diligence memo.
Be evidence-led, balanced, and explicit about uncertainty. Cite sources wherever available, never invent
precise figures, and flag missing or stale data. Use confidence scores from 1-10 for material claims.
At the end of every answer, include a compact section named DD_FRAMEWORK_SIGNAL with any relevant
scores, probabilities, or recommendation evidence that can support a downstream dashboard.
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
            return text.strip()
        return extract_responses_text(response).strip()
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
    return completion.choices[0].message.content.strip()


def extract_responses_text(response: Any) -> str:
    chunks: list[str] = []
    for item in getattr(response, "output", []) or []:
        for content in getattr(item, "content", []) or []:
            text = getattr(content, "text", None)
            if text:
                chunks.append(text)
    return "\n".join(chunks)


def build_step_1_prompt(ticker: str) -> str:
    return f"""
You are an institutional equity research analyst. Research {ticker}:
1. BUSINESS MODEL: Products, revenue breakdown (%), installed base, ASP, pre/early/profitable?
2. MOAT: Primary advantage? Durability (1-10)? Why?
3. KEY METRICS: Q revenue ($M), YoY growth (%), Top 3 competitors
4. CONFIDENCE: Rate each answer (1-10). Missing data?
Format: 800 words, cite sources.
""".strip()


def build_step_2_prompt(step_1: str) -> str:
    return f"""
Based on Step 1: {step_1}
1. HOSPITAL ECONOMICS: Replacement cycle (yrs)? Switching cost ($M + months)? Payback period (months)?
2. REIMBURSEMENT: CPT code? Rate per procedure? 5yr trend? Surgeon margin? Adoption threshold?
3. GROWTH: What drives it? What stalls it? Current adoption rate (%)?
4. UNIT ECONOMICS: Equipment margin (%), consumables margin (%), payback via recurring ($)?
Format: 900 words, show confidence.
""".strip()


def build_step_3_prompt(step_1: str, step_2: str) -> str:
    return f"""
Based on Steps 1-2: {step_1}

{step_2}
1. BULL (3yrs): 3 advantages, inflection timeline, what goes RIGHT, probability (%)
2. BEAR: 3 threats, incumbent response, stall timeline, probability (%)
3. ASSESSMENT: Which likely? Market pricing in vs. missing? Asymmetry?
4. MOAT: 3yr strength (1-10), 5yr strength (1-10)
Format: 1000 words, balanced evidence.
""".strip()


def run_analysis(ticker: str, force_refresh: bool = False) -> DDResult:
    cached = load_cache(ticker)
    if cached and not force_refresh:
        return cached

    client = get_openai_client()
    model = get_model()

    step_1_box = st.empty()
    step_2_box = st.empty()
    step_3_box = st.empty()

    with step_1_box.container():
        with st.spinner("Step 1: researching business model, moat, metrics, and confidence..."):
            step_1 = call_openai(client, build_step_1_prompt(ticker), model, step_name="Step 1")
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
        "Moat Score": find_score(
            combined,
            [r"moat(?:\s+score|\s+strength|\s+durability)?\D{0,40}(\d{1,2})\s*/\s*10"],
            default=6,
        ),
        "Adoption Clarity": score_adoption_clarity(combined),
        "Competitive Position": score_competitive_position(combined),
        "Thesis Strength": score_thesis_strength(combined),
    }
    return {name: clamp_score(score) for name, score in scores.items()}


def find_score(text: str, patterns: list[str], default: int) -> int:
    for pattern in patterns:
        match = re.search(pattern, text, flags=re.IGNORECASE)
        if match:
            return clamp_score(int(match.group(1)))
    return default


def score_adoption_clarity(text: str) -> int:
    score = 6
    positive_terms = ["adoption rate", "installed base", "replacement cycle", "payback", "cpt", "reimbursement"]
    uncertainty_terms = ["unknown", "not disclosed", "insufficient", "missing", "limited data"]
    score += min(2, sum(1 for term in positive_terms if term in text.lower()) // 2)
    score -= min(3, sum(1 for term in uncertainty_terms if term in text.lower()))
    return clamp_score(score)


def score_competitive_position(text: str) -> int:
    score = 6
    lower = text.lower()
    positives = ["differentiated", "leading", "proprietary", "regulatory", "installed base", "recurring"]
    negatives = ["incumbent", "commodity", "low switching", "price pressure", "crowded", "competition"]
    score += min(3, sum(1 for term in positives if term in lower))
    score -= min(3, sum(1 for term in negatives if term in lower) // 2)
    return clamp_score(score)


def score_thesis_strength(text: str) -> int:
    bull = extract_probability(text, "bull")
    bear = extract_probability(text, "bear")
    moat = find_score(text, [r"3yr\s+strength\D{0,30}(\d{1,2})\s*/\s*10"], default=6)
    if bull is None or bear is None:
        return moat
    spread = bull - bear
    if spread >= 25:
        return clamp_score(moat + 2)
    if spread >= 10:
        return clamp_score(moat + 1)
    if spread <= -20:
        return clamp_score(moat - 3)
    if spread <= -5:
        return clamp_score(moat - 1)
    return moat


def extract_probability(text: str, label: str) -> int | None:
    pattern = rf"{label}\D{{0,120}}(?:probability|prob\.)\D{{0,20}}(\d{{1,3}})\s*%"
    match = re.search(pattern, text, flags=re.IGNORECASE | re.DOTALL)
    if not match:
        pattern = rf"{label}\D{{0,80}}(\d{{1,3}})\s*%"
        match = re.search(pattern, text, flags=re.IGNORECASE | re.DOTALL)
    if not match:
        return None
    return max(0, min(100, int(match.group(1))))


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
    company = f"\n**Company:** {result.company_name}" if result.company_name else ""
    return f"""# {APP_TITLE}

**Ticker:** {result.ticker}{company}
**Model:** {result.model}
**Cached on:** {cached_on_label(result.cached_at)}
**Recommendation:** {result.recommendation}
**Reason:** {result.recommendation_reason}

## Dashboard

{scores}

## Step 1 - Business, Moat, Metrics

{result.step_1}

## Step 2 - Economics, Reimbursement, Growth

{result.step_2}

## Step 3 - Bull/Bear Assessment

{result.step_3}
"""


def render_dashboard(result: DDResult) -> None:
    st.subheader("Dashboard")
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
    with st.container(border=True):
        st.markdown(f"### {title}")
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
    st.caption("Three-step institutional equity research workflow with cached ticker memos and exportable outputs.")

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
            Enter a ticker, then run the workflow. Cached analyses return instantly unless refresh is enabled.
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
