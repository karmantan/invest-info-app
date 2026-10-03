from __future__ import annotations

from datetime import date
import html
import re

import pandas as pd


def global_css() -> str:
    """Shared dashboard theme and responsive, non-scrolling record layouts."""
    return """
    <style>
    :root { --ink:#16221c; --muted:#5d6962; --paper:#f6f5ef; --surface:#fff;
      --line:#d8ddd8; --green:#54715e; --sidebar:#edf0e9; --sidebar-selected:#d9dfd7; }
    .stApp { background:var(--paper); color:var(--ink); }
    h1,h2,h3 { font-family:Georgia,serif; letter-spacing:-.02em; }
    .block-container { padding-top:1.8rem; max-width:1220px; }
    .action { background:var(--surface); border:1px solid var(--line); border-left:5px solid var(--green);
      padding:1.1rem 1.3rem; border-radius:6px; margin:.8rem 0 1.3rem; }
    .action h2 { margin:0; font-size:1.45rem; }
    .action p { color:var(--muted); margin:.3rem 0 0; }
    .stMetric { background:var(--surface); border:1px solid var(--line); padding:12px; border-radius:6px; }

    /* The sidebar is styled only here. Explicit descendant colors prevent page CSS/theme regressions. */
    [data-testid="stSidebar"] { background:var(--sidebar); border-right:1px solid var(--line); color:var(--ink); }
    [data-testid="stSidebar"] :is(p,span,label,div,a,small,summary) { color:var(--ink); }
    [data-testid="stSidebar"] [data-testid="stCaptionContainer"] :is(p,span) { color:var(--muted); }
    [data-testid="stSidebar"] [role="radiogroup"] label { border-radius:6px; padding:.22rem .42rem; }
    [data-testid="stSidebar"] [role="radiogroup"] label:hover { background:#e1e5df; }
    [data-testid="stSidebar"] [role="radiogroup"] label:has(input:checked) { background:var(--sidebar-selected); font-weight:650; }
    [data-testid="stSidebar"] button, [data-testid="stSidebar"] summary,
    [data-testid="stSidebar"] [data-baseweb="select"] > div { color:var(--ink); border-color:#aeb8b0; }
    [data-testid="stSidebar"] svg { fill:currentColor; color:var(--ink); }

    .qc-card-grid { display:grid; grid-template-columns:repeat(3,minmax(0,1fr)); gap:.75rem; margin:.6rem 0 1rem; }
    .qc-card { min-width:0; background:var(--surface); border:1px solid var(--line); border-radius:7px;
      padding:.85rem .95rem; overflow-wrap:anywhere; }
    .qc-card h3 { margin:0 0 .55rem; font-size:1.05rem; line-height:1.25; }
    .qc-fields { display:grid; grid-template-columns:minmax(7rem,38%) minmax(0,1fr); gap:.34rem .6rem; }
    .qc-label { color:var(--muted); font-size:.77rem; font-weight:650; line-height:1.3; }
    .qc-value { color:var(--ink); font-size:.88rem; line-height:1.35; min-width:0; overflow-wrap:anywhere; }
    .qc-card details { border-top:1px solid var(--line); margin-top:.65rem; padding-top:.48rem; }
    .qc-card summary { color:var(--green); cursor:pointer; font-size:.84rem; font-weight:650; }
    .qc-card details .qc-fields { margin-top:.55rem; }
    .qc-horizon { border-top:1px solid var(--line); padding:.7rem 0 .15rem; }
    .qc-horizon:first-of-type { border-top:0; padding-top:0; }
    .qc-horizon h4 { color:var(--ink); font-size:.92rem; line-height:1.3; margin:0 0 .45rem; }
    .qc-info-panel { display:grid; grid-template-columns:repeat(4,minmax(0,1fr)); gap:.65rem;
      background:var(--surface); border:1px solid var(--line); border-radius:7px; padding:.8rem .9rem; margin:.4rem 0 .65rem; }
    .qc-info-item { min-width:0; }
    .qc-info-label { color:var(--muted); font-size:.75rem; font-weight:650; line-height:1.25; margin-bottom:.18rem; }
    .qc-info-value { color:var(--ink); font-size:.95rem; font-weight:550; line-height:1.35; overflow-wrap:anywhere; }
    .qc-table { width:100%; table-layout:fixed; border-collapse:collapse; background:var(--surface); font-size:.88rem; }
    .qc-table th,.qc-table td { border-bottom:1px solid var(--line); padding:.62rem .68rem; text-align:left;
      vertical-align:top; white-space:normal; overflow-wrap:anywhere; word-break:normal; }
    .qc-table th { background:#eef1eb; font-size:.79rem; }
    .qc-table-wrap { width:100%; max-width:100%; overflow:visible; }
    @media (prefers-color-scheme:dark) {
      :root { --ink:#edf2ee; --muted:#b7c1ba; --paper:#111713; --surface:#19211c;
        --line:#3c4940; --sidebar:#172019; --sidebar-selected:#334138; }
      [data-testid="stSidebar"] [role="radiogroup"] label:hover { background:#29352d; }
      .qc-table th { background:#253129; }
    }
    @media (max-width:1100px) { .qc-card-grid { grid-template-columns:repeat(2,minmax(0,1fr)); } }
    @media (max-width:800px) {
      .qc-card-grid { grid-template-columns:minmax(0,1fr); }
      .qc-info-panel { grid-template-columns:repeat(2,minmax(0,1fr)); }
      .qc-table,.qc-table tbody,.qc-table tr,.qc-table td { display:block; width:100%; }
      .qc-table thead { display:none; }
      .qc-table tr { border:1px solid var(--line); border-radius:6px; margin-bottom:.65rem; padding:.4rem .55rem; }
      .qc-table td { border:0; padding:.25rem 0; }
      .qc-table td::before { content:attr(data-label); color:var(--muted); display:block; font-size:.75rem; font-weight:650; }
    }
    </style>
    """


def _safe(value: object) -> str:
    if value is None or (not isinstance(value, (list, dict)) and pd.isna(value)):
        value = "Not available"
    return html.escape(str(value))


def compact_table_html(rows: list[dict[str, object]], column_widths: list[str] | None = None) -> str:
    if not rows:
        return ""
    headers = list(rows[0])
    widths = column_widths or [f"{100 / len(headers):.3f}%"] * len(headers)
    colgroup = "".join(f'<col style="width:{_safe(width)}">' for width in widths)
    head = "".join(f"<th>{_safe(label)}</th>" for label in headers)
    body = "".join("<tr>" + "".join(
        f'<td data-label="{_safe(label)}">{_safe(row.get(label))}</td>' for label in headers
    ) + "</tr>" for row in rows)
    return f'<div class="qc-table-wrap"><table class="qc-table"><colgroup>{colgroup}</colgroup><thead><tr>{head}</tr></thead><tbody>{body}</tbody></table></div>'


def responsive_cards_html(records: list[dict[str, object]], title_key: str,
                          visible_keys: list[str], detail_keys: list[str] | None = None) -> str:
    cards = []
    for record in records:
        fields = "".join(f'<div class="qc-label">{_safe(key)}</div><div class="qc-value">{_safe(record.get(key))}</div>' for key in visible_keys)
        details = ""
        if detail_keys:
            detail_fields = "".join(f'<div class="qc-label">{_safe(key)}</div><div class="qc-value">{_safe(record.get(key))}</div>' for key in detail_keys)
            details = f'<details><summary>More details</summary><div class="qc-fields">{detail_fields}</div></details>'
        cards.append(f'<article class="qc-card"><h3>{_safe(record.get(title_key))}</h3><div class="qc-fields">{fields}</div>{details}</article>')
    return '<div class="qc-card-grid">' + "".join(cards) + "</div>"


def opportunity_cards_html(records: list[dict[str, object]]) -> str:
    """Render one asset card containing all of that asset's horizon records."""
    grouped: dict[str, list[dict[str, object]]] = {}
    for record in records:
        grouped.setdefault(str(record.get("Investment")), []).append(record)

    cards = []
    detail_keys = ["Exact horizon", "P10", "P5", "Model score/reliability", "Model version",
                   "Canonical status", "Canonical forecast status", "Sample size", "Benchmark",
                   "Risk-policy code", "Technical risk flags", "Feature values", "As-of date"]
    for investment, horizons in grouped.items():
        sections = []
        for record in horizons:
            fields = "".join(
                f'<div class="qc-label">{_safe(key)}</div><div class="qc-value">{_safe(record.get(key))}</div>'
                for key in ["Current view", "Risk", "Historical evidence", "Why"]
            )
            detail_fields = "".join(
                f'<div class="qc-label">{_safe(key)}</div><div class="qc-value">{_safe(record.get(key))}</div>'
                for key in detail_keys
            )
            sections.append(
                f'<section class="qc-horizon"><h4>{_safe(record.get("Time horizon"))}</h4>'
                f'<div class="qc-fields">{fields}</div><details><summary>More details</summary>'
                f'<div class="qc-fields">{detail_fields}</div></details></section>'
            )
        cards.append(f'<article class="qc-card"><h3>{_safe(investment)}</h3>{"".join(sections)}</article>')
    return '<div class="qc-card-grid qc-opportunity-grid">' + "".join(cards) + "</div>"


def info_panel_html(items: list[tuple[object, object]]) -> str:
    fields = "".join(
        f'<div class="qc-info-item"><div class="qc-info-label">{_safe(label)}</div>'
        f'<div class="qc-info-value">{_safe(value)}</div></div>'
        for label, value in items
    )
    return f'<div class="qc-info-panel">{fields}</div>'


ASSET_NAMES = {
    "SPY": "S&P 500",
    "URTH": "MSCI World",
    "XLK": "US Technology",
    "SMH": "Semiconductors",
    "XBI": "US Biotechnology",
    "XLE": "US Energy",
    "EEM": "Emerging Markets",
}

STATUS_LABELS = {
    "NO EDGE": "No opportunity detected",
    "NO MODEL EDGE": "No opportunity detected",
    "MODEL INTERESTING": "Historically promising",
    "MODEL INTERESTING — RISK GATE PASSES": "Promising setup — risk checks passed",
    "MODEL INTERESTING — SOFT VOLATILITY FLAG": "Promising setup — unusually large price swings",
    "MODEL INTERESTING — MARKET DOWNTREND VETO": "Promising setup — blocked by broad-market decline",
    "MODEL INTERESTING — EXTREME DRAWDOWN VETO": "Promising setup — blocked after a very large fall",
    "MODEL INTERESTING — RISK VETO": "Promising setup — blocked by risk warning",
    "PAPER BUY": "Paper position opened",
    "PAPER HOLD": "Paper position being tracked",
    "PAPER REDUCE": "Paper position would be reduced",
    "PAPER EXIT": "Paper position closed",
    "WATCH": "Worth watching",
    "PASS": "Risk checks passed",
    "FAIL": "Blocked by risk rules",
    "REJECT": "Blocked by risk rules",
}

RISK_LABELS = {
    "high_asset_volatility": "Price swings are unusually large",
    "high_market_volatility": "Market-wide uncertainty is unusually high",
    "broad_market_downtrend": "Broad US market is in a sustained decline",
    "extreme_asset_drawdown": "Investment has fallen more than 20% from its recent high",
    "none": "No current warning",
}

MODEL_LABELS = {
    "ridge": "Statistical model (Ridge)",
    "gradient_boosting": "Flexible machine-learning model (Gradient boosting)",
    "momentum": "Momentum rule",
    "unconditional": "Historical-average baseline",
    "dip": "Dip-buying rule",
}

GLOSSARY = {
    "Asset / ticker": "An asset is an investment. A ticker is its short exchange code; for example, SMH is the code used for the Semiconductors research proxy.",
    "Paper trade": "A hypothetical position tracked with no real money. Nothing is sent to a broker.",
    "Passive benchmark": "A broad-market investment used as a simple comparison for the strategy.",
    "Excess return": "The investment's return minus the benchmark's return over the same period.",
    "Momentum": "The tendency for investments that have performed relatively well recently to continue doing so for a time.",
    "Volatility": "How widely and quickly a price moves. Higher volatility means larger price swings and greater uncertainty.",
    "Drawdown": "The percentage fall from a recent high.",
    "Bad-case outcome": "A historical result near the bottom 10% of comparable outcomes. Roughly 1 in 10 was worse.",
    "Severe-case outcome": "A historical result near the bottom 5% of comparable outcomes. Roughly 1 in 20 was worse.",
    "Risk check": "A fixed set of conditions that can block or flag an otherwise promising setup.",
    "Historical backtest": "A test of a fixed strategy on past data. It is research, not a forecast or money earned.",
    "Prospective paper test": "A forward-looking test that records hypothetical decisions as they occur, without real money.",
    "Model": "A repeatable calculation that combines market indicators to rank historical setups.",
    "Benchmark": "The broad-market comparison used to judge whether a result added value.",
    "SPY": "A US-listed ETF used here as the S&P 500 research proxy.",
    "URTH": "A US-listed ETF used here as the MSCI World research proxy.",
}


def asset_name(ticker: object, secondary: bool = True) -> str:
    code = "" if ticker is None else str(ticker).upper()
    friendly = ASSET_NAMES.get(code, code or "Not available")
    return f"{friendly} ({code})" if secondary and code in ASSET_NAMES else friendly


def horizon_label(days: object, technical: bool = False) -> str:
    try:
        value = int(days)
    except (TypeError, ValueError):
        return "Not available"
    labels = {21: "About 1 month", 63: "About 3 months", 126: "About 6 months"}
    friendly = labels.get(value, f"{value} trading days")
    return f"{friendly} ({value} trading days)" if technical and value in labels else friendly


def status_label(value: object) -> str:
    raw = "" if value is None else str(value).strip().upper()
    return STATUS_LABELS.get(raw, raw.replace("_", " ").capitalize() if raw else "Not available")


def risk_label(value: object) -> str:
    raw = "none" if value is None or not str(value).strip() else str(value).strip()
    parts = [p.strip() for p in raw.split(",")]
    return "; ".join(RISK_LABELS.get(p, p.replace("_", " ").capitalize()) for p in parts)


def model_label(value: object) -> str:
    raw = "" if value is None else str(value).strip().lower()
    return MODEL_LABELS.get(raw, raw.replace("_", " ").title() or "Not available")


def percent(value: object, digits: int = 1) -> str:
    try:
        if pd.isna(value):
            return "Not available"
        return f"{float(value):.{digits}%}"
    except (TypeError, ValueError):
        return "Not available"


def money(value: object, decimals: int = 0) -> str:
    try:
        if pd.isna(value):
            return "Not available"
        return f"€{float(value):,.{decimals}f}"
    except (TypeError, ValueError):
        return "Not available"


def friendly_reason(raw: object, asset: object = None, horizon: object = None, status: object = None, risk: object = None) -> str:
    text = "" if raw is None else str(raw)
    code = "" if status is None else str(status).upper()
    name = asset_name(asset, secondary=False)
    period = horizon_label(horizon).lower()
    if "21-day research signal" in text:
        if code == "WATCH":
            return f"{name} is worth watching over {period}, but short-term real-money recommendations are disabled."
        return f"The short-term research does not currently identify a strong enough opportunity in {name}."
    if "outside the selected cohort" in text:
        return f"The historical model does not currently find enough evidence that {name} is likely to outperform over {period}."
    if "another simultaneous horizon" in text:
        warning = risk_label(risk)
        suffix = "" if warning == "No current warning" else f" {warning}, but this does not automatically block the current risk rules."
        return f"{name} ranks among the stronger medium-term historical opportunities.{suffix}"
    if "longest simultaneously triggered horizon" in text:
        return f"The historical setup remains positive for {name}, and current risk rules allow a paper position to be tracked over {period}."
    cleaned = re.sub(r"\bPolicy\s*2\b", "current risk rules", text, flags=re.I)
    cleaned = cleaned.replace("Frozen ridge", "The historical model").replace("top cohort", "stronger group")
    cleaned = cleaned.replace("zero-capital prospective test", "paper test with no real money")
    return cleaned.strip().rstrip(".") + "." if cleaned.strip() else "No further explanation is available."


def days_remaining(entry_date: object, horizon_days: object, as_of: date | None = None) -> int | None:
    try:
        start = pd.Timestamp(entry_date).date()
        elapsed = len(pd.bdate_range(start, as_of or date.today())) - 1
        return max(int(horizon_days) - max(elapsed, 0), 0)
    except (TypeError, ValueError):
        return None
