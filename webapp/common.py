"""Shared state, cached data loading, formatting and chart styling for the website."""
from __future__ import annotations
import json, os, sys
from pathlib import Path
import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path: sys.path.insert(0, str(ROOT))

from src.web import model
from src.web import portfolio as pf
from src.web.market import IsinResolver, PriceStore
from src.web.settings import data_dir, load_universe, load_web_config, merge_settings
from src.web.store import WebStore

DEMO = os.getenv("INVEST_DEMO_PRICES") == "1"

# Categorical series colours (validated reference palette, fixed order). The
# portfolio / chosen ETF is always slot 1, so colour follows the entity.
SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#4a3aa7"]
MUTED = "#8a8985"

SHORT = {"label": "Short term", "min": 1, "max": 12, "default": 12, "unit": "months",
         "blurb": "Up to a year. Prices are mostly random over a few months, so fees, taxes and risk matter more than the forecast."}
MEDIUM = {"label": "Medium term", "min": 1, "max": 5, "default": 3, "unit": "years",
          "blurb": "One to five years. Long enough for the expected return to start showing, short enough that a bad market can still leave you down."}
LONG = {"label": "Long term", "min": 5, "max": 30, "default": 10, "unit": "years",
        "blurb": "Five years or more. Time smooths out ups and downs; low costs and broad diversification win."}


# ---------------------------------------------------------------------------- data access
@st.cache_resource
def get_store() -> WebStore:
    return WebStore()


def get_settings() -> dict:
    return merge_settings(load_web_config(), get_store().settings_overrides())


@st.cache_resource
def get_universe() -> pd.DataFrame:
    return load_universe()


@st.cache_resource
def price_store() -> PriceStore:
    fetcher = None
    if DEMO:
        from src.web.demo import synthetic_chart
        fetcher = synthetic_chart
    return PriceStore(data_dir() / "prices", load_web_config()["data"].get("max_age_hours", 12), fetcher=fetcher)


@st.cache_resource
def resolver() -> IsinResolver:
    search = None
    if DEMO: search = lambda isin: []
    return IsinResolver(data_dir() / "isin_symbols.json", load_web_config().get("isin_overrides", {}), get_universe(), search=search)


@st.cache_data(ttl=1800, show_spinner=False)
def load_prices(symbols: tuple[str, ...]) -> dict:
    """{symbol: (EUR price series or None, metadata)}; cached for 30 minutes per symbol set."""
    return price_store().many(list(symbols))


def prices_for(symbols) -> dict[str, pd.Series]:
    return {k: v[0] for k, v in load_prices(tuple(sorted(set(s for s in symbols if s)))).items() if v[0] is not None}


@st.cache_data(ttl=1800, show_spinner=False)
def _estimates(settings_json: str) -> dict:
    settings = json.loads(settings_json); uni = get_universe(); m = settings["model"]
    symbols = tuple(sorted(set(uni.ticker) | {m["market_ticker"], m["cash_ticker"]}))
    loaded = load_prices(symbols)
    series = {k: v[0] for k, v in loaded.items()}
    rf = model.risk_free_rate(series.get(m["cash_ticker"]), m["risk_free_fallback"])
    out, errors = {}, {}
    for _, etf in uni.iterrows():
        s = series.get(etf.ticker)
        if s is None: errors[etf.ticker] = "no price data"; continue
        try: out[etf.ticker] = model.estimate(s, series.get(m["market_ticker"]), etf.to_dict(), rf, settings)
        except Exception as exc: errors[etf.ticker] = str(exc)
    dates = [e.last_date for e in out.values()]
    names = {k: v[1].get("name") for k, v in loaded.items()}
    return {"estimates": out, "rf": rf, "errors": errors, "data_date": max(dates) if dates else None, "yahoo_names": names}


def estimates(settings: dict) -> dict:
    with st.spinner("Loading market data… (the first visit of the day can take ~20 seconds)"):
        return _estimates(json.dumps(settings, sort_keys=True))


def portfolio_context(settings: dict) -> dict | None:
    """Latest statement, mapped to price symbols, with daily value history."""
    latest = get_store().latest()
    if latest is None: return None
    mapped = pf.map_holdings(latest["holdings"], latest["crypto"], resolver(), settings.get("crypto_symbols", {}))
    with st.spinner("Loading prices for your holdings…"):
        prices = prices_for(list(mapped.symbol.dropna()) + [b["ticker"] for b in settings["benchmarks"]])
    as_of = pd.Timestamp(latest["date"]) if latest.get("date") else None
    values = pf.value_history(mapped, prices, as_of)
    checked = pf.price_check(mapped, prices, as_of)
    live = pf.live_values(checked, values)
    return {**latest, "mapped": checked, "prices": prices, "values": values, "live": live}


# ---------------------------------------------------------------------------- formatting
def eur(x, decimals: int = 0) -> str:
    if x is None or (isinstance(x, float) and pd.isna(x)): return "—"
    sign = "-" if x < 0 else ""
    return f"{sign}€{abs(x):,.{decimals}f}"


def signed_eur(x, decimals: int = 0) -> str:
    return ("+" if x >= 0 else "") + eur(x, decimals)


def pct(x, decimals: int = 1, signed: bool = False) -> str:
    if x is None or (isinstance(x, float) and pd.isna(x)): return "—"
    text = f"{abs(x) * 100:.{decimals}f}%"
    if x < 0: return "-" + text
    return ("+" + text) if signed else text


def horizon_text(years: float) -> str:
    months = int(round(years * 12))
    if months < 12: return f"{months} month" + ("s" if months != 1 else "")
    if months % 12 == 0: return f"{months // 12} year" + ("s" if months != 12 else "")
    return f"{months / 12:.1f} years"


def chance_text(p: float) -> str:
    """Probability in words a non-specialist reads correctly."""
    if p < 0.005: return "almost never"
    n = max(1, round(p * 10))
    return f"about {n} in 10" if p >= 0.05 else f"about {max(1, round(p * 100))} in 100"


def chart_layout(fig, height: int = 380, y_title: str | None = None, y_format: str | None = None):
    fig.update_layout(height=height, margin=dict(l=8, r=8, t=36, b=8), hovermode="x unified",
                      legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0, title=None))
    fig.update_yaxes(title=y_title, tickformat=y_format, zeroline=True, gridcolor="rgba(128,128,128,0.15)")
    fig.update_xaxes(showgrid=False)
    return fig


def demo_banner() -> None:
    if DEMO: st.warning("Demo mode: prices are synthetic test data with no market meaning. Unset INVEST_DEMO_PRICES for real prices.", icon="🧪")


def disclaimer() -> None:
    st.caption("Estimates from a statistical model, not a promise and not personal financial advice. Taxes are simplified — check with a tax adviser for anything important.")


def go(page: str) -> None:
    """Switch to another page of the site (keys as in app.PAGES)."""
    st.switch_page(st.session_state["_pages"][page])
