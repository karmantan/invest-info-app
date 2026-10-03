"""Portfolio analytics: what the holdings from the latest statement are worth and how they performed.

Trade Republic statements contain holdings, not the transaction history, so the
performance line answers: "how did the investments I hold now do over this
period, compared with the market?" (buy-and-hold of today's quantities).
"""
from __future__ import annotations
import numpy as np
import pandas as pd

PERIODS = ["1M", "3M", "6M", "YTD", "1Y", "3Y", "5Y", "10Y"]


def holding_kind(name: str | None, isin: str | None) -> str:
    text = (name or "").lower()
    if any(k in text for k in ("(acc)", "(dist)", " etf", "ucits", "etc")): return "ETF / fund"
    return "Single stock"


def map_holdings(holdings: pd.DataFrame, crypto: pd.DataFrame | None, resolver, crypto_symbols: dict) -> pd.DataFrame:
    rows = []
    for _, h in holdings.iterrows():
        rows.append({"name": h.security_name or h["isin"], "isin": h["isin"], "kind": holding_kind(h.security_name, h["isin"]), "quantity": h.quantity,
                     "statement_value": float(h.market_value_eur or 0), "symbol": resolver.resolve(h["isin"])})
    for _, c in (crypto if crypto is not None else pd.DataFrame()).iterrows():
        rows.append({"name": c["name"] or c.symbol, "isin": None, "kind": "Crypto", "quantity": c.quantity,
                     "statement_value": float(c.market_value_eur or 0), "symbol": crypto_symbols.get(str(c.symbol or "").upper())})
    return pd.DataFrame(rows, columns=["name", "isin", "kind", "quantity", "statement_value", "symbol"])


def _anchor_scale(h: pd.Series, s: pd.Series, as_of: pd.Timestamp | None) -> tuple[float | None, str]:
    """Scale Yahoo prices so they equal the statement value on the statement date.

    Only relative price moves from Yahoo are used, so a listing in another unit
    cannot distort values. A price that disagrees with the statement by more than
    25% probably belongs to a different instrument and is not used."""
    s = s.dropna()
    if not len(s) or not h.quantity or not h.statement_value: return None, "no price data"
    anchor = s[s.index <= as_of] if as_of is not None else s
    if not len(anchor): return None, "price history starts after the statement date"
    price = float(anchor.iloc[-1])
    if price <= 0: return None, "no price data"
    statement_price = h.statement_value / float(h.quantity)
    if not 0.75 <= price / statement_price <= 1.33: return None, "price does not match the statement — symbol may be wrong"
    return h.statement_value / price, "ok"


def price_check(mapped: pd.DataFrame, prices: dict[str, pd.Series], as_of: pd.Timestamp | None) -> pd.DataFrame:
    out = mapped.copy()
    out["price_status"] = [("no symbol found" if not h.symbol else _anchor_scale(h, prices[h.symbol], as_of)[1] if prices.get(h.symbol) is not None else "no price data") for _, h in mapped.iterrows()]
    return out


def value_history(mapped: pd.DataFrame, prices: dict[str, pd.Series], as_of: pd.Timestamp | None = None) -> pd.DataFrame:
    """Daily EUR value of each position held at today's quantity."""
    columns = {}
    for i, h in mapped.iterrows():
        s = prices.get(h.symbol)
        if s is None: continue
        scale, _ = _anchor_scale(h, s, as_of)
        if scale is None: continue
        columns[i] = s.dropna() * scale
    if not columns: return pd.DataFrame()
    frame = pd.DataFrame(columns).sort_index()
    frame = frame[frame.index.dayofweek < 5]
    # Carry the last price across a holiday on one exchange, but never back-fill before a listing.
    return frame.ffill(limit=5)


def period_start(end: pd.Timestamp, label: str) -> pd.Timestamp:
    if label == "YTD": return pd.Timestamp(year=end.year, month=1, day=1)
    unit = label[-1]; n = int(label[:-1])
    return end - (pd.DateOffset(months=n) if unit == "M" else pd.DateOffset(years=n))


def buy_and_hold_return(values: pd.DataFrame, start: pd.Timestamp) -> pd.Series:
    """Chain-linked cumulative return; positions without a price on a day sit out that day."""
    v = values[values.index >= start]
    if len(v) < 2: return pd.Series(dtype=float)
    prev = v.shift(1)
    valid = v.notna() & prev.notna()
    gain = (v - prev).where(valid).sum(axis=1)
    base = prev.where(valid).sum(axis=1)
    daily = (gain / base.replace(0, np.nan)).fillna(0.0)
    daily.iloc[0] = 0.0
    return (1 + daily).cumprod() - 1


def index_return(series: pd.Series, start: pd.Timestamp) -> pd.Series:
    s = series[series.index >= start].dropna()
    return s / s.iloc[0] - 1 if len(s) else s


def coverage(values: pd.DataFrame, mapped: pd.DataFrame, start: pd.Timestamp) -> float:
    """Share of today's statement value whose prices go back to the period start."""
    total = mapped.statement_value.sum()
    if values.empty or not total: return 0.0
    first = values[values.index >= start]
    if first.empty: return 0.0
    ok = [i for i in values.columns if pd.notna(first[i].iloc[0])]
    return float(mapped.loc[ok, "statement_value"].sum() / total)


def position_returns(values: pd.DataFrame, mapped: pd.DataFrame, start: pd.Timestamp) -> pd.DataFrame:
    rows = []
    v = values[values.index >= start]
    for i in v.columns:
        s = v[i].dropna()
        if len(s) < 2: continue
        rows.append({"name": mapped.loc[i, "name"], "kind": mapped.loc[i, "kind"], "return": float(s.iloc[-1] / s.iloc[0] - 1),
                     "gain_eur": float(s.iloc[-1] - s.iloc[0]), "since": str(s.index[0].date())})
    return pd.DataFrame(rows).sort_values("return", ascending=False) if rows else pd.DataFrame(rows)


def live_values(mapped: pd.DataFrame, values: pd.DataFrame) -> pd.DataFrame:
    """Latest market value per position; falls back to the statement value when no usable price exists."""
    out = mapped.copy()
    latest, dates = [], []
    for i, h in out.iterrows():
        s = values[i].dropna() if i in values.columns else pd.Series(dtype=float)
        if len(s): latest.append(float(s.iloc[-1])); dates.append(str(s.index[-1].date()))
        else: latest.append(h.statement_value); dates.append(None)
    out["live_value"] = latest; out["price_date"] = dates
    return out


def monthly_returns(series: pd.Series) -> pd.Series:
    return series.resample("ME").last().pct_change().dropna()


def portfolio_series(values: pd.DataFrame) -> pd.Series:
    """A price-like index of the current holdings (for correlations)."""
    if values.empty: return pd.Series(dtype=float)
    return 1 + buy_and_hold_return(values, values.index[0])


def correlation(a: pd.Series, b: pd.Series, min_months: int = 24) -> float | None:
    if a is None or b is None or not len(a) or not len(b): return None
    both = pd.concat([monthly_returns(a), monthly_returns(b)], axis=1, join="inner").dropna()
    if len(both) < min_months: return None
    return float(both.corr().iloc[0, 1])


def diversification_label(corr: float | None) -> str:
    if corr is None: return "Unknown"
    if corr < 0.5: return "Adds a lot of variety"
    if corr < 0.8: return "Adds some variety"
    return "Moves like what you own"


def whatif_value(row: pd.Series, prices: pd.Series | None) -> dict | None:
    """Current value of a saved pretend investment (before selling costs and tax)."""
    if prices is None or not len(prices): return None
    s = prices.dropna(); entry = pd.Timestamp(row.entry_date)
    if row["mode"] == "monthly":
        buys = s[s.index >= entry].resample("MS").first().dropna()
        units = float((row.amount_eur / buys).sum()); paid = float(row.amount_eur * len(buys))
    else:
        units = float(row.amount_eur / row.entry_price); paid = float(row.amount_eur)
    value = units * float(s.iloc[-1])
    return {"value": value, "paid": paid, "gain": value - paid, "return": value / paid - 1 if paid else 0.0, "date": str(s.index[-1].date())}
