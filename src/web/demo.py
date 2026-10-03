"""Synthetic prices for offline tests and screenshots (INVEST_DEMO_PRICES=1).
They have no market meaning; the website shows a banner whenever they are used."""
from __future__ import annotations
import hashlib
import numpy as np
import pandas as pd

PROFILE = {"XEON": (0.02, 0.003), "EUNH": (0.025, 0.05), "EUN5": (0.03, 0.05), "4GLD": (0.06, 0.15),
           "BTC-EUR": (0.4, 0.7), "ETH-EUR": (0.4, 0.8)}


def synthetic_chart(symbol: str) -> tuple[pd.Series, dict]:
    seed = int(hashlib.sha256(symbol.encode()).hexdigest()[:8], 16)
    rng = np.random.default_rng(seed)
    if symbol.startswith("EUR") and symbol.endswith("=X"):
        dates = pd.bdate_range(end=pd.Timestamp("2026-10-02"), periods=252 * 20)
        return pd.Series(1.1 * np.exp(np.cumsum(rng.normal(0, 0.005, len(dates)))), index=dates), {"currency": symbol[3:6], "name": symbol}
    mu, vol = PROFILE.get(symbol.split(".")[0], (0.05 + 0.06 * rng.random(), 0.12 + 0.18 * rng.random()))
    years = int(rng.integers(6, 20))
    dates = pd.bdate_range(end=pd.Timestamp("2026-10-02"), periods=252 * years)
    steps = rng.normal(mu / 252 - vol ** 2 / 504, vol / np.sqrt(252), len(dates))
    currency = "USD" if "." not in symbol and "-" not in symbol else "EUR"
    return pd.Series(50 * np.exp(np.cumsum(steps)), index=dates), {"currency": currency, "name": f"Demo {symbol}"}
