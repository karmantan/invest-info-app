import numpy as np
import pandas as pd
import pytest
from src.web import model
from src.web.settings import load_universe, load_web_config

S = load_web_config()
U = load_universe().set_index("ticker", drop=False)


def gbm(mu, vol, years=12, seed=1, start=100.0):
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range("2014-01-01", periods=252 * years)
    steps = rng.normal(mu / 252 - vol ** 2 / 504, vol / np.sqrt(252), len(dates))
    return pd.Series(start * np.exp(np.cumsum(steps)), index=dates)


def test_universe_is_complete_and_valid():
    u = load_universe()
    assert u.ticker.is_unique and u["isin"].is_unique and len(u) >= 25
    assert u["isin"].str.match(r"^[A-Z]{2}[A-Z0-9]{9}[0-9]$").all()
    assert set(u.tax_class) <= {"equity", "bond", "gold_etc", "mixed"}
    assert u.ter.between(0, 0.01).all() and u.spread_pct.between(0, 0.01).all()


def test_estimate_recovers_volatility_and_market_beta():
    market = gbm(0.07, 0.15, seed=2)
    est = model.estimate(market, market, U.loc["EUNL.DE"].to_dict(), 0.02, S)
    assert est.vol_long == pytest.approx(0.15, abs=0.03)
    assert est.beta == pytest.approx(1.0, abs=0.01)
    assert est.capm_return == pytest.approx(0.02 + 0.045 - 0.002)
    assert 0 < est.history_weight < 0.5


def test_money_market_expected_is_cash_rate_minus_cost():
    cash = pd.Series(100 * 1.02 ** (np.arange(2000) / 252), index=pd.bdate_range("2018-01-01", periods=2000))
    rf = model.risk_free_rate(cash, 0.03)
    assert rf == pytest.approx(0.02 + 0.001, abs=0.002)
    est = model.estimate(cash, gbm(0.07, 0.15), U.loc["XEON.DE"].to_dict(), rf, S)
    assert est.beta == 0 and est.momentum_tilt == 0
    assert est.expected_return == pytest.approx(rf - 0.001)


def test_lump_sum_outcomes_are_ordered_and_net_of_costs():
    est = model.estimate(gbm(0.07, 0.15), gbm(0.07, 0.15), U.loc["EUNL.DE"].to_dict(), 0.02, S)
    etf = U.loc["EUNL.DE"].to_dict()
    short, long = model.lump_sum(1000, 0.5, est, etf, S), model.lump_sum(1000, 20, est, etf, S)
    for r in (short, long):
        assert r["p10"] < r["p25"] < r["p50"] < r["p75"] < r["p90"]
        assert 0 < r["prob_loss"] < 1 and r["costs"] > 2
    assert long["prob_loss"] < short["prob_loss"]          # time diversification of loss probability
    assert long["mean"] > short["mean"] and long["tax"] > short["tax"]


def test_zero_volatility_gives_deterministic_after_tax_result():
    est = model.Estimate("X", 0.05, 0.05, None, 0, 0, 1, 1e-6, 1e-6, 0, 0, True, 0, 0, 0.02, 1, "2026-01-01")
    etf = {"spread_pct": 0.0, "tax_class": "equity"}
    s = {**S, "trade_republic": {**S["trade_republic"], "order_fee_eur": 0.0}, "tax": {**S["tax"], "saver_allowance_eur": 0}}
    r = model.lump_sum(1000, 10, est, etf, s)
    gain = 1000 * 1.05 ** 10 - 1000
    assert r["mean"] == pytest.approx(1000 + gain - gain * 0.7 * 0.26375, rel=1e-4)
    assert r["prob_loss"] == 0


def test_savings_plan_paid_in_and_path():
    est = model.estimate(gbm(0.07, 0.15), gbm(0.07, 0.15), U.loc["EUNL.DE"].to_dict(), 0.02, S)
    out, path = model.savings_plan(100, 5, est, U.loc["EUNL.DE"].to_dict(), S, paths=500)
    assert out["paid_in"] == 6000 and path.paid_in.iloc[-1] == 6000
    assert out["p10"] < out["p50"] < out["p90"]


def test_rank_sorts_and_includes_baseline_comparison():
    m = gbm(0.07, 0.15)
    ests = {t: model.estimate(gbm(0.06, 0.1 + 0.01 * i, seed=i), m, U.loc[t].to_dict(), 0.02, S) for i, t in enumerate(U.ticker[:8])}
    ranked = model.rank(ests, U.reset_index(drop=True), 1000, 3, S, "balance")
    assert len(ranked) == 8 and ranked.score.is_monotonic_decreasing
    assert model.rank(ests, U.reset_index(drop=True), 1000, 3, S, "safety").prob_loss.is_monotonic_increasing


def test_historical_windows():
    s = pd.Series(np.linspace(100, 200, 1000), index=pd.bdate_range("2020-01-01", periods=1000))
    h = model.historical_windows(s, 1)
    assert h["share_positive"] == 1 and h["windows"] == 1000 - 252
    assert model.historical_windows(s.iloc[:200], 1) is None
