"""Expected-return and risk model behind the ETF ideas and the what-if simulator.

Building blocks (all standard and deliberately simple):

* **CAPM anchor** – expected return = cash rate + beta × equity risk premium − fund cost.
  Beta is measured against MSCI World and shrunk toward its asset-class prior (Blume).
* **Own history, shrunk** – the fund's historical return above cash (over the same years)
  gets a Bayesian weight years / (years + (vol / tau)^2), so a short, noisy or lucky
  history cannot dominate; the remaining estimation error widens long-horizon ranges.
* **Trend tilt** – 12-1 month time-series momentum (Moskowitz, Ooi & Pedersen 2012),
  capped and fading over ~6 months, so it only matters for short horizons.
* **Volatility term structure** – today's EWMA volatility (RiskMetrics) fades back
  to the long-run volatility.
* **Log-normal outcomes** – the value after T years is log-normal; percentiles,
  averages and the chance of a loss are then computed *after* Trade Republic
  costs and German tax, exactly per outcome rather than on the average.
"""
from __future__ import annotations
from dataclasses import dataclass, asdict
import math
import numpy as np
import pandas as pd
from scipy.stats import norm
from src.web.costs import buy_cost, cash_account, effective_tax_rate, sell_net

TRADING_DAYS = 252
PRIOR_BETA = {"equity": 1.0, "bond": 0.1, "money": 0.0, "gold": 0.1}
DIVIDEND_YIELD = {"equity": 0.02, "bond": 0.025, "money": 0.0, "gold": 0.0}
TAG_YIELD = {"dividend": 0.05, "realestate": 0.035}


@dataclass
class Estimate:
    ticker: str
    expected_return: float      # long-run yearly return (arithmetic, net of fund cost)
    capm_return: float
    history_return: float | None
    history_years: float
    history_weight: float
    beta: float
    vol_long: float
    vol_now: float
    momentum_signal: float
    momentum_tilt: float
    trend_up: bool
    drawdown_1y: float
    dividend_tax_drag: float
    rf: float
    last_price: float
    last_date: str
    mean_uncertainty: float = 0.0   # standard error of the expected return itself (per year)

    def to_dict(self) -> dict: return asdict(self)


def _monthly_log_returns(prices: pd.Series) -> pd.Series:
    return np.log(prices.resample("ME").last()).diff().dropna()


def risk_free_rate(cash_prices: pd.Series | None, fallback: float, ter: float = 0.001) -> float:
    """Current cash rate implied by the overnight-rate ETF's last six months."""
    if cash_prices is None or len(cash_prices.dropna()) < 130: return fallback
    s = cash_prices.dropna()
    window = s.iloc[-127:]
    days = (window.index[-1] - window.index[0]).days
    if days <= 0: return fallback
    annual = (window.iloc[-1] / window.iloc[0]) ** (365.25 / days) - 1 + ter
    return float(np.clip(annual, 0.0, 0.08)) if np.isfinite(annual) else fallback


def _cash_return_over(cash: pd.Series | None, start: pd.Timestamp, end: pd.Timestamp, ter: float = 0.001) -> float | None:
    """Yearly return of the overnight-rate ETF over a window: what plain cash earned back then."""
    if cash is None: return None
    c = cash.dropna(); c = c[(c.index >= start) & (c.index <= end)]
    if len(c) < 60: return None
    days = (c.index[-1] - c.index[0]).days
    return float((c.iloc[-1] / c.iloc[0]) ** (365.25 / days) - 1 + ter) if days > 0 else None


def estimate(prices: pd.Series, market: pd.Series | None, etf: dict, rf: float, settings: dict, cash_history: pd.Series | None = None) -> Estimate:
    m = settings["model"]
    s = prices.dropna().astype(float)
    s = s[s > 0]
    if len(s) < 60: raise ValueError(f"{etf['ticker']}: not enough price history")
    s = s[s.index >= s.index[-1] - pd.DateOffset(years=m.get("history_years", 20))]
    asset = etf["asset_class"]
    years = (s.index[-1] - s.index[0]).days / 365.25
    daily = np.log(s).diff().dropna()
    monthly = _monthly_log_returns(s)
    vol_long = float(monthly.std() * math.sqrt(12)) if len(monthly) >= 24 else float(daily.std() * math.sqrt(TRADING_DAYS))
    lam = m.get("ewma_lambda", 0.94)
    vol_now = float(math.sqrt((daily ** 2).ewm(alpha=1 - lam).mean().iloc[-1] * TRADING_DAYS))
    floor = 0.003 if asset == "money" else 0.02
    vol_long = max(vol_long, floor); vol_now = max(vol_now, floor)

    prior = PRIOR_BETA.get(asset, 1.0); beta = prior
    if market is not None and asset != "money":
        mm = _monthly_log_returns(market.dropna())
        both = pd.concat([monthly, mm], axis=1, join="inner").dropna()
        if len(both) >= 24 and both.iloc[:, 1].var() > 0:
            raw = both.cov().iloc[0, 1] / both.iloc[:, 1].var()
            beta = 0.67 * raw + 0.33 * prior
    capm = rf + beta * m["equity_risk_premium"] - etf["ter"]

    history = float(math.exp(monthly.mean() * 12 + vol_long ** 2 / 2) - 1) if len(monthly) >= 12 else None
    # Judge history by its return *above cash over the same years* (rates were near zero for much of
    # 2009-2021), then shrink that toward the CAPM premium. Bayesian weight: the prior says a fund's true
    # edge is uncertain by about `prior_alpha_uncertainty` a year; the history's own noise is vol / sqrt(years).
    tau = m.get("prior_alpha_uncertainty", 0.015)
    if asset == "money" or history is None:
        weight, uncertainty = 0.0, 0.002
        expected = capm
    else:
        k = (vol_long / tau) ** 2
        weight = years / (years + k)
        uncertainty = math.sqrt(1 / (1 / tau ** 2 + years / vol_long ** 2))
        past_cash = _cash_return_over(cash_history, s.index[0], s.index[-1])
        history_excess = history - (past_cash if past_cash is not None else rf)
        expected = rf + weight * history_excess + (1 - weight) * (capm - rf)

    signal = 0.0
    if asset != "money" and len(s) > TRADING_DAYS:
        r12_1 = math.log(s.iloc[-22] / s.iloc[-TRADING_DAYS])
        signal = float(np.clip(r12_1 / (vol_long * math.sqrt(11 / 12)), -2, 2))
    tilt = m.get("momentum_strength", 0.02) * signal

    drag = 0.0
    if etf.get("distributing"):
        yield_ = max([DIVIDEND_YIELD.get(asset, 0.0)] + [TAG_YIELD[t] for t in etf.get("tags", []) if t in TAG_YIELD])
        exemption = settings["tax"].get("partial_exemption", {}).get(etf["tax_class"], 0.0)
        drag = yield_ * (1 - exemption) * effective_tax_rate(settings["tax"])

    last_year = s.iloc[-TRADING_DAYS:]
    return Estimate(
        ticker=etf["ticker"], expected_return=float(expected), capm_return=float(capm), history_return=history,
        history_years=float(years), history_weight=float(weight), beta=float(beta), vol_long=vol_long, vol_now=vol_now,
        momentum_signal=signal, momentum_tilt=float(tilt), trend_up=bool(s.iloc[-1] >= s.iloc[-200:].mean()),
        drawdown_1y=float(s.iloc[-1] / last_year.max() - 1), dividend_tax_drag=float(drag), rf=float(rf),
        last_price=float(s.iloc[-1]), last_date=str(s.index[-1].date()), mean_uncertainty=float(uncertainty))


def log_moments(est: Estimate, years: float, settings: dict) -> tuple[float, float]:
    """Mean and standard deviation of log(value_T / value_0) before costs and tax."""
    m = settings["model"]; tau = m.get("momentum_decay_years", 0.5); tau_v = m.get("vol_decay_years", 0.25)
    a_m = tau * (1 - math.exp(-years / tau)); a_v = tau_v * (1 - math.exp(-years / tau_v))
    log_expected = years * math.log1p(est.expected_return - est.dividend_tax_drag) + est.momentum_tilt * a_m
    variance = est.vol_now ** 2 * a_v + est.vol_long ** 2 * max(0.0, years - a_v)
    # The expected return is itself an estimate; its error compounds with time (predictive distribution).
    variance += (est.mean_uncertainty * years) ** 2
    return log_expected - variance / 2, math.sqrt(max(variance, 1e-12))


def _summary(net: np.ndarray, tax: np.ndarray, costs: np.ndarray, paid_in: float) -> dict:
    order = np.sort(net)
    q = lambda p: float(np.quantile(order, p))
    return {"paid_in": paid_in, "mean": float(net.mean()), "p10": q(.10), "p25": q(.25), "p50": q(.50), "p75": q(.75), "p90": q(.90),
            "prob_loss": float((net < paid_in - 1e-9).mean()), "tax": float(tax.mean()), "costs": float(costs.mean()), "std": float(net.std())}


def lump_sum(amount: float, years: float, est: Estimate, etf: dict, settings: dict, points: int = 2001) -> dict:
    """One-off purchase now, everything sold after `years`. Deterministic quantile grid."""
    years = max(years, 1 / 12)
    invested, buy_costs = buy_cost(amount, etf["spread_pct"], settings["trade_republic"]["order_fee_eur"])
    mean, sd = log_moments(est, years, settings)
    z = norm.ppf((np.arange(points) + 0.5) / points)
    gross = invested * np.exp(mean + sd * z)
    net, tax, selling = sell_net(gross, amount, etf, years, settings)
    out = _summary(net, tax, buy_costs + selling, amount)
    out.update({"gross_median": float(invested * math.exp(mean)), "years": years, "mode": "lump"})
    return out


def lump_sum_path(amount: float, years: float, est: Estimate, etf: dict, settings: dict) -> pd.DataFrame:
    """Outcome bands if you sold at each month along the way (for the fan chart)."""
    months = max(1, int(round(years * 12)))
    step = max(1, months // 120)
    grid = sorted(set(list(range(step, months + 1, step)) + [months]))
    rows = [{"years": 0.0, "paid_in": amount, "p10": amount, "p50": amount, "p90": amount, "mean": amount}]
    for mo in grid:
        r = lump_sum(amount, mo / 12, est, etf, settings, points=401)
        rows.append({"years": mo / 12, "paid_in": amount, "p10": r["p10"], "p50": r["p50"], "p90": r["p90"], "mean": r["mean"]})
    return pd.DataFrame(rows)


def savings_plan(monthly: float, years: float, est: Estimate, etf: dict, settings: dict, paths: int = 3000, seed: int = 7) -> tuple[dict, pd.DataFrame]:
    """Monthly savings plan (free on Trade Republic) simulated by Monte Carlo, sold at the end."""
    m = settings["model"]; tau = m.get("momentum_decay_years", 0.5); tau_v = m.get("vol_decay_years", 0.25)
    months = max(1, int(round(years * 12)))
    rng = np.random.default_rng(seed)
    fee = settings["trade_republic"].get("savings_plan_fee_eur", 0.0)
    per_buy = max(0.0, monthly - fee) * (1 - etf["spread_pct"] / 2)
    t = (np.arange(months) + 0.5) / 12
    drift = math.log1p(est.expected_return - est.dividend_tax_drag) + est.momentum_tilt * np.exp(-t / tau)
    var = est.vol_long ** 2 + (est.vol_now ** 2 - est.vol_long ** 2) * np.exp(-t / tau_v)
    shocks = rng.standard_normal((paths, months))
    # Each path draws its own long-run return, reflecting that the expected return is uncertain.
    drift_error = est.mean_uncertainty * rng.standard_normal((paths, 1))
    log_r = (drift + drift_error) / 12 - var / 24 - est.mean_uncertainty ** 2 * t / 12 + np.sqrt(var / 12) * shocks
    value = np.zeros(paths); rows = [{"years": 0.0, "paid_in": 0.0, "p10": 0.0, "p50": 0.0, "p90": 0.0, "mean": 0.0}]
    step = max(1, months // 120)
    for i in range(months):
        value = (value + per_buy) * np.exp(log_r[:, i])
        paid = monthly * (i + 1)
        if (i + 1) % step == 0 or i + 1 == months:
            net, _, _ = sell_net(value, paid, etf, (i + 1) / 12, settings)
            rows.append({"years": (i + 1) / 12, "paid_in": paid, "p10": float(np.quantile(net, .1)), "p50": float(np.median(net)), "p90": float(np.quantile(net, .9)), "mean": float(net.mean())})
    paid_in = monthly * months
    net, tax, selling = sell_net(value, paid_in, etf, years, settings)
    costs = selling + months * (monthly - per_buy)
    out = _summary(net, tax, costs, paid_in); out.update({"years": years, "mode": "monthly"})
    return out, pd.DataFrame(rows)


def cash_path(amount: float, years: float, settings: dict, monthly: float = 0.0) -> pd.DataFrame:
    result = cash_account(amount, years, settings, monthly=monthly)
    return pd.DataFrame({"years": np.arange(len(result["path"])) / 12, "value": result["path"]})


def historical_windows(prices: pd.Series, years: float) -> dict | None:
    """What actually happened to every past holding period of the same length (before costs and tax)."""
    s = prices.dropna()
    h = int(round(years * TRADING_DAYS))
    if h < 1 or len(s) < h + 126: return None
    values = s.to_numpy()
    returns = values[h:] / values[:-h] - 1
    return {"windows": int(len(returns)), "history_years": round((s.index[-1] - s.index[0]).days / 365.25, 1),
            "share_positive": float((returns > 0).mean()), "median": float(np.median(returns)),
            "worst": float(returns.min()), "best": float(returns.max()), "start": str(s.index[0].date())}


def risk_level(vol: float) -> str:
    if vol < 0.03: return "Very low"
    if vol < 0.08: return "Low"
    if vol < 0.16: return "Medium"
    if vol < 0.24: return "High"
    return "Very high"


def rank(estimates: dict[str, Estimate], universe: pd.DataFrame, amount: float, years: float, settings: dict, sort: str = "balance") -> pd.DataFrame:
    """Score every ETF for one amount and horizon, after costs and tax."""
    cash = cash_account(amount, years, settings)["net"]
    rows = []
    for _, etf in universe.iterrows():
        est = estimates.get(etf.ticker)
        if est is None: continue
        r = lump_sum(amount, years, est, etf.to_dict(), settings, points=801)
        rows.append({"ticker": etf.ticker, "name": etf["name"], "category": etf.category, "isin": etf["isin"],
                     "expected": r["mean"], "profit": r["mean"] - amount, "p10": r["p10"], "p50": r["p50"], "p90": r["p90"],
                     "prob_loss": r["prob_loss"], "tax": r["tax"], "costs": r["costs"],
                     "vs_cash": r["mean"] - cash, "score": (r["mean"] - cash) / max(r["std"], 1e-6),
                     "risk": risk_level(est.vol_long), "vol": est.vol_long, "trend_up": est.trend_up,
                     "yearly_expected": est.expected_return, "description": etf.get("description", "")})
    frame = pd.DataFrame(rows)
    if frame.empty: return frame
    key = {"balance": ("score", False), "return": ("expected", False), "safety": ("prob_loss", True)}[sort]
    return frame.sort_values(key[0], ascending=key[1]).reset_index(drop=True)
