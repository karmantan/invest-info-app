import numpy as np
import pandas as pd

from src.features.medium import build_medium_panel
from src.models.medium import medium_walk_forward


def _inputs(periods=1700):
    dates = pd.bdate_range("2000-01-03", periods=periods)
    prices = pd.concat([
        pd.DataFrame({"date": dates, "ticker": ticker, "close": 100 * np.exp(np.arange(periods) * slope)})
        for ticker, slope in (("SPY", .0002), ("URTH", .00015), ("XLK", .0003))
    ], ignore_index=True)
    fx = pd.DataFrame({"date": dates, "close": 1.1})
    macro = pd.DataFrame({"date": dates, "vix": 20., "dgs2": 2., "dgs10": 3., "dgs2_change_21d": 0., "dgs10_change_21d": 0., "curve_2s10s": 1.})
    fundamentals = pd.DataFrame(columns=["ticker", "available_date", "stability_score"])
    return dates, prices, fx, macro, fundamentals


def test_medium_forward_targets_are_future_eur_returns():
    dates, prices, fx, macro, fundamentals = _inputs(400)
    panel = build_medium_panel(prices, fx, macro, fundamentals, {})
    x = panel[panel.ticker == "XLK"].reset_index(drop=True)
    i = 260
    expected = x.loc[i + 63, "close_eur"] / x.loc[i, "close_eur"] - 1
    assert np.isclose(x.loc[i, "forward_eur_63d"], expected)
    assert x.loc[i, "target_end_63d"] == x.loc[i + 63, "date"]


def test_walk_forward_purges_targets_and_spaces_cohorts():
    _, prices, fx, macro, fundamentals = _inputs()
    panel = build_medium_panel(prices, fx, macro, fundamentals, {})
    features = ["eur_return_63d", "drawdown_252d", "vix"]
    predictions = medium_walk_forward(panel, features, 63, min_train=300)
    assert not predictions.empty
    ridge = predictions[predictions.model == "ridge"]
    for _, fold in ridge.groupby(ridge.date.dt.year):
        unique_dates = fold.date.sort_values().drop_duplicates()
        offsets = panel.date.drop_duplicates().sort_values().reset_index(drop=True)
        positions = offsets[offsets.isin(unique_dates)].index.to_numpy()
        assert np.all(np.diff(positions) >= 63)
        assert (fold.target_end_63d > fold.date).all()
