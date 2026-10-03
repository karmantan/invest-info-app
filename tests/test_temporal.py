import pandas as pd
import numpy as np
import pytest
from src.features import build_features, assert_point_in_time
from src.models import yearly_walk_forward
from src.models.real_experiment import strict_walk_forward

def prices(n=800):
    return pd.DataFrame({"date":pd.bdate_range("2018-01-01",periods=n),"ticker":"X","close":100+np.arange(n)*.1})

def test_forward_return_uses_future_only_as_target():
    f=build_features(prices(),horizon=21); row=f.iloc[100]
    assert row["forward_return_21d"]==pytest.approx(f.iloc[121].close/row.close-1)
    assert_point_in_time(f,["return_5d","drawdown_63d"],"forward_return_21d")

def test_walk_forward_is_chronological():
    f=build_features(prices(1600)); r=yearly_walk_forward(f,["return_5d","return_21d","drawdown_63d","volatility_21d"],"forward_return_21d",min_train=252)
    assert not r.predictions.empty
    assert (r.predictions.training_end<r.predictions.date).all()

def test_duplicate_observations_rejected():
    p=prices(100); p=pd.concat([p,p.iloc[[2]]]);
    with pytest.raises(ValueError,match="duplicate"): build_features(p)

def test_real_walk_forward_purges_overlapping_training_targets():
    dates=pd.bdate_range("2014-01-01",periods=2200); rows=[]
    for ticker in ["A","B","C"]:
        x=pd.DataFrame({"date":dates,"target_end_date":pd.Series(dates).shift(-21),"ticker":ticker,"forward_excess_21d":.01,"forward_eur_21d":.02,"benchmark_forward_eur_21d":.01,"eur_return_21d":.01,"f":np.arange(len(dates),dtype=float)})
        rows.append(x)
    result=strict_walk_forward(pd.concat(rows),["f"],min_train=756)
    assert not result.empty
    assert (result.training_end<result.date).all()
    for _,group in result.groupby(["model","ticker"]):
        positions=pd.Series(group.date.sort_values()).map({d:i for i,d in enumerate(dates)})
        assert (positions.diff().dropna()>=21).all()
