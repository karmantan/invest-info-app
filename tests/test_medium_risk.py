import numpy as np
import pandas as pd

from src.reporting.medium_risk_audit import add_point_in_time_flags, policy_weights


def _panel(n=300):
    dates=pd.bdate_range("2020-01-01",periods=n)
    base=pd.DataFrame({"date":dates,"ticker":"SPY","vix":np.arange(n,dtype=float),"volatility_63d":np.arange(n,dtype=float)/100,"distance_ma200":.01,"eur_return_63d":.02,"drawdown_252d":-.05})
    asset=base.assign(ticker="SMH",volatility_63d=np.arange(n,dtype=float)/50)
    return pd.concat([base,asset],ignore_index=True)


def test_expanding_percentiles_do_not_use_current_observation():
    panel=_panel(); flagged=add_point_in_time_flags(panel); row=flagged[(flagged.ticker=="SMH")].iloc[252]
    expected=np.quantile(np.arange(252,dtype=float)/50,.8)
    assert np.isclose(row.asset_vol_p80,expected)
    assert row.high_asset_volatility


def test_policies_match_prespecified_gate_logic():
    x=pd.DataFrame({"risk_flag_count":[0,1,2,1],"broad_market_downtrend":[False,False,False,True],"extreme_asset_drawdown":[False,False,False,False]})
    w=policy_weights(x)
    assert w.policy_1_two_flags.tolist()==[1,1,0,1]
    assert w.policy_2_hard_flags.tolist()==[1,1,1,0]
    assert w.policy_3_any_flag.tolist()==[1,0,0,0]
    assert w.risk_sizing.tolist()==[1,.5,0,0]
