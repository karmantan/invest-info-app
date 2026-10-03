import pandas as pd
import numpy as np
from src.features.real_panel import build_real_panel

def test_eur_conversion_and_excess_target_are_explicit():
    dates=pd.bdate_range("2020-01-01",periods=100); prices=[]
    for ticker,growth in [("SPY",1.001),("ETF",1.002)]: prices.append(pd.DataFrame({"date":dates,"ticker":ticker,"close":100*growth**np.arange(100),"source":"real","currency":"USD","adjustment":"adjusted close"}))
    fx=pd.DataFrame({"date":dates,"close":1.1*1.0005**np.arange(100)})
    macro=pd.DataFrame({"date":dates,"dgs2":1,"dgs10":2,"vix":20,"curve_2s10s":1,"dgs2_change_1d":0,"dgs2_change_5d":0,"dgs2_change_21d":0,"dgs10_change_1d":0,"dgs10_change_5d":0,"dgs10_change_21d":0})
    panel=build_real_panel(pd.concat(prices),fx,macro)
    row=panel[(panel.ticker=="ETF")].iloc[50]
    assert row.forward_local_21d>row.forward_eur_21d
    assert abs(row.forward_excess_21d-(row.forward_eur_21d-row.benchmark_forward_eur_21d))<1e-12
