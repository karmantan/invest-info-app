from __future__ import annotations
import numpy as np
import pandas as pd

def build_real_panel(prices: pd.DataFrame, fx: pd.DataFrame, macro: pd.DataFrame, benchmark="SPY") -> pd.DataFrame:
    p=prices.copy(); p["date"]=pd.to_datetime(p.date); p=p.sort_values(["ticker","date"])
    fx=fx[["date","close"]].rename(columns={"close":"usd_per_eur"}).sort_values("date")
    frames=[]
    for ticker,x in p.groupby("ticker"):
        x=x.sort_values("date").copy(); x=pd.merge_asof(x,fx,on="date",direction="backward",tolerance=pd.Timedelta("5d"))
        x["close_eur"]=x["close"]/x["usd_per_eur"]
        for d in (1,5,21,63):
            x[f"local_return_{d}d"]=x.close.pct_change(d,fill_method=None); x[f"eur_return_{d}d"]=x.close_eur.pct_change(d,fill_method=None)
        x["drawdown_63d"]=x.close_eur/x.close_eur.rolling(63,min_periods=40).max()-1
        x["volatility_21d"]=x.close_eur.pct_change(fill_method=None).rolling(21,min_periods=15).std()*np.sqrt(252)
        x["forward_local_5d"]=x.close.shift(-5)/x.close-1; x["forward_eur_5d"]=x.close_eur.shift(-5)/x.close_eur-1
        x["forward_local_21d"]=x.close.shift(-21)/x.close-1; x["forward_eur_21d"]=x.close_eur.shift(-21)/x.close_eur-1
        x["target_end_date"]=x.date.shift(-21); frames.append(x)
    panel=pd.concat(frames,ignore_index=True)
    bench=panel[panel.ticker==benchmark][["date","forward_eur_21d","eur_return_21d"]].rename(columns={"forward_eur_21d":"benchmark_forward_eur_21d","eur_return_21d":"broad_momentum_21d"})
    panel=panel.merge(bench,on="date",how="left"); panel["forward_excess_21d"]=panel.forward_eur_21d-panel.benchmark_forward_eur_21d
    m=macro.copy().sort_values("date"); m["date"]=pd.to_datetime(m.date)
    # One observation-day lag is conservative: macro closes are never used before the next trading date.
    value_cols=[c for c in m.columns if c!="date"]
    m[value_cols]=m[value_cols].shift(1)
    panel=pd.merge_asof(panel.sort_values("date"),m,on="date",direction="backward",tolerance=pd.Timedelta("7d"))
    panel["eurusd_change_21d"]=panel.groupby("ticker").usd_per_eur.pct_change(21,fill_method=None)
    return panel.sort_values(["date","ticker"]).reset_index(drop=True)

FEATURES=["eur_return_5d","eur_return_21d","eur_return_63d","broad_momentum_21d","drawdown_63d","volatility_21d","dgs2","dgs2_change_5d","dgs2_change_21d","dgs10","dgs10_change_5d","dgs10_change_21d","curve_2s10s","vix","eurusd_change_21d"]
