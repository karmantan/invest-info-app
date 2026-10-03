from __future__ import annotations
import numpy as np
import pandas as pd

MEDIUM_FEATURES=["eur_return_5d","eur_return_21d","eur_return_63d","eur_return_126d","eur_return_252d","drawdown_63d","drawdown_126d","drawdown_252d","distance_ma50","distance_ma200","volatility_63d","relative_spy_63d","relative_urth_63d","vix","dgs2","dgs10","dgs2_change_21d","dgs10_change_21d","curve_2s10s","eurusd_change_21d","fundamental_stability_score","fundamental_coverage"]

def _fundamental_daily(dates,etf,fundamentals,members):
    values=[]
    for ticker in members:
        f=fundamentals[fundamentals.ticker==ticker][["available_date","stability_score"]].dropna().sort_values("available_date"); left=pd.DataFrame({"date":dates})
        values.append(pd.merge_asof(left,f,left_on="date",right_on="available_date",direction="backward").stability_score.rename(ticker))
    if not values: return pd.DataFrame({"fundamental_stability_score":np.nan,"fundamental_coverage":0.0},index=range(len(dates)))
    matrix=pd.concat(values,axis=1); return pd.DataFrame({"fundamental_stability_score":matrix.median(axis=1,skipna=True),"fundamental_coverage":matrix.notna().mean(axis=1)})

def build_medium_panel(prices,fx,macro,fundamentals,membership,benchmark="SPY"):
    p=prices.copy(); p.date=pd.to_datetime(p.date); fx=fx[["date","close"]].rename(columns={"close":"usd_per_eur"}).sort_values("date"); frames=[]
    for ticker,x in p.groupby("ticker"):
        x=x.sort_values("date").copy(); x=pd.merge_asof(x,fx,on="date",direction="backward",tolerance=pd.Timedelta("5d")); x["close_eur"]=x.close/x.usd_per_eur
        for d in (5,21,63,126,252): x[f"eur_return_{d}d"]=x.close_eur.pct_change(d,fill_method=None)
        for d in (63,126,252): x[f"drawdown_{d}d"]=x.close_eur/x.close_eur.rolling(d,min_periods=max(20,d//2)).max()-1
        x["distance_ma50"]=x.close_eur/x.close_eur.rolling(50,min_periods=40).mean()-1; x["distance_ma200"]=x.close_eur/x.close_eur.rolling(200,min_periods=150).mean()-1; x["volatility_63d"]=x.close_eur.pct_change(fill_method=None).rolling(63,min_periods=40).std()*np.sqrt(252)
        for h in (63,126): x[f"forward_eur_{h}d"]=x.close_eur.shift(-h)/x.close_eur-1; x[f"target_end_{h}d"]=x.date.shift(-h)
        fd=_fundamental_daily(x.date.reset_index(drop=True),ticker,fundamentals,membership.get(ticker,[])); x["fundamental_stability_score"]=fd.fundamental_stability_score.values; x["fundamental_coverage"]=fd.fundamental_coverage.values; frames.append(x)
    panel=pd.concat(frames,ignore_index=True)
    for bench in ("SPY","URTH"):
        b=panel[panel.ticker==bench][["date","eur_return_63d","forward_eur_63d","forward_eur_126d"]].rename(columns={"eur_return_63d":f"{bench.lower()}_momentum_63d","forward_eur_63d":f"{bench.lower()}_forward_63d","forward_eur_126d":f"{bench.lower()}_forward_126d"}); panel=panel.merge(b,on="date",how="left")
    panel["relative_spy_63d"]=panel.eur_return_63d-panel.spy_momentum_63d; panel["relative_urth_63d"]=panel.eur_return_63d-panel.urth_momentum_63d
    for h in (63,126): panel[f"forward_excess_{h}d"]=panel[f"forward_eur_{h}d"]-panel[f"spy_forward_{h}d"]
    m=macro.copy().sort_values("date"); m.date=pd.to_datetime(m.date); cols=[c for c in m if c!="date"]; m[cols]=m[cols].shift(1); panel=pd.merge_asof(panel.sort_values("date"),m,on="date",direction="backward",tolerance=pd.Timedelta("7d")); panel["eurusd_change_21d"]=panel.groupby("ticker").usd_per_eur.pct_change(21,fill_method=None)
    panel["drawdown_band"]=pd.cut(panel.drawdown_252d,[-np.inf,-.20,-.10,-.05,np.inf],labels=["EXTREME DRAWDOWN","LARGE DIP","MODERATE DIP","NORMAL VOLATILITY"])
    panel["fundamental_state"]=pd.cut(panel.fundamental_stability_score,[-np.inf,-5,-2,1,np.inf],labels=["SEVERELY DETERIORATING","DETERIORATING","STABLE","IMPROVING"]).astype(object); panel.loc[(panel.fundamental_coverage<.6)|panel.fundamental_stability_score.isna(),"fundamental_state"]="INSUFFICIENT DATA"
    return panel.sort_values(["date","ticker"]).reset_index(drop=True)
