from __future__ import annotations
import numpy as np
import pandas as pd

def enrich_event_returns(events,prices,panel,macro_state):
    p=prices.copy(); p["date"]=pd.to_datetime(p.date); event_rows=[]
    macro=macro_state.copy(); macro["date"]=pd.to_datetime(macro.date); macro["vix_change_1d"]=macro.vix.diff(); macro=macro.set_index("date")
    for _,event in events.iterrows():
        etfs=[event.etf] if event.etf!="ALL" else sorted(panel.ticker.unique())
        for etf in etfs:
            x=p[p.ticker==etf].sort_values("date").reset_index(drop=True); dates=x.date.to_numpy(); idx=np.searchsorted(dates,np.datetime64(event.effective_market_date))
            if idx>=len(x): continue
            row=event.to_dict(); row["etf"]=etf; row["effective_market_date"]=x.date.iloc[idx]; row["event_day_etf_return"]=x.close.iloc[idx]/x.close.iloc[idx-1]-1 if idx else np.nan
            if idx>=5:
                row["pre_5d_return"]=x.close.iloc[idx-1]/x.close.iloc[idx-5]-1
            spy=p[p.ticker=="SPY"].set_index("date").close
            for h in (5,10,21):
                if idx+h<len(x):
                    raw=x.close.iloc[idx+h]/x.close.iloc[idx]-1; start=x.date.iloc[idx]; end=x.date.iloc[idx+h]; bench=spy.asof(end)/spy.asof(start)-1; row[f"post_{h}d_return"]=raw; row[f"post_{h}d_excess"]=raw-bench
            if event.event_type=="major_earnings":
                company=p[p.ticker==event.company_ticker].sort_values("date").reset_index(drop=True)
                ci=np.searchsorted(company.date.to_numpy(),np.datetime64(row["effective_market_date"]))
                row["event_day_company_return"]=company.close.iloc[ci]/company.close.iloc[ci-1]-1 if 0<ci<len(company) else np.nan
            if row["effective_market_date"] in macro.index:
                m=macro.loc[row["effective_market_date"]]; row["two_year_yield_event_change"]=m.get("dgs2_change_1d"); row["ten_year_yield_event_change"]=m.get("dgs10_change_1d"); row["vix_event_change"]=m.get("vix_change_1d",np.nan); row["vix_event_level"]=m.get("vix",np.nan)
            event_rows.append(row)
    enriched=pd.DataFrame(event_rows)
    for i,r in enriched.iterrows():
        relevant=enriched[(enriched.etf==r.etf)&(enriched.effective_market_date>r.effective_market_date)&(enriched.effective_market_date<=r.effective_market_date+pd.offsets.BDay(21))]
        enriched.loc[i,"overlapping_events_next_21d"]=len(relevant)
    return enriched

def event_study_summary(enriched):
    rows=[]
    data=enriched.copy(); data["reaction_group"]=np.where(data.event_day_etf_return>=0,"nonnegative_event_day","negative_event_day")
    for event_type,x in data.groupby("event_type"):
        groups=[("all",x)]
        if event_type=="major_earnings": groups += list(x.groupby("reaction_group"))
        for group,g in groups:
            for h in (5,10,21):
                v=g[f"post_{h}d_excess"].dropna(); rows.append({"event_type":event_type,"group":group,"horizon_days":h,"sample_size":len(v),"mean_excess":v.mean(),"median_excess":v.median(),"probability_outperform":(v>0).mean(),"p10":v.quantile(.1),"overlap_rate":(g.overlapping_events_next_21d>0).mean()})
    return pd.DataFrame(rows)

EVENT_FEATURES=["days_since_major_earnings","earnings_event_today","sector_event_day_return","days_since_cpi","cpi_event_today","days_since_fomc","fomc_event_today","target_rate_change_event","two_year_yield_event_change","ten_year_yield_event_change","vix_event_change"]

def add_event_features(panel,enriched):
    out=panel.copy().sort_values(["ticker","date"])
    for col in EVENT_FEATURES: out[col]=np.nan if col.startswith("days_since") else 0.0
    for ticker,xidx in out.groupby("ticker").groups.items():
        idx=list(xidx); dates=out.loc[idx,"date"]
        relevant=enriched[(enriched.etf==ticker)|(enriched.event_type.isin(["cpi","fomc"]))].copy()
        for kind,prefix in (("major_earnings","major_earnings"),("cpi","cpi"),("fomc","fomc")):
            ev=relevant[relevant.event_type==kind].sort_values("effective_market_date"); event_dates=pd.Series(ev.effective_market_date.dropna().unique())
            if len(event_dates):
                left=pd.DataFrame({"date":dates.values,"_idx":idx}).sort_values("date"); right=pd.DataFrame({"event_date":event_dates}).sort_values("event_date")
                merged=pd.merge_asof(left,right,left_on="date",right_on="event_date",direction="backward"); elapsed=(merged.date-merged.event_date).dt.days.clip(upper=365).fillna(365); out.loc[merged._idx,f"days_since_{prefix}"]=elapsed.values
            else: out.loc[idx,f"days_since_{prefix}"]=365
            today_col={"major_earnings":"earnings_event_today","cpi":"cpi_event_today","fomc":"fomc_event_today"}[kind]; counts=ev.groupby("effective_market_date").size(); out.loc[idx,today_col]=dates.map(counts).fillna(0).values
        day=relevant.groupby("effective_market_date").agg(sector_event_day_return=("event_day_etf_return","mean"),target_rate_change_event=("target_rate_change","mean"),two_year_yield_event_change=("two_year_yield_event_change","mean"),ten_year_yield_event_change=("ten_year_yield_event_change","mean"),vix_event_change=("vix_event_change","mean"))
        for col in day.columns: out.loc[idx,col]=dates.map(day[col]).fillna(0).values
    return out
