from __future__ import annotations
import hashlib,json
from datetime import datetime,timezone
import numpy as np
import pandas as pd
from src.config import ROOT,load_config
from src.features.medium import MEDIUM_FEATURES,build_medium_panel
from src.fundamentals.sec_xbrl import FUNDAMENTAL_COMPANIES,ingest_fundamentals
from src.models.medium import medium_walk_forward
from src.reporting.audit_21d import PERIODS,bootstrap,costs

MEMBERSHIP={etf:[ticker for ticker,meta in FUNDAMENTAL_COMPANIES.items() if meta[2]==etf] for etf in ("SMH","XLK","XLE")}
SIZES=(500,1000,2000,3000,5000)

def ingest(refresh=False):
    f=ingest_fundamentals(refresh); print(f.groupby("etf").stability_score.agg(["count","min","max"]).to_string()); print("SEC XBRL fundamentals cached by filing date.")

def dip_studies(panel):
    rows=[]
    for ticker,x in panel.groupby("ticker"):
        x=x.sort_values("date").copy()
        for threshold in (.05,.10,.15,.20):
            crossing=(x.drawdown_252d<=-threshold)&(x.drawdown_252d.shift(1)>-threshold); candidates=x[crossing].copy(); kept=[]; last=None
            for i,r in candidates.iterrows():
                if last is None or (r.date-last).days>=63: kept.append(i); last=r.date
            e=x.loc[kept]
            conditions={"all":pd.Series(True,index=e.index),"stable_fundamentals":e.fundamental_state.isin(["STABLE","IMPROVING"]),"deteriorating_fundamentals":e.fundamental_state.isin(["DETERIORATING","SEVERELY DETERIORATING"]),"high_vix":e.vix>=30,"low_vix":e.vix<30,"rising_rates":e.dgs2_change_21d>0,"falling_rates":e.dgs2_change_21d<=0,"positive_252d_momentum":e.eur_return_252d>0,"negative_252d_momentum":e.eur_return_252d<=0}
            for condition,mask in conditions.items():
                g=e[mask]
                for h in (63,126):
                    raw=g[f"forward_eur_{h}d"].dropna(); excess=g.loc[raw.index,f"forward_excess_{h}d"]
                    rows.append({"ticker":ticker,"drawdown_threshold":threshold,"condition":condition,"horizon_days":h,"sample_size":len(raw),"mean_return":raw.mean(),"median_return":raw.median(),"probability_positive":(raw>0).mean(),"mean_excess":excess.mean(),"probability_outperform":(excess>0).mean(),"p10":raw.quantile(.1),"p25":raw.quantile(.25),"worst":raw.min()})
    return pd.DataFrame(rows)

def model_summary(pred,cfg,horizon):
    rows=[]; selected=[]; size=1000; mc,pc=costs(size,cfg)
    for model,x in pred.groupby("model"):
        rank=x.groupby("date").prediction.rank(pct=True,method="first"); s=x[rank>.66].copy(); selected.append(s.assign(selection="top_bucket")); target=f"forward_excess_{horizon}d"; raw=f"forward_eur_{horizon}d"
        rows.append({"horizon_days":horizon,"model":model,"observations":len(x),"correlation":x.prediction.corr(x[target]),"directional_accuracy":(np.sign(x.prediction)==np.sign(x[target])).mean(),"selected":len(s),"mean_return":s[raw].mean(),"mean_excess":s[target].mean(),"median_excess":s[target].median(),"probability_outperform":(s[target]>0).mean(),"p10_return":s[raw].quantile(.1),"worst_return":s[raw].min(),"tactical_net_value_added_eur":(s[target]*size-(mc-pc)).sum(),"wealth_value_added_eur":(s[target]*size-mc).sum()})
    return pd.DataFrame(rows),pd.concat(selected,ignore_index=True)

def robustness(selected,horizon):
    target=f"forward_excess_{horizon}d"; rows=[]
    for model,x in selected.groupby("model"):
        for lo,hi,label in PERIODS:
            g=x[x.date.dt.year.between(lo,hi)]; rows.append({"dimension":"period","value":label,"horizon_days":horizon,"model":model,"sample_size":len(g),"mean_excess":g[target].mean(),"median_excess":g[target].median(),"win_rate":(g[target]>0).mean(),"p10":g[target].quantile(.1)})
        for ticker,g in x.groupby("ticker"): rows.append({"dimension":"asset","value":ticker,"horizon_days":horizon,"model":model,"sample_size":len(g),"mean_excess":g[target].mean(),"median_excess":g[target].median(),"win_rate":(g[target]>0).mean(),"p10":g[target].quantile(.1)})
    return pd.DataFrame(rows)

def outliers(selected,horizon):
    target=f"forward_excess_{horizon}d"; rows=[]
    for model,g in selected.groupby("model"):
        v=g[target].sort_values(ascending=False).dropna().to_numpy(); row={"horizon_days":horizon,"model":model,"sample_size":len(v),"mean":v.mean(),"median":np.median(v),"win_rate":np.mean(v>0),"p10":np.quantile(v,.1),"worst":v[-1],"best":v[0],"mean_without_best_1":v[1:].mean(),"mean_without_best_2":v[2:].mean(),"mean_without_best_3":v[3:].mean()}; row.update(bootstrap(v)); rows.append(row)
    return pd.DataFrame(rows)

def analogue_for(panel,ticker,horizon=126,n=20):
    cols=["drawdown_252d","eur_return_63d","eur_return_126d","eur_return_252d","volatility_63d","vix","dgs2","dgs10","dgs2_change_21d","dgs10_change_21d","eurusd_change_21d","fundamental_stability_score"]
    x=panel[panel.ticker==ticker].sort_values("date").copy(); current=x.iloc[-1]; usable=[c for c in cols if pd.notna(pd.to_numeric(current[c],errors="coerce"))]; hist=x.dropna(subset=[f"forward_eur_{horizon}d"]+usable).copy()
    if hist.empty: return pd.DataFrame()
    hist[usable]=hist[usable].apply(pd.to_numeric,errors="coerce"); scale=hist[usable].std().replace(0,np.nan); usable=[c for c in usable if pd.notna(scale[c])]; squared=((hist[usable]-pd.to_numeric(current[usable],errors="coerce"))/scale[usable])**2; hist["distance"]=np.sqrt(squared.astype(float).mean(axis=1)); candidates=hist.sort_values("distance"); chosen=[]
    for i,r in candidates.iterrows():
        if all(abs((r.date-d).days)>=63 for d in chosen): chosen.append(r.date)
        if len(chosen)>=n: break
    result=candidates[candidates.date.isin(chosen)].copy(); result["similarity_score"]=1/(1+result.distance); return result[["date","similarity_score",f"forward_eur_{horizon}d",f"spy_forward_{horizon}d",f"forward_excess_{horizon}d"]]

def scanner(panel):
    rows=[]; analogues=[]
    for ticker,x in panel.groupby("ticker"):
        latest=x.sort_values("date").iloc[-1]; a63=analogue_for(panel,ticker,63); a126=analogue_for(panel,ticker,126); analogues += [a.assign(ticker=ticker,horizon_days=h) for a,h in ((a63,63),(a126,126))]
        state=latest.fundamental_state; dip=latest.drawdown_band
        if dip=="NORMAL VOLATILITY": status="NOT A DIP"
        elif state=="INSUFFICIENT DATA": status="DIP — FUNDAMENTALS UNCLEAR"
        elif state in ("DETERIORATING","SEVERELY DETERIORATING"): status="DIP — FUNDAMENTALS DETERIORATING"
        else: status="DIP — FUNDAMENTALS STABLE"
        p63=(a63.forward_eur_63d>0).mean() if len(a63) else np.nan; p126=(a126.forward_eur_126d>0).mean() if len(a126) else np.nan; tails=[v for v in [a63.forward_eur_63d.quantile(.1) if len(a63) else np.nan,a126.forward_eur_126d.quantile(.1) if len(a126) else np.nan] if pd.notna(v)]; p10=min(tails) if tails else np.nan; probabilities=[v for v in (p63,p126) if pd.notna(v)]; rebound=np.mean(probabilities) if probabilities else np.nan
        score=min(abs(latest.drawdown_252d)/.20,1)*25 + (25 if state in ("STABLE","IMPROVING") else -40 if state in ("DETERIORATING","SEVERELY DETERIORATING") else -10) + (25*((rebound-.5)*2) if pd.notna(rebound) else 0) + (10 if latest.eur_return_252d>0 else -5) + (15 if pd.notna(p10) and p10>-.10 else -15); score=min(60,max(0,score))
        reason=f"{dip}; fundamentals {str(state).lower()}; reliable historical ETF valuation unavailable; closest analogues were positive {rebound:.0%} of the time." if pd.notna(rebound) else f"{dip}; fundamentals {str(state).lower()}; reliable historical ETF valuation and sufficient analogues unavailable."
        rows.append({"asset":ticker,"as_of_date":str(latest.date.date()),"drawdown_252d":latest.drawdown_252d,"valuation_state":"UNAVAILABLE — price decline is not proof of cheapness","fundamental_state":state,"fundamental_coverage":latest.fundamental_coverage,"analogue_mean_63d":a63.forward_eur_63d.mean() if len(a63) else np.nan,"analogue_benchmark_63d":a63.spy_forward_63d.mean() if len(a63) else np.nan,"analogue_mean_126d":a126.forward_eur_126d.mean() if len(a126) else np.nan,"analogue_benchmark_126d":a126.spy_forward_126d.mean() if len(a126) else np.nan,"p10_downside":p10,"historical_rebound_probability":rebound,"analogue_sample_size":min(len(a63),len(a126)),"dip_opportunity_score":score,"confidence":"LOW" if latest.fundamental_coverage<.6 else "MEDIUM","research_status":status,"reason":reason})
    return pd.DataFrame(rows),pd.concat(analogues,ignore_index=True)

def allocations(scan,panel,cfg):
    rows=[]; mc,pc=None,None
    for _,r in scan.iterrows():
        for size in SIZES:
            mc,pc=costs(size,cfg); expected=(r.analogue_mean_63d+r.analogue_mean_126d)/2; passive=(r.analogue_benchmark_63d+r.analogue_benchmark_126d)/2
            rows.append({"asset":r.asset,"allocation_eur":size,"expected_63d_eur":size*r.analogue_mean_63d,"passive_63d_eur":size*r.analogue_benchmark_63d,"expected_126d_eur":size*r.analogue_mean_126d,"passive_126d_eur":size*r.analogue_benchmark_126d,"fresh_money_value_added_eur":size*(expected-passive)-(mc-pc),"wealth_value_added_eur":size*(expected-passive)-mc,"p10_eur":size*r.p10_downside,"model_roundtrip_cost_eur":mc,"benchmark_roundtrip_cost_eur":pc})
    return pd.DataFrame(rows)

def passive_comparisons(selected,panel,cfg):
    """Compare unchanged selections with both pre-specified passive references."""
    rows=[]
    for horizon,x in selected.groupby("horizon_days"):
        urth=panel[panel.ticker=="URTH"][["date",f"forward_eur_{horizon}d"]].rename(columns={f"forward_eur_{horizon}d":"urth_return"})
        x=x.merge(urth,on="date",how="left")
        for model,g in x.groupby("model"):
            for benchmark,column in (("SPY",f"spy_forward_{horizon}d"),("URTH","urth_return")):
                q=g.dropna(subset=[column]); size=1000; mc,pc=costs(size,cfg); excess=q[f"forward_eur_{horizon}d"]-q[column]
                rows.append({"horizon_days":horizon,"model":model,"benchmark":benchmark,"sample_size":len(q),"model_mean_return":q[f"forward_eur_{horizon}d"].mean(),"passive_mean_return":q[column].mean(),"mean_excess":excess.mean(),"median_excess":excess.median(),"probability_outperform":(excess>0).mean(),"tactical_net_value_added_eur":(excess*size-(mc-pc)).sum(),"wealth_value_added_eur":(excess*size-mc).sum()})
    return pd.DataFrame(rows)

def run():
    cfg=load_config(); prices=pd.read_parquet(ROOT/"data/processed/real_prices.parquet"); fx=pd.read_parquet(ROOT/"data/processed/eurusd.parquet"); macro=pd.read_parquet(ROOT/"data/processed/macro_state.parquet"); fundamentals=pd.read_parquet(ROOT/"data/processed/company_fundamentals.parquet"); panel=build_medium_panel(prices,fx,macro,fundamentals,MEMBERSHIP); panel.to_parquet(ROOT/"data/processed/medium_panel.parquet",index=False)
    valuation=[{"asset":t,"quality":"UNAVAILABLE","plain_language":"Reliable point-in-time historical ETF valuation is unavailable; a price decline is not evidence that the ETF is cheap.","limitation":"No historical aggregate valuation or defensible historical constituent weights from implemented free sources."} for t in cfg["research_proxies"]]; (ROOT/"reports/medium_valuation_quality.json").write_text(json.dumps(valuation,indent=2))
    dips=dip_studies(panel); dips.to_csv(ROOT/"reports/medium_dip_studies.csv",index=False); traps=dips[(dips.condition=="deteriorating_fundamentals")&(dips.drawdown_threshold>=.10)]; traps.to_csv(ROOT/"reports/medium_value_trap_study.csv",index=False)
    summaries=[]; robustness_rows=[]; outlier_rows=[]; selected_rows=[]
    for h in (63,126):
        pred=medium_walk_forward(panel,MEDIUM_FEATURES,h); pred.to_csv(ROOT/f"reports/medium_{h}d_predictions.csv",index=False); summary,selected=model_summary(pred,cfg,h); summaries.append(summary); selected_rows.append(selected.assign(horizon_days=h)); robustness_rows.append(robustness(selected,h)); outlier_rows.append(outliers(selected,h))
    pd.concat(summaries).to_csv(ROOT/"reports/medium_model_results.csv",index=False); pd.concat(robustness_rows).to_csv(ROOT/"reports/medium_robustness.csv",index=False); pd.concat(outlier_rows).to_csv(ROOT/"reports/medium_outlier_sensitivity.csv",index=False)
    passive_comparisons(pd.concat(selected_rows,ignore_index=True),panel,cfg).to_csv(ROOT/"reports/medium_passive_comparison.csv",index=False)
    scan,analogues=scanner(panel); scan.to_csv(ROOT/"reports/medium_current_scanner.csv",index=False); analogues.to_csv(ROOT/"reports/medium_analogues.csv",index=False); allocations(scan,panel,cfg).to_csv(ROOT/"reports/medium_allocation_sensitivity.csv",index=False)
    spec={"features":MEDIUM_FEATURES,"horizons":[63,126],"models":{"ridge":{"alpha":10},"gradient_boosting":{"max_depth":3,"learning_rate":.05,"max_iter":100,"l2_regularization":1,"random_state":42}},"evaluation":"annual expanding walk-forward, purged target completion, non-overlapping horizon-spaced cohorts","dip_crossings":[.05,.10,.15,.20],"valuation":"unavailable for ETF proxies","fundamental_score":"pre-specified annual SEC XBRL deterioration score"}; h=hashlib.sha256(json.dumps(spec,sort_keys=True).encode()).hexdigest(); meta={"experiment_id":f"medium-dip-{h[:12]}","mode":"real_medium_term_research","generated_timestamp_utc":datetime.now(timezone.utc).isoformat(),"short_term_control_hash":json.loads((ROOT/"reports/current_real_experiment.json").read_text())["model_configuration_hash"],"specification_hash":h,"specification":spec,"recommendation_status":"RESEARCH ONLY — LIVE RECOMMENDATIONS DISABLED","valuation_coverage":"UNAVAILABLE for all ETF proxies","fundamental_coverage":"Major-constituent approximation for SMH, XLK, XLE; unavailable for SPY, URTH, XBI, EEM","point_in_time_limitations":["SEC annual facts are available only from filing dates and can be stale between filings.","ETF aggregation is an unweighted median of a fixed major-company subset, not historical ETF constituent weights.","Historical ETF valuation is unavailable and excluded.","Free data do not support historical forward valuation expectations."]}; (ROOT/"reports/medium_research_metadata.json").write_text(json.dumps(meta,indent=2)); print(json.dumps(meta,indent=2)); print("\nSCANNER\n",scan.to_string(index=False)); print("\nMODELS\n",pd.concat(summaries).to_string(index=False))

if __name__=="__main__": run()
