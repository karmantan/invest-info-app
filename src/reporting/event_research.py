from __future__ import annotations
import hashlib, json
from datetime import datetime, timezone
import numpy as np
import pandas as pd
from src.config import ROOT, load_config
from src.data.adapters import YahooChartPrices
from src.events.ingestion import COMPANIES, ingest_event_library
from src.events.study import EVENT_FEATURES, add_event_features, enrich_event_returns, event_study_summary
from src.features.real_panel import FEATURES
from src.models.real_experiment import strict_walk_forward
from src.reporting.audit_21d import PERIODS, MODELS, bootstrap, costs, top_bucket
from src.portfolio.allocation import allocate

def ingest(refresh=False):
    events=ingest_event_library(refresh); company=[]
    for ticker in COMPANIES:
        try: company.append(YahooChartPrices().fetch(ticker,refresh))
        except Exception as exc: print(f"{ticker}: company-price download failed: {exc}")
    if not company: raise RuntimeError("No company prices available for earnings event study")
    pd.concat(company,ignore_index=True).to_parquet(ROOT/"data/processed/event_company_prices.parquet",index=False)
    print(events.groupby("event_type").size().to_string()); print("Event library and company prices cached.")

def metrics(pred,cfg,label):
    selected=top_bucket(pred); size=1000; mc,pc=costs(size,cfg); rows=[]
    for model,a in pred.groupby("model"):
        s=selected[selected.model==model]; equity=(1+s.groupby("date").forward_excess_21d.mean()).cumprod(); downside=s.forward_excess_21d[s.forward_excess_21d<0].std()
        rows.append({"specification":label,"model":model,"observations":len(a),"prediction_correlation":a.prediction.corr(a.forward_excess_21d),"directional_accuracy":(np.sign(a.prediction)==np.sign(a.forward_excess_21d)).mean(),"selected":len(s),"top_mean_excess":s.forward_excess_21d.mean(),"top_median_excess":s.forward_excess_21d.median(),"outperformance_probability":(s.forward_excess_21d>0).mean(),"p10":s.forward_excess_21d.quantile(.1),"downside_deviation":downside,"gross_value_added_eur":(s.forward_excess_21d*size).sum(),"tactical_net_value_added_eur":(s.forward_excess_21d*size-(mc-pc)).sum(),"wealth_counterfactual_value_added_eur":(s.forward_excess_21d*size-mc).sum()})
    return pd.DataFrame(rows),selected

def period_sector(pred,selected,label):
    rows=[]; sectors=[]
    for model,a in pred.groupby("model"):
        s=selected[selected.model==model]
        for lo,hi,period in PERIODS:
            aa=a[a.date.dt.year.between(lo,hi)]; ss=s[s.date.dt.year.between(lo,hi)]; rows.append({"specification":label,"model":model,"period":period,"observations":len(aa),"correlation":aa.prediction.corr(aa.forward_excess_21d),"directional_accuracy":(np.sign(aa.prediction)==np.sign(aa.forward_excess_21d)).mean(),"selected":len(ss),"mean_excess":ss.forward_excess_21d.mean(),"median_excess":ss.forward_excess_21d.median(),"outperformance_probability":(ss.forward_excess_21d>0).mean(),"p10":ss.forward_excess_21d.quantile(.1)})
        for ticker,aa in a.groupby("ticker"):
            ss=s[s.ticker==ticker]; sectors.append({"specification":label,"model":model,"ticker":ticker,"observations":len(aa),"correlation":aa.prediction.corr(aa.forward_excess_21d),"selected":len(ss),"mean_excess":ss.forward_excess_21d.mean(),"median_excess":ss.forward_excess_21d.median(),"outperformance_probability":(ss.forward_excess_21d>0).mean(),"p10":ss.forward_excess_21d.quantile(.1)})
    return pd.DataFrame(rows),pd.DataFrame(sectors)

def influence(selected,label):
    rows=[]
    for model,s in selected.groupby("model"):
        v=s.forward_excess_21d.sort_values(ascending=False).to_numpy(); row={"specification":label,"model":model,"sample_size":len(v),"mean":v.mean(),"median":np.median(v),"best":v[0],"worst":v[-1],"mean_without_best_1":v[1:].mean(),"mean_without_best_2":v[2:].mean(),"mean_without_best_3":v[3:].mean()}; row.update(cluster_bootstrap(s,"date","forward_excess_21d")); rows.append(row)
    return pd.DataFrame(rows)

def cluster_bootstrap(frame,cluster_col,value_col,seed=42,iterations=5000):
    groups=[g[value_col].dropna().to_numpy() for _,g in frame.groupby(cluster_col)]; rng=np.random.default_rng(seed)
    if len(groups)<10: return {"mean_ci_low":np.nan,"mean_ci_high":np.nan,"median_ci_low":np.nan,"median_ci_high":np.nan,"bootstrap_clusters":len(groups)}
    means=[]; medians=[]
    for _ in range(iterations):
        chosen=rng.integers(0,len(groups),len(groups)); values=np.concatenate([groups[i] for i in chosen]); means.append(values.mean()); medians.append(np.median(values))
    return {"mean_ci_low":float(np.quantile(means,.025)),"mean_ci_high":float(np.quantile(means,.975)),"median_ci_low":float(np.quantile(medians,.025)),"median_ci_high":float(np.quantile(medians,.975)),"bootstrap_clusters":len(groups)}

def event_influence(enriched):
    rows=[]
    for event_type,g in enriched.groupby("event_type"):
        for h in (5,10,21):
            v=g[f"post_{h}d_excess"].dropna().sort_values(ascending=False).to_numpy()
            row={"event_type":event_type,"horizon_days":h,"sample_size":len(v),"mean":v.mean(),"median":np.median(v),"best":v[0],"worst":v[-1],"mean_without_best_1":v[1:].mean(),"mean_without_best_2":v[2:].mean(),"mean_without_best_3":v[3:].mean()}; row.update(cluster_bootstrap(g.dropna(subset=[f"post_{h}d_excess"]),"event_id",f"post_{h}d_excess")); rows.append(row)
    return pd.DataFrame(rows)

def risk_veto(pred,cfg,label):
    rows=[]
    for _,r in pred.iterrows():
        forecast={"expected_return":r.prediction+r.benchmark_forward_eur_21d,"expected_excess_return":r.prediction,"p10":r.prediction+r.residual_p10,"sample_size":r.training_rows,"probability_loss_gt_5pct":r.residual_prob_loss5,"confidence":min(.8,.5+abs(r.prediction)*3)}
        decision=allocate(forecast,0,1000,12500,0,cfg); rows.append({"specification":label,"model":r.model,"accepted":decision.action=="BUY","actual_excess":r.forward_excess_21d})
    x=pd.DataFrame(rows); return x.groupby(["specification","model","accepted"]).agg(signals=("actual_excess","size"),mean_excess=("actual_excess","mean"),median_excess=("actual_excess","median"),outperformance=("actual_excess",lambda s:(s>0).mean()),p10=("actual_excess",lambda s:s.quantile(.1))).reset_index()

def diffusion(enriched):
    x=enriched[(enriched.event_type=="major_earnings")&(enriched.etf=="SMH")].copy(); x["event_reaction"]=np.where(x.event_day_etf_return>=0,"nonnegative","negative"); x["yield_environment"]=np.where(x.two_year_yield_event_change>0,"rising_2y","nonrising_2y"); x["vix_environment"]=np.where(x.vix_event_change>=30,"high_vix","lower_vix"); rows=[]
    for dimension in ("event_reaction","yield_environment","vix_environment"):
        for group,g in x.groupby(dimension):
            for h in (5,10,21):
                v=g[f"post_{h}d_excess"].dropna(); rows.append({"dimension":dimension,"group":group,"horizon_days":h,"sample_size":len(v),"mean_excess":v.mean(),"median_excess":v.median(),"outperformance_probability":(v>0).mean(),"p10":v.quantile(.1)})
    return pd.DataFrame(rows)

def run():
    cfg=load_config(); events=pd.read_parquet(ROOT/"data/processed/events.parquet"); etf=pd.read_parquet(ROOT/"data/processed/real_prices.parquet"); company=pd.read_parquet(ROOT/"data/processed/event_company_prices.parquet"); prices=pd.concat([etf,company],ignore_index=True); panel=pd.read_parquet(ROOT/"data/processed/real_feature_panel.parquet"); panel["date"]=pd.to_datetime(panel.date); macro=pd.read_parquet(ROOT/"data/processed/macro_state.parquet")
    enriched=enrich_event_returns(events,prices,panel,macro); enriched.to_parquet(ROOT/"data/processed/events_enriched.parquet",index=False); study=event_study_summary(enriched); study.to_csv(ROOT/"reports/event_studies.csv",index=False); diffusion(enriched).to_csv(ROOT/"reports/event_semiconductor_diffusion.csv",index=False); event_influence(enriched).to_csv(ROOT/"reports/event_study_outlier_sensitivity.csv",index=False)
    augmented_panel=add_event_features(panel,enriched); augmented= strict_walk_forward(augmented_panel,FEATURES+EVENT_FEATURES); augmented=augmented[augmented.model.isin(MODELS)].copy(); augmented.to_csv(ROOT/"reports/event_augmented_predictions.csv",index=False)
    base=pd.read_csv(ROOT/"reports/real_walk_forward_predictions.csv",parse_dates=["date","target_end_date","training_end"]); base=base[base.model.isin(MODELS)]
    base_metrics,base_selected=metrics(base,cfg,"frozen_base"); event_metrics,event_selected=metrics(augmented,cfg,"event_augmented"); combined=pd.concat([base_metrics,event_metrics],ignore_index=True); combined.to_csv(ROOT/"reports/event_model_comparison.csv",index=False)
    numeric=[c for c in combined.columns if c not in ("specification","model")]; delta=[]
    for model in MODELS:
        b=base_metrics[base_metrics.model==model].iloc[0]; e=event_metrics[event_metrics.model==model].iloc[0]; delta.append({"model":model,**{f"delta_{c}":e[c]-b[c] for c in numeric}})
    pd.DataFrame(delta).to_csv(ROOT/"reports/event_incremental_value.csv",index=False)
    bp,bs=period_sector(base,base_selected,"frozen_base"); ep,es=period_sector(augmented,event_selected,"event_augmented"); pd.concat([bp,ep]).to_csv(ROOT/"reports/event_results_by_period.csv",index=False); pd.concat([bs,es]).to_csv(ROOT/"reports/event_results_by_sector.csv",index=False); pd.concat([influence(base_selected,"frozen_base"),influence(event_selected,"event_augmented")]).to_csv(ROOT/"reports/event_outlier_sensitivity.csv",index=False); pd.concat([risk_veto(base,cfg,"frozen_base"),risk_veto(augmented,cfg,"event_augmented")]).to_csv(ROOT/"reports/event_risk_veto.csv",index=False)
    event_counts=events.groupby("event_type").agg(count=("event_id","size"),first=("effective_market_date","min"),last=("effective_market_date","max")).reset_index().to_dict("records"); frozen_hash=json.loads((ROOT/"reports/current_real_experiment.json").read_text())["model_configuration_hash"]; event_spec={"control_hash":frozen_hash,"event_features":EVENT_FEATURES,"models":"same frozen ridge and gradient boosting hyperparameters","surprise_policy":"unavailable consensus remains missing; no proxy surprise","days_since_policy":"calendar days capped at 365; 365 means no observed event in the preceding year"}; event_hash=hashlib.sha256(json.dumps(event_spec,sort_keys=True).encode()).hexdigest(); metadata={"experiment_id":f"event-21d-{event_hash[:12]}","mode":"real_event_research","generated_timestamp_utc":datetime.now(timezone.utc).isoformat(),"frozen_control_hash":frozen_hash,"event_specification_hash":event_hash,"event_specification":event_spec,"coverage":event_counts,"recommendation_status":"INSUFFICIENT REAL-DATA EVIDENCE","point_in_time_limitations":["SEC Item 2.02 filing timestamp may follow the actual earnings announcement.","Historical consensus estimates are unavailable; surprise fields are missing.","BLS archive automation was blocked; CPI coverage is PARTIAL and limited to the successfully cached 2020 official calendar.","BLS CPI values are current non-seasonally-adjusted series matched to official release dates, not stored historical release vintages.","FOMC statement dates are official; exact release times are not parsed per event.","Historical constituent importance is a documented major-company approximation, not historical ETF weights.","Event outcomes overlap heavily; confidence intervals resample event IDs, but results remain descriptive rather than independent-event causal estimates."]}
    (ROOT/"reports/event_research_metadata.json").write_text(json.dumps(metadata,indent=2,default=str)); print(json.dumps(metadata,indent=2,default=str)); print("\nEVENT STUDIES\n",study.to_string(index=False)); print("\nMODEL COMPARISON\n",combined.to_string(index=False)); print("\nINCREMENTAL\n",pd.DataFrame(delta).to_string(index=False))

if __name__=="__main__": run()
