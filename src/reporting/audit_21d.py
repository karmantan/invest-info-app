from __future__ import annotations
import hashlib, json
from datetime import datetime, timezone
from pathlib import Path
import numpy as np
import pandas as pd
from src.config import ROOT, load_config
from src.features.real_panel import FEATURES
from src.models.real_experiment import leave_one_asset_out
from src.risk.metrics import maximum_drawdown

PERIODS=[(2005,2009,"2005–2009"),(2010,2014,"2010–2014"),(2015,2019,"2015–2019"),(2020,2022,"2020–2022"),(2023,9999,"2023–latest")]
MODELS=("ridge","gradient_boosting")

def frozen_spec(cfg):
    spec={"features":FEATURES,"target":"21-trading-day EUR excess return versus SPY","models":{"ridge":{"alpha":10,"imputer":"median","scaler":"StandardScaler"},"gradient_boosting":{"max_depth":3,"learning_rate":.05,"max_iter":100,"l2_regularization":1,"random_state":42,"imputer":"median"}},"top_bucket_cutoff":.66,"prediction_spacing":21,"minimum_training_rows":756,"risk_thresholds":cfg["risk"],"allocation_thresholds":cfg["portfolio"]}
    encoded=json.dumps(spec,sort_keys=True,separators=(",",":")).encode(); return spec,hashlib.sha256(encoded).hexdigest()

def top_bucket(pred):
    pieces=[]
    for model,x in pred.groupby("model"):
        selected=x[x.groupby("date").prediction.rank(pct=True,method="first")>.66].copy(); selected["model"]=model; pieces.append(selected)
    return pd.concat(pieces,ignore_index=True)

def costs(size,cfg):
    fixed=cfg["costs"]["buy_fee_eur"]+cfg["costs"]["sell_fee_eur"]
    return fixed+size*2*cfg["costs"]["estimated_spread_bps"]/10000, fixed+size*2*cfg["costs"]["estimated_benchmark_spread_bps"]/10000

def cost_audit(top,cfg,secondary):
    rows=[]
    for size in (500,1000,2000,3000,5000,6000):
        mc,pc=costs(size,cfg)
        for model,x in top.groupby("model"):
            for benchmark,col in (("SPY","benchmark_forward_eur_21d"),("URTH","urth_forward_eur_21d")):
                comparable=x.dropna(subset=[col]); gross_model=float((comparable.forward_eur_21d*size).sum()); gross_passive=float((comparable[col]*size).sum()); n=len(comparable)
                wealth=gross_model-n*mc-gross_passive
                rows.append({"model":model,"benchmark":benchmark,"allocation_eur":size,"signals":n,"model_gross_profit_eur":gross_model,"passive_gross_profit_eur":gross_passive,"model_costs_eur":n*mc,"passive_costs_eur":n*pc,"model_net_profit_eur":gross_model-n*mc,"passive_net_profit_eur":gross_passive-n*pc,"tactical_net_value_added_eur":gross_model-gross_passive-n*(mc-pc),"tactical_net_value_added_pct_per_trade":(gross_model-gross_passive-n*(mc-pc))/(n*size),"wealth_counterfactual_value_added_eur":wealth,"wealth_counterfactual_value_added_pct_per_trade":wealth/(n*size)})
    return pd.DataFrame(rows)

def bootstrap(values, seed=42, iterations=5000):
    x=np.asarray(values,float); rng=np.random.default_rng(seed)
    if len(x)<10: return {"mean_ci_low":np.nan,"mean_ci_high":np.nan,"median_ci_low":np.nan,"median_ci_high":np.nan}
    samples=rng.choice(x,(iterations,len(x)),replace=True)
    means=samples.mean(axis=1); medians=np.median(samples,axis=1)
    return {"mean_ci_low":float(np.quantile(means,.025)),"mean_ci_high":float(np.quantile(means,.975)),"median_ci_low":float(np.quantile(medians,.025)),"median_ci_high":float(np.quantile(medians,.975))}

def risk_audit(risk,panel,cfg):
    context=panel[["date","ticker","vix","curve_2s10s"]].copy(); context["regime"]=np.select([context.vix>=30,context.curve_2s10s<0],["high_vix","inverted_curve"],default="normal")
    x=risk.merge(context,on=["date","ticker"],how="left"); x["period"]=x.date.dt.year.map(lambda y:next(label for lo,hi,label in PERIODS if lo<=y<=hi))
    rows=[]; concentration=[]
    for model in MODELS:
        accepted=x[(x.model==model)&x.accepted].copy(); v=accepted.actual_excess.sort_values(ascending=False).to_numpy(); base={"model":model,"accepted":len(v),"mean_excess":float(np.mean(v)),"median_excess":float(np.median(v)),"outperformance_rate":float(np.mean(v>0)),"p10":float(np.quantile(v,.1)),"p25":float(np.quantile(v,.25)),"worst":float(np.min(v)),"best":float(np.max(v)),"mean_without_best_1":float(np.mean(v[1:])),"mean_without_best_2":float(np.mean(v[2:])),"mean_without_best_3":float(np.mean(v[3:]))}; base.update(bootstrap(v)); rows.append(base)
        for dimension in ("ticker","period","regime"):
            for value,g in accepted.groupby(dimension): concentration.append({"model":model,"dimension":dimension,"value":value,"accepted":len(g),"mean_excess":g.actual_excess.mean(),"median_excess":g.actual_excess.median(),"outperformance_rate":(g.actual_excess>0).mean()})
    return pd.DataFrame(rows),pd.DataFrame(concentration)

def period_audit(pred,top,cfg):
    rows=[]; size=1000; mc,pc=costs(size,cfg)
    for model in MODELS:
        allm=pred[pred.model==model]
        selected=top[top.model==model]
        for lo,hi,label in PERIODS:
            a=allm[allm.date.dt.year.between(lo,hi)]; s=selected[selected.date.dt.year.between(lo,hi)]
            rows.append({"model":model,"period":label,"predictions":len(a),"selected":len(s),"correlation":a.prediction.corr(a.forward_excess_21d),"directional_accuracy":(np.sign(a.prediction)==np.sign(a.forward_excess_21d)).mean(),"top_mean_excess":s.forward_excess_21d.mean(),"top_median_excess":s.forward_excess_21d.median(),"outperformance_probability":(s.forward_excess_21d>0).mean(),"p10":s.forward_excess_21d.quantile(.1),"gross_value_added_eur":(s.forward_excess_21d*size).sum(),"tactical_net_value_added_eur":(s.forward_excess_21d*size-(mc-pc)).sum()})
    return pd.DataFrame(rows)

def asset_audit(pred,top,risk):
    rows=[]
    for model in MODELS:
        for ticker,a in pred[pred.model==model].groupby("ticker"):
            s=top[(top.model==model)&(top.ticker==ticker)]; accepted=risk[(risk.model==model)&risk.accepted&(risk.ticker==ticker)]
            rows.append({"model":model,"ticker":ticker,"observations":len(a),"mean_prediction":a.prediction.mean(),"realized_mean_return":a.forward_eur_21d.mean(),"correlation":a.prediction.corr(a.forward_excess_21d),"directional_accuracy":(np.sign(a.prediction)==np.sign(a.forward_excess_21d)).mean(),"selected":len(s),"selected_mean_excess":s.forward_excess_21d.mean(),"selected_median_excess":s.forward_excess_21d.median(),"selected_outperformance_probability":(s.forward_excess_21d>0).mean() if len(s) else np.nan,"selected_p10":s.forward_excess_21d.quantile(.1),"risk_veto_acceptance_count":len(accepted)})
    return pd.DataFrame(rows)

def sequential_paths(top,cfg):
    rows=[]; size=1.0
    for model,x in top.groupby("model"):
        by=x.groupby("date").agg(model_return=("forward_eur_21d","mean"),passive_return=("benchmark_forward_eur_21d","mean")).sort_index()
        fixed_rate=(cfg["costs"]["buy_fee_eur"]+cfg["costs"]["sell_fee_eur"])/1000
        trade_cost=fixed_rate+2*cfg["costs"]["estimated_spread_bps"]/10000
        model_path=(1+by.model_return-trade_cost).cumprod(); passive_path=(1+by.passive_return).cumprod(); excess_path=(1+by.model_return-by.passive_return-trade_cost).cumprod()
        rows.append({"model":model,"periods":len(by),"sequential_model_ending_multiple":model_path.iloc[-1],"buy_and_maintain_passive_ending_multiple":passive_path.iloc[-1],"sequential_model_drawdown":maximum_drawdown(model_path),"passive_drawdown":maximum_drawdown(passive_path),"synthetic_cumulative_excess_drawdown":maximum_drawdown(excess_path)})
    return pd.DataFrame(rows)

def run():
    cfg=load_config(); spec,spec_hash=frozen_spec(cfg); generated=datetime.now(timezone.utc).isoformat(); experiment_id=f"real-21d-frozen-{spec_hash[:12]}"
    pred=pd.read_csv(ROOT/"reports/real_walk_forward_predictions.csv",parse_dates=["date","target_end_date","training_end"]); panel=pd.read_parquet(ROOT/"data/processed/real_feature_panel.parquet"); panel["date"]=pd.to_datetime(panel.date)
    if pred.model.nunique()!=4 or len(pred)!=6808: raise RuntimeError("Current real prediction artifact has unexpected shape; refusing stale/mixed audit")
    urth=panel[panel.ticker=="URTH"][["date","forward_eur_21d"]].rename(columns={"forward_eur_21d":"urth_forward_eur_21d"}); pred=pred.merge(urth,on="date",how="left",validate="many_to_one")
    top=top_bucket(pred); risk=pd.read_csv(ROOT/"reports/real_risk_decisions.csv",parse_dates=["date"])
    cost=cost_audit(top,cfg,urth); risk_summary,risk_concentration=risk_audit(risk,panel,cfg); periods=period_audit(pred,top,cfg); assets=asset_audit(pred,top,risk); paths=sequential_paths(top,cfg)
    loao_path=ROOT/"reports/audit_loao_predictions.csv"
    if loao_path.exists():
        loao=pd.read_csv(loao_path,parse_dates=["date","target_end_date","training_end"])
        if len(loao)!=3404 or set(loao.model)!=set(MODELS): raise RuntimeError("Cached leave-one-asset-out artifact has unexpected provenance")
    else: loao=leave_one_asset_out(panel,FEATURES)
    loao_top=top_bucket(loao); compare_rows=[]
    for validation,all_predictions,selected in (("pooled",pred[pred.model.isin(MODELS)],top[top.model.isin(MODELS)]),("leave_one_asset_out",loao,loao_top)):
        for model,a in all_predictions.groupby("model"):
            s=selected[selected.model==model]; compare_rows.append({"model":model,"validation":validation,"all_observations":len(a),"prediction_correlation":a.prediction.corr(a.forward_excess_21d),"directional_accuracy":(np.sign(a.prediction)==np.sign(a.forward_excess_21d)).mean(),"selected_observations":len(s),"selected_mean_excess":s.forward_excess_21d.mean(),"selected_median_excess":s.forward_excess_21d.median(),"selected_outperformance":(s.forward_excess_21d>0).mean()})
    loao_summary=pd.DataFrame(compare_rows)
    u=top[top.model=="unconditional"].copy(); u["period"]=u.date.dt.year.map(lambda y:next(label for lo,hi,label in PERIODS if lo<=y<=hi)); unconditional_rows=[]
    for dimension in ("ticker","period"):
        for value,g in u.groupby(dimension): unconditional_rows.append({"dimension":dimension,"value":value,"selected":len(g),"mean_excess":g.forward_excess_21d.mean(),"gross_value_added_eur":(g.forward_excess_21d*1000).sum()})
    unconditional=pd.DataFrame(unconditional_rows); unconditional["note"]="Global expanding-training mean prediction is identical across assets on a date; method='first' tie-breaking makes top-bucket membership depend on sorted ticker order."
    outputs={"audit_cost_sensitivity.csv":cost,"audit_risk_veto.csv":risk_summary,"audit_risk_concentration.csv":risk_concentration,"audit_subperiods.csv":periods,"audit_assets.csv":assets,"audit_leave_one_asset_out.csv":loao_summary,"audit_unconditional.csv":unconditional,"audit_drawdowns.csv":paths,"audit_loao_predictions.csv":loao}
    for name,frame in outputs.items(): frame.to_csv(ROOT/"reports"/name,index=False)
    metadata={"experiment_id":experiment_id,"dataset_type":"genuine historical market and macro data","mode":"real","generated_timestamp_utc":generated,"sample_count_per_method":1702,"total_prediction_rows":len(pred),"model_configuration_hash":spec_hash,"frozen_specification":spec,"source_prediction_artifact":"real_walk_forward_predictions.csv","artifact_consistency_note":"The real pooled experiment has 1,702 asset-date observations per method. The separate demo artifact has 2,757 observations per ticker/model because it uses longer deterministic synthetic histories and a different command; it is not comparable research evidence.","audit_artifacts":list(outputs),"dashboard_status":"audit_only_insufficient_evidence"}
    (ROOT/"reports/audit_metadata.json").write_text(json.dumps(metadata,indent=2),encoding="utf-8"); (ROOT/"reports/current_real_experiment.json").write_text(json.dumps(metadata,indent=2),encoding="utf-8")
    print(json.dumps({"experiment_id":experiment_id,"configuration_hash":spec_hash,"artifacts":list(outputs)},indent=2)); print("\nCOST @ €1000 / SPY\n",cost[(cost.allocation_eur==1000)&(cost.benchmark=="SPY")].to_string(index=False)); print("\nRISK\n",risk_summary.to_string(index=False)); print("\nLOAO\n",loao_summary.to_string(index=False)); print("\nPATHS\n",paths.to_string(index=False))

if __name__=="__main__": run()
