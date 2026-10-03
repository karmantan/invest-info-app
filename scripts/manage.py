from __future__ import annotations
import argparse, json
from pathlib import Path
import numpy as np
import pandas as pd
from src.config import ROOT, load_config
from src.storage import Database
from src.data.adapters import AlphaVantagePrices, FredAlfredAdapter, YahooChartPrices, FredCsvAdapter, price_quality
from src.features import build_features
from src.models import yearly_walk_forward
from src.features.real_panel import build_real_panel, FEATURES
from src.models.real_experiment import strict_walk_forward, statistical_metrics, ranked_results, bucket_results
from src.portfolio.allocation import allocate

def demo_prices():
    rng=np.random.default_rng(42); dates=pd.bdate_range("2012-01-02",periods=3300); frames=[]
    for i,ticker in enumerate(["SPY","URTH","SMH","XBI","XLE"]):
        returns=rng.normal(.00025+i*.00001,.009+i*.0015,len(dates)); close=100*np.exp(np.cumsum(returns))
        frames.append(pd.DataFrame({"date":dates,"ticker":ticker,"close":close,"source":"DEMONSTRATION ONLY — synthetic"}))
    return pd.concat(frames,ignore_index=True)

def cmd_init(_): Database().initialize(); print("Initialized state/invest.db")
def cmd_update(args):
    cfg=load_config(); frames=[]
    for item in cfg["universe"]:
        try: frames.append(AlphaVantagePrices().fetch(item["ticker"],args.refresh))
        except RuntimeError as e: print(f"{item['ticker']}: {e}")
    if frames: pd.concat(frames).to_parquet(ROOT/"data/processed/prices.parquet",index=False); print("Updated cached market prices")
def cmd_ingest_real(args):
    cfg=load_config(); processed=ROOT/"data/processed"; processed.mkdir(parents=True,exist_ok=True); frames=[]; quality=[]
    preferred=AlphaVantagePrices(); fallback=YahooChartPrices(); has_alpha=bool(__import__("os").getenv("ALPHA_VANTAGE_API_KEY"))
    for item in cfg["universe"]:
        ticker=item["ticker"]
        try:
            frame=preferred.fetch(ticker,args.refresh) if has_alpha else fallback.fetch(ticker,args.refresh)
            if "currency" not in frame: frame["currency"]="USD"
            if "adjustment" not in frame: frame["adjustment"]="adjusted close"
            frames.append(frame); quality.append(price_quality(frame,ticker,cfg.get("research_proxies",{}).get(ticker)))
            print(f"{ticker}: {len(frame)} rows cached")
        except Exception as exc:
            quality.append({"status":"FAILED","ticker":ticker,"provider":"Alpha Vantage" if has_alpha else "Yahoo Finance chart","warnings":[str(exc)]}); print(f"{ticker}: FAILED — {exc}")
    try:
        fx=fallback.fetch("EURUSD=X",args.refresh); fx.to_parquet(processed/"eurusd.parquet",index=False); quality.append(price_quality(fx,"EURUSD=X",{"represents":"USD per EUR exchange rate","actual_instrument":None}))
    except Exception as exc: quality.append({"status":"FAILED","ticker":"EURUSD=X","provider":"Yahoo Finance chart","warnings":[str(exc)]}); fx=None
    macro_parts={}
    for sid in ("DGS2","DGS10","VIXCLS"):
        try:
            col={"DGS2":"dgs2","DGS10":"dgs10","VIXCLS":"vix"}[sid]
            f=FredCsvAdapter().fetch(sid,args.refresh); macro_parts[sid]=f[["date","value"]].rename(columns={"value":col}); quality.append({"status":"WORKING" if f.value.notna().sum()>1000 else "PARTIAL","provider":"Federal Reserve Economic Data (FRED)","ticker":sid,"download_timestamp_utc":pd.Timestamp.now(tz="UTC").isoformat(),"earliest_date":str(f.date.min().date()),"latest_date":str(f.date.max().date()),"rows":int(f.value.notna().sum()),"adjustment":"not applicable; latest FRED vintage","currency":"percent/index","missing_observations":int(f.value.isna().sum()),"proxy":None,"warnings":["Latest-vintage FRED CSV; not an ALFRED vintage archive"]})
        except Exception as exc: quality.append({"status":"FAILED","ticker":sid,"provider":"FRED","warnings":[str(exc)]})
    if frames: pd.concat(frames,ignore_index=True).to_parquet(processed/"real_prices.parquet",index=False)
    if len(macro_parts)==3:
        macro=macro_parts["DGS2"].merge(macro_parts["DGS10"],on="date",how="outer").merge(macro_parts["VIXCLS"],on="date",how="outer").sort_values("date")
        macro[["dgs2","dgs10","vix"]]=macro[["dgs2","dgs10","vix"]].ffill(limit=5); macro["curve_2s10s"]=macro.dgs10-macro.dgs2
        for col in ("dgs2","dgs10"):
            for d in (1,5,21): macro[f"{col}_change_{d}d"]=macro[col].diff(d)
        macro.to_parquet(processed/"macro_state.parquet",index=False)
    (ROOT/"reports/data_quality.json").write_text(json.dumps(quality,indent=2),encoding="utf-8")
    failed=[q for q in quality if q["status"]=="FAILED"]
    print(f"Data-quality manifest written; {len(failed)} failed datasets. Re-run the same command to resume from cache.")
    if failed or not frames or fx is None or len(macro_parts)<3: raise SystemExit(2)

def cmd_experiment(_):
    cfg=load_config(); processed=ROOT/"data/processed"; required=[processed/"real_prices.parquet",processed/"eurusd.parquet",processed/"macro_state.parquet"]
    missing=[str(p) for p in required if not p.exists()]
    if missing: raise SystemExit("Missing genuine datasets: "+", ".join(missing))
    prices=pd.read_parquet(required[0]); fx=pd.read_parquet(required[1]); macro=pd.read_parquet(required[2])
    if prices.source.str.contains("synthetic",case=False).any(): raise SystemExit("Synthetic data rejected by real experiment")
    panel=build_real_panel(prices,fx,macro,cfg["app"]["principal_benchmark"]); panel.to_parquet(processed/"real_feature_panel.parquet",index=False)
    pred=strict_walk_forward(panel,FEATURES); 
    if pred.empty: raise SystemExit("Insufficient genuine history for walk-forward experiment")
    costs=cfg["costs"]; nominal=cfg["app"]["nominal_test_allocation_eur"]
    stats=statistical_metrics(pred); ranked,selected=ranked_results(pred,nominal,costs["buy_fee_eur"],costs["sell_fee_eur"],costs["estimated_spread_bps"]); buckets=bucket_results(pred)
    risk_rows=[]
    for _,row in pred.iterrows():
        forecast={"expected_return":row.prediction+row.benchmark_forward_eur_21d,"expected_excess_return":row.prediction,"p10":row.prediction+row.residual_p10,"sample_size":row.training_rows,"probability_loss_gt_5pct":row.residual_prob_loss5,"confidence":min(.8,.5+abs(row.prediction)*3)}
        decision=allocate(forecast,0,nominal,nominal/.08,0,cfg)
        risk_rows.append({"model":row.model,"date":row.date,"ticker":row.ticker,"accepted":decision.action=="BUY","vetoes":"; ".join(decision.vetoes),"actual_excess":row.forward_excess_21d,"actual_return":row.forward_eur_21d,"benchmark_return":row.benchmark_forward_eur_21d})
    risk=pd.DataFrame(risk_rows); risk["after_cost_excess"]=risk.actual_excess-(costs["buy_fee_eur"]+costs["sell_fee_eur"])/nominal-2*costs["estimated_spread_bps"]/10000
    risk_summary=risk.groupby(["model","accepted"]).agg(signals=("ticker","size"),mean_excess=("actual_excess","mean"),mean_after_cost_excess=("after_cost_excess","mean"),median_excess=("actual_excess","median"),probability_outperform=("actual_excess",lambda s:(s>0).mean())).reset_index()
    coverage={"prediction_start":str(pred.date.min().date()),"prediction_end":str(pred.date.max().date()),"target_end":str(pred.target_end_date.max().date()),"non_overlapping_rebalance_spacing_trading_days":21,"benchmark":cfg["app"]["principal_benchmark"],"currency":"EUR","feature_policy":"features at date close; FRED values lagged one observation day; training purged until target end before test start"}
    pred.to_csv(ROOT/"reports/real_walk_forward_predictions.csv",index=False); stats.to_csv(ROOT/"reports/real_statistical_results.csv",index=False); ranked.to_csv(ROOT/"reports/real_ranked_results.csv",index=False); buckets.to_csv(ROOT/"reports/real_signal_buckets.csv",index=False); risk.to_csv(ROOT/"reports/real_risk_decisions.csv",index=False); risk_summary.to_csv(ROOT/"reports/real_risk_summary.csv",index=False); (ROOT/"reports/real_experiment_metadata.json").write_text(json.dumps(coverage,indent=2),encoding="utf-8")
    print("COVERAGE",json.dumps(coverage)); print("STATISTICAL\n",stats.to_string(index=False)); print("RANKED\n",ranked.to_string(index=False)); print("RISK VETO\n",risk_summary.to_string(index=False)); print("Real-data experiment artifacts written to reports/.")
def cmd_demo(_):
    (ROOT/"data/processed").mkdir(parents=True,exist_ok=True); demo_prices().to_parquet(ROOT/"data/processed/demo_prices.parquet",index=False); print("Created clearly labeled synthetic demonstration data")
def cmd_train(args):
    path=ROOT/"data/processed"/("demo_prices.parquet" if args.demo else "prices.parquet")
    if not path.exists(): raise SystemExit(f"Missing {path}; run update-data or demo-data first")
    frame=build_features(pd.read_parquet(path),horizon=21); features=["return_5d","return_21d","return_63d","drawdown_63d","volatility_21d"]
    results={}
    out=[]
    for ticker,part in frame.groupby("ticker"):
        for model in ("ridge","gradient_boosting"):
            result=yearly_walk_forward(part,features,"forward_return_21d",model); results[f"{ticker}:{model}"]=result.metrics
            if not result.predictions.empty: out.append(result.predictions.assign(ticker=ticker))
    (ROOT/"reports").mkdir(exist_ok=True); (ROOT/"reports/demo_synthetic_model_results.json").write_text(json.dumps({"experiment_id":"demo-synthetic-21d-v1","dataset_type":"deterministic synthetic demonstration","mode":"demo","generated_timestamp_utc":pd.Timestamp.now(tz="UTC").isoformat(),"sample_count_by_model":{k:v.get("observations",0) for k,v in results.items()},"results":results},indent=2),encoding="utf-8")
    if out: pd.concat(out).to_csv(ROOT/"reports/demo_synthetic_walk_forward_predictions.csv",index=False)
    print(json.dumps(results,indent=2)); print("Results are research evidence, not proof of profitability.")
def main():
    parser=argparse.ArgumentParser(); sub=parser.add_subparsers(required=True)
    sub.add_parser("init-db").set_defaults(fn=cmd_init)
    u=sub.add_parser("update-data"); u.add_argument("--refresh",action="store_true"); u.set_defaults(fn=cmd_update)
    i=sub.add_parser("ingest-real"); i.add_argument("--refresh",action="store_true"); i.set_defaults(fn=cmd_ingest_real)
    sub.add_parser("experiment-real").set_defaults(fn=cmd_experiment)
    sub.add_parser("demo-data").set_defaults(fn=cmd_demo)
    t=sub.add_parser("train"); t.add_argument("--demo",action="store_true"); t.set_defaults(fn=cmd_train)
    args=parser.parse_args(); args.fn(args)
if __name__=="__main__": main()
