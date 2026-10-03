from __future__ import annotations

import hashlib
import json
import os
import smtplib
import sqlite3
from datetime import datetime, timezone
from email.message import EmailMessage
from pathlib import Path
from zoneinfo import ZoneInfo

import joblib
import numpy as np
import pandas as pd

from src.config import ROOT, load_config
from src.features.medium import MEDIUM_FEATURES, build_medium_panel
from src.features.real_panel import FEATURES, build_real_panel
from src.models.real_experiment import _model
from src.reporting.audit_21d import costs
from src.reporting.medium_risk_audit import FLAG_COLUMNS, add_point_in_time_flags
from src.storage import Database
from dashboard.ui import asset_name, friendly_reason, horizon_label, status_label


BERLIN = ZoneInfo("Europe/Berlin")
MODEL_DIR = ROOT / "state/models"
REPORT_DIR = ROOT / "reports/prospective"


def _hash_file(path: Path) -> str:
    h=hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda:f.read(1024*1024),b""): h.update(chunk)
    return h.hexdigest()


def _json_hash(value) -> str:
    return hashlib.sha256(json.dumps(value,sort_keys=True,separators=(",",":"),default=str).encode()).hexdigest()


def _clean_dict(row, columns):
    result={}
    for col in columns:
        value=row[col]
        if pd.isna(value): result[col]=None
        elif isinstance(value,(np.integer,np.floating,np.bool_)): result[col]=value.item()
        elif isinstance(value,pd.Timestamp): result[col]=value.isoformat()
        else: result[col]=value
    return result


def _data_snapshot():
    paths=[ROOT/"data/processed/real_prices.parquet",ROOT/"data/processed/eurusd.parquet",ROOT/"data/processed/macro_state.parquet",ROOT/"data/processed/company_fundamentals.parquet"]
    missing=[str(p) for p in paths if not p.exists()]
    if missing: raise RuntimeError("Missing required data: "+", ".join(missing))
    return max(datetime.fromtimestamp(p.stat().st_mtime,timezone.utc) for p in paths).isoformat(),_json_hash({p.name:_hash_file(p) for p in paths})


def build_current_panels(cfg):
    prices=pd.read_parquet(ROOT/"data/processed/real_prices.parquet"); fx=pd.read_parquet(ROOT/"data/processed/eurusd.parquet"); macro=pd.read_parquet(ROOT/"data/processed/macro_state.parquet")
    if prices.source.astype(str).str.contains("synthetic|demo",case=False,regex=True).any(): raise RuntimeError("Synthetic/demo prices rejected")
    fundamentals=pd.read_parquet(ROOT/"data/processed/company_fundamentals.parquet")
    membership={"SMH":["NVDA","AMD","AVGO","INTC","QCOM","MU","AMAT","TXN"],"XLK":["AAPL","MSFT","NVDA","AVGO","CRM","ORCL"],"XLE":["XOM","CVX","COP","SLB","EOG"]}
    short=build_real_panel(prices,fx,macro,cfg["app"]["principal_benchmark"])
    medium=build_medium_panel(prices,fx,macro,fundamentals,membership)
    short.to_parquet(ROOT/"data/processed/prospective_21d_panel.parquet",index=False); medium.to_parquet(ROOT/"data/processed/prospective_medium_panel.parquet",index=False)
    return short,medium,prices,fx,macro


def quality_check(prices,fx,macro,market_date):
    warnings=[]; hard=[]
    quality_path=ROOT/"reports/data_quality.json"
    if quality_path.exists():
        for item in json.loads(quality_path.read_text()):
            if item.get("status")=="FAILED": hard.append(f"{item.get('ticker')}: provider failed")
    latest_prices=prices.groupby("ticker").date.max()
    for ticker,last in latest_prices.items():
        gap=len(pd.bdate_range(pd.Timestamp(last)+pd.offsets.BDay(1),market_date))
        if gap>1: hard.append(f"{ticker}: stale price ({str(last)[:10]})")
    fx_latest=pd.to_datetime(fx.date).max(); macro_latest=pd.to_datetime(macro.date).max()
    if len(pd.bdate_range(fx_latest+pd.offsets.BDay(1),market_date))>1: hard.append(f"FX stale ({fx_latest.date()})")
    if len(pd.bdate_range(macro_latest+pd.offsets.BDay(1),market_date))>2: hard.append(f"macro stale ({macro_latest.date()})")
    return {"hard":hard,"warnings":warnings,"price_latest":str(pd.to_datetime(prices.date).max().date()),"fx_latest":str(fx_latest.date()),"macro_latest":str(macro_latest.date())}


def initialize_frozen_models(short,medium,meta):
    MODEL_DIR.mkdir(parents=True,exist_ok=True); manifest_path=MODEL_DIR/"prospective_manifest.json"
    if manifest_path.exists(): return json.loads(manifest_path.read_text())
    models={}
    for horizon,panel,features,target,end in ((21,short,FEATURES,"forward_excess_21d","target_end_date"),(63,medium,MEDIUM_FEATURES,"forward_excess_63d","target_end_63d"),(126,medium,MEDIUM_FEATURES,"forward_excess_126d","target_end_126d")):
        latest=pd.to_datetime(panel.date).max(); train=panel[(panel[end]<latest)&panel[target].notna()].copy(); model=_model("ridge"); model.fit(train[features],train[target]); path=MODEL_DIR/f"frozen_ridge_{horizon}d.joblib"; joblib.dump(model,path)
        models[str(horizon)]={"path":str(path.relative_to(ROOT)),"sha256":_hash_file(path),"training_rows":len(train),"training_data_end":str(pd.to_datetime(train[end]).max()),"features":features}
    manifest={"created_at":datetime.now(timezone.utc).isoformat(),"frozen_21d_research_hash":meta["short_hash"],"frozen_medium_research_hash":meta["medium_hash"],"models":models}
    manifest_path.write_text(json.dumps(manifest,indent=2)); return manifest


def ensure_experiment(db,manifest,risk_hash,cfg_hash,snapshot_ts):
    experiment_id="prospective-paper-"+_json_hash({"manifest":manifest,"risk":risk_hash,"config":cfg_hash})[:12]
    with db.connect() as c:
        row=c.execute("SELECT * FROM prospective_experiments WHERE experiment_id=?",(experiment_id,)).fetchone()
        if not row:
            c.execute("INSERT INTO prospective_experiments(experiment_id,started_at,frozen_21d_hash,frozen_medium_hash,risk_policy_hash,configuration_hash,data_snapshot_timestamp,mode,real_capital_enabled) VALUES(?,?,?,?,?,?,?,?,0)",(experiment_id,datetime.now(timezone.utc).isoformat(),manifest["frozen_21d_research_hash"],manifest["frozen_medium_research_hash"],risk_hash,cfg_hash,snapshot_ts,"paper"))
    return experiment_id


def current_forecasts(short,medium,manifest):
    outputs=[]
    for horizon,panel,features in ((21,short,FEATURES),(63,medium,MEDIUM_FEATURES),(126,medium,MEDIUM_FEATURES)):
        x=panel.sort_values("date").groupby("ticker").tail(1).copy(); model=joblib.load(ROOT/manifest["models"][str(horizon)]["path"]); x["forecast"]=model.predict(x[features]); x["forecast_rank"]=x.forecast.rank(pct=True,method="first"); x["top_cohort"]=x.forecast_rank>.66; x["horizon_days"]=horizon
        if horizon==21:
            x["active_risk_flags"]="none"; x["policy_2_pass"]=False
        else:
            x=add_point_in_time_flags(panel).sort_values("date").groupby("ticker").tail(1).merge(x[["ticker","forecast","forecast_rank","top_cohort","horizon_days"]],on="ticker",how="inner",suffixes=("","_forecast")); x["active_risk_flags"]=x.apply(lambda r:", ".join(f for f in FLAG_COLUMNS if bool(r[f])) or "none",axis=1); x["policy_2_pass"]=~(x.broad_market_downtrend|x.extreme_asset_drawdown)
        outputs.append((horizon,x,features))
    return outputs


def _eur_series(prices,fx,ticker):
    p=prices[prices.ticker==ticker][["date","close"]].copy(); p.date=pd.to_datetime(p.date); f=fx[["date","close"]].rename(columns={"close":"eurusd"}).copy(); f.date=pd.to_datetime(f.date)
    return pd.merge_asof(p.sort_values("date"),f.sort_values("date"),on="date",direction="backward",tolerance=pd.Timedelta("5d")).assign(close_eur=lambda z:z.close/z.eurusd)


def update_realized_outcomes(c,experiment_id,prices,fx,now):
    pending=c.execute("SELECT * FROM paper_signals WHERE experiment_id=? AND realized_at IS NULL",(experiment_id,)).fetchall()
    spy=_eur_series(prices,fx,"SPY").set_index("date"); urth=_eur_series(prices,fx,"URTH").set_index("date")
    for row in pending:
        asset=_eur_series(prices,fx,row["asset"]).reset_index(drop=True); dates=asset.index[asset.date>=pd.Timestamp(row["observed_date"])].tolist()
        if len(dates)<=row["horizon_days"]: continue
        start=asset.loc[dates[0]]; end=asset.loc[dates[row["horizon_days"]]]; end_date=end.date
        if pd.Timestamp(row["observed_date"]) not in spy.index or end_date not in spy.index: continue
        raw=end.close_eur/start.close_eur-1; sr=spy.loc[end_date].close_eur/spy.loc[pd.Timestamp(row["observed_date"])].close_eur-1
        ur=np.nan
        if pd.Timestamp(row["observed_date"]) in urth.index and end_date in urth.index: ur=urth.loc[end_date].close_eur/urth.loc[pd.Timestamp(row["observed_date"])].close_eur-1
        c.execute("UPDATE paper_signals SET realized_at=?,realized_return=?,spy_return=?,urth_return=?,excess_spy=?,excess_urth=? WHERE id=?",(now,raw,sr,None if pd.isna(ur) else ur,raw-sr,None if pd.isna(ur) else raw-ur,row["id"]))
    challenger=c.execute("SELECT * FROM prospective_model_forecasts WHERE experiment_id=? AND realized_at IS NULL",(experiment_id,)).fetchall()
    for row in challenger:
        asset=_eur_series(prices,fx,row["asset"]).reset_index(drop=True); dates=asset.index[asset.date>=pd.Timestamp(row["observed_date"])].tolist()
        if len(dates)<=row["horizon_days"]: continue
        start=asset.loc[dates[0]]; end=asset.loc[dates[row["horizon_days"]]]; end_date=end.date
        if pd.Timestamp(row["observed_date"]) not in spy.index or end_date not in spy.index: continue
        raw=end.close_eur/start.close_eur-1; benchmark=spy.loc[end_date].close_eur/spy.loc[pd.Timestamp(row["observed_date"])].close_eur-1
        c.execute("UPDATE prospective_model_forecasts SET realized_at=?,realized_return=?,benchmark_return=?,excess_return=? WHERE id=?",(now,raw,benchmark,raw-benchmark,row["id"]))


def _latest_snapshot_value(c):
    row=c.execute("SELECT total_value_eur FROM portfolio_snapshots WHERE total_value_eur IS NOT NULL ORDER BY snapshot_date DESC,id DESC LIMIT 1").fetchone()
    return float(row[0]) if row else None


def _position_price(prices,fx,ticker,date=None):
    s=_eur_series(prices,fx,ticker); s=s[s.date<=pd.Timestamp(date)] if date else s
    return float(s.iloc[-1].close_eur),str(s.iloc[-1].date.date())


def close_expired(c,experiment_id,prices,fx,market_date,run_id,now):
    changes=[]
    positions=c.execute("SELECT * FROM paper_positions WHERE experiment_id=? AND closed_date IS NULL",(experiment_id,)).fetchall()
    for p in positions:
        asset=_eur_series(prices,fx,p["asset"]); elapsed=len(asset[(asset.date>pd.Timestamp(p["entry_date"]))&(asset.date<=market_date)])
        if elapsed<p["horizon_days"]: continue
        current,_=_position_price(prices,fx,p["asset"],market_date); gross=current/p["entry_price"]-1; mc,_=costs(p["allocation_eur"],load_config()); net=gross-mc/p["allocation_eur"]
        spy_now,_=_position_price(prices,fx,"SPY",market_date); urth_now,_=_position_price(prices,fx,"URTH",market_date)
        spy=spy_now/p["benchmark_spy_entry"]-1; urth=urth_now/p["benchmark_urth_entry"]-1
        c.execute("UPDATE paper_positions SET current_status='PAPER EXIT',closed_date=?,closed_price=?,gross_return=?,net_return=?,spy_return=?,urth_return=?,simulated_cost_eur=? WHERE id=?",(str(market_date.date()),current,gross,net,spy,urth,mc,p["id"]))
        changes.append((p["asset"],p["horizon_days"],p["current_status"],"PAPER EXIT","Frozen horizon completed",1))
    return changes


def send_alerts(changes):
    required=("SMTP_HOST","SMTP_FROM","ALERT_TO"); missing=[k for k in required if not os.getenv(k)]
    if missing: return "NOT CONFIGURED — "+", ".join(missing)
    host=os.environ["SMTP_HOST"]; port=int(os.getenv("SMTP_PORT","587")); username=os.getenv("SMTP_USERNAME"); password=os.getenv("SMTP_PASSWORD")
    for change in changes:
        asset,horizon,previous,new,reason,_=change; msg=EmailMessage(); msg["Subject"]=f"Quiet Capital: {asset} paper signal changed"; msg["From"]=os.environ["SMTP_FROM"]; msg["To"]=os.environ["ALERT_TO"]
        msg["Subject"]=f"Quiet Capital: {asset_name(asset)} — {status_label(new)}"
        msg.set_content(f"Investment: {asset_name(asset)}\nPrevious: {status_label(previous)}\nNow: {status_label(new)}\nTime horizon: {horizon_label(horizon)}\nWhy: {friendly_reason(reason,asset,horizon,new)}\n\nThis is a paper signal using no real money. It is not a trade instruction.\n")
        with smtplib.SMTP(host,port,timeout=30) as smtp:
            if os.getenv("SMTP_STARTTLS","true").lower()=="true": smtp.starttls()
            if username and password: smtp.login(username,password)
            smtp.send_message(msg)
    return "SENT"


def send_test_email():
    required=("SMTP_HOST","SMTP_FROM","ALERT_TO"); missing=[k for k in required if not os.getenv(k)]
    if missing: raise RuntimeError("Email is not configured. Set: "+", ".join(missing))
    msg=EmailMessage(); msg["Subject"]="Quiet Capital: email configuration test"; msg["From"]=os.environ["SMTP_FROM"]; msg["To"]=os.environ["ALERT_TO"]
    msg.set_content("Quiet Capital email alerts are configured. This is a configuration test, not a model signal or trade instruction.\n")
    host=os.environ["SMTP_HOST"]; port=int(os.getenv("SMTP_PORT","587")); username=os.getenv("SMTP_USERNAME"); password=os.getenv("SMTP_PASSWORD")
    with smtplib.SMTP(host,port,timeout=30) as smtp:
        if os.getenv("SMTP_STARTTLS","true").lower()=="true": smtp.starttls()
        if username and password: smtp.login(username,password)
        smtp.send_message(msg)
    return "TEST EMAIL SENT"


def meaningful_email_transition(previous,new):
    return previous!=new and (new in ("PAPER BUY","PAPER REDUCE","PAPER EXIT") or (previous=="PAPER HOLD" and new=="MODEL INTERESTING — RISK VETO"))


def write_reviews(c,experiment_id,market_date):
    REPORT_DIR.mkdir(parents=True,exist_ok=True)
    positions=pd.read_sql_query("SELECT * FROM paper_positions WHERE experiment_id=?",c,params=(experiment_id,))
    changes=pd.read_sql_query("SELECT * FROM paper_status_changes WHERE experiment_id=?",c,params=(experiment_id,))
    signals=pd.read_sql_query("SELECT * FROM paper_signals WHERE experiment_id=?",c,params=(experiment_id,))
    openp=positions[positions.closed_date.isna()]
    changed_dates=pd.to_datetime(changes.changed_at,errors="coerce",utc=True).dt.tz_localize(None) if len(changes) else pd.Series(dtype="datetime64[ns]")
    recent=changes[changed_dates>=pd.Timestamp(market_date).tz_localize(None)-pd.Timedelta(days=7)] if len(changes) else changes
    weekly=f"# Weekly paper review — through {market_date.date()}\n\nResearch-only, zero capital. No model retraining.\n\nOpen positions: {len(openp)}\n\nStatus changes in trailing seven days: {len(recent)}\n\nData failures are listed in prospective_runs. Upcoming expiries are derived from fixed 63/126-session horizons.\n"
    (REPORT_DIR/"weekly-current.md").write_text(weekly)
    if market_date.weekday()==0: (REPORT_DIR/f"weekly-{market_date.date()}.md").write_text(weekly)
    complete=positions[positions.closed_date.notna()]; gross=complete.gross_return.mean() if len(complete) else np.nan; spy=complete.spy_return.mean() if len(complete) else np.nan; urth=complete.urth_return.mean() if len(complete) else np.nan
    monthly=f"# Monthly prospective review — {market_date.strftime('%Y-%m')}\n\nProspective paper records only; backtests excluded.\n\nCompleted trades: {len(complete)}\nOpen positions: {positions.closed_date.isna().sum() if len(positions) else 0}\nMean gross return: {gross}\nMean SPY return: {spy}\nMean URTH return: {urth}\nSignals with outcomes: {signals.realized_at.notna().sum() if len(signals) else 0}\n"
    (REPORT_DIR/"monthly-current.md").write_text(monthly)
    if market_date.day<=3: (REPORT_DIR/f"monthly-{market_date.strftime('%Y-%m')}.md").write_text(monthly)


def write_performance(c,experiment_id):
    REPORT_DIR.mkdir(parents=True,exist_ok=True); positions=pd.read_sql_query("SELECT * FROM paper_positions WHERE experiment_id=? ORDER BY closed_date,id",c,params=(experiment_id,)); complete=positions[positions.closed_date.notna()].copy() if len(positions) else positions
    def drawdown(column):
        if complete.empty: return None
        path=np.r_[1.,(1+complete[column].astype(float)).cumprod().to_numpy()]; return maximum_drawdown(path)
    result={"experiment_id":experiment_id,"mode":"prospective_paper_only","completed_paper_trades":len(complete),"currently_open":int(positions.closed_date.isna().sum()) if len(positions) else 0,"gross_return":complete.gross_return.mean() if len(complete) else None,"net_simulated_return":complete.net_return.mean() if len(complete) else None,"spy_return":complete.spy_return.mean() if len(complete) else None,"urth_return":complete.urth_return.mean() if len(complete) else None,"excess_spy":(complete.gross_return-complete.spy_return).mean() if len(complete) else None,"excess_urth":(complete.gross_return-complete.urth_return).mean() if len(complete) else None,"win_rate":(complete.gross_return>0).mean() if len(complete) else None,"benchmark_outperformance_rate":(complete.gross_return>complete.spy_return).mean() if len(complete) else None,"p10":complete.gross_return.quantile(.1) if len(complete)>=10 else None,"maximum_drawdown":drawdown("net_return"),"spy_maximum_drawdown":drawdown("spy_return"),"urth_maximum_drawdown":drawdown("urth_return"),"turnover":len(complete),"simulated_costs_eur":complete.simulated_cost_eur.sum() if len(complete) else 0.,"small_sample_warning":None if len(complete)>=20 else "Fewer than 20 completed paper trades; no confidence inference is permitted."}
    (REPORT_DIR/"current_performance.json").write_text(json.dumps(result,indent=2,default=lambda v:None if pd.isna(v) else float(v))); return result


def run_daily(ingestion_failed=None):
    cfg=load_config(); db=Database(); db.initialize(); now=datetime.now(timezone.utc).isoformat(); local_now=datetime.now(BERLIN)
    config_hash=_hash_file(ROOT/"config/default.yaml"); short_meta=json.loads((ROOT/"reports/current_real_experiment.json").read_text()); medium_meta=json.loads((ROOT/"reports/medium_research_metadata.json").read_text()); risk_meta=json.loads((ROOT/"reports/medium_risk_metadata.json").read_text())
    snapshot_ts,snapshot_hash=_data_snapshot(); short,medium,prices,fx,macro=build_current_panels(cfg); market_date=pd.to_datetime(prices.date).max().normalize(); manifest=initialize_frozen_models(short,medium,{"short_hash":short_meta["model_configuration_hash"],"medium_hash":medium_meta["specification_hash"]}); experiment_id=ensure_experiment(db,manifest,risk_meta["risk_specification_hash"],config_hash,snapshot_ts)
    from src.model_governance import initialize_versions,log_challenger_forecasts,maybe_schedule_challenger,status as governance_status
    initialize_versions(db); maybe_schedule_challenger(local_now,db); governance=governance_status(db); suspend_entries=governance["model_validity"]!="NORMAL"
    with db.connect() as c:
        existing=c.execute("SELECT * FROM prospective_runs WHERE experiment_id=? AND market_date=?",(experiment_id,str(market_date.date()))).fetchone()
        if existing and existing["status"]=="SUCCESS": return {"status":"ALREADY_COMPLETED","run_id":existing["id"],"experiment_id":experiment_id,"market_date":str(market_date.date())}
        if existing: run_id=existing["id"]; c.execute("UPDATE prospective_runs SET started_at=?,status='RUNNING',error_type=NULL,error_message=NULL WHERE id=?",(now,run_id))
        else: run_id=c.execute("INSERT INTO prospective_runs(experiment_id,market_date,started_at,status,data_snapshot_timestamp,data_snapshot_hash) VALUES(?,?,?,?,?,?)",(experiment_id,str(market_date.date()),now,"RUNNING",snapshot_ts,snapshot_hash)).lastrowid
        quality=quality_check(prices,fx,macro,market_date)
        if ingestion_failed: quality["hard"].append("provider refresh failed: "+ingestion_failed)
        c.execute("UPDATE prospective_runs SET price_latest_date=?,macro_latest_date=?,fx_latest_date=? WHERE id=?",(quality["price_latest"],quality["macro_latest"],quality["fx_latest"],run_id))
        if quality["hard"]:
            message="; ".join(quality["hard"]); previous=c.execute("SELECT asset,horizon_days,recommendation FROM paper_signals WHERE experiment_id=? AND observed_date=(SELECT MAX(observed_date) FROM paper_signals WHERE experiment_id=?)",(experiment_id,experiment_id)).fetchall(); actionable=[r for r in previous if r["recommendation"] in ("PAPER BUY","PAPER HOLD","MODEL INTERESTING — RISK GATE PASSES")]
            stale_changes=[(r["asset"],r["horizon_days"],r["recommendation"],"SIGNAL SUPPRESSED — DATA STALE",message,1) for r in actionable]
            for change in stale_changes: c.execute("INSERT INTO paper_status_changes(experiment_id,run_id,changed_at,asset,horizon_days,previous_status,new_status,reason,alert_required) VALUES(?,?,?,?,?,?,?,?,?)",(experiment_id,run_id,now,*change))
            try: email_status=send_alerts(stale_changes) if stale_changes else "NO OTHERWISE ACTIONABLE SIGNAL — NO EMAIL"
            except Exception as exc: email_status=f"FAILED — {type(exc).__name__}: {exc}"
            c.execute("UPDATE paper_status_changes SET email_status=? WHERE run_id=? AND alert_required=1",(email_status,run_id)); c.execute("UPDATE prospective_runs SET status='SUPPRESSED_DATA_STALE',completed_at=?,error_type='DATA_QUALITY',error_message=? WHERE id=?",(now,message,run_id)); return {"status":"SIGNAL SUPPRESSED — DATA STALE","errors":quality["hard"],"run_id":run_id,"experiment_id":experiment_id,"email_status":email_status}
        update_realized_outcomes(c,experiment_id,prices,fx,now); changes=close_expired(c,experiment_id,prices,fx,market_date,run_id,now)
        closed_today={x[0] for x in changes if x[3]=="PAPER EXIT"}
        open_positions={r["asset"]:r for r in c.execute("SELECT * FROM paper_positions WHERE experiment_id=? AND closed_date IS NULL",(experiment_id,)).fetchall()}
        forecasts=current_forecasts(short,medium,manifest); candidate={}
        for horizon,x,features in forecasts:
            for _,r in x.iterrows():
                if horizon in (63,126) and r.top_cohort and r.policy_2_pass: candidate.setdefault(r.ticker,[]).append(horizon)
        portfolio_value=_latest_snapshot_value(c); inserted={}
        for horizon,x,features in forecasts:
            hist=pd.read_csv(ROOT/f"reports/medium_{horizon}d_predictions.csv") if horizon in (63,126) else None
            if hist is not None:
                hist=hist[hist.model=="ridge"]; hist=hist[hist.groupby("date").prediction.rank(pct=True,method="first")>.66]
            for _,r in x.iterrows():
                current=open_positions.get(r.ticker); flags=r.active_risk_flags; pass_policy=bool(r.policy_2_pass) if horizon in (63,126) else False
                if horizon==21: recommendation="WATCH" if r.top_cohort else "NO EDGE"; reason="Frozen 21-day research signal; live short-term recommendations remain disabled."
                elif current: recommendation="MODEL INTERESTING — RISK VETO" if not pass_policy else "PAPER HOLD"; reason="Existing zero-capital paper position; fixed horizon remains in force."
                elif suspend_entries and r.top_cohort: recommendation="MODEL VALIDITY REVIEW"; reason="Incumbent validity is under structural review; new paper entries are suspended without modifying the model."
                elif r.ticker in candidate and horizon==max(candidate[r.ticker]): recommendation="PAPER BUY"; reason="Frozen top cohort with Policy 2 pass; one coherent position uses the longest simultaneously triggered horizon."
                elif r.top_cohort and not pass_policy: recommendation="MODEL INTERESTING — RISK VETO"; reason="Frozen top cohort rejected by Policy 2 hard gate."
                elif r.top_cohort: recommendation="MODEL INTERESTING — RISK GATE PASSES"; reason="Eligible research forecast represented by another simultaneous horizon or existing capital constraint."
                else: recommendation="NO EDGE"; reason="Frozen forecast is outside the selected cohort."
                if current and horizon==current["horizon_days"]:
                    c.execute("UPDATE paper_positions SET current_status=?,risk_status=? WHERE id=?",("PAPER HOLD" if pass_policy else "MODEL INTERESTING — RISK VETO","PASS" if pass_policy else "VETO",current["id"]))
                price=float(r.close_eur); eurusd=float(r.usd_per_eur); asset_hist=hist[hist.ticker==r.ticker][f"forward_eur_{horizon}d"] if hist is not None else pd.Series(dtype=float)
                model_hash=manifest["models"][str(horizon)]["sha256"]; feature_json=json.dumps(_clean_dict(r,features),sort_keys=True)
                version="short-ridge-v1" if horizon==21 else "medium-ridge-v1"
                sig_id=c.execute("INSERT OR IGNORE INTO paper_signals(experiment_id,run_id,observed_date,observed_at,asset,horizon_days,forecast,forecast_rank,top_cohort,active_risk_flags,policy_2_pass,recommendation,benchmark,price,eurusd,expected_return,historical_p10,historical_p5,confidence,model_hash,configuration_hash,data_snapshot_timestamp,feature_values_json,decision_reason,model_version) VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",(experiment_id,run_id,str(market_date.date()),now,r.ticker,horizon,float(r.forecast),float(r.forecast_rank),int(r.top_cohort),flags,int(pass_policy),recommendation,"SPY",price,eurusd,float(r.forecast),asset_hist.quantile(.1) if len(asset_hist) else None,asset_hist.quantile(.05) if len(asset_hist) else None,"LOW" if len(asset_hist)<20 else "MEDIUM",model_hash,config_hash,snapshot_ts,feature_json,reason,version)).lastrowid
                if not sig_id: continue
                inserted[(r.ticker,horizon)]=sig_id
                prev=c.execute("SELECT recommendation FROM paper_signals WHERE experiment_id=? AND asset=? AND horizon_days=? AND observed_date<? ORDER BY observed_date DESC LIMIT 1",(experiment_id,r.ticker,horizon,str(market_date.date()))).fetchone(); previous=prev[0] if prev else None
                meaningful=meaningful_email_transition(previous,recommendation)
                if previous!=recommendation: changes.append((r.ticker,horizon,previous,recommendation,reason,int(meaningful)))
        for asset,horizons in candidate.items():
            if asset in open_positions or asset in closed_today or suspend_entries: continue
            chosen=max(horizons); sig_id=inserted.get((asset,chosen))
            if not sig_id: continue
            entry,_=_position_price(prices,fx,asset,market_date); spy,_=_position_price(prices,fx,"SPY",market_date); urth,_=_position_price(prices,fx,"URTH",market_date); sleeve=portfolio_value*.05 if portfolio_value else None; allocation=sleeve or 1000.
            c.execute("INSERT INTO paper_positions(experiment_id,asset,entry_signal_id,entry_date,entry_timestamp,entry_price,allocation_eur,portfolio_sleeve_eur,normalized_allocation_eur,horizon_days,triggered_horizons_json,benchmark_spy_entry,benchmark_urth_entry,original_thesis,current_status,risk_status) VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",(experiment_id,asset,sig_id,str(market_date.date()),now,entry,allocation,sleeve,1000.,chosen,json.dumps(horizons),spy,urth,"Frozen ridge top cohort; Policy 2 passed; zero-capital prospective test.","PAPER HOLD","PASS"))
        for change in changes:
            c.execute("INSERT INTO paper_status_changes(experiment_id,run_id,changed_at,asset,horizon_days,previous_status,new_status,reason,alert_required) VALUES(?,?,?,?,?,?,?,?,?)",(experiment_id,run_id,now,*change))
        log_challenger_forecasts(c,experiment_id,market_date,now,medium)
        alert_changes=[x for x in changes if x[-1]]
        try: email_status=send_alerts(alert_changes) if alert_changes else "NO MATERIAL CHANGE — NO EMAIL"
        except Exception as exc: email_status=f"FAILED — {type(exc).__name__}: {exc}"
        if alert_changes: c.execute("UPDATE paper_status_changes SET email_status=? WHERE run_id=? AND alert_required=1",(email_status,run_id))
        c.execute("UPDATE prospective_runs SET status='SUCCESS',completed_at=? WHERE id=?",(datetime.now(timezone.utc).isoformat(),run_id)); write_performance(c,experiment_id); write_reviews(c,experiment_id,market_date)
    return {"status":"SUCCESS","run_id":run_id,"experiment_id":experiment_id,"market_date":str(market_date.date()),"signals_inserted":len(inserted),"meaningful_changes":len([x for x in changes if x[-1]]),"email_status":email_status,"portfolio_basis":"5% of reconciled value" if portfolio_value else "normalized €1,000 only — no reconciled portfolio value"}


def record_failure(message):
    """Best-effort provider-failure record used by the common scheduled runner."""
    marker=ROOT/"logs/last_ingestion_failure.txt"; marker.parent.mkdir(exist_ok=True); marker.write_text(datetime.now(timezone.utc).isoformat()+" "+message+"\n")
