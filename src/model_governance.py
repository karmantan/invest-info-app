from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone

import joblib
import pandas as pd

from src.config import ROOT
from src.features.medium import MEDIUM_FEATURES
from src.models.real_experiment import _model
from src.storage import Database


VALID_REASONS={"INITIAL","SCHEDULED_RETRAIN","ANNUAL_REVIEW","STRUCTURAL_BREAK_REVIEW"}
TRAINING_PROCEDURE="Frozen ridge(alpha=10) pipeline with median imputation and StandardScaler; all point-in-time rows whose forward target ended before the creation data date. No feature, target, threshold, or risk-rule changes."


def _sha(path): return hashlib.sha256(path.read_bytes()).hexdigest()


def initialize_versions(db: Database | None=None):
    db=db or Database(); db.initialize(); manifest=json.loads((ROOT/"state/models/prospective_manifest.json").read_text()); risk=json.loads((ROOT/"reports/medium_risk_metadata.json").read_text()); config_hash=hashlib.sha256((ROOT/"config/default.yaml").read_bytes()).hexdigest(); created=manifest["created_at"]
    definitions=[("short-ridge-v1","short_ridge","21",manifest["frozen_21d_research_hash"]),("medium-ridge-v1","medium_ridge","126",manifest["frozen_medium_research_hash"])]
    with db.connect() as c:
        for version,family,key,parent in definitions:
            m=manifest["models"][key]; features=m["features"]
            artifacts={k:v for k,v in manifest["models"].items() if (family=="short_ridge" and k=="21") or (family=="medium_ridge" and k in ("63","126"))}
            c.execute("INSERT OR IGNORE INTO model_versions(version,model_family,role,created_at,training_end_date,feature_specification_json,training_procedure,configuration_hash,risk_policy_hash,creation_reason,artifact_manifest_json,parent_version,validity) VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?)",(version,family,"INCUMBENT",created,m["training_data_end"],json.dumps(features,sort_keys=True),TRAINING_PROCEDURE,config_hash,risk["risk_specification_hash"],"INITIAL",json.dumps(artifacts,sort_keys=True),parent,"NORMAL"))
        # One-time attachment of already-created first-run records, then restore the immutable trigger.
        c.execute("DROP TRIGGER IF EXISTS paper_signals_immutable")
        c.execute("UPDATE paper_signals SET model_version=CASE WHEN horizon_days=21 THEN 'short-ridge-v1' ELSE 'medium-ridge-v1' END WHERE model_version IS NULL")
    db.initialize()


def status(db: Database | None=None):
    db=db or Database(); initialize_versions(db)
    with db.connect() as c:
        rows=[dict(r) for r in c.execute("SELECT * FROM model_versions ORDER BY created_at,id")]
    medium=next(r for r in rows if r["model_family"]=="medium_ridge" and r["role"]=="INCUMBENT")
    challenger=next((r for r in rows if r["model_family"]=="medium_ridge" and r["role"]=="CHALLENGER"),None)
    created=pd.Timestamp(medium["created_at"]); next_retrain=created+pd.DateOffset(months=3); annual=created+pd.DateOffset(years=1)
    return {"incumbent":medium["version"],"last_trained":medium["training_end_date"],"next_scheduled_retrain":next_retrain.date().isoformat(),"next_annual_review":annual.date().isoformat(),"model_validity":medium["validity"],"challenger":challenger["version"] if challenger else None,"paper_trading":"ENABLED","real_capital":"DISABLED"}


def create_challenger(reason="SCHEDULED_RETRAIN", db: Database | None=None):
    if reason not in VALID_REASONS-{"INITIAL"}: raise ValueError("Invalid challenger creation reason")
    db=db or Database(); initialize_versions(db); panel=pd.read_parquet(ROOT/"data/processed/prospective_medium_panel.parquet"); panel.date=pd.to_datetime(panel.date); latest=panel.date.max()
    with db.connect() as c:
        incumbent=dict(c.execute("SELECT * FROM model_versions WHERE model_family='medium_ridge' AND role='INCUMBENT'").fetchone())
        existing=c.execute("SELECT version FROM model_versions WHERE model_family='medium_ridge' AND role='CHALLENGER'").fetchone()
        if existing: return existing[0]
        count=c.execute("SELECT COUNT(*) FROM model_versions WHERE model_family='medium_ridge'").fetchone()[0]; version=f"medium-ridge-v{count+1}"
    artifacts={}
    out=ROOT/"state/models"; out.mkdir(parents=True,exist_ok=True)
    for horizon in (63,126):
        target=f"forward_excess_{horizon}d"; end=f"target_end_{horizon}d"; train=panel[(panel[end]<latest)&panel[target].notna()]; model=_model("ridge"); model.fit(train[MEDIUM_FEATURES],train[target]); path=out/f"{version}_{horizon}d.joblib"; joblib.dump(model,path); artifacts[str(horizon)]={"path":str(path.relative_to(ROOT)),"sha256":_sha(path),"training_rows":len(train)}
    training_end=max(str(pd.to_datetime(panel.loc[panel[f"target_end_{h}d"]<latest,f"target_end_{h}d"]).max()) for h in (63,126))
    with db.connect() as c:
        c.execute("INSERT INTO model_versions(version,model_family,role,created_at,training_end_date,feature_specification_json,training_procedure,configuration_hash,risk_policy_hash,creation_reason,artifact_manifest_json,parent_version,validity) VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?)",(version,"medium_ridge","CHALLENGER",datetime.now(timezone.utc).isoformat(),training_end,json.dumps(MEDIUM_FEATURES,sort_keys=True),TRAINING_PROCEDURE,incumbent["configuration_hash"],incumbent["risk_policy_hash"],reason,json.dumps(artifacts,sort_keys=True),incumbent["version"],"NORMAL"))
    return version


def maybe_schedule_challenger(now=None,db=None):
    state=status(db); today=pd.Timestamp(now or datetime.now(timezone.utc)).date()
    if today>=pd.Timestamp(state["next_scheduled_retrain"]).date() and not state["challenger"]: return create_challenger("SCHEDULED_RETRAIN",db)
    return state["challenger"]


def require_structural_review(create=False,db=None):
    db=db or Database(); initialize_versions(db)
    with db.connect() as c: c.execute("UPDATE model_versions SET validity='REVIEW REQUIRED' WHERE model_family='medium_ridge' AND role='INCUMBENT'")
    return create_challenger("STRUCTURAL_BREAK_REVIEW",db) if create else None


def log_challenger_forecasts(c,experiment_id,market_date,observed_at,panel):
    rows=c.execute("SELECT * FROM model_versions WHERE model_family='medium_ridge' AND role='CHALLENGER'").fetchall()
    if not rows: return 0
    from src.reporting.medium_risk_audit import FLAG_COLUMNS,add_point_in_time_flags
    context=add_point_in_time_flags(panel); latest=context.sort_values("date").groupby("ticker").tail(1); inserted=0
    for version_row in rows:
        artifacts=json.loads(version_row["artifact_manifest_json"])
        for horizon in (63,126):
            model=joblib.load(ROOT/artifacts[str(horizon)]["path"]); x=latest.copy(); x["forecast"]=model.predict(x[MEDIUM_FEATURES]); x["rank"]=x.forecast.rank(pct=True,method="first")
            for _,r in x.iterrows():
                flags=", ".join(f for f in FLAG_COLUMNS if bool(r[f])) or "none"; policy=not (r.broad_market_downtrend or r.extreme_asset_drawdown)
                inserted+=c.execute("INSERT OR IGNORE INTO prospective_model_forecasts(experiment_id,model_version,observed_date,observed_at,asset,horizon_days,forecast,forecast_rank,top_cohort,active_risk_flags,policy_2_pass,feature_values_json,model_hash) VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?)",(experiment_id,version_row["version"],str(market_date.date()),observed_at,r.ticker,horizon,float(r.forecast),float(r["rank"]),int(r["rank"]>.66),flags,int(policy),json.dumps({f:None if pd.isna(r[f]) else float(r[f]) for f in MEDIUM_FEATURES},sort_keys=True),artifacts[str(horizon)]["sha256"])).rowcount
    return inserted
