import sqlite3
import json

import pytest
import numpy as np
import pandas as pd

from src.portfolio.reconciliation_service import reconcile_confirmed_snapshot,sleeve_values
from src.prospective import meaningful_email_transition,send_test_email
from src.storage import Database
from src.features.medium import MEDIUM_FEATURES


def test_confirmed_reconciliation_and_sleeves(tmp_path):
    db=Database(tmp_path/"state.db"); db.initialize(); snapshot={"snapshot_date":"2026-09-01","cash_eur":1000.,"total_value_eur":20000.,"source_file":"statement.pdf","warnings":[],"holdings":[{"security_name":"ETF","isin":"DE000ABC1234","ticker":None,"quantity":10.,"displayed_price":1900.,"market_value_eur":19000.,"asset_type":"security","confidence":1.}]}
    with pytest.raises(ValueError): reconcile_confirmed_snapshot(snapshot,False,db)
    reconcile_confirmed_snapshot(snapshot,True,db)
    with db.connect() as c: assert tuple(c.execute("SELECT snapshot_date,total_value_eur FROM portfolio_snapshots").fetchone())==("2026-09-01",20000.)
    assert sleeve_values(20000)=={"5%":1000.,"7.5%":1500.,"10%":2000.}


def test_failed_parse_does_not_overwrite_confirmed_snapshot(tmp_path):
    db=Database(tmp_path/"state.db"); db.initialize()
    good={"snapshot_date":"2026-09-01","cash_eur":0.,"total_value_eur":1000.,"source_file":"good.pdf","warnings":[],"holdings":[{"security_name":"ETF","isin":"DE000ABC1234","market_value_eur":1000.}]}
    reconcile_confirmed_snapshot(good,True,db)
    failed={"snapshot_date":"2026-09-02","cash_eur":0.,"total_value_eur":None,"source_file":"failed.pdf","warnings":[],"holdings":[]}
    with pytest.raises(ValueError): reconcile_confirmed_snapshot(failed,True,db)
    with db.connect() as c:
        assert c.execute("SELECT COUNT(*) FROM portfolio_snapshots").fetchone()[0] == 1
        assert c.execute("SELECT source_file FROM portfolio_snapshots").fetchone()[0] == "good.pdf"


def test_email_transition_allowlist():
    assert meaningful_email_transition("WATCH","PAPER BUY")
    assert meaningful_email_transition("PAPER HOLD","MODEL INTERESTING — RISK VETO")
    assert meaningful_email_transition("PAPER HOLD","PAPER EXIT")
    assert not meaningful_email_transition("NO EDGE","NO EDGE")
    assert not meaningful_email_transition("NO EDGE","WATCH")
    assert not meaningful_email_transition("PAPER BUY","PAPER HOLD")


def test_authenticated_test_email_path(monkeypatch):
    sent=[]
    class SMTP:
        def __init__(self,host,port,timeout): sent.append((host,port,timeout))
        def __enter__(self): return self
        def __exit__(self,*args): pass
        def starttls(self): sent.append("tls")
        def login(self,user,password): sent.append((user,password))
        def send_message(self,msg): sent.append(msg["Subject"])
    monkeypatch.setattr("src.prospective.smtplib.SMTP",SMTP)
    for key,value in {"SMTP_HOST":"smtp.gmail.com","SMTP_PORT":"587","SMTP_FROM":"owner@example.com","ALERT_TO":"owner@example.com","SMTP_USERNAME":"owner@example.com","SMTP_PASSWORD":"secret"}.items(): monkeypatch.setenv(key,value)
    assert send_test_email()=="TEST EMAIL SENT"
    assert "tls" in sent and "Quiet Capital: email configuration test" in sent


def test_model_versions_are_immutable_and_challenger_does_not_replace_incumbent(tmp_path):
    db=Database(tmp_path/"state.db"); db.initialize(); base=("medium-ridge-v1","medium_ridge","INCUMBENT","2026-09-01","2026-08-31","[]","frozen procedure","cfg","risk","INITIAL","{}",None,"NORMAL")
    challenger=("medium-ridge-v2","medium_ridge","CHALLENGER","2026-12-02","2026-12-01","[]","frozen procedure","cfg","risk","SCHEDULED_RETRAIN","{}","medium-ridge-v1","NORMAL")
    with db.connect() as c:
        c.execute("INSERT INTO model_versions(version,model_family,role,created_at,training_end_date,feature_specification_json,training_procedure,configuration_hash,risk_policy_hash,creation_reason,artifact_manifest_json,parent_version,validity) VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?)",base)
        c.execute("INSERT INTO model_versions(version,model_family,role,created_at,training_end_date,feature_specification_json,training_procedure,configuration_hash,risk_policy_hash,creation_reason,artifact_manifest_json,parent_version,validity) VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?)",challenger)
        assert c.execute("SELECT version FROM model_versions WHERE role='INCUMBENT'").fetchone()[0]=="medium-ridge-v1"
        with pytest.raises(sqlite3.IntegrityError): c.execute("UPDATE model_versions SET configuration_hash='changed' WHERE version='medium-ridge-v1'")


def test_quarterly_retrain_creates_separate_challenger_artifacts(tmp_path,monkeypatch):
    import src.model_governance as governance
    for directory in ("state/models","reports","config","data/processed"): (tmp_path/directory).mkdir(parents=True,exist_ok=True)
    manifest={"created_at":"2026-09-02T00:00:00+00:00","frozen_21d_research_hash":"short","frozen_medium_research_hash":"medium","models":{"21":{"path":"state/models/old21","sha256":"a","training_rows":10,"training_data_end":"2026-08-01","features":[]},"63":{"path":"state/models/old63","sha256":"b","training_rows":10,"training_data_end":"2026-08-01","features":MEDIUM_FEATURES},"126":{"path":"state/models/old126","sha256":"c","training_rows":10,"training_data_end":"2026-08-01","features":MEDIUM_FEATURES}}}
    (tmp_path/"state/models/prospective_manifest.json").write_text(json.dumps(manifest)); (tmp_path/"reports/medium_risk_metadata.json").write_text(json.dumps({"risk_specification_hash":"risk"})); (tmp_path/"config/default.yaml").write_text("test: true\n")
    dates=pd.bdate_range("2020-01-01",periods=300); panel=pd.DataFrame({"date":dates,"ticker":"SMH"})
    for f in MEDIUM_FEATURES: panel[f]=np.linspace(0,1,len(panel))
    for h in (63,126): panel[f"forward_excess_{h}d"]=np.linspace(-.1,.1,len(panel)); panel[f"target_end_{h}d"]=panel.date.shift(-h)
    panel.to_parquet(tmp_path/"data/processed/prospective_medium_panel.parquet",index=False)
    monkeypatch.setattr(governance,"ROOT",tmp_path); db=Database(tmp_path/"state/test.db")
    version=governance.create_challenger("SCHEDULED_RETRAIN",db)
    with db.connect() as c:
        assert c.execute("SELECT version FROM model_versions WHERE role='INCUMBENT' AND model_family='medium_ridge'").fetchone()[0]=="medium-ridge-v1"
        assert c.execute("SELECT creation_reason FROM model_versions WHERE version=?",(version,)).fetchone()[0]=="SCHEDULED_RETRAIN"
    assert (tmp_path/f"state/models/{version}_63d.joblib").exists()
