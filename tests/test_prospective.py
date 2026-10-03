import sqlite3

import pandas as pd
import pytest

from src.prospective import quality_check
from src.storage import Database


def test_paper_signal_provenance_is_immutable_but_outcome_can_append(tmp_path):
    db=Database(tmp_path/"paper.db"); db.initialize()
    with db.connect() as c:
        c.execute("INSERT INTO prospective_experiments(experiment_id,started_at,frozen_21d_hash,frozen_medium_hash,risk_policy_hash,configuration_hash,data_snapshot_timestamp) VALUES('e','now','a','b','c','d','now')")
        run=c.execute("INSERT INTO prospective_runs(experiment_id,market_date,started_at,status) VALUES('e','2026-09-01','now','SUCCESS')").lastrowid
        c.execute("INSERT INTO paper_signals(experiment_id,run_id,observed_date,observed_at,asset,horizon_days,forecast,forecast_rank,top_cohort,active_risk_flags,policy_2_pass,recommendation,benchmark,model_hash,configuration_hash,data_snapshot_timestamp,feature_values_json,decision_reason) VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",('e',run,'2026-09-01','now','SMH',63,.02,.9,1,'none',1,'PAPER BUY','SPY','m','c','now','{}','frozen rule'))
        with pytest.raises(sqlite3.IntegrityError): c.execute("UPDATE paper_signals SET forecast=.99 WHERE asset='SMH'")
        c.execute("UPDATE paper_signals SET realized_return=.03 WHERE asset='SMH'")
        assert c.execute("SELECT realized_return FROM paper_signals").fetchone()[0]==.03


def test_duplicate_market_day_is_database_idempotent(tmp_path):
    db=Database(tmp_path/"paper.db"); db.initialize()
    with db.connect() as c:
        c.execute("INSERT INTO prospective_runs(experiment_id,market_date,started_at,status) VALUES('e','2026-09-01','now','SUCCESS')")
        with pytest.raises(sqlite3.IntegrityError): c.execute("INSERT INTO prospective_runs(experiment_id,market_date,started_at,status) VALUES('e','2026-09-01','later','SUCCESS')")


def test_stale_required_data_suppresses_signals(tmp_path):
    dates=pd.to_datetime(['2026-08-20'])
    prices=pd.DataFrame({'ticker':['SPY'],'date':dates,'close':[100.],'source':['real']})
    fx=pd.DataFrame({'date':dates,'close':[1.1]}); macro=pd.DataFrame({'date':dates,'vix':[20.]})
    result=quality_check(prices,fx,macro,pd.Timestamp('2026-09-01'))
    assert result['hard']


def test_backtest_and_prospective_tables_are_separate(tmp_path):
    db=Database(tmp_path/"paper.db"); db.initialize()
    with db.connect() as c:
        tables={r[0] for r in c.execute("SELECT name FROM sqlite_master WHERE type='table'")}
    assert 'model_results' in tables and 'paper_signals' in tables and 'prospective_runs' in tables
