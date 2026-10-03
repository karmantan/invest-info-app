import json
import sqlite3

import numpy as np
import pandas as pd
import pytest

from src.discovery import (DISCOVERY_EXPERIMENT_ID, FROZEN_VALIDATION_EXPERIMENT_ID, VALIDATION_UNIVERSE,
    build_panel, catalogue, group_opportunities, promote, screen, start_simulated_tracking,
    stop_watching, watch_candidate, watchlist)
from src.storage import Database


def synthetic_prices():
    dates=pd.bdate_range("2019-01-01",periods=900); frames=[]
    for ticker,drift in (("SPY",.0002),("URTH",.00015),("AAA",.0004),("BNDX",.00005),("EURUSD=X",0.0)):
        close=100*np.exp(np.cumsum(np.full(len(dates),drift)+np.sin(np.arange(len(dates))/30)*.0001))
        frames.append(pd.DataFrame({"date":dates,"ticker":ticker,"close":close,"volume":2_000_000,"source":"test","currency":"USD","adjustment":"adjusted close"}))
    return pd.concat(frames,ignore_index=True)


def test_catalogue_is_broad_separate_and_nonleveraged():
    universe=catalogue()
    assert 100 <= len(universe) <= 300
    assert not VALIDATION_UNIVERSE.intersection(universe.ticker)
    assert {"Equity","Commodity","Fixed Income"}.issubset(set(universe.asset_class))
    assert not universe.structure.str.contains("leveraged|inverse|ETN",case=False,regex=True).any()


def test_screen_has_explicit_eligibility_and_fixed_income_logic():
    meta=pd.DataFrame([
        {"ticker":"AAA","friendly_name":"Example Equity","region":"United States","sector_theme":"Broad Market","asset_class":"Equity","currency":"USD","benchmark_category":"US Equity","launch_date":pd.Timestamp("2010-01-01"),"research_proxy_status":"test","structure":"ordinary ETF"},
        {"ticker":"BNDX","friendly_name":"Example Bonds","region":"Global","sector_theme":"Bonds","asset_class":"Fixed Income","currency":"USD","benchmark_category":"Bonds","launch_date":pd.Timestamp("2010-01-01"),"research_proxy_status":"test","structure":"ordinary bond ETF"},
    ])
    panel=build_panel(synthetic_prices())
    result=screen(panel,meta,top_n=2)
    assert set(result.eligibility)=={"ELIGIBLE"}
    bond=result[result.ticker=="BNDX"].iloc[0]
    assert "rates_state" in bond.features
    assert result.discovery_score.between(0,100).all()


def test_discovery_tables_are_separate_and_promotion_is_prospective(tmp_path):
    db=Database(tmp_path/"test.db"); db.initialize()
    with db.connect() as c:
        tables={r[0] for r in c.execute("SELECT name FROM sqlite_master WHERE type='table'")}
        assert {"discovery_runs","discovery_results","discovery_promotions"}.issubset(tables)
        run_id=c.execute("INSERT INTO discovery_runs(discovery_experiment_id,market_date,started_at,status,universe_size,eligible_count,excluded_count,shortlist_count) VALUES(?,?,?,?,?,?,?,?)",(DISCOVERY_EXPERIMENT_ID,"2026-09-01","2026-09-01T12:00:00Z","SUCCESS",153,100,53,1)).lastrowid
        c.execute("INSERT INTO discovery_results(run_id,discovery_experiment_id,observed_date,ticker,eligibility,discovery_score,rank,shortlisted,status,risk_level,risk_flags_json,data_quality_json,evidence_63d_json,evidence_126d_json,explanation,benchmark,feature_snapshot_json) VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",(run_id,DISCOVERY_EXPERIMENT_ID,"2026-09-01","AAA","ELIGIBLE",75,1,1,"Research candidate","Lower","[]","{}","{}","{}","Why","SPY","{}"))
    promotion_id=promote("AAA","Explicit paper review",1000,126,db)
    with db.connect() as c:
        p=c.execute("SELECT * FROM discovery_promotions WHERE id=?",(promotion_id,)).fetchone()
        assert p["discovery_date"]=="2026-09-01"
        assert p["promotion_date"]>p["discovery_date"]
        assert c.execute("SELECT COUNT(*) FROM paper_positions").fetchone()[0]==0
        assert c.execute("SELECT COUNT(*) FROM paper_signals").fetchone()[0]==0


def test_promotion_rejects_unshortlisted_asset(tmp_path):
    db=Database(tmp_path/"test.db"); db.initialize()
    with pytest.raises(ValueError): promote("NOPE","reason",1000,63,db)


def _candidate(db, ticker="VHT", rank=1, shortlisted=1, market_date="2026-09-01"):
    with db.connect() as c:
        run_id=c.execute("INSERT INTO discovery_runs(discovery_experiment_id,market_date,started_at,status,universe_size,eligible_count,excluded_count,shortlist_count) VALUES(?,?,?,?,?,?,?,?)",
            (DISCOVERY_EXPERIMENT_ID,market_date,market_date+"T12:00:00Z","SUCCESS",153,100,53,1)).lastrowid
        c.execute("""INSERT INTO discovery_results(run_id,discovery_experiment_id,observed_date,ticker,eligibility,
          discovery_score,rank,shortlisted,status,risk_level,risk_flags_json,data_quality_json,evidence_63d_json,
          evidence_126d_json,explanation,benchmark,feature_snapshot_json,overlap_warning)
          VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
          (run_id,DISCOVERY_EXPERIMENT_ID,market_date,ticker,"ELIGIBLE",75,rank,shortlisted,
           "Interesting medium-term setup","Lower","[]",json.dumps({"median_dollar_volume_63d":20_000_000}),
           json.dumps({"sample_size":20,"median_excess":.02,"p10":-.08,"p5":-.12}),
           json.dumps({"sample_size":20,"median_excess":.03,"p10":-.10,"p5":-.15}),"Medium-term evidence is positive.","SPY",
           json.dumps({"current_price_eur":100,"price_source":"Test adjusted close","drawdown":-.04}),"Approximate category overlap: High"))
    return run_id


def test_opportunity_group_retains_related_etfs_for_compare():
    rows=pd.DataFrame([
        {"ticker":"VHT","friendly_name":"US Healthcare","benchmark_category":"US Healthcare","rank":1},
        {"ticker":"XLV","friendly_name":"US Healthcare","benchmark_category":"US Healthcare","rank":2},
        {"ticker":"IYH","friendly_name":"US Healthcare","benchmark_category":"US Healthcare","rank":3},
    ])
    groups=group_opportunities(rows)
    assert len(groups)==1
    assert groups[0]["leading"]["ticker"]=="VHT"
    assert [x["ticker"] for x in groups[0]["candidates"]]==["VHT","XLV","IYH"]


def test_watch_persists_restart_and_stop_preserves_audit_history(tmp_path):
    path=tmp_path/"test.db"; db=Database(path); db.initialize(); _candidate(db)
    watch_candidate("VHT",db,now="2026-09-02T09:00:00+00:00")
    restarted_db=Database(path); restarted_db.initialize()
    assert list(watchlist(restarted_db).ticker)==["VHT"]
    stop_watching("VHT",restarted_db,now="2026-09-02T10:00:00+00:00")
    with restarted_db.connect() as c:
        assert c.execute("SELECT active FROM discovery_watchlist WHERE ticker='VHT'").fetchone()[0]==0
        assert [r[0] for r in c.execute("SELECT event_type FROM discovery_watch_history ORDER BY id")]==["STARTED","STOPPED"]


def test_watched_etf_remains_visible_after_leaving_shortlist(tmp_path):
    db=Database(tmp_path/"test.db"); db.initialize(); _candidate(db); watch_candidate("VHT",db)
    _candidate(db,shortlisted=0,market_date="2026-09-02")
    watched=watchlist(db)
    assert watched.iloc[0].ticker=="VHT"
    assert watched.iloc[0].current_shortlisted==0


def test_simulated_tracking_is_prospective_isolated_and_has_two_horizons(tmp_path):
    db=Database(tmp_path/"test.db"); db.initialize(); _candidate(db)
    with db.connect() as c:
        c.execute("INSERT INTO prospective_experiments(experiment_id,started_at,frozen_21d_hash,frozen_medium_hash,risk_policy_hash,configuration_hash,data_snapshot_timestamp,mode,real_capital_enabled) VALUES(?,?,?,?,?,?,?,?,?)",
                  (FROZEN_VALIDATION_EXPERIMENT_ID,"2026-01-01","a","b","c","d","e","paper",0))
    position_id=start_simulated_tracking("VHT",db,now="2026-09-02T10:00:00+00:00")
    with db.connect() as c:
        p=c.execute("SELECT * FROM discovery_positions WHERE id=?",(position_id,)).fetchone()
        assert p["tracking_timestamp"]=="2026-09-02T10:00:00+00:00" and p["market_date"]=="2026-09-01"
        assert p["normalized_notional_eur"]==1000 and p["discovery_experiment_id"]==DISCOVERY_EXPERIMENT_ID
        assert c.execute("SELECT COUNT(*) FROM discovery_position_horizons WHERE position_id=?",(position_id,)).fetchone()[0]==2
        assert c.execute("SELECT COUNT(*) FROM paper_positions").fetchone()[0]==0
        assert c.execute("SELECT COUNT(*) FROM trades").fetchone()[0]==0
        assert c.execute("SELECT experiment_id FROM prospective_experiments").fetchone()[0]==FROZEN_VALIDATION_EXPERIMENT_ID
    with pytest.raises(ValueError): start_simulated_tracking("VHT",db,now="2026-09-02T11:00:00+00:00")


def test_simulated_tracking_entry_cannot_be_backdated(tmp_path):
    db=Database(tmp_path/"test.db"); db.initialize(); _candidate(db)
    with pytest.raises(ValueError,match="backdated"):
        start_simulated_tracking("VHT",db,now="2026-08-31T10:00:00+00:00")
