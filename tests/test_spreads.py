from datetime import datetime,timedelta,timezone

import pytest

from src.spreads import collect,validate_quote
from src.storage import Database

UTC=timezone.utc
CFG={"provider":"Yahoo Finance via yfinance","source_label":"Indicative Yahoo/Xetra research data","cooldown_seconds":3600,
     "suspicious_spread_bps":100,"mappings":[{"isin":"IE00BYXG2H39","yahoo_ticker":"2B70.DE","exchange":"Xetra","exchange_aliases":["GER"],"currency":"EUR"}]}
MAPPING=CFG["mappings"][0]

def quote(**updates):
    raw={"ticker":"2B70.DE","exchange":"XETRA","exchange_code":"GER","currency":"EUR","bid":10,"ask":10.02,
         "bid_size":50,"ask_size":75,"regular_market_time":1,"market_state":"REGULAR","provider_fields":[]}
    raw.update(updates); return raw

def test_validation_units_and_last_trade_time_is_not_quote_time():
    now=datetime(2026,9,4,9,tzinfo=UTC); attempt,obs=validate_quote(quote(),MAPPING,now,now+timedelta(seconds=1),CFG)
    assert attempt["status"]=="OBSERVED"; assert obs["midpoint"]==pytest.approx(10.01)
    assert obs["spread_currency"]==pytest.approx(.02); assert obs["spread_percent"]==pytest.approx(.02/10.01*100)
    assert obs["spread_bps"]==pytest.approx(.02/10.01*10000)
    assert obs["quote_timestamp_utc"] is None and "UNVERIFIED" in obs["quote_freshness"]

@pytest.mark.parametrize("updates,reason",[
 ({"bid":None},"missing"),({"bid":float("nan")},"nonfinite"),({"bid":0},"positive"),({"bid":11,"ask":10},"Crossed"),
 ({"currency":"USD"},"currency"),({"exchange":"NYSE","exchange_code":"NYQ"},"exchange")])
def test_invalid_quotes_are_rejected_without_a_fake_zero_spread(updates,reason):
    attempt,obs=validate_quote(quote(**updates),MAPPING,datetime.now(UTC),datetime.now(UTC),CFG)
    assert attempt["status"]=="REJECTED" and reason.lower() in attempt["reason"].lower(); assert obs is None

def test_zero_and_wide_quotes_are_flagged():
    _,zero=validate_quote(quote(ask=10),MAPPING,datetime.now(UTC),datetime.now(UTC),CFG)
    _,wide=validate_quote(quote(ask=11),MAPPING,datetime.now(UTC),datetime.now(UTC),CFG)
    assert "zero-width" in zero["reason"] and zero["spread_currency"]==0
    assert "suspiciously wide" in wide["reason"]

def test_delayed_quote_and_unavailable_zero_sizes_are_explicit():
    _,obs=validate_quote(quote(exchange_data_delayed_by=15,bid_size=0,ask_size=0),MAPPING,datetime.now(UTC),datetime.now(UTC),CFG)
    assert "delayed by 15 minutes" in obs["reason"]
    assert obs["bid_size"] is None and obs["ask_size"] is None

def test_append_only_persistence_and_future_availability_exclusion(tmp_path):
    db=Database(tmp_path/"db.sqlite"); db.initialize(); now=datetime(2026,9,4,9,tzinfo=UTC)
    attempt,obs=validate_quote(quote(),MAPPING,now,now,CFG); db.record_spread_attempt(attempt,obs)
    with db.connect() as c:
        with pytest.raises(Exception): c.execute("UPDATE spread_observations SET spread_bps=0")
        c.execute("INSERT INTO portfolio_snapshots(snapshot_date) VALUES('2026-09-04')"); sid=c.execute("SELECT MAX(id) FROM portfolio_snapshots").fetchone()[0]
        c.execute("INSERT INTO holdings(snapshot_id,security_name,isin) VALUES(?,?,?)",(sid,"ETF",MAPPING["isin"]))
    assert db.latest_spreads((now-timedelta(seconds=1)).isoformat())[0]["observation_collected_at_utc"] is None
    assert db.latest_spreads(now.isoformat())[0]["spread_bps"]==pytest.approx(obs["spread_bps"])

def test_rate_limit_creates_failure_stops_requests_and_persists_hour_cooldown(tmp_path):
    class Limited:
        calls=0
        def fetch(self,ticker): self.calls+=1; raise RuntimeError("HTTP 429 Too Many Requests")
    db=Database(tmp_path/"db.sqlite"); cfg={**CFG,"mappings":[MAPPING,{**MAPPING,"isin":"OTHER","yahoo_ticker":"OTHER.DE"}]}; adapter=Limited(); now=datetime(2026,9,4,9,tzinfo=UTC)
    result=collect(db,cfg,adapter,now,force=True); assert adapter.calls==1 and result["results"][0]["status"]=="RATE_LIMITED"
    assert collect(db,cfg,adapter,now+timedelta(minutes=59),force=True)["status"]=="COOLDOWN" and adapter.calls==1
    with db.connect() as c: assert c.execute("SELECT COUNT(*) FROM spread_attempts WHERE status='RATE_LIMITED'").fetchone()[0]==1

def test_schema_is_additive_and_idempotent(tmp_path):
    db=Database(tmp_path/"db.sqlite"); db.initialize(); db.initialize()
    with db.connect() as c:
        assert c.execute("SELECT COUNT(*) FROM sqlite_master WHERE type='table' AND name LIKE 'spread_%'").fetchone()[0]==3

def test_spread_scheduler_is_isolated_from_prospective_scheduler():
    from src.config import ROOT
    spread=(ROOT/"scripts/com.quietcapital.spreads.plist.template").read_text()
    prospective=(ROOT/"scripts/com.quietcapital.prospective.plist.template").read_text()
    assert "com.quietcapital.spreads" in spread and "StartInterval" in spread and "900" in spread
    assert "com.quietcapital.prospective" in prospective and "<integer>6</integer>" in prospective
    assert "spreads" not in prospective
