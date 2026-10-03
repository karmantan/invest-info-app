from datetime import datetime, timezone

from src.app_status import data_status, runtime_identity
from src.storage import Database


def test_runtime_identity_uses_working_tree_and_real_paths():
    identity = runtime_identity(datetime(2026, 9, 4, tzinfo=timezone.utc))
    assert identity["build"].startswith("working-tree ")
    assert identity["entry_point"].endswith("dashboard/app.py")
    assert identity["database"].endswith("state/invest.db")


def test_data_status_counts_current_holdings_and_latest_collection(tmp_path):
    db = Database(tmp_path / "invest.db"); db.initialize()
    with db.connect() as c:
        sid=c.execute("INSERT INTO portfolio_snapshots(snapshot_date) VALUES('2026-09-04')").lastrowid
        c.execute("INSERT INTO holdings(snapshot_id,security_name,isin) VALUES(?,?,?)",(sid,"Mapped","ONE"))
        c.execute("INSERT INTO holdings(snapshot_id,security_name,isin) VALUES(?,?,?)",(sid,"Unmapped","TWO"))
    attempt={"requested_at_utc":"2026-09-04T08:00:00+00:00","collected_at_utc":"2026-09-04T08:00:01+00:00","available_at_utc":"2026-09-04T08:00:01+00:00","isin":"ONE","ticker":"ONE.DE","expected_exchange":"Xetra","expected_currency":"EUR","provider":"Yahoo","status":"OBSERVED","quality":"USABLE_WITH_FLAGS","reason":"delayed","http_status":None,"provider_metadata_json":"{}"}
    observation={**attempt,"exchange":"Xetra","currency":"EUR","bid":1.0,"ask":1.01,"bid_size":None,"ask_size":None,"midpoint":1.005,"spread_currency":.01,"spread_percent":.995,"spread_bps":99.5,"quote_timestamp_utc":None,"quote_freshness":"unverified","source_label":"test"}
    db.record_spread_attempt(attempt,observation)
    result=data_status(db,{"timeout_seconds":90,"mappings":[{"isin":"ONE","verified":True}]},datetime(2026,9,4,9,tzinfo=timezone.utc))
    assert (result["verified_holdings"],result["total_holdings"],result["usable_latest"]) == (1,2,1)
