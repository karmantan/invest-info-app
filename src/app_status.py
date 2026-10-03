from __future__ import annotations

import hashlib
import os
import json
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

from src.config import ROOT


BUILD_INPUTS = (
    "dashboard/app.py", "dashboard/ui.py", "src/app_status.py", "src/config.py",
    "src/storage.py", "src/spreads.py", "config/default.yaml", "config/spreads.yaml",
)


def runtime_identity(started_at: datetime | None = None) -> dict:
    """Identify the actual working-tree bytes used by this dashboard process."""
    startup = started_at or (datetime.fromisoformat(os.environ["INVEST_DASHBOARD_STARTED_AT"])
                             if os.getenv("INVEST_DASHBOARD_STARTED_AT") else datetime.now(timezone.utc))
    digest = hashlib.sha256()
    for relative in BUILD_INPUTS:
        path = ROOT / relative
        digest.update(relative.encode())
        digest.update(path.read_bytes())
    return {
        "build": f"working-tree {digest.hexdigest()[:12]}",
        "started_at": startup.isoformat(),
        "project_dir": str(ROOT),
        "python": sys.executable,
        "entry_point": str(ROOT / "dashboard/app.py"),
        "database": str(ROOT / "state/invest.db"),
    }


def data_status(db, spread_cfg: dict, now: datetime | None = None) -> dict:
    now = now or datetime.now(timezone.utc)
    mapped = {row["isin"] for row in spread_cfg.get("mappings", []) if row.get("verified")}
    batch_gap = timedelta(seconds=float(spread_cfg.get("timeout_seconds", 90)) + 30)
    with db.connect() as connection:
        update = connection.execute(
            "SELECT completed_at,market_date,price_latest_date,macro_latest_date,fx_latest_date "
            "FROM prospective_runs WHERE status='SUCCESS' ORDER BY completed_at DESC,id DESC LIMIT 1"
        ).fetchone()
        holdings = connection.execute(
            "SELECT isin FROM holdings WHERE snapshot_id=(SELECT MAX(id) FROM portfolio_snapshots)"
        ).fetchall()
        attempts = connection.execute(
            "SELECT id,requested_at_utc,collected_at_utc,isin,status,quality,reason "
            "FROM spread_attempts ORDER BY requested_at_utc DESC,id DESC"
        ).fetchall()
        cooldown = connection.execute(
            "SELECT value FROM spread_collector_state WHERE key='cooldown_until_utc'"
        ).fetchone()

        latest_batch = []
        if attempts:
            newest = datetime.fromisoformat(attempts[0]["requested_at_utc"])
            for attempt in attempts:
                requested = datetime.fromisoformat(attempt["requested_at_utc"])
                if newest - requested <= batch_gap:
                    latest_batch.append(attempt)
                else:
                    break
        batch_ids = [row["id"] for row in latest_batch]
        usable_isins: set[str] = set()
        if batch_ids:
            marks = ",".join("?" for _ in batch_ids)
            usable_isins = {row[0] for row in connection.execute(
                f"SELECT DISTINCT isin FROM spread_observations WHERE attempt_id IN ({marks}) "
                "AND quality IN ('USABLE','USABLE_WITH_FLAGS')", batch_ids
            )}

    latest_success = next((row for row in attempts if row["status"] == "OBSERVED"), None)
    failures = [row for row in latest_batch if row["status"] != "OBSERVED"]
    cooldown_until = datetime.fromisoformat(cooldown["value"]) if cooldown else None
    active_cooldown = cooldown_until if cooldown_until and cooldown_until > now else None
    holding_isins = [row["isin"] for row in holdings]
    datasets = {}
    quality_path = ROOT / "reports/data_quality.json"
    quality = json.loads(quality_path.read_text(encoding="utf-8")) if quality_path.exists() else []
    quality_by_ticker = {str(row.get("ticker")): row for row in quality}
    for filename, label, date_column in (
        ("real_prices.parquet", "Prices", "date"),
        ("eurusd.parquet", "EUR/USD", "date"),
        ("macro_state.parquet", "Macro (FRED)", "date"),
    ):
        path = ROOT / "data/processed" / filename
        latest_date = None
        if path.exists():
            import pandas as pd
            frame = pd.read_parquet(path, columns=[date_column])
            if not frame.empty:
                latest_date = str(pd.to_datetime(frame[date_column]).max().date())
        datasets[label] = {"latest_date": latest_date}
    ingestion_times = [row.get("download_timestamp_utc") for row in quality if row.get("download_timestamp_utc")]
    quality_failures = [row for row in quality if row.get("status") == "FAILED"]
    with db.connect() as connection:
        paper_attempt = connection.execute(
            "SELECT started_at,status,error_type,error_message FROM prospective_runs ORDER BY started_at DESC,id DESC LIMIT 1"
        ).fetchone()
        failed_paper = connection.execute(
            "SELECT started_at,status,error_type,error_message FROM prospective_runs "
            "WHERE status!='SUCCESS' ORDER BY started_at DESC,id DESC LIMIT 1"
        ).fetchone()
    attempt_times = [x for x in (ingestion_times + ([paper_attempt["started_at"]] if paper_attempt else [])) if x]
    return {
        "update": dict(update) if update else None,
        "total_holdings": len(holdings),
        "verified_holdings": sum(isin in mapped for isin in holding_isins),
        "latest_attempt": attempts[0]["requested_at_utc"] if attempts else None,
        "latest_success": latest_success["collected_at_utc"] if latest_success else None,
        "latest_batch_size": len(latest_batch),
        "usable_latest": sum(isin in usable_isins for isin in holding_isins),
        "failures": [dict(row) for row in failures],
        "unmapped": sum(isin not in mapped for isin in holding_isins),
        "cooldown_until": active_cooldown.isoformat() if active_cooldown else None,
        "datasets": datasets,
        "ingestion_attempt": max(ingestion_times) if ingestion_times else None,
        "provider_failures": quality_failures,
        "paper_attempt": dict(paper_attempt) if paper_attempt else None,
        "paper_failure": dict(failed_paper) if failed_paper else None,
        "workflow_attempt": max(attempt_times) if attempt_times else None,
    }
