"""Website persistence: statements, pretend investments and settings, in the same SQLite file as the research app."""
from __future__ import annotations
import json
from datetime import datetime, timezone
from pathlib import Path
import pandas as pd
from src.storage import Database
from src.web.settings import data_dir

WEB_SCHEMA = """
CREATE TABLE IF NOT EXISTS snapshot_extras (
 snapshot_id INTEGER PRIMARY KEY, brokerage_eur REAL, crypto_eur REAL, cash_eur REAL,
 reported_total_eur REAL, uploaded_at TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS crypto_holdings (
 snapshot_id INTEGER NOT NULL, name TEXT, symbol TEXT, quantity REAL, displayed_price REAL, market_value_eur REAL
);
CREATE TABLE IF NOT EXISTS whatif_positions (
 id INTEGER PRIMARY KEY, created_at TEXT NOT NULL, ticker TEXT NOT NULL, name TEXT NOT NULL, isin TEXT,
 mode TEXT NOT NULL, amount_eur REAL NOT NULL, horizon_years REAL NOT NULL, entry_date TEXT NOT NULL,
 entry_price REAL NOT NULL, expected_net_eur REAL, p10_eur REAL, p90_eur REAL, note TEXT, closed_at TEXT
);
CREATE TABLE IF NOT EXISTS app_settings (key TEXT PRIMARY KEY, value_json TEXT NOT NULL, updated_at TEXT NOT NULL);
"""


def _now() -> str: return datetime.now(timezone.utc).isoformat(timespec="seconds")


class WebStore:
    def __init__(self, path: str | Path | None = None):
        self.db = Database(path or data_dir() / "invest.db")
        self.db.initialize()
        with self.db.connect() as c: c.executescript(WEB_SCHEMA)

    # ---- Trade Republic statements -------------------------------------------------
    def save_statement(self, parsed: dict) -> int:
        holdings = [h for h in parsed.get("holdings", []) if h.get("market_value_eur") is not None]
        brokerage = round(sum(float(h["market_value_eur"]) for h in holdings), 2)
        crypto = float(parsed.get("crypto_value_eur") or sum(float(c.get("market_value_eur") or 0) for c in parsed.get("crypto_holdings", [])))
        cash = float(parsed.get("cash_eur") or 0)
        snapshot = {**parsed, "holdings": holdings, "cash_eur": cash, "total_value_eur": round(brokerage + crypto + cash, 2),
                    "source_file": Path(str(parsed.get("source_file") or "upload.pdf")).name}
        sid = self.db.insert_snapshot(snapshot)
        with self.db.connect() as c:
            c.execute("INSERT INTO snapshot_extras VALUES(?,?,?,?,?,?)", (sid, brokerage, crypto, cash, parsed.get("reported_total_financial_assets_eur"), _now()))
            c.executemany("INSERT INTO crypto_holdings VALUES(?,?,?,?,?,?)", [(sid, x.get("name"), x.get("symbol"), x.get("quantity"), x.get("displayed_price"), x.get("market_value_eur")) for x in parsed.get("crypto_holdings", [])])
        return sid

    def snapshots(self) -> pd.DataFrame:
        with self.db.connect() as c:
            return pd.read_sql_query("""SELECT s.id, s.snapshot_date, s.total_value_eur, s.source_file, e.brokerage_eur, e.crypto_eur,
                COALESCE(e.cash_eur, s.cash_eur) AS cash_eur, e.uploaded_at FROM portfolio_snapshots s
                LEFT JOIN snapshot_extras e ON e.snapshot_id = s.id ORDER BY s.snapshot_date DESC, s.id DESC""", c)

    def latest(self) -> dict | None:
        snaps = self.snapshots()
        if snaps.empty: return None
        row = snaps.iloc[0]; sid = int(row.id)
        with self.db.connect() as c:
            holdings = pd.read_sql_query("SELECT security_name, isin, quantity, displayed_price, market_value_eur FROM holdings WHERE snapshot_id=?", c, params=(sid,))
            crypto = pd.read_sql_query("SELECT name, symbol, quantity, displayed_price, market_value_eur FROM crypto_holdings WHERE snapshot_id=?", c, params=(sid,))
        brokerage = row.brokerage_eur if pd.notna(row.brokerage_eur) else float(holdings.market_value_eur.sum())
        cash = float(row.cash_eur or 0)
        crypto_value = row.crypto_eur if pd.notna(row.crypto_eur) else max(0.0, float(row.total_value_eur or 0) - brokerage - cash)
        return {"id": sid, "date": row.snapshot_date, "holdings": holdings, "crypto": crypto, "cash": cash,
                "brokerage": float(brokerage), "crypto_value": float(crypto_value), "total": float(row.total_value_eur or brokerage + cash + crypto_value)}

    def delete_snapshot(self, sid: int) -> None:
        with self.db.connect() as c:
            for table, column in (("holdings", "snapshot_id"), ("crypto_holdings", "snapshot_id"), ("snapshot_extras", "snapshot_id"), ("portfolio_snapshots", "id")):
                c.execute(f"DELETE FROM {table} WHERE {column}=?", (sid,))

    # ---- settings ----------------------------------------------------------------------
    def settings_overrides(self) -> dict:
        with self.db.connect() as c:
            row = c.execute("SELECT value_json FROM app_settings WHERE key='overrides'").fetchone()
        return json.loads(row[0]) if row else {}

    def save_settings(self, overrides: dict) -> None:
        with self.db.connect() as c:
            c.execute("INSERT INTO app_settings VALUES('overrides',?,?) ON CONFLICT(key) DO UPDATE SET value_json=excluded.value_json, updated_at=excluded.updated_at", (json.dumps(overrides), _now()))

    # ---- pretend investments -------------------------------------------------------------
    def add_whatif(self, **row) -> int:
        cols = ["ticker", "name", "isin", "mode", "amount_eur", "horizon_years", "entry_date", "entry_price", "expected_net_eur", "p10_eur", "p90_eur", "note"]
        with self.db.connect() as c:
            cur = c.execute(f"INSERT INTO whatif_positions(created_at,{','.join(cols)}) VALUES(?{',?' * len(cols)})", (_now(), *[row.get(k) for k in cols]))
            return int(cur.lastrowid)

    def whatifs(self, include_closed: bool = False) -> pd.DataFrame:
        with self.db.connect() as c:
            q = "SELECT * FROM whatif_positions" + ("" if include_closed else " WHERE closed_at IS NULL") + " ORDER BY id DESC"
            return pd.read_sql_query(q, c)

    def close_whatif(self, wid: int) -> None:
        with self.db.connect() as c: c.execute("UPDATE whatif_positions SET closed_at=? WHERE id=?", (_now(), wid))
