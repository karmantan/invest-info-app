from __future__ import annotations
import json, sqlite3
from contextlib import contextmanager
from pathlib import Path
from src.config import ROOT

SCHEMA = """
CREATE TABLE IF NOT EXISTS market_prices (ticker TEXT, date TEXT, close REAL, source TEXT, PRIMARY KEY(ticker,date));
CREATE TABLE IF NOT EXISTS macro_data (series_id TEXT, observation_date TEXT, vintage_date TEXT, value REAL, source TEXT, PRIMARY KEY(series_id,observation_date,vintage_date));
CREATE TABLE IF NOT EXISTS events (id INTEGER PRIMARY KEY, event_date TEXT, event_type TEXT, asset TEXT, actual REAL, consensus REAL, surprise REAL, point_in_time_quality TEXT, source TEXT);
CREATE TABLE IF NOT EXISTS portfolio_snapshots (id INTEGER PRIMARY KEY, snapshot_date TEXT, imported_at TEXT DEFAULT CURRENT_TIMESTAMP, cash_eur REAL, total_value_eur REAL, source_file TEXT, warnings_json TEXT);
CREATE TABLE IF NOT EXISTS holdings (snapshot_id INTEGER, security_name TEXT, isin TEXT, ticker TEXT, quantity REAL, displayed_price REAL, market_value_eur REAL, asset_type TEXT, confidence REAL);
CREATE TABLE IF NOT EXISTS recommendations (id INTEGER PRIMARY KEY, created_at TEXT DEFAULT CURRENT_TIMESTAMP, ticker TEXT, horizon_days INTEGER, action TEXT, amount_eur REAL, expected_return REAL, benchmark_return REAL, downside_p10 REAL, confidence REAL, reason TEXT, vetoes_json TEXT);
CREATE TABLE IF NOT EXISTS model_forecasts (id INTEGER PRIMARY KEY, as_of_date TEXT, ticker TEXT, horizon_days INTEGER, model_name TEXT, prediction REAL, training_end TEXT, sample_size INTEGER);
CREATE TABLE IF NOT EXISTS trades (id INTEGER PRIMARY KEY, trade_date TEXT, ticker TEXT, isin TEXT, amount_eur REAL, horizon_days INTEGER, original_thesis TEXT, expected_return REAL, expected_benchmark_return REAL, downside_estimate REAL, confidence REAL, review_triggers TEXT, exit_criteria TEXT, status TEXT, realized_result_eur REAL);
CREATE TABLE IF NOT EXISTS shadow_positions (trade_id INTEGER PRIMARY KEY, benchmark_ticker TEXT, principal_eur REAL, entry_price REAL, current_price REAL, benchmark_result_eur REAL);
CREATE TABLE IF NOT EXISTS model_results (id INTEGER PRIMARY KEY, run_at TEXT DEFAULT CURRENT_TIMESTAMP, ticker TEXT, horizon_days INTEGER, model_name TEXT, metrics_json TEXT, artifact_path TEXT);
CREATE TABLE IF NOT EXISTS project_expenses (id INTEGER PRIMARY KEY, expense_date TEXT, category TEXT, amount_eur REAL, description TEXT);
CREATE TABLE IF NOT EXISTS prospective_experiments (
 id INTEGER PRIMARY KEY, experiment_id TEXT UNIQUE NOT NULL, started_at TEXT NOT NULL,
 frozen_21d_hash TEXT NOT NULL, frozen_medium_hash TEXT NOT NULL, risk_policy_hash TEXT NOT NULL,
 configuration_hash TEXT NOT NULL, data_snapshot_timestamp TEXT NOT NULL, mode TEXT NOT NULL DEFAULT 'paper', real_capital_enabled INTEGER NOT NULL DEFAULT 0
);
CREATE TABLE IF NOT EXISTS prospective_runs (
 id INTEGER PRIMARY KEY, experiment_id TEXT NOT NULL, market_date TEXT NOT NULL, started_at TEXT NOT NULL,
 completed_at TEXT, status TEXT NOT NULL, data_snapshot_timestamp TEXT, data_snapshot_hash TEXT,
 price_latest_date TEXT, macro_latest_date TEXT, fx_latest_date TEXT, error_type TEXT, error_message TEXT,
 UNIQUE(experiment_id, market_date)
);
CREATE TABLE IF NOT EXISTS paper_signals (
 id INTEGER PRIMARY KEY, experiment_id TEXT NOT NULL, run_id INTEGER NOT NULL, observed_date TEXT NOT NULL,
 observed_at TEXT NOT NULL, asset TEXT NOT NULL, horizon_days INTEGER NOT NULL, forecast REAL,
 forecast_rank REAL, top_cohort INTEGER NOT NULL, active_risk_flags TEXT NOT NULL, policy_2_pass INTEGER NOT NULL,
 recommendation TEXT NOT NULL, benchmark TEXT NOT NULL, price REAL, eurusd REAL, expected_return REAL,
 historical_p10 REAL, historical_p5 REAL, confidence TEXT, model_hash TEXT NOT NULL, configuration_hash TEXT NOT NULL,
 data_snapshot_timestamp TEXT NOT NULL, feature_values_json TEXT NOT NULL, decision_reason TEXT NOT NULL, model_version TEXT,
 realized_at TEXT, realized_return REAL, spy_return REAL, urth_return REAL, excess_spy REAL, excess_urth REAL,
 UNIQUE(experiment_id, observed_date, asset, horizon_days)
);
CREATE TABLE IF NOT EXISTS paper_positions (
 id INTEGER PRIMARY KEY, experiment_id TEXT NOT NULL, asset TEXT NOT NULL, entry_signal_id INTEGER NOT NULL,
 entry_date TEXT NOT NULL, entry_timestamp TEXT NOT NULL, entry_price REAL NOT NULL, allocation_eur REAL NOT NULL,
 portfolio_sleeve_eur REAL, normalized_allocation_eur REAL NOT NULL, horizon_days INTEGER NOT NULL,
 triggered_horizons_json TEXT NOT NULL, benchmark_spy_entry REAL, benchmark_urth_entry REAL,
 original_thesis TEXT NOT NULL, current_status TEXT NOT NULL, risk_status TEXT NOT NULL,
 closed_date TEXT, closed_price REAL, gross_return REAL, net_return REAL, spy_return REAL, urth_return REAL,
 simulated_cost_eur REAL, UNIQUE(experiment_id, asset, entry_date)
);
CREATE UNIQUE INDEX IF NOT EXISTS one_open_paper_position_per_asset ON paper_positions(experiment_id,asset) WHERE closed_date IS NULL;
CREATE TABLE IF NOT EXISTS paper_status_changes (
 id INTEGER PRIMARY KEY, experiment_id TEXT NOT NULL, run_id INTEGER NOT NULL, changed_at TEXT NOT NULL,
 asset TEXT NOT NULL, horizon_days INTEGER NOT NULL, previous_status TEXT, new_status TEXT NOT NULL,
 reason TEXT NOT NULL, alert_required INTEGER NOT NULL DEFAULT 0, email_status TEXT
);
CREATE TABLE IF NOT EXISTS model_versions (
 id INTEGER PRIMARY KEY, version TEXT UNIQUE NOT NULL, model_family TEXT NOT NULL, role TEXT NOT NULL,
 created_at TEXT NOT NULL, training_end_date TEXT NOT NULL, feature_specification_json TEXT NOT NULL,
 training_procedure TEXT NOT NULL, configuration_hash TEXT NOT NULL, risk_policy_hash TEXT NOT NULL,
 creation_reason TEXT NOT NULL CHECK(creation_reason IN ('INITIAL','SCHEDULED_RETRAIN','ANNUAL_REVIEW','STRUCTURAL_BREAK_REVIEW')),
 artifact_manifest_json TEXT NOT NULL, parent_version TEXT, validity TEXT NOT NULL DEFAULT 'NORMAL', promoted_at TEXT
);
CREATE TABLE IF NOT EXISTS prospective_model_forecasts (
 id INTEGER PRIMARY KEY, experiment_id TEXT NOT NULL, model_version TEXT NOT NULL, observed_date TEXT NOT NULL,
 observed_at TEXT NOT NULL, asset TEXT NOT NULL, horizon_days INTEGER NOT NULL, forecast REAL, forecast_rank REAL,
 top_cohort INTEGER NOT NULL, active_risk_flags TEXT NOT NULL, policy_2_pass INTEGER NOT NULL,
 feature_values_json TEXT NOT NULL, model_hash TEXT NOT NULL, realized_at TEXT, realized_return REAL,
 benchmark_return REAL, excess_return REAL, UNIQUE(experiment_id,model_version,observed_date,asset,horizon_days)
);
CREATE TABLE IF NOT EXISTS discovery_runs (
 id INTEGER PRIMARY KEY, discovery_experiment_id TEXT NOT NULL, market_date TEXT NOT NULL,
 started_at TEXT NOT NULL, completed_at TEXT, status TEXT NOT NULL, universe_size INTEGER NOT NULL,
 eligible_count INTEGER NOT NULL DEFAULT 0, excluded_count INTEGER NOT NULL DEFAULT 0,
 shortlist_count INTEGER NOT NULL DEFAULT 0, data_snapshot_timestamp TEXT, notes_json TEXT,
 UNIQUE(discovery_experiment_id,market_date)
);
CREATE TABLE IF NOT EXISTS discovery_results (
 id INTEGER PRIMARY KEY, run_id INTEGER NOT NULL, discovery_experiment_id TEXT NOT NULL,
 observed_date TEXT NOT NULL, ticker TEXT NOT NULL, eligibility TEXT NOT NULL,
 exclusion_reason TEXT, discovery_score REAL, rank INTEGER, shortlisted INTEGER NOT NULL DEFAULT 0,
 status TEXT NOT NULL, risk_level TEXT, risk_flags_json TEXT NOT NULL, data_quality_json TEXT NOT NULL,
 evidence_63d_json TEXT, evidence_126d_json TEXT, explanation TEXT, benchmark TEXT NOT NULL,
 model_label TEXT NOT NULL DEFAULT 'OUT-OF-UNIVERSE RESEARCH', feature_snapshot_json TEXT NOT NULL,
 portfolio_warning TEXT, overlap_warning TEXT,
 UNIQUE(run_id,ticker)
);
CREATE TABLE IF NOT EXISTS discovery_promotions (
 id INTEGER PRIMARY KEY, discovery_experiment_id TEXT NOT NULL, ticker TEXT NOT NULL,
 discovery_date TEXT NOT NULL, promotion_date TEXT NOT NULL, reason TEXT NOT NULL,
 data_snapshot_json TEXT NOT NULL, discovery_score REAL, model_evidence_json TEXT,
 risk_state TEXT NOT NULL, horizon_days INTEGER NOT NULL, paper_allocation_eur REAL NOT NULL,
 benchmark TEXT NOT NULL, status TEXT NOT NULL DEFAULT 'OPEN', future_outcome_json TEXT,
 UNIQUE(discovery_experiment_id,ticker,promotion_date)
);
CREATE TABLE IF NOT EXISTS discovery_watchlist (
 id INTEGER PRIMARY KEY, discovery_experiment_id TEXT NOT NULL, ticker TEXT NOT NULL,
 first_watched_at TEXT NOT NULL, active INTEGER NOT NULL DEFAULT 1,
 reason_at_start TEXT NOT NULL, discovery_score_at_start REAL, discovery_rank_at_start INTEGER,
 status_at_start TEXT NOT NULL, risk_at_start TEXT NOT NULL, shortlisted_at_start INTEGER NOT NULL,
 UNIQUE(discovery_experiment_id,ticker)
);
CREATE TABLE IF NOT EXISTS discovery_watch_history (
 id INTEGER PRIMARY KEY, watch_id INTEGER NOT NULL, event_type TEXT NOT NULL,
 event_at TEXT NOT NULL, reason_snapshot TEXT, discovery_score_snapshot REAL,
 discovery_rank_snapshot INTEGER, status_snapshot TEXT, risk_snapshot TEXT,
 shortlisted_snapshot INTEGER NOT NULL
);
CREATE TABLE IF NOT EXISTS discovery_positions (
 id INTEGER PRIMARY KEY, discovery_experiment_id TEXT NOT NULL, ticker TEXT NOT NULL,
 friendly_name TEXT NOT NULL, economic_category TEXT NOT NULL, discovery_date TEXT NOT NULL,
 tracking_timestamp TEXT NOT NULL, market_date TEXT NOT NULL, entry_price REAL NOT NULL,
 entry_price_source TEXT NOT NULL, discovery_rank_at_entry INTEGER,
 discovery_score_at_entry REAL, reason_at_entry TEXT NOT NULL, risk_at_entry TEXT NOT NULL,
 drawdown_at_entry REAL, evidence_3m_json TEXT, evidence_6m_json TEXT,
 benchmark TEXT NOT NULL, model_version TEXT NOT NULL, data_quality_at_entry TEXT NOT NULL,
 portfolio_overlap_warning TEXT, normalized_notional_eur REAL NOT NULL DEFAULT 1000,
 status TEXT NOT NULL DEFAULT 'ACTIVE', stopped_at TEXT, stop_reason TEXT,
 UNIQUE(discovery_experiment_id,ticker,tracking_timestamp)
);
CREATE UNIQUE INDEX IF NOT EXISTS one_active_discovery_position_per_etf
 ON discovery_positions(discovery_experiment_id,ticker) WHERE status='ACTIVE';
CREATE TABLE IF NOT EXISTS discovery_position_horizons (
 id INTEGER PRIMARY KEY, position_id INTEGER NOT NULL, horizon_label TEXT NOT NULL,
 target_trading_days INTEGER NOT NULL, status TEXT NOT NULL DEFAULT 'TRACKING',
 completed_at TEXT, outcome_json TEXT, UNIQUE(position_id,target_trading_days)
);
CREATE TABLE IF NOT EXISTS discovery_alerts (
 id INTEGER PRIMARY KEY, run_id INTEGER NOT NULL, ticker TEXT NOT NULL, alert_type TEXT NOT NULL,
 created_at TEXT NOT NULL, message TEXT NOT NULL, email_status TEXT,
 UNIQUE(run_id,ticker,alert_type)
);
CREATE TABLE IF NOT EXISTS spread_attempts (
 id INTEGER PRIMARY KEY, requested_at_utc TEXT NOT NULL, collected_at_utc TEXT NOT NULL,
 available_at_utc TEXT NOT NULL, isin TEXT NOT NULL, ticker TEXT, expected_exchange TEXT,
 expected_currency TEXT, provider TEXT NOT NULL, status TEXT NOT NULL, quality TEXT NOT NULL,
 reason TEXT NOT NULL, http_status INTEGER, provider_metadata_json TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS spread_observations (
 id INTEGER PRIMARY KEY, attempt_id INTEGER NOT NULL UNIQUE, requested_at_utc TEXT NOT NULL,
 collected_at_utc TEXT NOT NULL, available_at_utc TEXT NOT NULL, isin TEXT NOT NULL,
 ticker TEXT NOT NULL, exchange TEXT NOT NULL, currency TEXT NOT NULL, bid REAL NOT NULL,
 ask REAL NOT NULL, bid_size REAL, ask_size REAL, midpoint REAL NOT NULL,
 spread_currency REAL NOT NULL, spread_percent REAL NOT NULL, spread_bps REAL NOT NULL,
 quote_timestamp_utc TEXT, quote_freshness TEXT NOT NULL, provider TEXT NOT NULL,
 source_label TEXT NOT NULL, quality TEXT NOT NULL, reason TEXT NOT NULL,
 provider_metadata_json TEXT NOT NULL,
 FOREIGN KEY(attempt_id) REFERENCES spread_attempts(id)
);
CREATE TABLE IF NOT EXISTS spread_collector_state (
 key TEXT PRIMARY KEY, value TEXT NOT NULL, updated_at_utc TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS spread_observations_lookup
 ON spread_observations(isin,available_at_utc DESC,id DESC);
CREATE INDEX IF NOT EXISTS spread_attempts_lookup
 ON spread_attempts(isin,available_at_utc DESC,id DESC);
CREATE TRIGGER IF NOT EXISTS discovery_position_entry_immutable
BEFORE UPDATE ON discovery_positions
WHEN NEW.discovery_experiment_id IS NOT OLD.discovery_experiment_id OR NEW.ticker IS NOT OLD.ticker
 OR NEW.friendly_name IS NOT OLD.friendly_name OR NEW.economic_category IS NOT OLD.economic_category
 OR NEW.discovery_date IS NOT OLD.discovery_date OR NEW.tracking_timestamp IS NOT OLD.tracking_timestamp
 OR NEW.market_date IS NOT OLD.market_date OR NEW.entry_price IS NOT OLD.entry_price
 OR NEW.entry_price_source IS NOT OLD.entry_price_source OR NEW.discovery_rank_at_entry IS NOT OLD.discovery_rank_at_entry
 OR NEW.discovery_score_at_entry IS NOT OLD.discovery_score_at_entry OR NEW.reason_at_entry IS NOT OLD.reason_at_entry
 OR NEW.risk_at_entry IS NOT OLD.risk_at_entry OR NEW.drawdown_at_entry IS NOT OLD.drawdown_at_entry
 OR NEW.evidence_3m_json IS NOT OLD.evidence_3m_json OR NEW.evidence_6m_json IS NOT OLD.evidence_6m_json
 OR NEW.benchmark IS NOT OLD.benchmark OR NEW.model_version IS NOT OLD.model_version
 OR NEW.data_quality_at_entry IS NOT OLD.data_quality_at_entry
 OR NEW.portfolio_overlap_warning IS NOT OLD.portfolio_overlap_warning
 OR NEW.normalized_notional_eur IS NOT OLD.normalized_notional_eur
BEGIN SELECT RAISE(ABORT, 'discovery position entry is immutable'); END;
DROP TRIGGER IF EXISTS paper_signals_immutable;
CREATE TRIGGER paper_signals_immutable
BEFORE UPDATE ON paper_signals
WHEN NEW.experiment_id IS NOT OLD.experiment_id OR NEW.run_id IS NOT OLD.run_id OR NEW.observed_date IS NOT OLD.observed_date
 OR NEW.observed_at IS NOT OLD.observed_at OR NEW.asset IS NOT OLD.asset OR NEW.horizon_days IS NOT OLD.horizon_days
 OR NEW.forecast IS NOT OLD.forecast OR NEW.forecast_rank IS NOT OLD.forecast_rank OR NEW.top_cohort IS NOT OLD.top_cohort
 OR NEW.active_risk_flags IS NOT OLD.active_risk_flags OR NEW.policy_2_pass IS NOT OLD.policy_2_pass
 OR NEW.recommendation IS NOT OLD.recommendation OR NEW.benchmark IS NOT OLD.benchmark OR NEW.price IS NOT OLD.price
 OR NEW.eurusd IS NOT OLD.eurusd OR NEW.expected_return IS NOT OLD.expected_return OR NEW.historical_p10 IS NOT OLD.historical_p10
 OR NEW.historical_p5 IS NOT OLD.historical_p5 OR NEW.confidence IS NOT OLD.confidence OR NEW.model_hash IS NOT OLD.model_hash
 OR NEW.configuration_hash IS NOT OLD.configuration_hash OR NEW.data_snapshot_timestamp IS NOT OLD.data_snapshot_timestamp
 OR NEW.feature_values_json IS NOT OLD.feature_values_json OR NEW.decision_reason IS NOT OLD.decision_reason OR NEW.model_version IS NOT OLD.model_version
BEGIN SELECT RAISE(ABORT, 'paper signal provenance is immutable'); END;
CREATE TRIGGER IF NOT EXISTS model_versions_immutable
BEFORE UPDATE ON model_versions
WHEN NEW.version IS NOT OLD.version OR NEW.model_family IS NOT OLD.model_family OR NEW.created_at IS NOT OLD.created_at
 OR NEW.training_end_date IS NOT OLD.training_end_date OR NEW.feature_specification_json IS NOT OLD.feature_specification_json
 OR NEW.training_procedure IS NOT OLD.training_procedure OR NEW.configuration_hash IS NOT OLD.configuration_hash
 OR NEW.risk_policy_hash IS NOT OLD.risk_policy_hash OR NEW.creation_reason IS NOT OLD.creation_reason
 OR NEW.artifact_manifest_json IS NOT OLD.artifact_manifest_json OR NEW.parent_version IS NOT OLD.parent_version
BEGIN SELECT RAISE(ABORT, 'model version specification is immutable'); END;
CREATE TRIGGER IF NOT EXISTS challenger_forecasts_immutable
BEFORE UPDATE ON prospective_model_forecasts
WHEN NEW.experiment_id IS NOT OLD.experiment_id OR NEW.model_version IS NOT OLD.model_version OR NEW.observed_date IS NOT OLD.observed_date
 OR NEW.observed_at IS NOT OLD.observed_at OR NEW.asset IS NOT OLD.asset OR NEW.horizon_days IS NOT OLD.horizon_days
 OR NEW.forecast IS NOT OLD.forecast OR NEW.forecast_rank IS NOT OLD.forecast_rank OR NEW.top_cohort IS NOT OLD.top_cohort
 OR NEW.active_risk_flags IS NOT OLD.active_risk_flags OR NEW.policy_2_pass IS NOT OLD.policy_2_pass
 OR NEW.feature_values_json IS NOT OLD.feature_values_json OR NEW.model_hash IS NOT OLD.model_hash
BEGIN SELECT RAISE(ABORT, 'prospective challenger forecast is immutable'); END;
CREATE TRIGGER IF NOT EXISTS spread_attempts_immutable
BEFORE UPDATE ON spread_attempts BEGIN SELECT RAISE(ABORT, 'spread attempts are append-only'); END;
CREATE TRIGGER IF NOT EXISTS spread_observations_immutable
BEFORE UPDATE ON spread_observations BEGIN SELECT RAISE(ABORT, 'spread observations are append-only'); END;
"""

class Database:
    def __init__(self, path: str | Path = ROOT / "state/invest.db"):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
    @contextmanager
    def connect(self):
        connection = sqlite3.connect(self.path)
        connection.row_factory = sqlite3.Row
        try:
            yield connection
            connection.commit()
        finally:
            connection.close()
    def initialize(self):
        with self.connect() as connection:
            connection.executescript(SCHEMA)
            columns={row[1] for row in connection.execute("PRAGMA table_info(paper_signals)")}
            if "model_version" not in columns: connection.execute("ALTER TABLE paper_signals ADD COLUMN model_version TEXT")
    def insert_snapshot(self, snapshot: dict) -> int:
        with self.connect() as c:
            cur = c.execute("INSERT INTO portfolio_snapshots(snapshot_date,cash_eur,total_value_eur,source_file,warnings_json) VALUES(?,?,?,?,?)", (snapshot["snapshot_date"], snapshot.get("cash_eur"), snapshot.get("total_value_eur"), snapshot.get("source_file"), json.dumps(snapshot.get("warnings", []))))
            sid = cur.lastrowid
            c.executemany("INSERT INTO holdings VALUES(?,?,?,?,?,?,?,?,?)", [(sid,h.get("security_name"),h.get("isin"),h.get("ticker"),h.get("quantity"),h.get("displayed_price"),h.get("market_value_eur"),h.get("asset_type","security"),h.get("confidence",0)) for h in snapshot.get("holdings",[])])
        return sid

    def record_spread_attempt(self, attempt: dict, observation: dict | None = None) -> int:
        """Append a request outcome and, when valid, its quote observation atomically."""
        attempt_columns = ("requested_at_utc","collected_at_utc","available_at_utc","isin","ticker",
                           "expected_exchange","expected_currency","provider","status","quality","reason",
                           "http_status","provider_metadata_json")
        with self.connect() as c:
            cur=c.execute(f"INSERT INTO spread_attempts({','.join(attempt_columns)}) VALUES({','.join('?' for _ in attempt_columns)})",
                          tuple(attempt.get(k) for k in attempt_columns))
            attempt_id=int(cur.lastrowid)
            if observation:
                columns=("requested_at_utc","collected_at_utc","available_at_utc","isin","ticker","exchange",
                         "currency","bid","ask","bid_size","ask_size","midpoint","spread_currency",
                         "spread_percent","spread_bps","quote_timestamp_utc","quote_freshness","provider",
                         "source_label","quality","reason","provider_metadata_json")
                c.execute(f"INSERT INTO spread_observations(attempt_id,{','.join(columns)}) VALUES(?,{','.join('?' for _ in columns)})",
                          (attempt_id,)+tuple(observation.get(k) for k in columns))
        return attempt_id

    def latest_spreads(self, as_of_utc: str | None = None):
        """Return current holdings with only data that was available by the requested cutoff."""
        cutoff=as_of_utc or "9999-12-31T23:59:59+00:00"
        with self.connect() as c:
            return c.execute("""
              WITH current_holdings AS (
                SELECT h.* FROM holdings h WHERE h.snapshot_id=(SELECT MAX(id) FROM portfolio_snapshots)
              )
              SELECT h.security_name,h.isin,h.ticker AS holding_ticker,
                a.id AS attempt_id,a.requested_at_utc AS attempt_requested_at_utc,
                a.collected_at_utc AS attempt_collected_at_utc,a.status AS attempt_status,
                a.quality AS attempt_quality,a.reason AS attempt_reason,a.ticker AS mapped_ticker,
                o.collected_at_utc AS observation_collected_at_utc,o.available_at_utc,
                o.exchange,o.currency,o.bid,o.ask,o.bid_size,o.ask_size,o.midpoint,
                o.spread_currency,o.spread_percent,o.spread_bps,o.quote_freshness,
                o.provider,o.source_label,o.quality AS observation_quality,o.reason AS observation_reason
              FROM current_holdings h
              LEFT JOIN spread_attempts a ON a.id=(SELECT id FROM spread_attempts x
                WHERE x.isin=h.isin AND x.available_at_utc<=? ORDER BY x.available_at_utc DESC,x.id DESC LIMIT 1)
              LEFT JOIN spread_observations o ON o.id=(SELECT id FROM spread_observations y
                WHERE y.isin=h.isin AND y.available_at_utc<=? ORDER BY y.available_at_utc DESC,y.id DESC LIMIT 1)
              ORDER BY h.market_value_eur DESC,h.security_name
            """,(cutoff,cutoff)).fetchall()
