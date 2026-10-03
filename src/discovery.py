from __future__ import annotations

import hashlib
import json
import os
import smtplib
from datetime import datetime, timezone
from email.message import EmailMessage
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from src.config import ROOT
from src.data.adapters import YahooChartPrices
from src.features.medium import MEDIUM_FEATURES
from src.storage import Database

DISCOVERY_EXPERIMENT_ID = "discovery-screen-v1"
FROZEN_VALIDATION_EXPERIMENT_ID = "prospective-paper-a1a8c1c4f515"
VALIDATION_UNIVERSE = {"SPY", "URTH", "XLK", "SMH", "XBI", "XLE", "EEM"}
CATALOGUE_PATH = ROOT / "config/discovery_universe.csv"
PRICES_PATH = ROOT / "data/processed/discovery_prices.parquet"
RESULTS_PATH = ROOT / "reports/discovery_current.csv"
GENERALIZATION_PATH = ROOT / "reports/discovery_generalization.csv"


def catalogue(path: Path = CATALOGUE_PATH) -> pd.DataFrame:
    frame = pd.read_csv(path, parse_dates=["launch_date"])
    if not frame.ticker.is_unique:
        raise ValueError("Discovery tickers must be unique")
    overlap = VALIDATION_UNIVERSE.intersection(frame.ticker)
    if overlap:
        raise ValueError(f"Discovery catalogue must exclude validation assets: {sorted(overlap)}")
    return frame


def refresh_prices(refresh: bool = False, adapter=None) -> tuple[pd.DataFrame, list[dict]]:
    adapter = adapter or YahooChartPrices()
    frames, failures = [], []
    # Benchmarks are research inputs only and never become discovery candidates.
    research_inputs = sorted(VALIDATION_UNIVERSE) + ["EURUSD=X"]
    for ticker in list(catalogue().ticker) + research_inputs:
        try:
            part = adapter.fetch(ticker, refresh=refresh).copy()
            part["ticker"] = ticker
            frames.append(part)
        except Exception as exc:
            failures.append({"ticker": ticker, "error": f"{type(exc).__name__}: {exc}"})
    if frames:
        incoming = pd.concat(frames, ignore_index=True)
        if PRICES_PATH.exists() and not refresh:
            incoming = pd.concat([pd.read_parquet(PRICES_PATH), incoming], ignore_index=True)
        incoming = incoming.drop_duplicates(["ticker", "date"], keep="last").sort_values(["ticker", "date"])
        PRICES_PATH.parent.mkdir(parents=True, exist_ok=True)
        incoming.to_parquet(PRICES_PATH, index=False)
        return incoming, failures
    if PRICES_PATH.exists():
        return pd.read_parquet(PRICES_PATH), failures
    raise RuntimeError("No discovery prices are available")


def build_panel(prices: pd.DataFrame, macro: pd.DataFrame | None = None) -> pd.DataFrame:
    p = prices.copy(); p["date"] = pd.to_datetime(p.date)
    fx = p[p.ticker == "EURUSD=X"][["date", "close"]].rename(columns={"close": "usd_per_eur"}).sort_values("date")
    frames = []
    for ticker, x in p[~p.ticker.isin(["EURUSD=X"])].groupby("ticker"):
        x = x.sort_values("date").drop_duplicates("date").copy()
        x = pd.merge_asof(x, fx, on="date", direction="backward", tolerance=pd.Timedelta("5d")) if not fx.empty else x.assign(usd_per_eur=1.0)
        x["close_eur"] = x.close / x.usd_per_eur
        ret = x.close_eur.pct_change(fill_method=None)
        for d in (5, 21, 63, 126, 252): x[f"eur_return_{d}d"] = x.close_eur.pct_change(d, fill_method=None)
        for d in (63, 126, 252): x[f"drawdown_{d}d"] = x.close_eur / x.close_eur.rolling(d, min_periods=max(20, d // 2)).max() - 1
        x["distance_ma50"] = x.close_eur / x.close_eur.rolling(50, min_periods=40).mean() - 1
        x["distance_ma200"] = x.close_eur / x.close_eur.rolling(200, min_periods=150).mean() - 1
        x["volatility_63d"] = ret.rolling(63, min_periods=40).std() * np.sqrt(252)
        for h in (63, 126):
            x[f"forward_eur_{h}d"] = x.close_eur.shift(-h) / x.close_eur - 1
            x[f"target_end_{h}d"] = x.date.shift(-h)
        x["observations"] = np.arange(1, len(x) + 1)
        x["missing_ratio_252d"] = x.close_eur.isna().rolling(252, min_periods=1).mean()
        x["median_dollar_volume_63d"] = np.nan
        if "volume" in x and x.volume.notna().any():
            x["median_dollar_volume_63d"] = (x.volume * x.close).rolling(63, min_periods=40).median()
        frames.append(x)
    panel = pd.concat(frames, ignore_index=True)
    for bench in ("SPY", "URTH"):
        b = panel[panel.ticker == bench][["date", "eur_return_63d", "forward_eur_63d", "forward_eur_126d"]].rename(columns={"eur_return_63d": f"{bench.lower()}_momentum_63d", "forward_eur_63d": f"{bench.lower()}_forward_63d", "forward_eur_126d": f"{bench.lower()}_forward_126d"})
        panel = panel.merge(b, on="date", how="left")
    panel["relative_spy_63d"] = panel.eur_return_63d - panel.spy_momentum_63d
    panel["relative_urth_63d"] = panel.eur_return_63d - panel.urth_momentum_63d
    for h in (63, 126): panel[f"forward_excess_{h}d"] = panel[f"forward_eur_{h}d"] - panel[f"spy_forward_{h}d"]
    if macro is not None and not macro.empty:
        m = macro.copy().sort_values("date"); m["date"] = pd.to_datetime(m.date)
        panel = pd.merge_asof(panel.sort_values("date"), m, on="date", direction="backward", tolerance=pd.Timedelta("7d"))
    for col in ("vix", "dgs2", "dgs10", "dgs2_change_21d", "dgs10_change_21d", "curve_2s10s"):
        if col not in panel: panel[col] = np.nan
    panel["eurusd_change_21d"] = panel.groupby("ticker").usd_per_eur.pct_change(21, fill_method=None)
    # Fundamentals are intentionally unavailable for the broad free-data universe.
    panel["fundamental_stability_score"] = np.nan
    panel["fundamental_coverage"] = 0.0
    return panel.sort_values(["date", "ticker"]).reset_index(drop=True)


def _eligibility(row: pd.Series, meta: pd.Series, as_of: pd.Timestamp) -> tuple[str, str | None]:
    structure = str(meta.structure).lower()
    if any(term in structure for term in ("leveraged", "inverse", "etn")):
        return "EXCLUDED STRUCTURE", "Leveraged, inverse, or note structure is not supported"
    age = (as_of - pd.Timestamp(meta.launch_date)).days
    if age < 3 * 365 or row.observations < 756:
        return "TOO NEW", "Less than about three years of price history"
    required = ["eur_return_21d", "eur_return_63d", "eur_return_126d", "eur_return_252d", "volatility_63d"]
    if any(pd.isna(row.get(c)) for c in required) or row.missing_ratio_252d > .08:
        return "LIMITED DATA", "Price history is incomplete or key screening fields are missing"
    if pd.notna(row.median_dollar_volume_63d) and row.median_dollar_volume_63d < 2_000_000:
        return "LOW LIQUIDITY", "Median daily traded value is below $2 million"
    if pd.isna(row.median_dollar_volume_63d):
        return "LIMITED DATA", "Reliable trading-volume data are unavailable"
    return "ELIGIBLE", None


def _percentile(series: pd.Series, value: float, higher: bool = True) -> float:
    clean = series.dropna()
    if clean.empty or pd.isna(value): return .5
    p = float((clean <= value).mean())
    return p if higher else 1 - p


def _analogue(panel: pd.DataFrame, ticker: str, horizon: int, n: int = 20) -> dict:
    x = panel[panel.ticker == ticker].sort_values("date").copy()
    cols = ["eur_return_63d", "eur_return_126d", "eur_return_252d", "drawdown_252d", "volatility_63d", "relative_spy_63d"]
    current = x.iloc[-1]; hist = x.dropna(subset=cols + [f"forward_eur_{horizon}d", f"forward_excess_{horizon}d"]).copy()
    if len(hist) < n: return {"sample_size": len(hist), "quality": "LIMITED", "median_excess": None, "outperformance_probability": None, "p10": None, "p5": None, "large_loss_probability": None}
    hist[cols] = hist[cols].apply(pd.to_numeric, errors="coerce")
    current_values = pd.to_numeric(current[cols], errors="coerce")
    scale = hist[cols].std().replace(0, np.nan)
    hist["distance"] = np.sqrt(((((hist[cols] - current_values) / scale) ** 2).mean(axis=1)).astype(float))
    selected, dates = [], []
    for idx, row in hist.sort_values("distance").iterrows():
        if all(abs((row.date - d).days) >= 63 for d in dates): selected.append(idx); dates.append(row.date)
        if len(selected) == n: break
    y = hist.loc[selected, f"forward_eur_{horizon}d"]; excess = hist.loc[selected, f"forward_excess_{horizon}d"]
    return {"sample_size": len(y), "quality": "GOOD" if len(y) >= 15 else "LIMITED", "median_excess": _finite(excess.median()), "outperformance_probability": _finite((excess > 0).mean()), "p10": _finite(y.quantile(.10)), "p5": _finite(y.quantile(.05)), "large_loss_probability": _finite((y < -.10).mean())}


def _finite(value):
    return None if value is None or not np.isfinite(value) else float(value)


def _status(score: float | None, risks: list[str], eligible: str) -> str:
    if eligible in ("TOO NEW", "LIMITED DATA"): return "Insufficient history"
    if eligible != "ELIGIBLE": return "No current opportunity"
    if risks and score is not None and score >= 60: return "Interesting but high risk"
    if score is not None and score >= 70: return "Interesting medium-term setup"
    if score is not None and score >= 55: return "Worth watching"
    return "No current opportunity"


def _explanation(row: pd.Series, risks: list[str]) -> str:
    if row.relative_spy_63d > .03 and row.eur_return_126d > 0:
        base = "Recent medium-term performance is stronger than the broad market, while the longer-term trend remains positive."
    elif row.drawdown_252d < -.05 and row.eur_return_252d > 0:
        base = "The ETF has fallen moderately from its recent high, but its longer-term trend remains positive."
    else:
        base = "Its combination of medium-term trend, relative performance, and current drawdown merits a closer look."
    if "extreme_asset_volatility" in risks: base += " Current price swings are unusually large."
    if "extreme_drawdown" in risks: base += " The current fall from its recent high is severe."
    if "broad_market_downtrend" in risks: base += " The broad market is below its long-term trend."
    return base


def screen(panel: pd.DataFrame, metadata: pd.DataFrame | None = None, top_n: int = 15) -> pd.DataFrame:
    metadata = catalogue() if metadata is None else metadata.copy()
    as_of = panel.date.max(); latest = panel.sort_values("date").groupby("ticker").tail(1).set_index("ticker")
    candidates = metadata.merge(latest, left_on="ticker", right_index=True, how="left", suffixes=("", "_price"))
    spy = latest.loc["SPY"] if "SPY" in latest.index else pd.Series(dtype=float)
    scored = []
    for _, row in candidates.iterrows():
        if pd.isna(row.get("date")):
            eligibility, reason = "LIMITED DATA", "No usable price history was downloaded"
        else:
            eligibility, reason = _eligibility(row, row, as_of)
        risks = []
        if pd.notna(row.get("drawdown_252d")) and row.drawdown_252d <= -.20: risks.append("extreme_drawdown")
        if pd.notna(row.get("volatility_63d")) and row.volatility_63d >= .45: risks.append("extreme_asset_volatility")
        if pd.notna(spy.get("distance_ma200")) and spy.distance_ma200 < 0: risks.append("broad_market_downtrend")
        score = None
        if eligibility == "ELIGIBLE":
            eligible_pool = candidates[candidates.observations >= 756]
            components = {
                "momentum": np.mean([_percentile(eligible_pool[f"eur_return_{d}d"], row[f"eur_return_{d}d"]) for d in (63, 126, 252)]),
                "relative_strength": np.mean([_percentile(eligible_pool[c], row[c]) for c in ("relative_spy_63d", "relative_urth_63d")]),
                "drawdown_state": _percentile(eligible_pool.drawdown_252d, row.drawdown_252d),
                "volatility": _percentile(eligible_pool.volatility_63d, row.volatility_63d, higher=False),
                "market_regime": 1.0 if pd.isna(spy.get("distance_ma200")) or spy.distance_ma200 >= 0 else .25,
                "risk_gate": 1.0 if not risks else .15,
                "data_quality": min(1.0, row.observations / 1260),
            }
            # Pre-specified V1 weights. They are deliberately not optimized on returns.
            if row.asset_class == "Fixed Income":
                components["rates_state"] = .75 if pd.isna(row.get("dgs10_change_21d")) or row.get("dgs10_change_21d") <= 0 else .35
                weights = {"momentum": .25, "relative_strength": .10, "drawdown_state": .10, "volatility": .15, "market_regime": .05, "risk_gate": .15, "data_quality": .10, "rates_state": .10}
            else:
                weights = {"momentum": .25, "relative_strength": .20, "drawdown_state": .10, "volatility": .10, "market_regime": .10, "risk_gate": .15, "data_quality": .10}
            score = 100 * sum(components[k] * weights[k] for k in weights)
        else: components = {}
        scored.append({**row[metadata.columns].to_dict(), "observed_date": as_of, "eligibility": eligibility, "exclusion_reason": reason, "discovery_score": score, "risk_flags": risks, "risk_level": "High" if risks else ("Moderate" if pd.notna(row.get("volatility_63d")) and row.volatility_63d > .28 else "Lower"), "features": components, "drawdown": _finite(row.get("drawdown_252d")), "momentum_63d": _finite(row.get("eur_return_63d")), "momentum_126d": _finite(row.get("eur_return_126d")), "relative_spy_63d": _finite(row.get("relative_spy_63d")), "median_dollar_volume_63d": _finite(row.get("median_dollar_volume_63d")), "observations": _finite(row.get("observations")), "current_price_eur": _finite(row.get("close_eur")), "price_source": row.get("source") or "Yahoo adjusted close converted to EUR"})
    out = pd.DataFrame(scored); out["rank"] = out.discovery_score.rank(ascending=False, method="first").astype("Int64")
    # Severe-risk names cannot displace ordinary eligible candidates at the top.
    safe = out[(out.eligibility == "ELIGIBLE") & out.risk_flags.map(lambda x: not x)].nsmallest(top_n, "rank")
    risky = out[(out.eligibility == "ELIGIBLE") & out.risk_flags.map(bool) & (out.discovery_score >= 60)].nlargest(3, "discovery_score")
    shortlist = set(pd.concat([safe, risky]).ticker)
    out["shortlisted"] = out.ticker.isin(shortlist)
    out["status"] = [_status(s, r, e) for s, r, e in zip(out.discovery_score, out.risk_flags, out.eligibility)]
    return out.sort_values(["shortlisted", "discovery_score"], ascending=[False, False], na_position="last")


def _portfolio_warnings(db: Database, result: pd.DataFrame) -> pd.DataFrame:
    validation_themes={"Technology":"US Technology","Semiconductors":"Semiconductors","Biotechnology":"US Biotechnology","Energy":"US Energy"}
    with db.connect() as c:
        holdings = pd.read_sql_query("SELECT h.security_name,h.ticker,h.market_value_eur FROM holdings h JOIN portfolio_snapshots p ON p.id=h.snapshot_id WHERE p.id=(SELECT MAX(id) FROM portfolio_snapshots)", c)
    if holdings.empty:
        result=result.copy(); result["portfolio_warning"]=None
        result["overlap_warning"]=[f"Approximate category overlap with the existing {validation_themes[t]} validation proxy; constituent overlap data are unavailable." if t in validation_themes else None for t in result.sector_theme]
        return result
    text = " ".join(holdings.security_name.fillna("").astype(str).str.lower())
    tickers = set(holdings.ticker.dropna().astype(str).str.upper())
    warnings, overlaps = [], []
    for _, row in result.iterrows():
        theme = str(row.sector_theme).lower()
        category_hit = any(token in text for token in theme.split() if len(token) > 4)
        warnings.append(f"Portfolio may already have significant {row.sector_theme.lower()} exposure." if category_hit else None)
        overlaps.append("This ETF is already held." if row.ticker in tickers else (f"Approximate category overlap with existing {row.sector_theme.lower()} exposure; constituent data are unavailable." if category_hit else (f"Approximate category overlap with the existing {validation_themes[row.sector_theme]} validation proxy; constituent overlap data are unavailable." if row.sector_theme in validation_themes else None)))
    result = result.copy(); result["portfolio_warning"] = warnings; result["overlap_warning"] = overlaps
    return result


def _frozen_predictions(panel: pd.DataFrame, result: pd.DataFrame) -> pd.DataFrame:
    equity_tickers = result[(result.shortlisted) & (result.asset_class == "Equity")].ticker
    current = panel[panel.ticker.isin(equity_tickers)].sort_values("date").groupby("ticker").tail(1).copy()
    for h in (63, 126):
        path = ROOT / f"state/models/frozen_ridge_{h}d.joblib"
        current[f"frozen_model_{h}d"] = np.nan
        if path.exists() and not current.empty:
            try: current[f"frozen_model_{h}d"] = joblib.load(path).predict(current[MEDIUM_FEATURES])
            except Exception: pass
    return current.set_index("ticker")


def deep_research(panel: pd.DataFrame, result: pd.DataFrame) -> pd.DataFrame:
    result = result.copy(); models = _frozen_predictions(panel, result)
    e63, e126, explanations = [], [], []
    for _, row in result.iterrows():
        if not row.shortlisted:
            e63.append(None); e126.append(None); explanations.append(None); continue
        a63 = _analogue(panel, row.ticker, 63); a126 = _analogue(panel, row.ticker, 126)
        if row.ticker in models.index:
            a63["frozen_model_input"] = _finite(models.loc[row.ticker, "frozen_model_63d"])
            a126["frozen_model_input"] = _finite(models.loc[row.ticker, "frozen_model_126d"])
        a63["label"] = a126["label"] = "OUT-OF-UNIVERSE RESEARCH"
        e63.append(a63); e126.append(a126); explanations.append(_explanation(pd.Series({"relative_spy_63d": row.relative_spy_63d, "eur_return_126d": row.momentum_126d, "drawdown_252d": row.drawdown, "eur_return_252d": row.features.get("momentum", 0)}), row.risk_flags))
    result["evidence_63d"] = e63; result["evidence_126d"] = e126; result["explanation"] = explanations
    return result


def generalization_test(panel: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for horizon in (63, 126):
        path = ROOT / f"state/models/frozen_ridge_{horizon}d.joblib"
        if not path.exists(): continue
        model = joblib.load(path)
        usable = panel.dropna(subset=[f"forward_excess_{horizon}d", f"forward_eur_{horizon}d", f"target_end_{horizon}d"]).copy()
        # Evaluation starts after the frozen model's training data. No fitting or retuning occurs here.
        manifest_path = ROOT / "state/models/prospective_manifest.json"
        manifest = json.loads(manifest_path.read_text()) if manifest_path.exists() else {}
        train_end = pd.Timestamp(manifest.get("models", {}).get(str(horizon), {}).get("training_data_end", "1900-01-01"))
        try: usable["prediction"] = model.predict(usable[MEDIUM_FEATURES])
        except Exception: continue
        dates = sorted(usable.date.unique())[::horizon]
        usable = usable[usable.date.isin(dates)]
        usable["selected"] = usable.groupby("date").prediction.rank(pct=True, method="first") > .66
        equity_discovery = set(catalogue().query("asset_class == 'Equity'").ticker)
        for label, tickers in (("Original validation universe", VALIDATION_UNIVERSE), ("Expanded discovery universe", equity_discovery)):
            x = usable[usable.ticker.isin(tickers) & usable.selected][f"forward_excess_{horizon}d"].dropna()
            raw = usable.loc[x.index, f"forward_eur_{horizon}d"]
            rows.append({"universe": label, "horizon_days": horizon, "observations": len(x), "mean_excess": _finite(x.mean()), "median_excess": _finite(x.median()), "outperformance_probability": _finite((x > 0).mean()), "p10": _finite(raw.quantile(.10)), "p5": _finite(raw.quantile(.05)), "large_loss_probability": _finite((raw < -.10).mean()), "model_specification": "frozen medium-ridge-v1; no refit or retuning", "evaluation_limit": f"Retrospective frozen-artifact test; dates through the original training end {train_end.date()} are not a prospective holdout"})
    return pd.DataFrame(rows)


def _json(value) -> str:
    return json.dumps(value, sort_keys=True, default=str, allow_nan=False)


def run(refresh: bool = False, db: Database | None = None, top_n: int = 15) -> dict:
    db = db or Database(); db.initialize(); started = datetime.now(timezone.utc).isoformat(); meta = catalogue()
    prices, failures = refresh_prices(refresh); macro_path = ROOT / "data/processed/macro_state.parquet"
    panel = build_panel(prices, pd.read_parquet(macro_path) if macro_path.exists() else None)
    result = _portfolio_warnings(db, deep_research(panel, screen(panel, meta, top_n)))
    market_date = str(pd.Timestamp(panel.date.max()).date()); snapshot = hashlib.sha256(pd.util.hash_pandas_object(prices[["ticker", "date", "close"]], index=False).values.tobytes()).hexdigest()
    with db.connect() as c:
        old = c.execute("SELECT id FROM discovery_runs WHERE discovery_experiment_id=? AND market_date=?", (DISCOVERY_EXPERIMENT_ID, market_date)).fetchone()
        same_day_previous=set(); same_day_alerts=[]
        if old:
            same_day_previous={r[0] for r in c.execute("SELECT ticker FROM discovery_results WHERE run_id=? AND shortlisted=1",(old[0],))}
            same_day_alerts=[dict(r) for r in c.execute("SELECT ticker,alert_type,created_at,message,email_status FROM discovery_alerts WHERE run_id=?",(old[0],))]
            c.execute("DELETE FROM discovery_results WHERE run_id=?", (old[0],)); c.execute("DELETE FROM discovery_alerts WHERE run_id=?", (old[0],)); c.execute("DELETE FROM discovery_runs WHERE id=?", (old[0],))
        eligible = int((result.eligibility == "ELIGIBLE").sum()); shortlisted = int(result.shortlisted.sum())
        run_id = c.execute("INSERT INTO discovery_runs(discovery_experiment_id,market_date,started_at,completed_at,status,universe_size,eligible_count,excluded_count,shortlist_count,data_snapshot_timestamp,notes_json) VALUES(?,?,?,?,?,?,?,?,?,?,?)", (DISCOVERY_EXPERIMENT_ID, market_date, started, datetime.now(timezone.utc).isoformat(), "SUCCESS", len(meta), eligible, len(meta)-eligible, shortlisted, snapshot, _json({"provider_failures": failures, "aum": "unavailable", "bid_ask_spreads": "unavailable"}))).lastrowid
        for alert in same_day_alerts:
            c.execute("INSERT INTO discovery_alerts(run_id,ticker,alert_type,created_at,message,email_status) VALUES(?,?,?,?,?,?)",(run_id,alert["ticker"],alert["alert_type"],alert["created_at"],alert["message"],alert["email_status"]))
        previous = same_day_previous | {r[0] for r in c.execute("SELECT DISTINCT ticker FROM discovery_results WHERE shortlisted=1")}
        for _, row in result.iterrows():
            snapshot_features={**row.features,"drawdown":row.drawdown,"momentum_63d":row.momentum_63d,"momentum_126d":row.momentum_126d,"relative_spy_63d":row.relative_spy_63d,"current_price_eur":row.current_price_eur,"price_source":row.price_source}
            c.execute("INSERT INTO discovery_results(run_id,discovery_experiment_id,observed_date,ticker,eligibility,exclusion_reason,discovery_score,rank,shortlisted,status,risk_level,risk_flags_json,data_quality_json,evidence_63d_json,evidence_126d_json,explanation,benchmark,feature_snapshot_json,portfolio_warning,overlap_warning) VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)", (run_id, DISCOVERY_EXPERIMENT_ID, market_date, row.ticker, row.eligibility, row.exclusion_reason, _finite(row.discovery_score), int(row["rank"]) if pd.notna(row["rank"]) else None, int(row.shortlisted), row.status, row.risk_level, _json(row.risk_flags), _json({"observations": row.observations, "median_dollar_volume_63d": row.median_dollar_volume_63d, "aum": None, "bid_ask_spread": None}), _json(row.evidence_63d) if row.evidence_63d else None, _json(row.evidence_126d) if row.evidence_126d else None, row.explanation, "SPY (with URTH shown as secondary comparator)", _json(snapshot_features), row.portfolio_warning, row.overlap_warning))
            if row.shortlisted and row.ticker not in previous:
                c.execute("INSERT INTO discovery_alerts(run_id,ticker,alert_type,created_at,message,email_status) VALUES(?,?,?,?,?,?)", (run_id, row.ticker, "NEW_SHORTLIST_ENTRY", started, f"{row.friendly_name} entered the discovery shortlist.", "PENDING"))
        _record_watched_candidate_alerts(c, run_id, started)
    export = result.copy()
    for col in ("risk_flags", "features", "evidence_63d", "evidence_126d"): export[col] = export[col].map(lambda v: _json(v) if isinstance(v, (dict, list)) else v)
    RESULTS_PATH.parent.mkdir(exist_ok=True); export.to_csv(RESULTS_PATH, index=False)
    generalization = generalization_test(panel); generalization.to_csv(GENERALIZATION_PATH, index=False)
    send_pending_alerts(db, run_id)
    return {"experiment_id": DISCOVERY_EXPERIMENT_ID, "validation_experiment_unchanged": FROZEN_VALIDATION_EXPERIMENT_ID, "market_date": market_date, "universe_size": len(meta), "eligible": int((result.eligibility == "ELIGIBLE").sum()), "excluded": int((result.eligibility != "ELIGIBLE").sum()), "shortlisted": int(result.shortlisted.sum()), "failures": len(failures)}


def promote(ticker: str, reason: str, allocation_eur: float, horizon_days: int, db: Database | None = None) -> int:
    if horizon_days not in (63, 126) or allocation_eur <= 0: raise ValueError("Promotion requires a 63/126-day horizon and positive paper allocation")
    db = db or Database(); db.initialize(); now = datetime.now(timezone.utc).isoformat()
    with db.connect() as c:
        row = c.execute("SELECT r.* FROM discovery_results r JOIN discovery_runs d ON d.id=r.run_id WHERE r.ticker=? AND r.shortlisted=1 ORDER BY d.market_date DESC LIMIT 1", (ticker.upper(),)).fetchone()
        if not row: raise ValueError("Only a prospectively shortlisted discovery candidate can be promoted")
        cur = c.execute("INSERT INTO discovery_promotions(discovery_experiment_id,ticker,discovery_date,promotion_date,reason,data_snapshot_json,discovery_score,model_evidence_json,risk_state,horizon_days,paper_allocation_eur,benchmark,status) VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?)", (DISCOVERY_EXPERIMENT_ID, ticker.upper(), row["observed_date"], now, reason, row["feature_snapshot_json"], row["discovery_score"], _json({"about_3_months": row["evidence_63d_json"], "about_6_months": row["evidence_126d_json"], "label": "OUT-OF-UNIVERSE RESEARCH"}), row["risk_level"], horizon_days, allocation_eur, "SPY", "OPEN"))
        return int(cur.lastrowid)


def group_opportunities(candidates: pd.DataFrame) -> list[dict]:
    """Group display records only; candidate scores and ranks remain untouched."""
    if candidates.empty:
        return []
    keys = [key for key in ("benchmark_category", "friendly_name", "ticker") if key in candidates]
    category_key = keys[0]
    groups = []
    for category, rows in candidates.sort_values("rank", na_position="last").groupby(category_key, sort=False, dropna=False):
        records = rows.to_dict("records")
        groups.append({"opportunity_name": str(category), "leading": records[0], "alternatives": records[1:], "candidates": records})
    return groups


def _latest_candidate(c, ticker: str, shortlisted: bool = False):
    condition = " AND r.shortlisted=1" if shortlisted else ""
    return c.execute(
        "SELECT r.* FROM discovery_results r JOIN discovery_runs d ON d.id=r.run_id "
        f"WHERE r.ticker=?{condition} ORDER BY d.market_date DESC,d.id DESC LIMIT 1", (ticker.upper(),)
    ).fetchone()


def watch_candidate(ticker: str, db: Database | None = None, now: str | None = None) -> int:
    db = db or Database(); db.initialize(); timestamp = now or datetime.now(timezone.utc).isoformat()
    with db.connect() as c:
        row = _latest_candidate(c, ticker)
        if not row or row["eligibility"] != "ELIGIBLE":
            raise ValueError("Only an eligible research candidate can be watched")
        existing = c.execute("SELECT * FROM discovery_watchlist WHERE discovery_experiment_id=? AND ticker=?", (DISCOVERY_EXPERIMENT_ID, ticker.upper())).fetchone()
        if existing and existing["active"]:
            return int(existing["id"])
        if existing:
            watch_id = int(existing["id"])
            c.execute("UPDATE discovery_watchlist SET active=1 WHERE id=?", (watch_id,))
            event = "RESTARTED"
        else:
            watch_id = int(c.execute(
                "INSERT INTO discovery_watchlist(discovery_experiment_id,ticker,first_watched_at,active,reason_at_start,discovery_score_at_start,discovery_rank_at_start,status_at_start,risk_at_start,shortlisted_at_start) VALUES(?,?,?,?,?,?,?,?,?,?)",
                (DISCOVERY_EXPERIMENT_ID, ticker.upper(), timestamp, 1, row["explanation"] or "Research candidate worth keeping on the radar.", row["discovery_score"], row["rank"], row["status"], row["risk_level"], row["shortlisted"])
            ).lastrowid)
            event = "STARTED"
        c.execute("INSERT INTO discovery_watch_history(watch_id,event_type,event_at,reason_snapshot,discovery_score_snapshot,discovery_rank_snapshot,status_snapshot,risk_snapshot,shortlisted_snapshot) VALUES(?,?,?,?,?,?,?,?,?)",
                  (watch_id,event,timestamp,row["explanation"],row["discovery_score"],row["rank"],row["status"],row["risk_level"],row["shortlisted"]))
        return watch_id


def stop_watching(ticker: str, db: Database | None = None, now: str | None = None) -> None:
    db = db or Database(); db.initialize(); timestamp = now or datetime.now(timezone.utc).isoformat()
    with db.connect() as c:
        watch = c.execute("SELECT * FROM discovery_watchlist WHERE discovery_experiment_id=? AND ticker=? AND active=1", (DISCOVERY_EXPERIMENT_ID,ticker.upper())).fetchone()
        if not watch: return
        row = _latest_candidate(c,ticker)
        c.execute("UPDATE discovery_watchlist SET active=0 WHERE id=?",(watch["id"],))
        c.execute("INSERT INTO discovery_watch_history(watch_id,event_type,event_at,reason_snapshot,discovery_score_snapshot,discovery_rank_snapshot,status_snapshot,risk_snapshot,shortlisted_snapshot) VALUES(?,?,?,?,?,?,?,?,?)",
                  (watch["id"],"STOPPED",timestamp,row["explanation"] if row else None,row["discovery_score"] if row else None,row["rank"] if row else None,row["status"] if row else None,row["risk_level"] if row else None,row["shortlisted"] if row else 0))


def watchlist(db: Database | None = None) -> pd.DataFrame:
    db = db or Database(); db.initialize()
    with db.connect() as c:
        return pd.read_sql_query("""SELECT w.*,r.explanation current_reason,r.discovery_score current_score,
          r.rank current_rank,r.status current_status,r.risk_level current_risk,r.shortlisted current_shortlisted,
          r.observed_date current_observed_date
          FROM discovery_watchlist w LEFT JOIN discovery_results r ON r.id=(SELECT r2.id FROM discovery_results r2
          JOIN discovery_runs d2 ON d2.id=r2.run_id WHERE r2.ticker=w.ticker ORDER BY d2.market_date DESC,d2.id DESC LIMIT 1)
          WHERE w.active=1 ORDER BY w.first_watched_at DESC""",c)


def describe_watch_changes(row) -> list[str]:
    changes=[]
    old_rank=row.get("discovery_rank_at_start"); new_rank=row.get("current_rank")
    if pd.notna(old_rank) and pd.notna(new_rank) and abs(int(new_rank)-int(old_rank)) >= 3:
        changes.append("Rank improved" if new_rank < old_rank else "Rank deteriorated")
    risk_order={"Lower":0,"Moderate":1,"High":2}; old_r=risk_order.get(row.get("risk_at_start")); new_r=risk_order.get(row.get("current_risk"))
    if old_r is not None and new_r is not None and old_r != new_r: changes.append("Risk increased" if new_r>old_r else "Risk decreased")
    if bool(row.get("shortlisted_at_start")) != bool(row.get("current_shortlisted")): changes.append("Entered shortlist" if row.get("current_shortlisted") else "Left shortlist")
    if row.get("current_status") == "No current opportunity": changes.append("Current setup no longer looks interesting")
    return changes or ["No meaningful change"]


def _prospective_entry_price(row, ticker: str) -> tuple[float, str, str]:
    features=json.loads(row["feature_snapshot_json"] or "{}")
    price=features.get("current_price_eur")
    if price is not None and np.isfinite(float(price)) and float(price)>0:
        return float(price),features.get("price_source") or "Latest available adjusted market close",row["observed_date"]
    if not PRICES_PATH.exists(): raise ValueError("No valid current market price is available")
    prices=pd.read_parquet(PRICES_PATH); prices["date"]=pd.to_datetime(prices.date)
    asset=prices[prices.ticker==ticker].sort_values("date")
    fx=prices[prices.ticker=="EURUSD=X"].sort_values("date")
    if asset.empty or fx.empty: raise ValueError("No valid current market price is available")
    market=asset.iloc[-1]; available_fx=fx[fx.date<=market.date]
    if available_fx.empty: raise ValueError("No valid current market price is available")
    eur=float(market.close)/float(available_fx.iloc[-1].close)
    if not np.isfinite(eur) or eur<=0: raise ValueError("No valid current market price is available")
    return eur,"Cached adjusted close converted to EUR",str(pd.Timestamp(market.date).date())


def start_simulated_tracking(ticker: str, db: Database | None = None, now: str | None = None) -> int:
    """Open one zero-capital discovery observation with two evaluation horizons."""
    db = db or Database(); db.initialize(); timestamp = now or datetime.now(timezone.utc).isoformat()
    with db.connect() as c:
        if c.execute("SELECT 1 FROM discovery_positions WHERE discovery_experiment_id=? AND ticker=? AND status='ACTIVE'",(DISCOVERY_EXPERIMENT_ID,ticker.upper())).fetchone():
            raise ValueError("Simulated tracking is already active for this ETF")
        row=_latest_candidate(c,ticker,shortlisted=True)
        if not row: raise ValueError("Only a current shortlisted research candidate can start simulated tracking")
        if pd.Timestamp(timestamp).date() < pd.Timestamp(row["observed_date"]).date():
            raise ValueError("A simulated tracking entry cannot be backdated")
        features=json.loads(row["feature_snapshot_json"] or "{}")
        price,price_source,market_date=_prospective_entry_price(row,ticker.upper())
        if pd.Timestamp(timestamp).date() < pd.Timestamp(market_date).date():
            raise ValueError("A simulated tracking entry cannot be backdated")
        meta=catalogue().set_index("ticker").loc[ticker.upper()]
        cur=c.execute("""INSERT INTO discovery_positions(discovery_experiment_id,ticker,friendly_name,economic_category,
          discovery_date,tracking_timestamp,market_date,entry_price,entry_price_source,discovery_rank_at_entry,
          discovery_score_at_entry,reason_at_entry,risk_at_entry,drawdown_at_entry,evidence_3m_json,evidence_6m_json,
          benchmark,model_version,data_quality_at_entry,portfolio_overlap_warning,normalized_notional_eur,status)
          VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
          (DISCOVERY_EXPERIMENT_ID,ticker.upper(),meta.friendly_name,meta.benchmark_category,row["observed_date"],timestamp,
           market_date,price,price_source,
           row["rank"],row["discovery_score"],row["explanation"] or "Current discovery candidate",row["risk_level"],
           features.get("drawdown"),row["evidence_63d_json"],row["evidence_126d_json"],row["benchmark"],
           "medium-ridge-v1 (frozen research input)",row["data_quality_json"],row["overlap_warning"],1000,"ACTIVE"))
        position_id=int(cur.lastrowid)
        c.executemany("INSERT INTO discovery_position_horizons(position_id,horizon_label,target_trading_days,status) VALUES(?,?,?,?)",
                      [(position_id,"About 3 months",63,"TRACKING"),(position_id,"About 6 months",126,"TRACKING")])
        return position_id


def stop_simulated_tracking(ticker: str, reason: str | None = None, db: Database | None = None, now: str | None = None) -> None:
    db=db or Database(); db.initialize(); timestamp=now or datetime.now(timezone.utc).isoformat()
    with db.connect() as c:
        c.execute("UPDATE discovery_positions SET status='STOPPED_EARLY',stopped_at=?,stop_reason=? WHERE discovery_experiment_id=? AND ticker=? AND status='ACTIVE'",
                  (timestamp,reason or "Stopped by the user",DISCOVERY_EXPERIMENT_ID,ticker.upper()))


def _record_watched_candidate_alerts(c, run_id: int, created_at: str) -> None:
    watched=c.execute("SELECT * FROM discovery_watchlist WHERE active=1").fetchall()
    for watch in watched:
        current=c.execute("SELECT * FROM discovery_results WHERE run_id=? AND ticker=?",(run_id,watch["ticker"])).fetchone()
        if not current: continue
        previous=c.execute("SELECT r.* FROM discovery_results r JOIN discovery_runs d ON d.id=r.run_id WHERE r.ticker=? AND r.run_id<>? ORDER BY d.market_date DESC,d.id DESC LIMIT 1",(watch["ticker"],run_id)).fetchone()
        events=[]
        if previous and not previous["shortlisted"] and current["shortlisted"]: events.append(("WATCH_ENTERED_SHORTLIST",f"{watch['ticker']} entered the current discovery shortlist."))
        if previous and previous["risk_level"] != "High" and current["risk_level"] == "High": events.append(("WATCH_RISK_DETERIORATED",f"{watch['ticker']} now has a material risk warning."))
        if previous and previous["risk_level"] == "High" and current["risk_level"] != "High": events.append(("WATCH_RISK_CLEARED",f"The material risk warning for {watch['ticker']} has cleared."))
        for kind,message in events:
            c.execute("INSERT OR IGNORE INTO discovery_alerts(run_id,ticker,alert_type,created_at,message,email_status) VALUES(?,?,?,?,?,?)",(run_id,watch["ticker"],kind,created_at,message,"PENDING"))
    market_date=c.execute("SELECT market_date FROM discovery_runs WHERE id=?",(run_id,)).fetchone()[0]
    for position in c.execute("SELECT * FROM discovery_positions WHERE status='ACTIVE'").fetchall():
        current=c.execute("SELECT * FROM discovery_results WHERE run_id=? AND ticker=?",(run_id,position["ticker"])).fetchone()
        if current and position["risk_at_entry"] != "High" and current["risk_level"] == "High":
            c.execute("INSERT OR IGNORE INTO discovery_alerts(run_id,ticker,alert_type,created_at,message,email_status) VALUES(?,?,?,?,?,?)",(run_id,position["ticker"],"TRACKING_RISK_WARNING",created_at,f"A material risk warning is active for simulated tracking of {position['ticker']}.","PENDING"))
        elapsed=int(np.busday_count(str(position["market_date"]),str(market_date)))
        for horizon in c.execute("SELECT * FROM discovery_position_horizons WHERE position_id=? AND status='TRACKING'",(position["id"],)).fetchall():
            if elapsed >= horizon["target_trading_days"]:
                c.execute("UPDATE discovery_position_horizons SET status='COMPLETE',completed_at=? WHERE id=?",(created_at,horizon["id"]))
                c.execute("INSERT OR IGNORE INTO discovery_alerts(run_id,ticker,alert_type,created_at,message,email_status) VALUES(?,?,?,?,?,?)",(run_id,position["ticker"],f"HORIZON_{horizon['target_trading_days']}_COMPLETE",created_at,f"The {horizon['horizon_label'].lower()} simulated tracking horizon completed for {position['ticker']}.","PENDING"))


def send_pending_alerts(db: Database, run_id: int) -> str:
    """Send only material discovery events; absence of SMTP is a recorded state, not an error."""
    required=("SMTP_HOST","SMTP_FROM","ALERT_TO"); missing=[key for key in required if not os.getenv(key)]
    with db.connect() as c: pending=c.execute("SELECT * FROM discovery_alerts WHERE run_id=? AND email_status='PENDING'",(run_id,)).fetchall()
    if not pending: return "NO MATERIAL DISCOVERY CHANGE — NO EMAIL"
    if missing:
        state="NOT CONFIGURED — "+", ".join(missing)
        with db.connect() as c: c.execute("UPDATE discovery_alerts SET email_status=? WHERE run_id=? AND email_status='PENDING'",(state,run_id))
        return state
    try:
        with smtplib.SMTP(os.environ["SMTP_HOST"],int(os.getenv("SMTP_PORT","587")),timeout=30) as smtp:
            if os.getenv("SMTP_STARTTLS","true").lower()=="true": smtp.starttls()
            if os.getenv("SMTP_USERNAME"): smtp.login(os.environ["SMTP_USERNAME"],os.environ.get("SMTP_PASSWORD",""))
            for row in pending:
                msg=EmailMessage(); msg["Subject"]="Quiet Capital: new ETF research candidate"; msg["From"]=os.environ["SMTP_FROM"]; msg["To"]=os.environ["ALERT_TO"]
                msg.set_content(row["message"]+"\n\nResearch only. No trade or validation-universe change was made."); smtp.send_message(msg)
        state="SENT"
    except Exception as exc: state=f"FAILED — {type(exc).__name__}: {exc}"
    with db.connect() as c: c.execute("UPDATE discovery_alerts SET email_status=? WHERE run_id=? AND email_status='PENDING'",(state,run_id))
    return state
