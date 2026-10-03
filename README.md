# Quiet Capital Research — V1

A local-first quantitative research, portfolio-monitoring, and decision-support application for a risk-averse private investor in Germany. It **does not place trades**, connect to Trade Republic, or request brokerage credentials. Every investment decision remains manual.

V1 is deliberately conservative: without a reconciled portfolio and genuine out-of-sample evidence it displays **NO HIGH-CONFIDENCE ACTION TODAY**. Synthetic demonstration data validate software behavior only and are never presented as investment evidence.

## Install

Requires Python 3.11 or newer.

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e '.[dev]'
cp .env.example .env
python scripts/manage.py init-db
```

Edit `.env` locally. Alpha Vantage and FRED keys are optional until their adapters are used. Set `SEC_USER_AGENT` to a real identifying name and email as required by SEC fair-access guidance. `.env` is ignored by Git.

## Launch the dashboard

```bash
./run.sh
```

Open the local address printed by Streamlit. The six pages are Today, Opportunities, Portfolio, Research, Performance, and Ledger.

## Upload and reconcile a Trade Republic PDF

On **Portfolio**, upload a text-based PDF, review every extracted field and warning, then explicitly confirm before saving. V1 recognizes dates, ISINs, euro values, and common cash labels. It intentionally reports uncertainty rather than guessing. Image-only PDFs require OCR, which is not included. A new PDF is appropriate after trades, cash movements, or periodic reconciliation—not for ordinary daily valuation.

## Update public data

Configure `ALPHA_VANTAGE_API_KEY`, then run:

```bash
python scripts/manage.py update-data
```

Add `--refresh` only when a cache refresh is intended. Price history is stored at `data/processed/prices.parquet`; provider caches are in `data/cache/`. FRED/ALFRED and SEC adapters are available in `src/data/adapters.py` and require explicit series/CIK integration in the next ingestion milestone. They never silently replace unavailable observations.

### First genuine-data 21-day experiment

```bash
python scripts/manage.py ingest-real
python scripts/manage.py experiment-real
```

Ingestion prefers Alpha Vantage when `ALPHA_VANTAGE_API_KEY` is configured and otherwise uses the Yahoo Finance chart endpoint as an explicit no-key fallback. Each ticker is cached independently, so a provider limit or interruption is resumable by running the same command again. `--refresh` deliberately replaces caches. FRED supplies DGS2, DGS10, and VIXCLS. EURUSD=X supplies USD per EUR; USD adjusted prices are divided by USD-per-EUR to obtain EUR investor values.

The genuine experiment uses 21-trading-day-spaced prediction cohorts, so evaluated outcomes do not overlap. At every annual fold, training observations are purged unless their complete 21-day target ended before the test period began. FRED state is lagged one observation day. Scaling and imputation are fitted inside each fold. Outputs are stored as `reports/real_*.csv`, with coverage metadata in `reports/real_experiment_metadata.json` and provider status in `reports/data_quality.json`.

Run the frozen accounting and robustness audit without changing features, models, thresholds, or hyperparameters:

```bash
python -m src.reporting.audit_21d
```

The audit separates (1) a tactical counterfactual that charges both the selected asset and passive substitute for matching round-trip costs and (2) a wealth counterfactual that treats tactical turnover as incremental relative to maintaining an existing passive holding. `reports/current_real_experiment.json` is the dashboard's provenance manifest. The dashboard rejects results whose real/demo mode or frozen-configuration hash does not match.

### Event-information research

The event experiment preserves the frozen control model and adds only the pre-specified, interpretable event fields documented in `reports/event_research_metadata.json`:

```bash
python scripts/event_research.py ingest
python scripts/event_research.py run
```

The library uses SEC 8-K Item 2.02 filing timestamps for major-company results, official BLS release calendars for CPI dates, and Federal Reserve statement archives for FOMC dates. Historical consensus earnings and CPI expectations are unavailable from the implemented free official sources; surprise fields remain missing and no substitute is invented. SEC filing time can lag the actual earnings announcement. BLS archive coverage may be partial when its site rejects automated historical requests. Event clusters and all 21-day overlaps are recorded rather than treated as independent observations.

### Medium-term dip research

The 63/126-trading-day experiment is a separate research module. It does not change the frozen 21-day or event controls and cannot generate a live BUY/SELL recommendation:

```bash
python scripts/medium_research.py ingest
python scripts/medium_research.py run
```

The first command caches annual SEC Company Facts for a fixed, documented subset of major SMH, XLK, and XLE companies. The second builds EUR-denominated 63/126-day targets, runs purged annual walk-forward tests on horizon-spaced cohorts, produces historical dip studies and transparent nearest analogues, and writes `reports/medium_*` artifacts. Re-running without `--refresh` uses the SEC cache.

Run the pre-specified downside-control audit without retraining the frozen medium-term models:

```bash
python scripts/medium_risk_audit.py
```

This reporting overlay applies four fixed point-in-time gates and three fixed combined policies plus one 50% sizing rule. VIX and asset-volatility 80th percentiles use only earlier history. Rejected observations remain cash for tactical-path statistics, while accepted-versus-rejected outcomes are reported separately. Outputs are `reports/medium_risk_*`; `medium_risk_metadata.json` binds them to the frozen medium-term configuration hash.

### Prospective zero-capital paper tracking

The prospective system is separate from every backtest. Its first successful daily run creates a permanent experiment start timestamp, persists the frozen 21/63/126-day ridge estimators, and records the model, Policy 2, configuration, and data-snapshot hashes. It never retrains those models automatically.

Run a complete daily cycle manually:

```bash
./scripts/run_daily.sh
```

The command refreshes genuine prices, EUR/USD, rates, and VIX using the existing ingestion path, applies the frozen models and Policy 2, appends immutable daily signal records, updates paper outcomes, and sends alerts only for material changes. Run logs are in `logs/`. Duplicate successful runs for the same market date are idempotent.

No Trade Republic connection or real order execution exists. Paper entries use 5% of the latest reconciled portfolio value when one is available. If no portfolio snapshot exists, the system uses normalized €1,000 zero-capital tracking and explicitly leaves the 5% euro amount unavailable.

Email is optional. Configure `SMTP_HOST`, `SMTP_PORT`, `SMTP_USERNAME`, `SMTP_PASSWORD`, `SMTP_FROM`, `SMTP_STARTTLS`, and `ALERT_TO` in the uncommitted `.env` file. No email is sent when nothing materially changes. Subjects and bodies identify every alert as a zero-capital paper signal. The alert allowlist is limited to a new paper entry, a held position entering a Policy 2 veto, paper reduce/exit, or stale/unavailable data suppressing an otherwise actionable state.

## Broad ETF discovery

`python scripts/discovery.py daily --refresh` runs the separate 153-ETF discovery screen. It caches free adjusted-price and volume data, applies explicit eligibility checks, deep-researches only the shortlist, and writes `reports/discovery_current.csv` plus the frozen-model generalization comparison. Its identifier is `discovery-screen-v1`; it never writes to the original prospective signal or position tables.

Discovery candidates use research language, never BUY/SELL. Explicit promotion to the separate paper ledger is available with `python scripts/discovery.py promote TICKER --reason "..." --allocation 1000 --horizon 126`. Promotion is timestamped when requested and cannot be backdated. Discovery email is limited to first shortlist entry, paper-review eligibility, or a major risk warning; small rank changes do not send mail.

The free V1 source supplies prices and trading volume when available. Historical AUM, bid-ask spreads, and constituent overlap are not available and are never fabricated; the dashboard labels those limitations and uses category overlap only as an approximate warning.

For Gmail, enable two-step verification on the Google account, create a dedicated app password, and place that app password—not the ordinary Google password—in `SMTP_PASSWORD`. Use `smtp.gmail.com`, port `587`, STARTTLS, and the Gmail address for both `SMTP_USERNAME` and `SMTP_FROM`. Keep `.env` readable only by the local user and never commit or paste it into reports. Other authenticated STARTTLS SMTP providers can use the same fields.

Verify delivery without creating a model signal:

```bash
./scripts/email_alerts.py test
```

The test message explicitly identifies itself as a configuration test. It does not create a signal, position, recommendation, or status transition.

Install the macOS LaunchAgent, inspect it, or remove it with:

```bash
./scripts/install_scheduler.sh
./scripts/scheduler_status.sh
./scripts/uninstall_scheduler.sh
```

The LaunchAgent runs daily at 06:00 local macOS time, which is Europe/Berlin on the intended machine. The uninstaller preserves the database, prospective artifacts, and logs. Installation writes only `~/Library/LaunchAgents/com.quietcapital.prospective.plist`.

Prospective signals and paper positions are stored in `state/invest.db`; current prospective performance and dated weekly/monthly reports are under `reports/prospective/`. Database triggers prevent edits to stored forecasts, ranks, risk flags, recommendations, features, model hashes, and rationales. Only future realized-outcome fields can be appended. Real-capital promotion remains a separate, unimplemented future decision gate.

### Model-update governance

Inspect the incumbent and schedule with:

```bash
./scripts/model_governance.py status
```

Daily data refreshes never retrain a model. The default quarterly checkpoint creates a challenger using the same ridge class, features, targets, training procedure, universe, and Policy 2 rules with an expanded training sample. It never overwrites or promotes the incumbent. Incumbent and challenger forecasts are stored separately and run prospectively in parallel.

Manual governance commands are:

```bash
./scripts/model_governance.py create-challenger --reason SCHEDULED_RETRAIN
./scripts/model_governance.py create-challenger --reason ANNUAL_REVIEW
./scripts/model_governance.py structural-review
./scripts/model_governance.py structural-review --create-challenger
```

Structural review is reserved for a documented regulatory, geopolitical, technological, or market-structure break. It marks the incumbent `REVIEW REQUIRED` and suspends new paper entries; it never silently modifies the incumbent. Recent losses alone are not a valid creation reason. Promotion of a challenger is intentionally not automated.

Reliable point-in-time historical aggregate valuation for the ETF proxies is not available from the implemented free sources. Valuation is therefore marked `UNAVAILABLE`; it is not imputed, and the valuation-aware baseline is not reported. ETF fundamental state is an unweighted, fixed-major-company approximation—not historical constituent-weighted ETF fundamentals. SPY, URTH, XBI, and EEM currently have no fundamental score. These limitations prevent high-confidence classifications.

## Train and run walk-forward validation

For a pipeline-only demonstration:

```bash
python scripts/manage.py demo-data
python scripts/manage.py train --demo
```

For cached provider prices:

```bash
python scripts/manage.py train
```

The optional synthetic pipeline stores clearly separated `reports/demo_synthetic_model_results.json` and `reports/demo_synthetic_walk_forward_predictions.csv` artifacts. It does not use random train/test splits and cannot feed dashboard recommendations.

## Tests

```bash
pytest
```

Tests cover chronological separation, target leakage guards, duplicate observations, missing-date behavior, forward returns, medium-term target purging and cohort spacing, benchmark alignment, configurable tax calculations, portfolio allocation, PDF uncertainty, reconciliation estimates, and capital double-allocation prevention.

## Storage and backup

SQLite state is in `state/invest.db`. Large observations and caches are Parquet. To back up while the app is stopped:

```bash
mkdir -p backups
sqlite3 state/invest.db ".backup 'backups/invest-$(date +%Y-%m-%d).db'"
```

Also copy `data/processed/`, `reports/`, and any manually maintained configuration. Database tables separately store market prices, macro vintages, events, snapshots, holdings, recommendations, forecasts, trades, passive shadow positions, model results, and project expenses. The access layer is isolated so PostgreSQL can replace SQLite later.

## Costs, tax, risk, and evidence

All fee, spread, German tax, saver-allowance, ETF partial-exemption, and loss-offset assumptions are configurable and approximate. Verify them with a qualified tax professional. The allocation layer applies evidence, downside, concentration, sleeve-size, cash, and minimum-trade vetoes. Forecast horizons cannot independently claim the same capital. Profit targets and fun money are monitoring fields only and never enter a forecast.

## Point-in-time and data limitations

- Alpha Vantage free-tier history and coverage can be limited; no subscription is purchased automatically.
- FRED revised series are not point-in-time correct unless explicit ALFRED vintages are requested and stored.
- SEC filings are available by filing date, but analyst expectations and historical constituent membership are not available in V1.
- Free data do not provide reliable historical earnings/revenue consensus surprises, ETF overlap, bid/ask history, or all event expectations. These are excluded from high-confidence decisions.
- The genuine 21-day experiment converts USD adjusted prices into EUR using historically aligned EUR/USD observations. Remaining FX limitations include provider timing differences, no intraday execution FX, no broker conversion charge, and no validation against the investor's actual EUR-traded UCITS instruments.
- PDF layouts vary. V1 is a validation-first parser scaffold, not a universal statement parser.
- Taxes are estimates, not tax advice. Unrealized/realized lot-level accounting needs confirmed trade records.
- Synthetic demonstration data are generated solely for deterministic software tests and have no market meaning.

## Safety and scope

This software provides research support, not individualized regulated advice. It never executes orders. Before recording any manual trade, review source dates, liquidity, costs, taxes, concentration, and the original thesis. No claim of profitability is made without genuine chronological out-of-sample evidence against the configured passive benchmark after costs and risk.
