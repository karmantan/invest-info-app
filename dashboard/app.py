from __future__ import annotations
import json, sys
from datetime import date,datetime,timezone
from pathlib import Path
import numpy as np
import pandas as pd
import streamlit as st

ROOT=Path(__file__).resolve().parents[1]; sys.path.insert(0,str(ROOT))
from dashboard.ui import ASSET_NAMES,GLOSSARY,asset_name,compact_table_html,days_remaining,friendly_reason,global_css,horizon_label,info_panel_html,model_label,money,opportunity_cards_html,percent,responsive_cards_html,risk_label,status_label
from src.config import load_config
from src.model_governance import status as governance_status
from src.portfolio.pdf_import import parse_pdf
from src.portfolio.reconciliation_service import reconcile_confirmed_snapshot,validate_confirmed_snapshot
from src.reporting.audit_21d import frozen_spec
from src.storage import Database
from src.spreads import load_spread_config
from src.app_status import data_status,runtime_identity
from src.discovery import describe_watch_changes,group_opportunities,start_simulated_tracking,stop_watching,watch_candidate,watchlist

st.set_page_config(page_title="Quiet Capital",page_icon="◌",layout="wide")
st.markdown(global_css(),unsafe_allow_html=True)
cfg=load_config(); db=Database(); db.initialize()
@st.cache_resource
def app_runtime(): return runtime_identity()
runtime=app_runtime()
meta_path=ROOT/"reports/current_real_experiment.json"; audit_path=ROOT/"reports/audit_cost_sensitivity.csv"
try:
    meta=json.loads(meta_path.read_text()); real_ready=meta.get("mode")=="real" and meta.get("model_configuration_hash")==frozen_spec(cfg)[1] and audit_path.exists()
except (FileNotFoundError,json.JSONDecodeError): meta={}; real_ready=False

def tables():
    with db.connect() as c:
        p=pd.read_sql_query("SELECT * FROM paper_positions ORDER BY entry_date DESC,id DESC",c)
        s=pd.read_sql_query("SELECT * FROM paper_signals ORDER BY observed_date DESC,id DESC",c)
        x=pd.read_sql_query("SELECT * FROM paper_status_changes WHERE run_id=(SELECT id FROM prospective_runs ORDER BY started_at DESC LIMIT 1) ORDER BY id",c)
    return p,s,x

def mark(p):
    path=ROOT/"data/processed/prospective_medium_panel.parquet"
    if p.empty or not path.exists(): return p.copy(),None
    latest=pd.read_parquet(path).sort_values("date").groupby("ticker").tail(1)[["ticker","close_eur","date"]]
    x=p.merge(latest,left_on="asset",right_on="ticker",how="left"); prices=latest.set_index("ticker").close_eur.to_dict()
    x["current_value_eur"]=x.allocation_eur*x.close_eur/x.entry_price; x["paper_pl_eur"]=x.current_value_eur-x.allocation_eur; x["paper_return"]=x.close_eur/x.entry_price-1
    x["spy_to_date"]=prices.get("SPY",pd.NA)/x.benchmark_spy_entry-1; x["urth_to_date"]=prices.get("URTH",pd.NA)/x.benchmark_urth_entry-1
    return x,latest.date.max()

def show_glossary():
    with st.expander("Asset Guide & Glossary",expanded=True):
        st.write("These US-listed ETFs are consistent research proxies. They may differ from the UCITS ETFs you actually own.")
        st.markdown(compact_table_html([{"Investment":n,"Ticker":t} for t,n in ASSET_NAMES.items()], ["75%","25%"]),unsafe_allow_html=True)
        for term,definition in GLOSSARY.items(): st.markdown(f"**{term}**  \n{definition}")

pages=["Today","Opportunities","Discover","Portfolio","Research","Performance","Ledger","Help / Glossary"]
requested_page=st.query_params.get("page")
if requested_page in pages:
    st.session_state["current_page"]=requested_page
    del st.query_params["page"]
page=st.sidebar.radio("Quiet Capital",pages,key="current_page")
st.sidebar.caption("Decision support only · Nothing is traded")
st.sidebar.caption("Paper mode · Real-money recommendations disabled")
with st.sidebar.expander("Asset Guide"):
    for ticker,name in ASSET_NAMES.items(): st.markdown(f"**{name}**  \nTicker: {ticker}")
    st.caption("US-listed research proxies; your UCITS ETFs may differ.")
paper,signals,changes=tables(); marked,marked_date=mark(paper)

if page=="Today":
    st.title("Today"); st.caption(f"Research snapshot · {date.today():%d %B %Y}")
    spread_cfg=load_spread_config(); status=data_status(db,spread_cfg)
    def shown_time(value):
        if not value: return "Not available"
        return pd.Timestamp(value).tz_convert("Europe/Berlin").strftime("%d %b %Y, %H:%M %Z")
    update=status["update"]
    actual_date=(update.get("price_latest_date") or update.get("market_date")) if update else None
    ingestion_time=status.get("ingestion_attempt")
    st.subheader("App and data status")
    st.markdown(info_panel_html([
        ("Running app build",runtime["build"]),("Dashboard started",shown_time(runtime["started_at"])),
        ("Latest successful data ingestion",shown_time(ingestion_time) if ingestion_time else "No successful ingestion recorded"),
        ("Latest successful paper-scanner run",shown_time(update["completed_at"]) if update else "No successful paper run recorded"),
        ("Prices latest date",status["datasets"]["Prices"]["latest_date"] or "Not available"),
        ("EUR/USD latest date",status["datasets"]["EUR/USD"]["latest_date"] or "Not available"),
        ("Macro latest date",status["datasets"]["Macro (FRED)"]["latest_date"] or "Not available"),
        ("Latest daily workflow attempt",shown_time(status.get("workflow_attempt"))),
        ("Latest spread attempt",shown_time(status["latest_attempt"])),
        ("Latest successful collection",shown_time(status["latest_success"])),
        ("Verified holding mappings",f'{status["verified_holdings"]} of {status["total_holdings"]}'),
        ("Usable in latest collection",f'{status["usable_latest"]} of {status["total_holdings"]} holdings'),
    ]),unsafe_allow_html=True)
    problems=[]
    if status["unmapped"]: problems.append(f'{status["unmapped"]} holdings do not have a verified Yahoo/Xetra mapping.')
    if status["failures"]: problems.append(f'{len(status["failures"])} requests in the latest collection were unavailable: '+"; ".join(sorted({r["reason"] for r in status["failures"] if r["reason"]}))+".")
    if status.get("provider_failures"): problems.append("Investment-data provider failures: " + "; ".join(f'{r.get("ticker", "dataset")}: {", ".join(r.get("warnings", []))}' for r in status["provider_failures"]))
    if status.get("paper_failure"): problems.append(f'Latest paper-scanner attempt failed: {status["paper_failure"].get("error_message") or status["paper_failure"].get("status")}.')
    if status["cooldown_until"]: problems.append(f'Collection is in cooldown until {shown_time(status["cooldown_until"])}.')
    st.warning(" ".join(problems) if problems else "No collection errors or active cooldown are recorded for the latest collection.")
    st.caption("Yahoo quotes are delayed; bid/ask freshness is unverified.")
    st.caption("Collected spreads are not yet used in model cost assumptions.")
    st.markdown("[View detailed spreads on the Portfolio page](?page=Portfolio#indicative-spreads)")
    st.markdown('<div class="action"><h2>Paper trading mode</h2><p>Quiet Capital is tracking hypothetical positions. No real money is recommended or invested.</p></div>',unsafe_allow_html=True)
    st.subheader("What changed?"); material=changes[changes.alert_required==1] if not changes.empty else changes
    if material.empty: st.info("No important signal change today. Existing paper positions continue to be tracked.")
    else:
        rows=[]
        for _,r in material.iterrows():
            q=signals[(signals.asset==r.asset)&(signals.horizon_days==r.horizon_days)]; risk=q.iloc[0].active_risk_flags if not q.empty else "none"
            rows.append({"Investment":asset_name(r.asset),"Previous":status_label(r.previous_status),"Now":status_label(r.new_status),"Time horizon":horizon_label(r.horizon_days),"Reason":friendly_reason(r.reason,r.asset,r.horizon_days,r.new_status,risk)})
        st.markdown(compact_table_html(rows,["18%","17%","17%","14%","34%"]),unsafe_allow_html=True)
        with st.expander("Technical details for today's changes"):
            tech=material.copy(); tech["Investment"]=tech.asset.map(asset_name); tech["Time horizon"]=tech.horizon_days.map(lambda v:horizon_label(v,True))
            st.dataframe(tech[["Investment","Time horizon","previous_status","new_status","reason"]].rename(columns={"previous_status":"Stored previous status","new_status":"Stored new status","reason":"Stored event reason"}),hide_index=True,use_container_width=True)
    st.subheader("Model status")
    try:
        g=governance_status(db); values=[("Current model","Medium-term Ridge v1"),("Last trained",pd.Timestamp(g["last_trained"]).strftime("%d %B %Y") if g.get("last_trained") else "Not available"),("Next scheduled retraining",pd.Timestamp(g["next_scheduled_retrain"]).strftime("%d %B %Y") if g.get("next_scheduled_retrain") else "Not available"),("Model validity",str(g.get("model_validity") or "Not available").capitalize()),("Challenger model",g.get("challenger") or "None"),("Mode","Paper trading"),("Real-money recommendations","Disabled")]
    except Exception: values=[(x,"Not available") for x in ["Current model","Last trained","Next scheduled retraining","Model validity","Challenger model"]]+[("Mode","Paper trading"),("Real-money recommendations","Disabled")]
    st.markdown(info_panel_html(values),unsafe_allow_html=True)
    st.caption("The statistical model combines several indicators while limiting overfitting. Technical name: Ridge. No challenger model currently exists.")
    st.subheader("Real-money decision")
    st.markdown(compact_table_html([{"Investment":"All investments","Recommendation":"No real-money recommendation","Amount":"Not active — paper mode","Expected return":"Not sufficiently validated","Historical downside":"See Opportunities"}]),unsafe_allow_html=True)
    st.info("A day with no trade can be a successful day. Profit goals never change forecasts or loosen risk rules.")

elif page=="Opportunities":
    st.title("Opportunities"); st.write("Historical research views in plain language. These are not real-money recommendations.")
    path=ROOT/"reports/medium_risk_current_scanner.csv"
    if not path.exists(): st.info("Medium-term research is still in progress. No completed scanner result is available yet.")
    else:
        scan=pd.read_csv(path); rows=[]
        for _,r in scan.iterrows():
            raw="Frozen forecast is outside the selected cohort." if r.forecast_status=="NOT TOP COHORT" else "Eligible research forecast represented by another simultaneous horizon or existing capital constraint."
            warning=risk_label(r.active_risk_flags)
            risk="Risk checks passed" if r.policy_2_result=="PASS" else "Blocked by risk rules"
            if warning != "No current warning": risk=warning
            rows.append({"Investment":asset_name(r.asset),"Current view":status_label(r.research_status),"Time horizon":horizon_label(r.horizon_days),"Risk":risk,"Historical evidence":f"Average excess return: {percent(r.forecast_excess)}; bad-case historical outcome: {percent(r.historical_p10)}","Why":friendly_reason(raw,r.asset,r.horizon_days,r.research_status,r.active_risk_flags),"Exact horizon":horizon_label(r.horizon_days,True),"P10":percent(r.historical_p10),"P5":percent(r.historical_p5),"Model score/reliability":r.confidence,"Model version":"Not included in current scanner artifact","Canonical status":r.research_status,"Canonical forecast status":r.forecast_status,"Sample size":"See model-comparison research","Benchmark":"Broad-market comparison","Risk-policy code":r.policy_2_result,"Technical risk flags":r.active_risk_flags,"Feature values":"Not included in current scanner artifact","As-of date":r.as_of_date})
        st.markdown(opportunity_cards_html(rows),unsafe_allow_html=True)
        st.caption("*Historical research result, not a forecast. Excess means the result above or below the broad-market comparison.")
        st.info("Bad-case historical outcome: roughly 1 in 10 comparable historical outcomes were worse.")
        with st.expander("Technical research details"):
            d=scan.copy(); d["Investment"]=d.asset.map(asset_name); d["Time horizon"]=d.horizon_days.map(lambda v:horizon_label(v,True)); d["Bad-case historical outcome (P10)"]=d.historical_p10.map(percent); d["Severe-case historical outcome (P5)"]=d.historical_p5.map(percent); d["Current risk rules (Policy 2)"]=d.policy_2_result.map(lambda v:"Pass" if v=="PASS" else "Fail"); d["Technical risk flags"]=d.active_risk_flags.map(risk_label)
            st.dataframe(d[["Investment","Time horizon","Bad-case historical outcome (P10)","Severe-case historical outcome (P5)","Current risk rules (Policy 2)","Technical risk flags","confidence"]].rename(columns={"confidence":"Model reliability"}),hide_index=True,use_container_width=True)
            st.caption("P10 and P5 are historical percentiles, not probabilities. Roughly 1 in 20 outcomes were worse than P5.")

elif page=="Discover":
    st.title("Discover"); st.write("A broad research screen for potentially interesting ETFs outside the frozen seven-investment validation universe.")
    st.info("Discovery results are research candidates, not real-money recommendations. They cannot enter the original paper experiment automatically.")
    universe=pd.read_csv(ROOT/"config/discovery_universe.csv")
    with db.connect() as c:
        run=pd.read_sql_query("SELECT * FROM discovery_runs ORDER BY market_date DESC,id DESC LIMIT 1",c)
        if run.empty: discovered=pd.DataFrame(); alerts=pd.DataFrame(); positions=pd.DataFrame(); discovery_horizons=pd.DataFrame()
        else:
            discovered=pd.read_sql_query("SELECT * FROM discovery_results WHERE run_id=? ORDER BY shortlisted DESC,rank",c,params=(int(run.iloc[0].id),))
            alerts=pd.read_sql_query("SELECT * FROM discovery_alerts WHERE run_id=?",c,params=(int(run.iloc[0].id),))
            positions=pd.read_sql_query("SELECT * FROM discovery_positions ORDER BY tracking_timestamp DESC",c)
            discovery_horizons=pd.read_sql_query("SELECT * FROM discovery_position_horizons",c)
    if run.empty:
        st.info("The first broad discovery scan has not run yet. The daily process will populate this page without changing the validation experiment.")
    else:
        summary=run.iloc[0]; cols=st.columns(5)
        cols[0].metric("Eligible ETFs scanned",int(summary.eligible_count)); cols[1].metric("Candidates shortlisted",int(summary.shortlist_count))
        cols[2].metric("New candidates today",len(alerts[alerts.alert_type=="NEW_SHORTLIST_ENTRY"])); cols[3].metric("High-risk candidates",int(((discovered.shortlisted==1)&(discovered.risk_level=="High")).sum())); cols[4].metric("Data-quality exclusions",int(summary.excluded_count))
        merged=discovered.merge(universe,on="ticker",how="left")
        region_options=sorted(merged.region.dropna().unique()); theme_options=sorted(merged.sector_theme.dropna().unique()); class_options=sorted(merged.asset_class.dropna().unique()); risk_options=sorted(merged.risk_level.dropna().unique())
        fcols=st.columns(4); region=fcols[0].multiselect("Region",region_options); theme=fcols[1].multiselect("Sector or theme",theme_options); asset_class=fcols[2].multiselect("Asset class",class_options); risk=fcols[3].multiselect("Risk level",risk_options)
        watched=watchlist(db); watched_tickers=set(watched.ticker) if not watched.empty else set()
        view=st.radio("Show",["Current shortlist","Watchlist"],horizontal=True)
        shown=merged[merged.shortlisted==1].copy() if view=="Current shortlist" else merged[merged.ticker.isin(watched_tickers)].copy()
        if region: shown=shown[shown.region.isin(region)]
        if theme: shown=shown[shown.sector_theme.isin(theme)]
        if asset_class: shown=shown[shown.asset_class.isin(asset_class)]
        if risk: shown=shown[shown.risk_level.isin(risk)]
        def evidence(e):
            if not e or e.get("sample_size",0)<10: return "Limited comparable history"
            p=e.get("outperformance_probability"); med=e.get("median_excess")
            if p is not None and p>=.6 and med is not None and med>0: return "Historically promising"
            if p is not None and p>=.5: return "Mixed but worth watching"
            return "No clear historical advantage"
        def comparison_reason(rows):
            if len(rows)<2: return "This opportunity currently has one qualifying ETF."
            a,b=rows.iloc[0],rows.iloc[1]; gap=float(a.discovery_score or 0)-float(b.discovery_score or 0)
            if abs(gap)<2: return "The evidence does not clearly distinguish these ETFs. They represent very similar exposures."
            af=json.loads(a.feature_snapshot_json or "{}"); bf=json.loads(b.feature_snapshot_json or "{}")
            if af.get("momentum_126d",0)>bf.get("momentum_126d",0) and a.risk_level==b.risk_level:
                return f"{a.ticker} currently ranks above {b.ticker} because its medium-term evidence is slightly stronger while historical downside is similar."
            return f"{a.ticker} currently ranks above {b.ticker} because its combined discovery evidence is stronger under the unchanged research screen."
        active_tickers=set(positions.loc[positions.status=="ACTIVE","ticker"]) if not positions.empty else set()
        for group in group_opportunities(shown):
            r=pd.Series(group["leading"]); alternatives=group["alternatives"]
            e63=json.loads(r.evidence_63d_json) if pd.notna(r.evidence_63d_json) else {}; e126=json.loads(r.evidence_126d_json) if pd.notna(r.evidence_126d_json) else {}
            title=group["opportunity_name"]
            with st.container(border=True):
                st.subheader(title)
                if alternatives: st.caption(f"{len(group['candidates'])} qualifying ETFs · Leading research candidate: {r.friendly_name} ({r.ticker}) · Alternatives: {', '.join(x['ticker'] for x in alternatives)}")
                else: st.caption(f"Research candidate: {r.friendly_name} ({r.ticker})")
                current_status=r.status if r.shortlisted else "No longer in the current shortlist"
                st.markdown(info_panel_html([("Current view",current_status),("Current risk",r.risk_level),("About 3 months",evidence(e63)),("About 6 months",evidence(e126))]),unsafe_allow_html=True)
                st.write(r.explanation)
                actions=st.columns(3)
                compare_label="Compare ETFs" if alternatives else "Compare"
                if actions[0].button(compare_label,key=f"compare-button-{r.ticker}",use_container_width=True): st.session_state[f"compare-{r.ticker}"]=not st.session_state.get(f"compare-{r.ticker}",False)
                is_watched=r.ticker in watched_tickers
                if actions[1].button("Watching" if is_watched else "Watch",key=f"watch-{r.ticker}",use_container_width=True):
                    if is_watched: stop_watching(r.ticker,db); st.success("Stopped watching. Watch history was preserved.")
                    else: watch_candidate(r.ticker,db); st.success("Watching. No simulated position was opened.")
                    st.rerun()
                tracking=r.ticker in active_tickers
                if actions[2].button("Simulated tracking active" if tracking else "Start simulated tracking",key=f"track-{r.ticker}",disabled=tracking,use_container_width=True): st.session_state[f"confirm-track-{r.ticker}"]=True
                if st.session_state.get(f"compare-{r.ticker}"):
                    st.markdown("#### Compact ETF comparison")
                    st.write(comparison_reason(pd.DataFrame(group["candidates"])))
                    compare_cards=[]
                    for item in group["candidates"]:
                        q=pd.Series(item); q63=json.loads(q.evidence_63d_json or "{}"); q126=json.loads(q.evidence_126d_json or "{}"); quality=json.loads(q.data_quality_json or "{}"); features=json.loads(q.feature_snapshot_json or "{}")
                        volume=quality.get("median_dollar_volume_63d")
                        overlap=q.overlap_warning or "Exact constituent overlap unavailable; no approximate category warning identified."
                        compare_cards.append({"Investment":f"{q.friendly_name} ({q.ticker})","Discovery rank":q["rank"],"About 3 months":evidence(q63),"About 6 months":evidence(q126),"Historical excess return":f"3 months: {percent(q63.get('median_excess'))}; 6 months: {percent(q126.get('median_excess'))}","Bad-case historical outcome":f"3 months: {percent(q63.get('p10'))}; 6 months: {percent(q126.get('p10'))}","Severe-case historical outcome":f"3 months: {percent(q63.get('p5'))}; 6 months: {percent(q126.get('p5'))}","Current drawdown":percent(features.get("drawdown")),"Volatility / risk level":q.risk_level,"Liquidity proxy":f"Median daily traded value ${volume:,.0f}" if volume else "Unavailable from current data source","Expense ratio":"Unavailable from current data source","Existing-portfolio overlap":q.portfolio_warning or "No category-level warning identified","Constituent overlap":overlap,"Data quality":q.eligibility,"Current discovery status":q.status})
                    st.markdown(responsive_cards_html(compare_cards,"Investment",["Discovery rank","About 3 months","About 6 months","Historical excess return","Bad-case historical outcome","Severe-case historical outcome","Current drawdown","Volatility / risk level","Liquidity proxy","Expense ratio","Existing-portfolio overlap","Constituent overlap","Data quality","Current discovery status"]),unsafe_allow_html=True)
                if st.session_state.get(f"confirm-track-{r.ticker}"):
                    st.markdown(f"#### Start simulated tracking?\n**{title} — {r.ticker}**")
                    st.write("This will record a pretend-money position from today's available market price. No real money will be invested. This position will be tracked separately from the original seven-investment experiment.")
                    st.write("Tracking horizon: About 3 months and About 6 months. One simulated €1,000 research position supports both evaluations.")
                    yes,no=st.columns(2)
                    if yes.button("Start tracking",key=f"confirm-yes-{r.ticker}",use_container_width=True):
                        try: start_simulated_tracking(r.ticker,db); st.session_state[f"confirm-track-{r.ticker}"]=False; st.success("Simulated tracking started. No real-money trade was created."); st.rerun()
                        except ValueError as exc: st.error(str(exc))
                    if no.button("Cancel",key=f"confirm-no-{r.ticker}",use_container_width=True): st.session_state[f"confirm-track-{r.ticker}"]=False; st.rerun()
        if shown.empty: st.info("No research candidates match the current filters.")
        if view=="Watchlist" and not watched.empty:
            st.subheader("Watchlist changes")
            watch_cards=[]
            names=universe.set_index("ticker").friendly_name.to_dict()
            for _,w in watched.iterrows(): watch_cards.append({"Investment":f"{names.get(w.ticker,w.ticker)} ({w.ticker})","Why I am watching it":w.reason_at_start,"Current discovery status":w.current_status if w.current_shortlisted else "No longer in the current shortlist","Current risk":w.current_risk,"Date added":w.first_watched_at,"What changed since it was added":"; ".join(describe_watch_changes(w))})
            st.markdown(responsive_cards_html(watch_cards,"Investment",["Why I am watching it","Current discovery status","Current risk","Date added","What changed since it was added"]),unsafe_allow_html=True)
        st.caption("Friendly names are primary. Tickers, exact horizons, frozen-model research inputs, and tail outcomes remain inside details.")
        with st.expander("Eligibility and exclusions"):
            counts=merged.eligibility.value_counts().to_dict(); st.markdown(compact_table_html([{"Eligibility status":k.title(),"ETFs":v} for k,v in counts.items()], ["70%","30%"]),unsafe_allow_html=True)
            st.caption("Free V1 data provide adjusted prices and trading volume when available. Reliable fund size and bid-ask history are not available and are shown as limitations, never estimated.")
        with st.expander("Discovery simulated tracking"):
            st.caption("Separate from the original validation experiment. Results are never combined into one headline.")
            if positions.empty: st.write("No discovery simulated tracking is active.")
            else:
                current_discovery_prices={}
                price_path=ROOT/"data/processed/discovery_prices.parquet"
                if price_path.exists():
                    raw_prices=pd.read_parquet(price_path); raw_prices["date"]=pd.to_datetime(raw_prices.date)
                    latest_fx=raw_prices[raw_prices.ticker=="EURUSD=X"].sort_values("date").tail(1)
                    if not latest_fx.empty:
                        usd_per_eur=float(latest_fx.iloc[0].close)
                        current_discovery_prices={x.ticker:(float(x.close)/usd_per_eur,pd.Timestamp(x.date)) for _,x in raw_prices[raw_prices.ticker!="EURUSD=X"].sort_values("date").groupby("ticker").tail(1).iterrows()}
                position_cards=[]
                latest_by_ticker=merged.set_index("ticker") if not merged.empty else pd.DataFrame()
                for _,p in positions.iterrows():
                    marked_price,marked_date=current_discovery_prices.get(p.ticker,(None,None)); simulated_result="Awaiting the next available market mark"
                    if marked_price is not None:
                        return_now=marked_price/float(p.entry_price)-1; simulated_result=f"{money(1000*return_now)} / {percent(return_now)}"
                    elapsed=int(np.busday_count(str(p.market_date),str(marked_date.date()))) if marked_date is not None else 0
                    horizon_rows=discovery_horizons[discovery_horizons.position_id==p.id].set_index("target_trading_days") if not discovery_horizons.empty else pd.DataFrame()
                    h3=horizon_rows.loc[63,"status"].title() if not horizon_rows.empty and 63 in horizon_rows.index else "Tracking"
                    h6=horizon_rows.loc[126,"status"].title() if not horizon_rows.empty and 126 in horizon_rows.index else "Tracking"
                    remaining=f"About 3 months: {max(0,63-elapsed)} trading days; About 6 months: {max(0,126-elapsed)} trading days"
                    current_risk=latest_by_ticker.loc[p.ticker,"risk_level"] if not latest_by_ticker.empty and p.ticker in latest_by_ticker.index else p.risk_at_entry
                    risk_change="No material change recorded" if current_risk==p.risk_at_entry else f"Changed from {p.risk_at_entry} to {current_risk}"
                    position_cards.append({"Investment":f"{p.friendly_name} ({p.ticker})","Status":"Simulated tracking active" if p.status=="ACTIVE" else "Stopped early","Started":p.tracking_timestamp,"Entry price":money(p.entry_price,2),"Current simulated result":simulated_result,"Time remaining":remaining,"Risk changes":risk_change,"About 3-month outcome status":h3,"About 6-month outcome status":h6,"Research value":"Simulated €1,000 research position","Market date":p.market_date,"Entry-price source":p.entry_price_source,"Benchmark":p.benchmark})
                st.markdown(responsive_cards_html(position_cards,"Investment",["Status","Started","Entry price","Current simulated result","Time remaining","Risk changes","About 3-month outcome status","About 6-month outcome status","Research value"],["Market date","Entry-price source","Benchmark"]),unsafe_allow_html=True)

elif page=="Portfolio":
    st.title("Portfolio")
    st.markdown('<div class="action"><h2>Upload your actual Trade Republic portfolio statement</h2><p>Upload your latest portfolio or net-worth PDF. Quiet Capital extracts your actual holdings and cash. Nothing is traded or sent to Trade Republic.</p></div>',unsafe_allow_html=True)
    uploaded=st.file_uploader("Upload Trade Republic statement",type=["pdf"],help="Parsed locally; no brokerage credentials are requested.")
    if uploaded:
        parsed=parse_pdf(uploaded); st.subheader("Review extracted portfolio")
        st.caption(f"Document type: {parsed.get('document_type','Unknown').title()}")
        for warning in parsed["warnings"]: st.warning(warning)
        holdings=pd.DataFrame(parsed["holdings"])
        if holdings.empty:
            st.error("We could read this Trade Republic statement, but could not reliably identify its individual holdings. No portfolio data has been changed."); edited=holdings
        else:
            holdings["Investment"]=holdings.apply(lambda r:r.get("security_name") or asset_name(r.get("ticker")),axis=1)
            summary=[]
            total=sum(float(v or 0) for v in holdings.market_value_eur)
            for _,r in holdings.iterrows(): summary.append({"Investment":r.Investment,"Units":r.get("quantity") if pd.notna(r.get("quantity")) else "Needs review","Current value":money(r.get("market_value_eur"),2),"Portfolio share":percent(float(r.get("market_value_eur") or 0)/total if total else 0)})
            st.markdown(compact_table_html(summary,["40%","18%","23%","19%"]),unsafe_allow_html=True)
            with st.expander("Review or correct extracted technical fields"):
                st.caption("Only fields marked Needs review or Missing require attention. Identifiers remain under More details.")
                confidence_rows=[]
                for _,row in holdings.iterrows():
                    field_confidence=row.get("field_confidence") or {}
                    confidence_rows.append({"Investment":row.Investment,"ISIN":row.get("isin"),"Ticker":row.get("ticker") or "Missing","Units confidence":field_confidence.get("quantity","Missing"),"Value confidence":field_confidence.get("market_value_eur","Needs review")})
                st.markdown(responsive_cards_html(confidence_rows,"Investment",[],["ISIN","Ticker","Units confidence","Value confidence"]),unsafe_allow_html=True)
                editable=holdings.drop(columns=["field_confidence","currency"],errors="ignore")
                edited=st.data_editor(editable,hide_index=True,use_container_width=True,num_rows="fixed",disabled=["confidence"])
        reviewed=parsed.copy(); reviewed["holdings"]=edited.drop(columns=["Investment"],errors="ignore").where(pd.notna(edited),None).to_dict("records")
        cash_label="Reviewed cash (€)" if parsed.get("cash_included") else "Manual cash (€) — not included in statement"
        reviewed["cash_eur"]=st.number_input(cash_label,min_value=0.0,value=float(parsed.get("cash_eur") or 0),step=1.0)
        reviewed["cash_source"]="extracted/reviewed" if parsed.get("cash_included") else "manual"
        securities_value=sum(float(h.get("market_value_eur") or 0) for h in reviewed["holdings"])
        crypto_value=float(reviewed.get("crypto_value_eur") or 0)
        reviewed["securities_value_eur"]=securities_value; reviewed["total_value_eur"]=securities_value+crypto_value+reviewed["cash_eur"] if reviewed["holdings"] else None
        reported=reviewed.get("reported_securities_value_eur"); difference=securities_value-reported if reported is not None else None
        reviewed["reconciliation_difference_eur"]=difference
        reviewed["material_reconciliation_mismatch"]=difference is not None and abs(difference)>reviewed.get("reconciliation_tolerance_eur",.02)
        st.metric("Securities value",money(securities_value,2) if reviewed["holdings"] else "Not available")
        if reported is not None:
            st.caption(f"Reported securities value: {money(reported,2)} · Sum of extracted holdings: {money(securities_value,2)} · Difference: {money(difference,2)}")
        if reviewed["holdings"]:
            if reviewed.get("crypto_value_eur") is not None: st.caption(f"Crypto wallet subtotal: {money(crypto_value,2)}")
            st.metric("Calculated total financial assets",money(reviewed["total_value_eur"],2))
        else: st.metric("Calculated total financial assets","Not available")
        errors=validate_confirmed_snapshot(reviewed)
        for error in errors: st.error(error)
        confirmed=st.checkbox("I reviewed the extracted date, cash, holdings, and uncertain fields")
        if st.button("Confirm portfolio snapshot",disabled=not confirmed or bool(errors)): reconcile_confirmed_snapshot(reviewed,confirmed,db); st.success("Portfolio snapshot confirmed. No holdings or trades were changed.")
        with st.expander("Technical extraction diagnostics"):
            st.json(parsed.get("diagnostics",{}))
    with db.connect() as c:
        snapshots=pd.read_sql_query("SELECT * FROM portfolio_snapshots ORDER BY id DESC",c); actual=pd.read_sql_query("SELECT * FROM holdings WHERE snapshot_id=(SELECT MAX(id) FROM portfolio_snapshots)",c)
    if snapshots.empty: st.info("Actual portfolio not reconciled yet. Upload a Trade Republic statement above.")
    else:
        st.subheader("Actual Portfolio"); st.metric("Confirmed portfolio value",money(snapshots.iloc[0].total_value_eur))
        if not actual.empty:
            total=float(snapshots.iloc[0].total_value_eur or 0); actual_rows=[]
            for _,r in actual.iterrows():
                actual_rows.append({"Investment":r.security_name or asset_name(r.ticker),"Units":r.quantity,"Current value":money(r.market_value_eur,2),"Portfolio share":percent(float(r.market_value_eur or 0)/total if total else 0),"Ticker":r.ticker,"ISIN":r.isin,"Price":money(r.market_value_eur/r.quantity,2) if r.quantity else "Not available"})
            st.markdown(compact_table_html([{k:r[k] for k in ["Investment","Units","Current value","Portfolio share"]} for r in actual_rows],["40%","18%","23%","19%"]),unsafe_allow_html=True)
            with st.expander("Holding identifiers and reconciliation details"):
                st.markdown(responsive_cards_html(actual_rows,"Investment",[],["Ticker","ISIN","Price"]),unsafe_allow_html=True)
        st.markdown('<span id="indicative-spreads"></span>',unsafe_allow_html=True)
        st.subheader("Indicative spreads")
        st.caption("Stored Yahoo/Xetra research observations only. The dashboard never requests Yahoo data during a rerender.")
        spread_cfg=load_spread_config(); mapped={x["isin"]:x for x in spread_cfg["mappings"] if x.get("verified")}; unresolved={x["isin"]:x["reason"] for x in spread_cfg.get("unresolved",[])}
        spread_rows=[]; now_utc=datetime.now(timezone.utc)
        for r in db.latest_spreads():
            mapping=mapped.get(r["isin"])
            if not mapping:
                spread_rows.append({"Investment":r["security_name"],"Listing":"Mapping needed","Latest attempt":"Not attempted","Latest observation":"Unavailable","Source":"Unavailable","Spread":"Unavailable","Quality":unresolved.get(r["isin"],"A verified European listing mapping is needed")}); continue
            observed=r["observation_collected_at_utc"]
            stale=bool(observed and (now_utc-datetime.fromisoformat(observed)).total_seconds()>float(spread_cfg["stale_after_seconds"]))
            quality=(r["observation_reason"] or "No usable observation yet")+("; stored value is stale" if stale else "")
            spread=(f'{r["spread_currency"]:.4f} {r["currency"]} · {r["spread_percent"]:.3f}% · {r["spread_bps"]:.1f} bps' if r["spread_currency"] is not None else "Unavailable")
            spread_rows.append({"Investment":r["security_name"],"Listing":f'{mapping["yahoo_ticker"]} · {mapping["exchange"]}',
              "Latest attempt":f'{r["attempt_collected_at_utc"] or "Never"} · {r["attempt_status"] or "No attempt"}',
              "Latest observation":observed or "No usable observation","Source":r["source_label"] or spread_cfg["source_label"],
              "Spread":spread,"Quality":quality if observed else (r["attempt_reason"] or "No collection attempt yet")})
        if spread_rows: st.markdown(responsive_cards_html(spread_rows,"Investment",["Listing","Spread","Quality"],["Latest attempt","Latest observation","Source"]),unsafe_allow_html=True)
        else: st.info("No confirmed holdings are available for spread mapping.")
    st.subheader("Paper Portfolio"); st.caption("Hypothetical positions generated automatically by the fixed research process. They are separate from actual holdings and use no real money.")
    if marked.empty: st.info("No paper positions have been opened yet. They will appear automatically when the fixed research and risk conditions are met.")
    else:
        cards=[]
        for _,r in marked.iterrows(): cards.append({"Investment":asset_name(r.asset),"Paper status":status_label(r.current_status),"Entry date":r.entry_date,"Current paper result":f"{money(r.paper_pl_eur)} / {percent(r.paper_return)}","Time remaining":f"{days_remaining(r.entry_date,r.horizon_days)} trading days","Risk":status_label(r.risk_status),"Entry price":money(r.entry_price,2),"Current price":money(r.close_eur,2),"S&P 500 comparison":percent(r.spy_to_date),"MSCI World comparison":percent(r.urth_to_date),"Paper allocation":money(r.allocation_eur),"Exact horizon":horizon_label(r.horizon_days,True),"Model version":r.get("model_version","Not available")})
        st.markdown(responsive_cards_html(cards,"Investment",["Paper status","Entry date","Current paper result","Time remaining","Risk"],["Entry price","Current price","S&P 500 comparison","MSCI World comparison","Paper allocation","Exact horizon","Model version"]),unsafe_allow_html=True); st.caption(f"Marked using available prices dated {pd.Timestamp(marked_date):%d %B %Y}.")

elif page=="Research":
    st.title("Research"); st.write("Technical evidence and implementation details live here. Historical tests never authorize a trade by themselves.")
    st.subheader("Model approaches")
    for name,explanation in [("Statistical model (Ridge)","A relatively simple statistical model that combines several indicators while limiting overfitting."),("Flexible machine-learning model (Gradient boosting)","A more flexible model that can capture nonlinear relationships. It did not outperform the simpler model reliably."),("Momentum rule","A simple strategy that favors investments that have recently been performing relatively well."),("Historical-average baseline","A deliberately simple comparison that assumes future returns resemble the historical average.")]: st.markdown(f"**{name}**  \n{explanation}")
    path=ROOT/"reports/medium_model_results.csv"
    if path.exists():
        results=pd.read_csv(path); cards=[]
        for _,r in results.iterrows(): cards.append({"Model approach":model_label(r.model),"Time horizon":horizon_label(r.horizon_days),"Average excess return":percent(r.mean_excess),"Median excess return":percent(r.median_excess),"Bad-case result":percent(r.p10_return),"Outperformance rate":percent(r.probability_outperform,0),"Exact horizon":horizon_label(r.horizon_days,True),"Sample size":r.observations,"Correlation":r.correlation,"Directional accuracy":percent(r.directional_accuracy),"Selected":r.selected,"Worst historical result":percent(r.worst_return)})
        st.markdown(responsive_cards_html(cards,"Model approach",["Time horizon","Average excess return","Median excess return","Bad-case result","Outperformance rate"],["Exact horizon","Sample size","Correlation","Directional accuracy","Selected","Worst historical result"]),unsafe_allow_html=True)
    else: st.info("No completed medium-term research artifact is available.")
    with st.expander("Current risk rules (technical name: Policy 2)"):
        st.write("The automatic risk block rejects a setup when the broad US market is in a sustained decline or the investment has fallen more than 20% from its recent high. Rules and thresholds are frozen.")
    show_glossary()

elif page=="Performance":
    st.title("Performance"); st.subheader("Historical backtest research")
    st.warning("These numbers aggregate repeated hypothetical historical research allocations. They are not money earned by your portfolio.")
    if real_ready:
        audit=pd.read_csv(audit_path); audit=audit[(audit.allocation_eur==1000)&(audit.benchmark=="SPY")]
        cards=[]
        for _,r in audit.iterrows(): cards.append({"Research approach":model_label(r.model),"Historical strategy result":money(r.model_net_profit_eur),"Passive comparison":money(r.passive_net_profit_eur),"Historical relative value":money(r.tactical_net_value_added_eur),"Relative value per allocation":percent(r.tactical_net_value_added_pct_per_trade),"Benchmark":r.benchmark,"Signals":r.signals,"Model costs":money(r.model_costs_eur),"Passive costs":money(r.passive_costs_eur),"Hypothetical allocation":money(r.allocation_eur)})
        st.markdown(responsive_cards_html(cards,"Research approach",["Historical strategy result","Passive comparison","Historical relative value","Relative value per allocation"],["Benchmark","Signals","Model costs","Passive costs","Hypothetical allocation"]),unsafe_allow_html=True); st.caption("Each card repeatedly applies a hypothetical €1,000 allocation to every qualifying historical signal, including simulated costs. Historical only — not a forecast.")
    else: st.info("Validated historical audit totals are not currently available. No placeholder profit is shown.")
    st.subheader("Actual portfolio performance"); st.info("Actual performance is unavailable until a confirmed portfolio history exists. Upload a Trade Republic statement on the Portfolio page.")
    st.subheader("Fun Money"); st.metric("Fun money available","€0"); st.caption("No realized model-assisted gains have been recorded yet. Paper profits and backtest results never count as Fun Money.")
    st.subheader("Prospective paper performance")
    completed=marked[marked.closed_date.notna()] if not marked.empty else marked; openp=marked[marked.closed_date.isna()] if not marked.empty else marked
    if marked.empty: st.info("No paper positions have been opened yet, so prospective performance cannot be marked.")
    elif "paper_return" not in marked: st.info("Paper positions exist, but current-price marking is unavailable because the local price panel is missing.")
    else:
        allocation=openp.allocation_eur.sum(); result=openp.paper_pl_eur.sum(); spy=(openp.spy_to_date*openp.allocation_eur).sum(); urth=(openp.urth_to_date*openp.allocation_eur).sum(); relative=(result-spy)/allocation if allocation else 0
        cols=st.columns(3); cols[0].metric("Open paper positions",len(openp)); cols[1].metric("Completed paper positions",len(completed)); cols[2].metric("Current unrealized paper result",f"{money(result)} / {percent(result/allocation if allocation else 0)}")
        cols=st.columns(3); cols[0].metric("S&P 500 comparison to date",f"{money(spy)} / {percent(spy/allocation if allocation else 0)}"); cols[1].metric("MSCI World comparison to date",f"{money(urth)} / {percent(urth/allocation if allocation else 0)}"); cols[2].metric("Current relative paper performance",percent(relative))
        position_cards=[]
        for _,r in openp.iterrows(): position_cards.append({"Investment":asset_name(r.asset),"Status":status_label(r.current_status),"Current paper result":f"{money(r.paper_pl_eur)} / {percent(r.paper_return)}","Time remaining":f"{days_remaining(r.entry_date,r.horizon_days)} trading days","Risk":status_label(r.risk_status),"Entry date":r.entry_date,"Entry price":money(r.entry_price,2),"Current price":money(r.close_eur,2),"Paper allocation":money(r.allocation_eur),"Exact horizon":horizon_label(r.horizon_days,True),"S&P 500 comparison":percent(r.spy_to_date),"MSCI World comparison":percent(r.urth_to_date)})
        if position_cards:
            st.markdown("#### Open paper positions")
            st.markdown(responsive_cards_html(position_cards,"Investment",["Status","Current paper result","Time remaining","Risk"],["Entry date","Entry price","Current price","Paper allocation","Exact horizon","S&P 500 comparison","MSCI World comparison"]),unsafe_allow_html=True)
        st.warning(f"These positions are still open. No final prospective performance conclusion can be drawn yet. Values use prices dated {pd.Timestamp(marked_date):%d %B %Y}.")

elif page=="Ledger":
    st.title("Ledger"); signal_tab,position_tab,real_tab=st.tabs(["Paper Signal Ledger","Paper Position Ledger","Real Trade Ledger"])
    with signal_tab:
        if signals.empty: st.info("No paper signals have been recorded yet.")
        else:
            cards=[]
            for _,r in signals.iterrows(): cards.append({"Investment":asset_name(r.asset),"Date":r.observed_date,"Status":status_label(r.recommendation),"Time horizon":horizon_label(r.horizon_days),"Risk":"Risk checks passed" if r.policy_2_pass else "Blocked by risk rules","Result/status":status_label(r.recommendation),"Reason":friendly_reason(r.decision_reason,r.asset,r.horizon_days,r.recommendation,r.active_risk_flags),"Model version":r.model_version if pd.notna(r.model_version) else "Not available","Exact horizon":horizon_label(r.horizon_days,True),"Canonical status":r.recommendation,"Risk-policy code":r.policy_2_pass,"Risk flags":r.active_risk_flags})
            st.markdown(responsive_cards_html(cards,"Investment",["Date","Status","Time horizon","Risk","Result/status"],["Reason","Model version","Exact horizon","Canonical status","Risk-policy code","Risk flags"]),unsafe_allow_html=True)
    with position_tab:
        if marked.empty: st.info("No paper positions have been opened yet.")
        else:
            rows=[]
            for _,r in marked.iterrows():
                remaining=days_remaining(r.entry_date,r.horizon_days)
                rows.append({"Investment":asset_name(r.asset),"Entry date":r.entry_date,"Status":status_label(r.current_status),"Time horizon":horizon_label(r.horizon_days),"Result/status":f"{money(r.paper_pl_eur)} / {percent(r.paper_return)}","Risk":status_label(r.risk_status),"Entry price":money(r.entry_price,2),"Paper allocation":money(r.allocation_eur),"Trading days remaining":remaining if remaining is not None else "Not available","S&P 500 comparison":percent(r.spy_to_date),"MSCI World comparison":percent(r.urth_to_date),"Exact horizon":horizon_label(r.horizon_days,True),"Model version":r.get("model_version","Not available")})
            st.markdown(responsive_cards_html(rows,"Investment",["Entry date","Status","Time horizon","Result/status","Risk"],["Entry price","Paper allocation","Trading days remaining","S&P 500 comparison","MSCI World comparison","Exact horizon","Model version"]),unsafe_allow_html=True); st.caption("Paper results are hypothetical and use no real money.")
    with real_tab:
        with db.connect() as c: trades=pd.read_sql_query("SELECT trade_date,ticker,isin,amount_eur,horizon_days,original_thesis,status,realized_result_eur FROM trades ORDER BY id DESC",c)
        if trades.empty: st.info("No real model-assisted trades have been recorded. This is legitimate while real-money recommendations are disabled.")
        else:
            cards=[]
            for _,r in trades.iterrows(): cards.append({"Investment":asset_name(r.ticker),"Date":r.trade_date,"Status":status_label(r.status),"Time horizon":horizon_label(r.horizon_days),"Result/status":money(r.realized_result_eur),"Reason":r.original_thesis,"ISIN":r.isin,"Amount":money(r.amount_eur),"Exact horizon":horizon_label(r.horizon_days,True),"Canonical status":r.status})
            st.markdown(responsive_cards_html(cards,"Investment",["Date","Status","Time horizon","Result/status"],["Reason","ISIN","Amount","Exact horizon","Canonical status"]),unsafe_allow_html=True)

else:
    st.title("Help / Glossary"); st.write("Plain-language definitions for everything you need to use Quiet Capital."); show_glossary()
