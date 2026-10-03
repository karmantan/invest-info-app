from __future__ import annotations
import pandas as pd
import streamlit as st
from common import demo_banner, eur, get_settings, get_store, pct, portfolio_context, signed_eur
from src.portfolio.pdf_import import parse_pdf


def _upload() -> None:
    st.subheader("Upload a new statement")
    st.caption("In the Trade Republic app: Profile → Documents → Account statements → **Vermögensübersicht / Wealth overview** (PDF). "
               "The file is read on this server and only the numbers are kept — no Trade Republic login is ever needed.")
    uploaded = st.file_uploader("Trade Republic PDF", type=["pdf"], label_visibility="collapsed")
    if not uploaded: return
    parsed = parse_pdf(uploaded)
    for w in parsed.get("warnings", []): st.warning(w)
    if not parsed.get("holdings"):
        st.error("No holdings could be read from this PDF. Is it the wealth overview (Vermögensübersicht)? Nothing was saved."); return
    c = st.columns(4)
    c[0].metric("Statement date", parsed.get("snapshot_date") or "Unknown")
    c[1].metric("Brokerage", eur(parsed.get("securities_value_eur"), 2))
    c[2].metric("Cash", eur(parsed.get("cash_eur"), 2))
    c[3].metric("Crypto", eur(parsed.get("crypto_value_eur"), 2))
    if parsed.get("reconciliation_difference_eur") is not None and not parsed.get("material_reconciliation_mismatch"):
        st.success(f"All {len(parsed['holdings'])} positions add up to the statement's brokerage total, and brokerage + crypto + cash match the reported total of {eur(parsed.get('reported_total_financial_assets_eur'), 2)}.")
    st.markdown("**Check the positions** (you can correct a value by clicking on it):")
    frame = pd.DataFrame(parsed["holdings"])[["security_name", "isin", "quantity", "displayed_price", "market_value_eur"]]
    edited = st.data_editor(frame, hide_index=True, width="stretch", num_rows="dynamic",
                            column_config={"security_name": "Name", "isin": "ISIN", "quantity": st.column_config.NumberColumn("Units", format="%.6f"),
                                           "displayed_price": st.column_config.NumberColumn("Price", format="€%.2f"), "market_value_eur": st.column_config.NumberColumn("Value", format="€%.2f")})
    crypto = parsed.get("crypto_holdings") or []
    if crypto:
        st.caption("Crypto: " + ", ".join(f"{c['name']} {eur(c['market_value_eur'], 2)}" for c in crypto))
    cash = st.number_input("Cash (€)", min_value=0.0, value=float(parsed.get("cash_eur") or 0.0), step=1.0)
    if parsed.get("snapshot_date") is None: st.error("The statement date could not be read, so it cannot be saved."); return
    if st.button("Save this statement", type="primary"):
        reviewed = {**parsed, "holdings": edited.dropna(subset=["market_value_eur"]).where(pd.notna(edited), None).to_dict("records"), "cash_eur": cash, "source_file": uploaded.name}
        get_store().save_statement(reviewed)
        st.success("Saved. The Overview page now uses this statement."); st.rerun()


def render() -> None:
    st.title("My Trade Republic")
    demo_banner()
    settings = get_settings()
    ctx = portfolio_context(settings)
    if ctx is not None:
        st.subheader(f"Statement of {ctx['date']}")
        live = ctx["live"].copy()
        live["Change since statement"] = live.live_value - live.statement_value
        total = float(live.live_value.sum()) or 1.0
        show = pd.DataFrame({"Holding": live["name"], "Type": live.kind, "Units": live.quantity, "On statement": live.statement_value, "Now (estimate)": live.live_value,
                             "Change": live["Change since statement"], "Share": live.live_value / total, "ISIN": live["isin"], "Price source": live.symbol.fillna("—"), "Price check": live.price_status})
        money = st.column_config.NumberColumn(format="€%.2f")
        st.dataframe(show.sort_values("Now (estimate)", ascending=False), hide_index=True, width="stretch", height=36 * (len(show) + 1) + 4,
                     column_config={"On statement": money, "Now (estimate)": money, "Change": money, "Units": st.column_config.NumberColumn(format="%.4f"),
                                    "Share": st.column_config.ProgressColumn("Share of invested", format="percent", min_value=0, max_value=1)})
        st.caption(f"Cash {eur(ctx['cash'], 2)} · Brokerage on statement {eur(ctx['brokerage'], 2)} · Crypto on statement {eur(ctx['crypto_value'], 2)} · "
                   f"Change since statement {signed_eur(float(live['Change since statement'].sum()))}. \"Price check\" flags holdings whose Yahoo price does not match the statement; they stay at the statement value.")
    else:
        st.info("No statement saved yet.")
    _upload()
    snaps = get_store().snapshots()
    if not snaps.empty:
        with st.expander(f"Saved statements ({len(snaps)})"):
            st.dataframe(snaps[["snapshot_date", "total_value_eur", "brokerage_eur", "cash_eur", "crypto_eur", "source_file", "uploaded_at"]]
                         .rename(columns={"snapshot_date": "Date", "total_value_eur": "Total", "brokerage_eur": "Brokerage", "cash_eur": "Cash", "crypto_eur": "Crypto", "source_file": "File", "uploaded_at": "Uploaded"}),
                         hide_index=True, width="stretch")
            if len(snaps) > 1:
                st.line_chart(snaps.set_index("snapshot_date").sort_index()[["total_value_eur"]].rename(columns={"total_value_eur": "Total on statement (includes deposits)"}))
            sid = st.selectbox("Delete a statement", snaps.id, format_func=lambda i: f"{snaps.set_index('id').loc[i, 'snapshot_date']} (#{i})")
            if st.button("Delete"): get_store().delete_snapshot(int(sid)); st.rerun()
