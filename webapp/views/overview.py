from __future__ import annotations
import pandas as pd
import plotly.graph_objects as go_
import streamlit as st
from common import MUTED, SERIES, chart_layout, demo_banner, eur, get_settings, get_store, go, pct, portfolio_context, prices_for, signed_eur
from src.web import portfolio as pf
from src.web.costs import effective_tax_rate


def _benchmark_chart(settings: dict, ctx: dict | None) -> None:
    st.subheader("How your investments did compared with the market")
    benchmarks = settings["benchmarks"]
    names = [b["name"] for b in benchmarks]
    period = st.segmented_control("Period", pf.PERIODS, default="1Y", key="overview_period") or "1Y"
    chosen = st.multiselect("Compare with", names, default=names[:2], max_selections=4, key="overview_benchmarks")
    prices = ctx["prices"] if ctx else prices_for([b["ticker"] for b in benchmarks])
    ends = [s.index.max() for s in prices.values() if s is not None and len(s)]
    if not ends: st.info("Market data is not available right now. Please try again later."); return
    end = max(ends); start = pf.period_start(end, period)
    fig = go_.Figure(); summary = []
    if ctx is not None and not ctx["values"].empty:
        mine = pf.buy_and_hold_return(ctx["values"], start)
        if len(mine):
            fig.add_trace(go_.Scatter(x=mine.index, y=mine, name="Your investments", line=dict(color=SERIES[0], width=2.5), hovertemplate="%{y:+.1%}"))
            summary.append(("your investments", float(mine.iloc[-1])))
    for i, b in enumerate(benchmarks):
        if b["name"] not in chosen or b["ticker"] not in prices: continue
        r = pf.index_return(prices[b["ticker"]], start)
        if not len(r): continue
        fig.add_trace(go_.Scatter(x=r.index, y=r, name=b["name"], line=dict(color=SERIES[(i + 1) % len(SERIES)], width=2), hovertemplate="%{y:+.1%}"))
        summary.append((b["name"], float(r.iloc[-1])))
    fig.add_hline(y=0, line=dict(color=MUTED, width=1))
    st.plotly_chart(chart_layout(fig, 420, "Return since start of period", ".0%"), width="stretch")
    if summary:
        parts = [f"**{name}** {pct(value, signed=True)}" for name, value in summary]
        st.markdown(f"Over the last **{period}**: " + " · ".join(parts))
    if ctx is not None:
        cover = pf.coverage(ctx["values"], ctx["mapped"], start)
        st.caption(f"\"Your investments\" = the holdings on your statement of {ctx['date']}, as if you had held them for the whole period "
                   f"(the statement has no purchase history). Prices cover {pct(cover, 0)} of their value from the start of this period; "
                   "younger or unmatched positions join the line when their data begins. Cash is left out so the comparison is fair.")
        moves = pf.position_returns(ctx["values"], ctx["mapped"], start)
        if not moves.empty:
            with st.expander("Best and worst holdings in this period"):
                show = moves.assign(Return=moves["return"].map(lambda v: pct(v, signed=True)), Gain=moves.gain_eur.map(signed_eur))
                st.dataframe(show[["name", "kind", "Return", "Gain", "since"]].rename(columns={"name": "Holding", "kind": "Type", "since": "Data since"}), hide_index=True, width="stretch")


def _breakdown(ctx: dict, settings: dict) -> None:
    live = ctx["live"]
    brokerage_live = float(live.loc[live.kind != "Crypto", "live_value"].sum())
    crypto_live = float(live.loc[live.kind == "Crypto", "live_value"].sum()) or ctx["crypto_value"]
    total_live = brokerage_live + crypto_live + ctx["cash"]
    st.caption(f"From your Trade Republic statement of **{ctx['date']}**, updated with today's prices where available. Cash is as on the statement.")
    cols = st.columns(4)
    cols[0].metric("Total", eur(total_live), signed_eur(total_live - ctx["total"]) + " since statement", border=True)
    cols[1].metric("Invested (brokerage)", eur(brokerage_live), signed_eur(brokerage_live - ctx["brokerage"]) + " since statement", border=True)
    cols[2].metric("Cash", eur(ctx["cash"]), f"{pct(ctx['cash'] / total_live if total_live else 0, 0)} of everything", delta_color="off", border=True)
    cols[3].metric("Crypto", eur(crypto_live), signed_eur(crypto_live - ctx["crypto_value"]) + " since statement", border=True)

    parts = pd.DataFrame({"part": ["Invested (brokerage)", "Cash", "Crypto"], "value": [brokerage_live, ctx["cash"], crypto_live]})
    fig = go_.Figure()
    for i, r in parts.iterrows():
        fig.add_trace(go_.Bar(y=["Your money"], x=[r.value], name=r.part, orientation="h", marker=dict(color=SERIES[i], line=dict(width=2, color="rgba(255,255,255,0.9)")),
                              text=f"{r.part}: {pct(r.value / total_live if total_live else 0, 0)}", textposition="inside", insidetextanchor="middle",
                              hovertemplate=f"{r.part}: {eur(r.value)}<extra></extra>"))
    fig.update_layout(barmode="stack", height=120, margin=dict(l=8, r=8, t=8, b=8), showlegend=False, xaxis=dict(visible=False), yaxis=dict(visible=False))
    st.plotly_chart(fig, width="stretch")

    left, right = st.columns([3, 2])
    with left:
        st.markdown("**Where your invested money is**")
        top = live.sort_values("live_value", ascending=True).tail(12)
        fig = go_.Figure(go_.Bar(x=top.live_value, y=top["name"], orientation="h", marker=dict(color=SERIES[0], cornerradius=4),
                                 hovertemplate="%{y}: €%{x:,.0f}<extra></extra>", text=[eur(v) for v in top.live_value], textposition="outside", cliponaxis=False))
        fig.update_layout(height=max(220, 28 * len(top) + 40), margin=dict(l=8, r=60, t=8, b=8), xaxis=dict(visible=False), yaxis=dict(automargin=True))
        st.plotly_chart(fig, width="stretch")
    with right:
        invested = live[live.kind != "Crypto"]
        total_inv = float(invested.live_value.sum()) or 1.0
        by_kind = live.groupby("kind").live_value.sum()
        st.markdown("**Things worth knowing**")
        notes = []
        biggest = invested.sort_values("live_value").iloc[-1] if not invested.empty else None
        if biggest is not None and biggest.live_value / total_inv > 0.25:
            notes.append(f"**{biggest['name']}** is {pct(biggest.live_value / total_inv, 0)} of your invested money. One fund that large drives most of your ups and downs.")
        single = float(by_kind.get("Single stock", 0.0))
        if single:
            notes.append(f"Single stocks: {eur(single)} ({pct(single / total_inv, 0)} of invested). Individual companies swing far more than funds.")
        rate = settings["trade_republic"]["cash_interest_rate"]; interest = ctx["cash"] * rate
        allowance = settings["tax"]["saver_allowance_eur"]
        notes.append(f"Your cash earns about **{eur(interest)} a year** at {pct(rate, 2)} interest — that uses {pct(min(1.0, interest / allowance) if allowance else 1, 0)} of your {eur(allowance)} tax-free allowance. "
                     f"Above the allowance, interest is taxed at {pct(effective_tax_rate(settings['tax']), 1)}.")
        if ctx["cash"] > 0.3 * total_live:
            notes.append(f"{pct(ctx['cash'] / total_live, 0)} of your money is cash. Keep an emergency reserve there; the **What if I invest?** page shows what the rest might do instead.")
        for n in notes: st.markdown("- " + n)
        if st.button("Try investing some of your cash →", type="primary"): go("simulator")
    problems = ctx["mapped"][ctx["mapped"].price_status != "ok"]
    if not problems.empty:
        with st.expander(f"{len(problems)} holding(s) shown at statement value (no matching live price)"):
            st.dataframe(problems[["name", "isin", "symbol", "price_status"]].rename(columns={"name": "Holding", "isin": "ISIN", "symbol": "Yahoo symbol", "price_status": "Why"}), hide_index=True, width="stretch")
            st.caption("You can fix a wrong or missing symbol under isin_overrides in config/web.yaml.")


def _pretend(settings: dict) -> None:
    rows = get_store().whatifs()
    if rows.empty: return
    st.subheader("Your pretend investments")
    st.caption("Saved from the What-if page. Shows what they would be worth now (before selling fee and tax), next to what the model expected.")
    prices = prices_for(list(rows.ticker))
    table = []
    for _, r in rows.iterrows():
        v = pf.whatif_value(r, prices.get(r.ticker))
        table.append({"Investment": r["name"], "Started": r.entry_date, "Type": "Monthly plan" if r["mode"] == "monthly" else "One-off",
                      "Paid in": eur(v["paid"]) if v else "—", "Worth now": eur(v["value"]) if v else "—", "Result": f"{signed_eur(v['gain'])} ({pct(v['return'], 1, True)})" if v else "—",
                      "Model expected at end": f"{eur(r.expected_net_eur)} after {r.horizon_years:g} years" if pd.notna(r.expected_net_eur) else "—", "id": r.id})
    frame = pd.DataFrame(table)
    st.dataframe(frame.drop(columns="id"), hide_index=True, width="stretch")
    with st.expander("Remove a pretend investment"):
        choice = st.selectbox("Pretend investment", frame.id, format_func=lambda i: f"{frame.set_index('id').loc[i, 'Investment']} — started {frame.set_index('id').loc[i, 'Started']}")
        if st.button("Remove"): get_store().close_whatif(int(choice)); st.rerun()


def render() -> None:
    st.title("Overview")
    demo_banner()
    settings = get_settings()
    ctx = portfolio_context(settings)
    if ctx is None:
        st.info("Upload your Trade Republic statement (Vermögensübersicht PDF) to see your cash, investments and performance here.", icon="📄")
        if st.button("Upload statement", type="primary"): go("portfolio")
    else:
        _breakdown(ctx, settings)
    _benchmark_chart(settings, ctx)
    _pretend(settings)
