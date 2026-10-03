from __future__ import annotations
import pandas as pd
import streamlit as st
from common import LONG, MEDIUM, SHORT, chance_text, demo_banner, disclaimer, estimates, eur, get_settings, get_store, get_universe, go, horizon_text, pct, portfolio_context, prices_for, signed_eur
from src.web import model
from src.web import portfolio as pf
from src.web.costs import cash_account

SORTS = {"balance": "Best balance of return and risk", "return": "Highest expected result", "safety": "Lowest chance of losing money"}


def _owned_and_variety(settings: dict, tickers: list[str]) -> tuple[dict, dict]:
    ctx = portfolio_context(settings)
    if ctx is None: return {}, {}
    uni = get_universe(); owned = {}
    held = ctx["live"].dropna(subset=["isin"])
    invested = float(ctx["live"].loc[ctx["live"].kind != "Crypto", "live_value"].sum()) or 1.0
    for _, h in held.iterrows():
        match = uni[uni["isin"] == h["isin"]]
        if not match.empty: owned[match.iloc[0].ticker] = h.live_value / invested
    mine = pf.portfolio_series(ctx["values"])
    prices = prices_for(tickers)
    variety = {t: pf.correlation(prices.get(t), mine) for t in tickers}
    return owned, variety


def _card(row: pd.Series, rank: int, amount: float, years: float, owned: dict, variety: dict, key: str) -> None:
    with st.container(border=True):
        head, action = st.columns([5, 1])
        head.markdown(f"**{rank}. {row['name']}**  \n{row.category} · ISIN `{row['isin']}` · Risk: **{row.risk}**")
        if row.description: head.caption(row.description)
        if action.button("Simulate", key=f"sim-{key}-{row.ticker}", width="stretch"):
            st.session_state["sim_ticker"] = row.ticker; st.session_state["sim_amount"] = amount; st.session_state["sim_months"] = int(round(years * 12)); go("simulator")
        c = st.columns(4)
        c[0].metric("Expected after fees & tax", eur(row.expected), signed_eur(row.profit))
        c[1].metric("Bad case (1 in 10)", eur(row.p10), signed_eur(row.p10 - amount))
        c[2].metric("Good case (1 in 10)", eur(row.p90), signed_eur(row.p90 - amount))
        c[3].metric("Chance of a loss", pct(row.prob_loss, 0), chance_text(row.prob_loss), delta_color="off")
        notes = [f"{signed_eur(row.vs_cash)} compared with leaving it as cash"]
        if row.ticker in owned: notes.append(f"You already own this ({pct(owned[row.ticker], 0)} of your invested money).")
        corr = variety.get(row.ticker)
        if corr is not None: notes.append(f"Next to what you own: **{pf.diversification_label(corr)}** (correlation {corr:.2f}).")
        notes.append("Recent trend: up" if row.trend_up else "Recent trend: down (below its 200-day average)")
        st.caption(" · ".join(notes))


def _horizon_tab(spec: dict, settings: dict, ests: dict, amount: float, sort: str, owned: dict, variety: dict, key: str) -> None:
    st.caption(spec["blurb"])
    if spec["unit"] == "months":
        months = st.slider("How long would you keep it?", spec["min"], spec["max"], spec["default"], format="%d months", key=f"h-{key}")
        years = months / 12
    else:
        years = float(st.slider("How long would you keep it?", spec["min"], spec["max"], spec["default"], format="%d years", key=f"h-{key}"))
    uni = get_universe()
    ranked = model.rank(ests, uni, amount, years, settings, sort)
    if ranked.empty: st.warning("No market data available right now."); return
    cash = cash_account(amount, years, settings)
    st.markdown(f"**Baseline — leave it in your Trade Republic cash account** ({pct(settings['trade_republic']['cash_interest_rate'], 2)} interest): "
                f"{eur(amount)} becomes **{eur(cash['net'])}** after {horizon_text(years)} and tax, with no risk of loss.")
    for i, (_, row) in enumerate(ranked.head(5).iterrows(), 1):
        _card(row, i, amount, years, owned, variety, key)
    with st.expander(f"All {len(ranked)} ETFs for {horizon_text(years)}"):
        table = ranked.assign(**{"Expected": ranked.expected.round(0), "Profit": ranked.profit.round(0), "Bad case (1 in 10)": ranked.p10.round(0), "Good case (1 in 10)": ranked.p90.round(0),
                                 "Chance of loss": ranked.prob_loss, "Fees": ranked.costs.round(2), "Tax": ranked.tax.round(0), "Expected per year (model)": ranked.yearly_expected})
        st.dataframe(table[["name", "category", "risk", "Expected", "Profit", "Bad case (1 in 10)", "Good case (1 in 10)", "Chance of loss", "Fees", "Tax", "Expected per year (model)", "isin"]]
                     .rename(columns={"name": "ETF", "category": "Type", "risk": "Risk", "isin": "ISIN"}), hide_index=True, width="stretch",
                     column_config={"Expected": st.column_config.NumberColumn(format="€%.0f"), "Profit": st.column_config.NumberColumn(format="€%.0f"),
                                    "Bad case (1 in 10)": st.column_config.NumberColumn(format="€%.0f"), "Good case (1 in 10)": st.column_config.NumberColumn(format="€%.0f"),
                                    "Chance of loss": st.column_config.ProgressColumn(format="percent", min_value=0, max_value=1),
                                    "Fees": st.column_config.NumberColumn(format="€%.2f"), "Tax": st.column_config.NumberColumn(format="€%.0f"),
                                    "Expected per year (model)": st.column_config.NumberColumn(format="percent")})


def render() -> None:
    st.title("ETF ideas")
    demo_banner()
    st.write("Every Trade Republic ETF on the list is scored for a holding period: what you would most likely have **after the €1 order fees, the bid-ask spread "
             "and German tax**, how wide the range of outcomes is, and how often you would end up with less than you put in.")
    settings = get_settings()
    data = estimates(settings)
    ests = data["estimates"]
    if not ests: st.error("Market data could not be loaded. Please try again in a few minutes."); return
    latest = get_store().latest()
    cash = latest["cash"] if latest else 0.0
    c1, c2 = st.columns([1, 2])
    amount = c1.number_input("Amount to invest (€)", min_value=50.0, max_value=float(max(cash, 1_000_000.0)), value=1000.0, step=50.0,
                             help=f"Your cash on the last statement: {eur(cash)}" if latest else None)
    sort = c2.radio("Order by", list(SORTS), format_func=SORTS.get, horizontal=True, index=0)
    if latest and amount > cash: st.warning(f"That is more than the {eur(cash)} cash on your last statement.")
    owned, variety = _owned_and_variety(settings, list(ests))
    tabs = st.tabs([SHORT["label"], MEDIUM["label"], LONG["label"]])
    for tab, spec, key in zip(tabs, (SHORT, MEDIUM, LONG), ("short", "medium", "long")):
        with tab: _horizon_tab(spec, settings, ests, amount, sort, owned, variety, key)
    with st.expander("How to read this"):
        st.markdown(
            "- **Expected after fees & tax** is the average result over all the outcomes the model considers, after buying, selling and paying tax.\n"
            "- **8 in 10 outcomes between**: in 1 of 10 cases you would end below the lower number, in 1 of 10 above the higher one.\n"
            "- **Best balance** ranks by extra result over cash per unit of uncertainty (a Sharpe ratio after costs and tax). It favours broad, cheap funds over narrow bets.\n"
            "- **Correlation** compares the fund's monthly moves with your current holdings: near 1 means it moves the same way, so it adds little variety.\n"
            "- Short horizons look worse than you might expect: the €2 round-trip fee, the spread and pure chance weigh heavily over a few months.")
    st.caption(f"Prices up to {data['data_date']}. Cash rate used by the model: {pct(data['rf'], 2)} (from the euro overnight-rate ETF).")
    if data["errors"]: st.caption("Not available right now: " + ", ".join(sorted(data["errors"])))
    disclaimer()
