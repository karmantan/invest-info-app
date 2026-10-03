from __future__ import annotations
from datetime import date
import pandas as pd
import plotly.graph_objects as go_
import streamlit as st
from common import MUTED, SERIES, chance_text, chart_layout, demo_banner, disclaimer, estimates, eur, get_settings, get_store, get_universe, horizon_text, pct, prices_for, signed_eur
from src.web import model
from src.web.costs import cash_account

HORIZON_MONTHS = [1, 2, 3, 6, 9, 12, 18, 24, 36, 48, 60, 84, 120, 180, 240, 300, 360]


def _label(months: int) -> str: return horizon_text(months / 12)


def _rgba(hex_color: str, alpha: float) -> str:
    h = hex_color.lstrip("#"); return f"rgba({int(h[:2], 16)},{int(h[2:4], 16)},{int(h[4:], 16)},{alpha})"


def _run(mode: str, amount: float, years: float, est, etf: dict, settings: dict):
    if mode == "monthly": return model.savings_plan(amount, years, est, etf, settings)
    return model.lump_sum(amount, years, est, etf, settings), model.lump_sum_path(amount, years, est, etf, settings)


def render() -> None:
    st.title("What if I invest?")
    demo_banner()
    settings = get_settings(); uni = get_universe(); data = estimates(settings); ests = data["estimates"]
    if not ests: st.error("Market data could not be loaded. Please try again in a few minutes."); return
    available = uni[uni.ticker.isin(ests)].reset_index(drop=True)
    latest = get_store().latest(); cash = latest["cash"] if latest else None
    label = lambda t: f"{available.set_index('ticker').loc[t, 'name']} · {available.set_index('ticker').loc[t, 'category']}"

    default = st.session_state.get("sim_ticker", "EUNL.DE")
    tickers = list(available.ticker)
    c1, c2 = st.columns([3, 2])
    ticker = c1.selectbox("ETF", tickers, index=tickers.index(default) if default in tickers else 0, format_func=label)
    mode = c2.segmented_control("How", ["lump", "monthly"], default="lump", format_func={"lump": "One-off amount", "monthly": "Monthly savings plan"}.get, key="sim_mode") or "lump"

    c1, c2 = st.columns(2)
    if mode == "lump":
        top = float(max(1000.0, round(cash or 50_000, -2)))
        start = float(min(top, st.session_state.get("sim_amount", 1000.0)))
        amount = c1.slider("Amount from your cash", 100.0, top, start, step=50.0, format="€%.0f",
                           help=f"Up to the {eur(cash)} cash on your last statement." if cash else "Upload a statement to use your real cash balance as the limit.")
    else:
        amount = c1.slider("Every month", 25.0, 3000.0, 100.0, step=25.0, format="€%.0f", help="Savings plans on Trade Republic have no order fee.")
    months_default = st.session_state.get("sim_months", 60)
    months_default = min(HORIZON_MONTHS, key=lambda m: abs(m - months_default))
    months = c2.select_slider("For how long", HORIZON_MONTHS, value=months_default, format_func=_label)
    years = months / 12
    others = st.multiselect("Compare with up to 2 other ETFs", [t for t in tickers if t != ticker], max_selections=2, format_func=label)

    etf = available.set_index("ticker").loc[ticker].to_dict() | {"ticker": ticker}
    est = ests[ticker]
    if etf.get("description"): st.caption(f"{etf['description']} ISIN {etf['isin']} · yearly fund cost {pct(etf['ter'], 2)}.")
    result, path = _run(mode, amount, years, est, etf, settings)
    paid = result["paid_in"]
    cash_res = cash_account(amount, years, settings, monthly=amount if mode == "monthly" else 0.0)

    st.divider()
    st.markdown(f"### {eur(paid)} {'paid in over' if mode == 'monthly' else 'invested for'} {horizon_text(years)} in {etf['name']}")
    m = st.columns(5)
    m[0].metric("Expected", eur(result["mean"]), signed_eur(result["mean"] - paid), border=True, help="After fees and tax: the average over all outcomes the model considers.")
    m[1].metric("Middle", eur(result["p50"]), signed_eur(result["p50"] - paid), border=True, help="Half of the outcomes are better, half are worse.")
    m[2].metric("Bad case", eur(result["p10"]), signed_eur(result["p10"] - paid), border=True, help="1 in 10 outcomes are worse than this.")
    m[3].metric("Good case", eur(result["p90"]), signed_eur(result["p90"] - paid), border=True, help="1 in 10 outcomes are better than this.")
    m[4].metric("Chance of loss", pct(result["prob_loss"], 0), chance_text(result["prob_loss"]), delta_color="off", border=True, help="Chance of getting back less than you put in, after fees and tax.")
    st.markdown(
        f"In plain words: you would put in **{eur(paid)}**. On average you could expect to have **{eur(result['mean'])}** after selling and tax "
        f"(a profit of **{signed_eur(result['mean'] - paid)}**). A bad case (1 in 10) leaves you with **{eur(result['p10'])}**, a good case (1 in 10) with **{eur(result['p90'])}** or more. "
        f"Leaving the money in your Trade Republic cash account instead would give **{eur(cash_res['net'])}** with no risk.")
    st.caption(f"Costs included: {eur(result['costs'], 2)} in order fees and spread · Expected tax: {eur(result['tax'])}"
               + (" · Gold held over a year is tax-free in Germany." if etf["tax_class"] == "gold_etc" and years > 1 else ""))

    fig = go_.Figure()
    fig.add_trace(go_.Scatter(x=path.years, y=path.p90, line=dict(width=0), showlegend=False, hoverinfo="skip"))
    fig.add_trace(go_.Scatter(x=path.years, y=path.p10, fill="tonexty", fillcolor=_rgba(SERIES[0], 0.18), line=dict(width=0), name="8 in 10 outcomes", hoverinfo="skip"))
    fig.add_trace(go_.Scatter(x=path.years, y=path.p50, name=f"{etf['name']} (middle outcome)", line=dict(color=SERIES[0], width=2.5), hovertemplate="€%{y:,.0f}"))
    fig.add_trace(go_.Scatter(x=path.years, y=path.p10, name="Bad case (1 in 10)", line=dict(color=SERIES[0], width=1, dash="dot"), hovertemplate="€%{y:,.0f}"))
    fig.add_trace(go_.Scatter(x=path.years, y=path.p90, name="Good case (1 in 10)", line=dict(color=SERIES[0], width=1, dash="dot"), hovertemplate="€%{y:,.0f}"))
    cash_line = model.cash_path(amount, years, settings, monthly=amount if mode == "monthly" else 0.0)
    fig.add_trace(go_.Scatter(x=cash_line.years, y=cash_line.value, name="Trade Republic cash account", line=dict(color=SERIES[1], width=2), hovertemplate="€%{y:,.0f}"))
    fig.add_trace(go_.Scatter(x=path.years, y=path.paid_in, name="Money you put in", line=dict(color=MUTED, width=1.5, dash="dash"), hovertemplate="€%{y:,.0f}"))
    rows = [{"ETF": etf["name"], "Expected": result["mean"], "Middle": result["p50"], "Bad case": result["p10"], "Good case": result["p90"], "Chance of loss": result["prob_loss"], "Fees": result["costs"], "Tax": result["tax"]}]
    for i, other in enumerate(others):
        o_etf = available.set_index("ticker").loc[other].to_dict() | {"ticker": other}
        o_res, o_path = _run(mode, amount, years, ests[other], o_etf, settings)
        fig.add_trace(go_.Scatter(x=o_path.years, y=o_path.p50, name=f"{o_etf['name']} (middle)", line=dict(color=SERIES[2 + i], width=2), hovertemplate="€%{y:,.0f}"))
        rows.append({"ETF": o_etf["name"], "Expected": o_res["mean"], "Middle": o_res["p50"], "Bad case": o_res["p10"], "Good case": o_res["p90"], "Chance of loss": o_res["prob_loss"], "Fees": o_res["costs"], "Tax": o_res["tax"]})
    rows.append({"ETF": "Trade Republic cash account", "Expected": cash_res["net"], "Middle": cash_res["net"], "Bad case": cash_res["net"], "Good case": cash_res["net"], "Chance of loss": 0.0, "Fees": 0.0, "Tax": cash_res["tax"]})
    fig.update_xaxes(title="Years from now")
    st.plotly_chart(chart_layout(fig, 440, "What you'd have if you sold then (after fees & tax)", "€,.0f"), width="stretch")

    money = st.column_config.NumberColumn(format="€%.0f")
    st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch",
                 column_config={"Expected": money, "Middle": money, "Bad case": money, "Good case": money, "Tax": money,
                                "Fees": st.column_config.NumberColumn(format="€%.2f"), "Chance of loss": st.column_config.ProgressColumn(format="percent", min_value=0, max_value=1)})

    st.subheader("Reality check: what actually happened before")
    series = prices_for([ticker]).get(ticker)
    hist = model.historical_windows(series, years) if series is not None else None
    if hist is None:
        st.info(f"This ETF's price history on Yahoo is too short to look at past {horizon_text(years)} periods.")
    else:
        st.markdown(f"Looking at every possible start day since {hist['start']} ({hist['windows']:,} overlapping periods), holding it for {horizon_text(years)} "
                    f"made money **{pct(hist['share_positive'], 0)}** of the time. The middle result was **{pct(hist['median'], 1, True)}**, "
                    f"the worst **{pct(hist['worst'], 1, True)}** and the best **{pct(hist['best'], 1, True)}** (before fees and tax).")
        st.caption("Overlapping periods are not independent, and the past is only one possible history — this checks the model's range, it does not replace it.")

    with st.expander("Model details for this ETF"):
        d = est
        st.markdown(
            f"- Expected return per year (before your fees and tax): **{pct(d.expected_return, 2)}**\n"
            f"  - Market-based estimate (CAPM): cash {pct(d.rf, 2)} + beta {d.beta:.2f} × equity premium {pct(settings['model']['equity_risk_premium'], 1)} − fund cost {pct(etf['ter'], 2)} = {pct(d.capm_return, 2)}\n"
            f"  - Its own history: {pct(d.history_return, 2) if d.history_return is not None else '—'} per year over {d.history_years:.1f} years, given **{pct(d.history_weight, 0)}** weight\n"
            f"- Trend signal (12-month momentum): {d.momentum_signal:+.2f} → {pct(d.momentum_tilt, 2, True)} per year at first, fading over about 6 months\n"
            f"- Volatility (typical yearly swing): now {pct(d.vol_now, 1)}, long-run {pct(d.vol_long, 1)}\n"
            f"- Fall from its 1-year high: {pct(d.drawdown_1y, 1)}\n"
            + (f"- Tax on yearly payouts (distributing fund): about {pct(d.dividend_tax_drag, 2)} per year\n" if d.dividend_tax_drag else "")
            + f"- Fund cost (TER) {pct(etf['ter'], 2)} per year · assumed bid-ask spread {pct(etf['spread_pct'], 2)} · tax class: {etf['tax_class']}")
        name = data["yahoo_names"].get(ticker)
        if name: st.caption(f"Price source: Yahoo symbol {ticker} ({name}). Check the ISIN {etf['isin']} in the Trade Republic app before buying.")

    with st.form("track"):
        st.markdown("**Keep an eye on this idea** — save it as a pretend investment and the Overview page will show how it really does from today.")
        note = st.text_input("Note (optional)")
        if st.form_submit_button("Start pretend investment"):
            get_store().add_whatif(ticker=ticker, name=etf["name"], isin=etf["isin"], mode=mode, amount_eur=amount, horizon_years=years,
                                   entry_date=est.last_date or str(date.today()), entry_price=est.last_price, expected_net_eur=result["mean"],
                                   p10_eur=result["p10"], p90_eur=result["p90"], note=note)
            st.success("Saved. You'll find it on the Overview page.")
    disclaimer()
