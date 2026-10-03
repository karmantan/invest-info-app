from __future__ import annotations
import streamlit as st
from common import DEMO, demo_banner, estimates, get_settings, get_store, pct, price_store
from src.web.costs import effective_tax_rate

CHURCH = {0.0: "No church tax", 0.08: "8% (Bavaria, Baden-Württemberg)", 0.09: "9% (other states)"}


def render() -> None:
    st.title("Settings & method")
    demo_banner()
    settings = get_settings(); store = get_store()
    with st.form("settings"):
        st.subheader("Your situation")
        c1, c2 = st.columns(2)
        interest = c1.number_input("Interest on Trade Republic cash (% per year)", 0.0, 10.0, settings["trade_republic"]["cash_interest_rate"] * 100, 0.05, help="Shown in the Trade Republic app; it follows the ECB deposit rate.")
        allowance = c2.number_input("Unused tax-free allowance this year (€)", 0.0, 2000.0, float(settings["tax"]["saver_allowance_eur"]), 50.0,
                                    help="Sparer-Pauschbetrag: €1,000 single, €2,000 jointly — minus what interest and dividends already use. Set up a Freistellungsauftrag at Trade Republic to use it.")
        church = c1.selectbox("Church tax", list(CHURCH), index=list(CHURCH).index(settings["tax"].get("church_tax_rate", 0.0)) if settings["tax"].get("church_tax_rate", 0.0) in CHURCH else 0, format_func=CHURCH.get)
        income_rate = c2.number_input("Personal income-tax rate (%) — only for gold sold within a year", 0.0, 45.0, settings["tax"]["personal_income_tax_rate"] * 100, 1.0)
        st.subheader("Model assumptions")
        c1, c2 = st.columns(2)
        erp = c1.number_input("Equity risk premium (% per year)", 0.0, 10.0, settings["model"]["equity_risk_premium"] * 100, 0.25,
                              help="How much more than cash the world stock market is expected to earn per year. Long-run studies put it around 3–6%.")
        fee = c2.number_input("Order fee per trade (€)", 0.0, 10.0, float(settings["trade_republic"]["order_fee_eur"]), 0.5)
        if st.form_submit_button("Save settings", type="primary"):
            overrides = store.settings_overrides()
            overrides.setdefault("trade_republic", {}).update({"cash_interest_rate": interest / 100, "order_fee_eur": fee})
            overrides.setdefault("tax", {}).update({"saver_allowance_eur": allowance, "church_tax_rate": church, "personal_income_tax_rate": income_rate / 100})
            overrides.setdefault("model", {}).update({"equity_risk_premium": erp / 100})
            store.save_settings(overrides); st.success("Saved."); st.rerun()
    st.caption(f"Your tax rate on investment gains: **{pct(effective_tax_rate(settings['tax']), 2)}** (25% Abgeltungsteuer + 5.5% Soli{' + church tax' if settings['tax'].get('church_tax_rate') else ''}).")
    if store.settings_overrides() and st.button("Reset to defaults"):
        store.save_settings({}); st.rerun()

    st.subheader("How the numbers are made")
    st.markdown("""
**1. Prices.** Daily prices (dividends included) of each fund's Xetra listing in euros, from Yahoo Finance, refreshed at most twice a day.

**2. Expected return per year.** Two standard estimates are blended:
- *Market-based (CAPM):* cash rate + the fund's sensitivity to world stocks (beta) × the equity risk premium − the fund's yearly cost (TER).
  This is the textbook answer to "what should a fund like this earn?" and does not chase past winners.
- *The fund's own history:* its realised yearly return. It gets more weight the longer and calmer its history (a Bayesian shrinkage); a 10-year
  history of a typical stock fund gets about one third of the weight, a wild theme fund much less.

**3. Short-term trend.** Funds that rose over the past 12 months (skipping the last month) have tended to keep drifting the same way for a few
months (time-series momentum). The model adds a small, capped tilt that fades within about half a year — it barely affects long horizons.

**4. Risk.** How much the fund swings (volatility): today's level (exponentially weighted, RiskMetrics) fading toward its long-run level.

**5. Range of outcomes.** From expected return and volatility, the value after your holding period follows a log-normal distribution — the
standard model behind most investment projections. Savings plans are simulated month by month (3,000 random paths).

**6. Costs and tax, per outcome.** For every possible outcome: €1 order fee to buy and to sell (savings plans: free), half the bid-ask spread
each way, then German tax on the gain: 30% of an equity fund's gain is tax-free (Teilfreistellung), your unused allowance is deducted, and the rest is
taxed at 26.375% (+ church tax). Bond and money-market funds get no partial exemption. Xetra-Gold is tax-free after one year.
The yearly *Vorabpauschale* is not modelled separately: it is tax paid in advance and credited when you sell.

**7. Reality check.** For each fund the simulator also shows what happened in every past period of the same length.

**Limits.** Markets can do things no model expects. History on Yahoo for some funds is short. Spreads and TERs are approximate.
This is decision support, not financial advice.
""")
    st.subheader("Data status")
    data = estimates(settings)
    st.write(f"Funds with data: {len(data['estimates'])} · Latest price: {data['data_date']} · Cash rate in use: {pct(data['rf'], 2)}" + (" · DEMO PRICES" if DEMO else ""))
    errors = {**data["errors"], **price_store().errors}
    if errors:
        with st.expander(f"{len(errors)} data problem(s)"):
            for k, v in errors.items(): st.write(f"**{k}**: {v}")
