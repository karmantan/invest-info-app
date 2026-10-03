"""Quiet Capital — hosted website entry point.

Run locally:   streamlit run webapp/app.py
On Render:     see render.yaml (APP_PASSWORD and a persistent disk are required).
"""
from __future__ import annotations
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import streamlit as st
from src.web.auth import LIMITER, password_matches, password_required, throttle

st.set_page_config(page_title="Quiet Capital", page_icon="◌", layout="wide")


def login_gate() -> None:
    try:
        needed, expected = password_required()
    except RuntimeError as exc:
        st.error(str(exc)); st.stop()
    if not needed or st.session_state.get("authenticated"): return
    st.title("Quiet Capital")
    st.caption("Private investment dashboard")
    if LIMITER.locked():
        st.error("Too many wrong passwords recently. Please try again in a few minutes."); st.stop()
    with st.form("login"):
        password = st.text_input("Password", type="password")
        submitted = st.form_submit_button("Log in", type="primary")
    if submitted:
        if password_matches(password, expected):
            st.session_state.authenticated = True; st.session_state.pop("failures", None); st.rerun()
        st.session_state.failures = st.session_state.get("failures", 0) + 1
        LIMITER.record_failure(); throttle(st.session_state.failures)
        st.error("Wrong password.")
    st.stop()


login_gate()

from views import ideas, overview, portfolio_page, settings_page, simulator  # noqa: E402  (after login)

PAGES = {
    "overview": st.Page(overview.render, title="Overview", icon=":material/space_dashboard:", url_path="overview", default=True),
    "ideas": st.Page(ideas.render, title="ETF ideas", icon=":material/lightbulb:", url_path="ideas"),
    "simulator": st.Page(simulator.render, title="What if I invest?", icon=":material/tune:", url_path="what-if"),
    "portfolio": st.Page(portfolio_page.render, title="My Trade Republic", icon=":material/account_balance_wallet:", url_path="portfolio"),
    "settings": st.Page(settings_page.render, title="Settings & method", icon=":material/settings:", url_path="settings"),
}
st.session_state["_pages"] = PAGES
navigation = st.navigation(list(PAGES.values()))
with st.sidebar:
    st.caption("Decision support only. Nothing is traded and no Trade Republic login is used.")
    if st.session_state.get("authenticated") and st.button("Log out", width="stretch"):
        st.session_state.clear(); st.rerun()
navigation.run()
