"""Every website page renders (demo prices, temporary data directory) and the login gate holds."""
import pytest
import streamlit as st
from streamlit.testing.v1 import AppTest
from src.config import ROOT

SAMPLE = {"snapshot_date": "2026-09-02", "cash_eur": 5000.0, "crypto_value_eur": 50.0,
          "holdings": [{"security_name": "Core MSCI World USD (Acc)", "isin": "IE00B4L5Y983", "quantity": 10, "displayed_price": 100, "market_value_eur": 1000.0},
                       {"security_name": "Tesla Inc.", "isin": "US88160R1014", "quantity": 1, "displayed_price": 300, "market_value_eur": 300.0}],
          "crypto_holdings": [{"name": "Bitcoin", "symbol": "BTC", "quantity": 0.001, "displayed_price": 50000, "market_value_eur": 50.0}]}


@pytest.fixture(autouse=True)
def demo_env(tmp_path, monkeypatch):
    monkeypatch.setenv("INVEST_DATA_DIR", str(tmp_path)); monkeypatch.setenv("INVEST_DEMO_PRICES", "1")
    monkeypatch.delenv("RENDER", raising=False)
    st.cache_resource.clear(); st.cache_data.clear()
    import webapp.common as common
    monkeypatch.setattr(common, "DEMO", True)
    from src.web.store import WebStore
    WebStore(tmp_path / "invest.db").save_statement(SAMPLE)
    yield
    st.cache_resource.clear(); st.cache_data.clear()


def view(name):
    def script(page, root):
        import importlib, sys
        sys.path[:0] = [root, root + "/webapp"]
        importlib.import_module(f"views.{page}").render()
    return AppTest.from_function(script, args=(name, str(ROOT)), default_timeout=120)


@pytest.mark.parametrize("page,title", [("overview", "Overview"), ("ideas", "ETF ideas"), ("simulator", "What if I invest?"),
                                         ("portfolio_page", "My Trade Republic"), ("settings_page", "Settings & method")])
def test_page_renders(page, title):
    at = view(page); at.run()
    assert not at.exception, at.exception
    assert at.title[0].value == title


def test_simulator_savings_plan_and_comparison():
    at = view("simulator")
    at.session_state["sim_mode"] = "monthly"; at.session_state["sim_ticker"] = "4GLD.DE"
    at.run()
    assert not at.exception
    at.multiselect[0].select("SXR8.DE").run()
    assert not at.exception
    assert any("Trade Republic cash account" in str(df.value) for df in at.dataframe)
    at.button[0].click().run()   # save as a pretend investment
    assert not at.exception and any("Saved" in s.value for s in at.success)


def test_login_gate_blocks_without_password(monkeypatch):
    monkeypatch.setenv("APP_PASSWORD", "s3cret")
    at = AppTest.from_file(str(ROOT / "webapp/app.py"), default_timeout=120); at.run()
    assert not at.exception and at.title[0].value == "Quiet Capital"
    at.text_input[0].input("s3cret"); at.button[0].click().run()
    assert not at.exception and at.session_state["authenticated"]
