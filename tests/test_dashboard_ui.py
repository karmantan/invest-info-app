from dashboard.ui import asset_name, compact_table_html, friendly_reason, global_css, horizon_label, info_panel_html, opportunity_cards_html, responsive_cards_html, risk_label, status_label
from pathlib import Path
from streamlit.testing.v1 import AppTest


def test_friendly_asset_and_horizon_labels():
    assert asset_name("SMH") == "Semiconductors (SMH)"
    assert horizon_label(63) == "About 3 months"
    assert horizon_label(126, technical=True) == "About 6 months (126 trading days)"


def test_canonical_codes_are_translated_without_changing_them():
    raw = "MODEL INTERESTING — RISK GATE PASSES"
    assert status_label(raw) == "Promising setup — risk checks passed"
    assert raw == "MODEL INTERESTING — RISK GATE PASSES"
    assert risk_label("high_asset_volatility") == "Price swings are unusually large"


def test_machine_reason_becomes_a_sentence():
    text = friendly_reason("Frozen top cohort with Policy 2 pass; one coherent position uses the longest simultaneously triggered horizon.", "SMH", 126, "PAPER BUY")
    assert "Semiconductors" in text
    assert "current risk rules" in text
    assert text.endswith(".")


def test_compact_table_is_fixed_wrapped_and_escaped():
    markup = compact_table_html([{"Investment": "Semiconductors <SMH>", "Result": "+5.1%"}])
    assert 'class="qc-table"' in markup
    assert "Semiconductors &lt;SMH&gt;" in markup
    css = global_css()
    assert "table-layout:fixed" in css
    assert "white-space:normal" in css
    assert "overflow-x" not in css


def test_cards_keep_secondary_fields_in_native_details():
    markup = responsive_cards_html(
        [{"Investment": "Semiconductors (SMH)", "Risk": "Risk checks passed", "P10": "−2.4%"}],
        "Investment", ["Risk"], ["P10"],
    )
    assert 'class="qc-card-grid"' in markup
    assert "<details>" in markup
    assert "More details" in markup
    assert "P10" in markup


def test_opportunities_group_horizons_into_one_card_per_investment():
    records = [
        {"Investment": "Semiconductors (SMH)", "Time horizon": "About 1 month", "Current view": "Worth watching", "Risk": "Risk checks passed", "Historical evidence": "One", "Why": "Short", "Exact horizon": "21 days"},
        {"Investment": "Semiconductors (SMH)", "Time horizon": "About 3 months", "Current view": "Historically promising", "Risk": "Price swings are unusually large", "Historical evidence": "Two", "Why": "Medium", "Exact horizon": "63 days"},
        {"Investment": "Emerging Markets (EEM)", "Time horizon": "About 6 months", "Current view": "No opportunity detected", "Risk": "Risk checks passed", "Historical evidence": "Three", "Why": "Long", "Exact horizon": "126 days"},
    ]
    markup = opportunity_cards_html(records)
    assert markup.count('<article class="qc-card">') == 2
    assert markup.count("Semiconductors (SMH)") == 1
    assert "About 1 month" in markup and "About 3 months" in markup
    assert "Worth watching" in markup and "Historically promising" in markup
    assert "No opportunity detected" in markup
    assert "One" in markup and "Two" in markup and "Three" in markup


def test_model_status_uses_compact_information_panel():
    markup = info_panel_html([("Current model", "Medium-term Ridge v1"), ("Mode", "Paper trading")])
    assert 'class="qc-info-panel"' in markup
    assert "Current model" in markup and "Medium-term Ridge v1" in markup
    css = global_css()
    assert ".qc-info-value" in css
    assert "font-size:.95rem" in css


def test_all_dashboard_pages_including_discover_render():
    app=AppTest.from_file(str(Path(__file__).parents[1]/"dashboard/app.py"),default_timeout=20).run()
    assert not app.exception
    radio=app.sidebar.radio[0]
    for page in ["Opportunities","Discover","Portfolio","Research","Performance","Ledger","Help / Glossary"]:
        radio.set_value(page); app.run(); assert not app.exception, page
        radio=app.sidebar.radio[0]


def test_discover_has_no_real_money_buy_or_sell_control():
    source=(Path(__file__).parents[1]/"dashboard/app.py").read_text()
    discover=source.split('elif page=="Discover":',1)[1].split('elif page=="Portfolio":',1)[0]
    assert 'button("Buy' not in discover
    assert 'button("Sell' not in discover
    assert 'button("Trade' not in discover
    assert "Start simulated tracking" in discover and '"Watch"' in discover and "Compare" in discover
