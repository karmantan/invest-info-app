from pathlib import Path

from src.portfolio.allocation import allocate
from src.portfolio.reconciliation import reconcile
from src.portfolio.pdf_import import _number, parse_text

CFG={"portfolio":{"minimum_trade_eur":100,"maximum_position_weight":.15,"maximum_sector_weight":.25,"tactical_sleeve_weight":.08},"risk":{"downside_lambda":2,"minimum_analogue_observations":20,"maximum_probability_loss_5pct":.2,"minimum_confidence_for_action":.6,"minimum_expected_excess_return":.01}}

def test_allocation_uses_one_total_target_not_horizon_sum():
    f={"expected_return":.08,"p10":-.02,"sample_size":100,"probability_loss_gt_5pct":.05,"confidence":.8,"expected_excess_return":.03}
    d=allocate(f,1600,5000,20000,1600,CFG)
    assert d.amount_now_eur==0  # tactical cap is already reached; no horizon double-count

def test_reconciliation_marks_snapshot_inference():
    a={"cash_eur":100,"holdings":[{"isin":"DE0001","quantity":2}]}; b={"cash_eur":50,"holdings":[{"isin":"DE0001","quantity":3}]}
    changes=reconcile(a,b); assert all(c["estimated"] for c in changes)

def test_pdf_parser_reports_uncertainty():
    parsed=parse_text("Portfolio 01.08.2026\nExample ETF DE000ABC1234 10,00 € 100,00 €")
    assert parsed["snapshot_date"]=="2026-08-01"; assert parsed["holdings"][0]["isin"]=="DE000ABC1234"; assert parsed["warnings"]


def test_literal_isin_and_empty_candidates_never_create_holdings():
    text="Portfolioübersicht\n"+"\n".join(f"ISIN: DE000ABC{i:04d}" for i in range(20))
    parsed=parse_text(text)
    assert parsed["holdings"]==[]
    assert parsed["diagnostics"]["accepted_holdings"]==0
    assert len(parsed["unmatched"])==20


def test_german_money_and_decimal_units_are_parsed():
    assert _number("1.234,56") == 1234.56
    assert _number("1\u00a0234,56") == 1234.56
    parsed=parse_text("""Portfolioübersicht 01.09.2026
US Technology ETF
ISIN: DE000ABC1234
0,123456 Stück
Marktwert 1.234,56 €
Gesamtwert Wertpapiere 1.234,56 €""")
    assert parsed["holdings"][0]["quantity"] == .123456
    assert parsed["holdings"][0]["market_value_eur"] == 1234.56


def test_interleaved_security_fields_remain_associated_and_reconcile_without_cash():
    parsed=parse_text("""Depotübersicht 01.09.2026
US Technology ETF
ISIN:
DE000ABC1234
12,34 Anteile
Marktwert
12.500,00 EUR
Emerging Markets ETF
ISIN: IE000ABC1234
2,5 Stück
Marktwert 20.804,64 €
Gesamtwert Wertpapiere 33.304,64 €""")
    assert [h["security_name"] for h in parsed["holdings"]] == ["US Technology ETF", "Emerging Markets ETF"]
    assert parsed["cash_eur"] is None
    assert parsed["securities_value_eur"] == 33304.64
    assert parsed["reconciliation_difference_eur"] == 0
    assert "Cash is not included in this statement." in parsed["warnings"]


def test_reported_total_mismatch_blocks_confirmation_and_manual_cash_is_allowed():
    parsed=parse_text("""Depotübersicht 01.09.2026
Example ETF
ISIN: DE000ABC1234
10 Stück
Marktwert 1.000,00 €
Gesamtwert Wertpapiere 1.100,00 €""")
    assert parsed["material_reconciliation_mismatch"]
    parsed["cash_eur"]=250
    parsed["cash_source"]="manual"
    parsed["total_value_eur"]=1250
    from src.portfolio.reconciliation_service import validate_confirmed_snapshot
    assert any("materially differs" in error for error in validate_confirmed_snapshot(parsed))


def test_trade_republic_positioned_wealth_overview_layout_is_structurally_parsed():
    # Synthetic values and identities reproduce the column/block layout only.
    layout = (Path(__file__).parent / "fixtures" / "trade_republic_wealth_overview_layout.txt").read_text().splitlines()
    text = "Vermögensübersicht 02.09.2026\n" + "\n".join(layout)
    parsed = parse_text(text, positioned_lines=layout)
    assert [h["security_name"] for h in parsed["holdings"]] == ["Sanitized Alpha Fund", "Sanitized Beta Fund"]
    assert [h["quantity"] for h in parsed["holdings"]] == [10, 5]
    assert [h["displayed_price"] for h in parsed["holdings"]] == [100, 500]
    assert [h["market_value_eur"] for h in parsed["holdings"]] == [1000, 2500]
    assert parsed["reported_securities_value_eur"] == 3500
    assert parsed["cash_eur"] == 450
    assert parsed["cash_source"] == "extracted"
    assert parsed["crypto_value_eur"] == 50
    assert parsed["reported_total_financial_assets_eur"] == 4000
    assert parsed["reconciliation_difference_eur"] == 0
    assert parsed["total_financial_assets_difference_eur"] == 0
