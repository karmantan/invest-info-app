import pytest
from src.benchmarks import shadow_result, compare_results
from src.tax import estimated_tax

def test_shadow_benchmark_and_value_added():
    s=shadow_result(1000,100,102); assert s["result_eur"]==pytest.approx(20)
    c=compare_results(1037,s["ending_value_eur"],1000); assert c["value_added_eur"]==pytest.approx(17)

def test_fee_independent_tax_is_configurable():
    settings={"capital_gains_rate":.25,"solidarity_surcharge_rate":.055,"church_tax_rate":0,"etf_partial_exemption_rate":.3}
    assert estimated_tax(1000,0,settings)==pytest.approx(184.625)

