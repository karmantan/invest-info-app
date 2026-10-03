import pandas as pd
from src.reporting.audit_21d import cost_audit, frozen_spec

CFG={"costs":{"buy_fee_eur":1,"sell_fee_eur":1,"estimated_spread_bps":10,"estimated_benchmark_spread_bps":10},"risk":{"threshold":"frozen"},"portfolio":{"threshold":"frozen"}}

def test_identical_tactical_costs_cancel_in_relative_value():
    top=pd.DataFrame({"model":["ridge"],"forward_eur_21d":[.03],"benchmark_forward_eur_21d":[.02],"urth_forward_eur_21d":[.01]})
    row=cost_audit(top,CFG,None).query("allocation_eur==1000 and benchmark=='SPY'").iloc[0]
    assert row.tactical_net_value_added_eur==10
    assert row.wealth_counterfactual_value_added_eur==6

def test_frozen_specification_hash_is_deterministic():
    assert frozen_spec(CFG)[1]==frozen_spec(CFG)[1]
