import numpy as np
import pytest
from src.web.costs import buy_cost, cash_account, effective_tax_rate, sell_net, tax_on_gain
from src.web.settings import load_web_config

S = load_web_config()


def test_effective_rate_with_and_without_church_tax():
    assert effective_tax_rate(S["tax"]) == pytest.approx(0.26375)
    assert effective_tax_rate({**S["tax"], "church_tax_rate": 0.09}) == pytest.approx(0.2799, abs=1e-4)


def test_equity_fund_exemption_then_allowance():
    tax = tax_on_gain(np.array([2000.0, 500.0, -300.0]), "equity", 3, S["tax"], allowance_eur=1000)
    assert tax[0] == pytest.approx(400 * 0.26375)
    assert tax[1] == 0 and tax[2] == 0


def test_bond_fund_has_no_partial_exemption():
    assert tax_on_gain(2000.0, "bond", 3, S["tax"], allowance_eur=0) == pytest.approx(2000 * 0.26375)


def test_gold_etc_tax_free_after_one_year():
    assert tax_on_gain(5000.0, "gold_etc", 1.5, S["tax"]) == 0
    assert tax_on_gain(5000.0, "gold_etc", 0.5, S["tax"]) == pytest.approx(1500)
    assert tax_on_gain(900.0, "gold_etc", 0.5, S["tax"]) == 0


def test_round_trip_costs_on_flat_market():
    invested, cost = buy_cost(1000, 0.002, 1.0)
    assert invested == pytest.approx(999 * 0.999) and cost == pytest.approx(1000 - invested)
    net, tax, selling = sell_net(invested, 1000, {"spread_pct": 0.002, "tax_class": "equity"}, 1, S)
    assert tax == 0 and float(net) == pytest.approx(invested * 0.999 - 1)


def test_cash_account_taxes_interest_above_allowance_yearly():
    small = cash_account(1000, 2, S)
    assert small["tax"] == 0 and small["net"] == pytest.approx(1000 * (1 + 0.02 / 12) ** 24)
    big = cash_account(100_000, 1, S)
    interest = 100_000 * ((1 + 0.02 / 12) ** 12 - 1)
    assert big["tax"] == pytest.approx((interest - 1000) * 0.26375)
