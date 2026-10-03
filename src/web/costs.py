"""Trade Republic order costs and German capital-gains tax, vectorised over outcomes.

All functions take amounts in EUR. Simplifications (explained on the website):
the saver allowance is applied once in the year of sale, the yearly
Vorabpauschale is ignored because it is prepaid tax credited at sale, and a
loss is shown as untaxed (in reality it can offset other gains).
"""
from __future__ import annotations
import numpy as np


def effective_tax_rate(tax: dict) -> float:
    """Abgeltungsteuer incl. Soli and church tax (church tax is deductible, §32d EStG)."""
    church = tax.get("church_tax_rate", 0.0)
    base = tax["capital_gains_rate"] / (1 + tax["capital_gains_rate"] * church)
    return base * (1 + tax.get("solidarity_surcharge_rate", 0.0) + church)


def tax_on_gain(gain, tax_class: str, years_held: float, tax: dict, allowance_eur: float | None = None):
    """Tax due when selling with the given realised gain (array-friendly)."""
    gain = np.asarray(gain, dtype=float)
    allowance = tax.get("saver_allowance_eur", 0.0) if allowance_eur is None else allowance_eur
    if tax_class == "gold_etc":
        # Physical-gold ETCs are private sales (§23 EStG): tax-free after one year,
        # otherwise the whole gain is taxed at the personal rate above the Freigrenze.
        if years_held > 1: return np.zeros_like(gain)
        limit = tax.get("private_sale_exemption_limit_eur", 1000)
        return np.where(gain >= limit, gain * tax.get("personal_income_tax_rate", 0.30), 0.0)
    exemption = tax.get("partial_exemption", {}).get(tax_class, 0.0)
    taxable = np.maximum(0.0, gain * (1 - exemption) - allowance)
    return taxable * effective_tax_rate(tax)


def buy_cost(amount: float, spread_pct: float, fee_eur: float) -> tuple[float, float]:
    """Returns (market value received, total cost paid) for a cash amount spent."""
    if amount <= fee_eur: return 0.0, amount
    invested = (amount - fee_eur) * (1 - spread_pct / 2)
    return invested, amount - invested


def sell_net(final_value, cost_basis, etf: dict, years_held: float, settings: dict):
    """Cash in hand after selling everything: (net, tax, selling costs)."""
    final_value = np.asarray(final_value, dtype=float)
    fee = settings["trade_republic"]["order_fee_eur"]
    proceeds = np.maximum(0.0, final_value * (1 - etf["spread_pct"] / 2) - fee)
    selling_costs = final_value - proceeds
    tax = tax_on_gain(proceeds - cost_basis, etf["tax_class"], years_held, settings["tax"])
    return proceeds - tax, tax, selling_costs


def cash_account(amount: float, years: float, settings: dict, monthly: float = 0.0) -> dict:
    """Leaving money in the Trade Republic cash account: interest is paid and taxed every year."""
    rate = settings["trade_republic"]["cash_interest_rate"]
    tax = settings["tax"]; eff = effective_tax_rate(tax); allowance = tax.get("saver_allowance_eur", 0.0)
    months = max(1, int(round(years * 12)))
    balance = amount; year_interest = 0.0; total_tax = 0.0; paid_in = amount; path = [amount]
    for month in range(1, months + 1):
        if monthly and month > 1: balance += monthly; paid_in += monthly
        interest = balance * rate / 12; balance += interest; year_interest += interest
        if month % 12 == 0 or month == months:
            due = max(0.0, year_interest - allowance) * eff
            balance -= due; total_tax += due; year_interest = 0.0
        path.append(balance)
    return {"net": balance, "tax": total_tax, "paid_in": paid_in, "path": np.array(path)}
