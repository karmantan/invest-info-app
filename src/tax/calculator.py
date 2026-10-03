def estimated_tax(gain_eur, remaining_allowance_eur, settings, etf_eligible=True, loss_offset_eur=0):
    taxable=max(0.0,gain_eur-loss_offset_eur-remaining_allowance_eur)
    if etf_eligible: taxable*=1-settings.get("etf_partial_exemption_rate",0)
    base=taxable*settings["capital_gains_rate"]
    return base*(1+settings.get("solidarity_surcharge_rate",0)+settings.get("church_tax_rate",0))

