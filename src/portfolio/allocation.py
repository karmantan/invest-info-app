from __future__ import annotations
from dataclasses import dataclass

@dataclass
class AllocationDecision:
    action: str; amount_now_eur: float; target_holding_eur: float; vetoes: list[str]; utility: float

def allocate(forecast: dict, current_holding_eur: float, cash_eur: float, portfolio_value_eur: float, sector_value_eur: float, config: dict):
    p=config["portfolio"]; r=config["risk"]; veto=[]
    expected=forecast["expected_return"]; downside=max(0,-forecast["p10"]); utility=expected-r["downside_lambda"]*downside
    if forecast.get("sample_size",0)<r["minimum_analogue_observations"]: veto.append("insufficient historical observations")
    if forecast.get("probability_loss_gt_5pct",1)>r["maximum_probability_loss_5pct"]: veto.append("tail-loss probability is too high")
    if forecast.get("confidence",0)<r["minimum_confidence_for_action"]: veto.append("evidence confidence is too weak")
    if forecast.get("expected_excess_return",0)<r["minimum_expected_excess_return"]: veto.append("expected value added after costs is too small")
    max_position=portfolio_value_eur*p["maximum_position_weight"]; max_sector=max(0,portfolio_value_eur*p["maximum_sector_weight"]-sector_value_eur+current_holding_eur)
    tactical_cap=portfolio_value_eur*p["tactical_sleeve_weight"]
    target=min(max_position,max_sector,tactical_cap*max(0,min(1,forecast.get("confidence",0))))
    amount=max(0,min(cash_eur,target-current_holding_eur)); amount=round(amount/50)*50
    if amount<p["minimum_trade_eur"]: veto.append("trade would be too small after costs")
    return AllocationDecision("BUY" if not veto and utility>0 else "NO ACTION", amount if not veto and utility>0 else 0, target, veto, utility)

