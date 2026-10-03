from __future__ import annotations

from src.storage import Database


def validate_confirmed_snapshot(snapshot: dict) -> list[str]:
    errors=[]
    if not snapshot.get("snapshot_date"): errors.append("A reconciliation date is required.")
    if not snapshot.get("holdings"): errors.append("At least one holding must be reviewed.")
    if snapshot.get("total_value_eur") is None or snapshot.get("total_value_eur",0)<=0: errors.append("A positive total portfolio value is required.")
    for i,h in enumerate(snapshot.get("holdings",[]),1):
        if not h.get("isin") and not h.get("security_name"): errors.append(f"Holding {i} needs a security name or ISIN.")
        if h.get("market_value_eur") is None: errors.append(f"Holding {i} needs a reviewed market value.")
    if snapshot.get("material_reconciliation_mismatch"):
        errors.append("The holdings total materially differs from the reported securities value. Resolve the reconciliation difference before confirming.")
    return errors


def reconcile_confirmed_snapshot(snapshot: dict, confirmed: bool, db: Database) -> int:
    if not confirmed: raise ValueError("Explicit confirmation is required before reconciliation.")
    errors=validate_confirmed_snapshot(snapshot)
    if errors: raise ValueError(" ".join(errors))
    return db.insert_snapshot(snapshot)


def sleeve_values(total_value_eur: float) -> dict[str,float]:
    return {"5%":total_value_eur*.05,"7.5%":total_value_eur*.075,"10%":total_value_eur*.10}
