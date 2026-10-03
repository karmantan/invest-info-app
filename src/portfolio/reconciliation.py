def reconcile(previous: dict | None, current: dict):
    if not previous: return [{"change":"initial snapshot","estimated":False}]
    old={h.get("isin") or h.get("security_name"):h for h in previous.get("holdings",[])}; new={h.get("isin") or h.get("security_name"):h for h in current.get("holdings",[])}; changes=[]
    for key in sorted(old.keys()|new.keys()):
        if key not in old: changes.append({"change":"new position","security":key,"estimated":True})
        elif key not in new: changes.append({"change":"removed position","security":key,"estimated":True})
        elif old[key].get("quantity")!=new[key].get("quantity"): changes.append({"change":"quantity changed","security":key,"from":old[key].get("quantity"),"to":new[key].get("quantity"),"estimated":True})
    if previous.get("cash_eur")!=current.get("cash_eur"): changes.append({"change":"cash changed","from":previous.get("cash_eur"),"to":current.get("cash_eur"),"estimated":True})
    return changes

