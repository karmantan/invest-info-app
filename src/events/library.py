EVENT_TYPES={
 "macro":["CPI","PCE","PAYROLLS","FOMC","TREASURY_YIELD_SHOCK"],
 "company_sector":["EARNINGS","REVENUE","GUIDANCE","ANALYST_REVISION"],
 "technology":["SEMICONDUCTOR_EARNINGS","AI_CAPEX","HYPERSCALER_SPEND","CHIP_DEMAND"],
 "biotechnology":["FDA_DECISION","CLINICAL_TRIAL","BIOTECH_MA","LICENSING"],
 "energy":["EIA_INVENTORY","OPEC","OIL_SHOCK","GEOPOLITICAL_ENERGY"],
}

def event_surprise(actual, consensus):
    return None if actual is None or consensus is None else actual-consensus

