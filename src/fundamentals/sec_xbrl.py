from __future__ import annotations
import json
from datetime import datetime, timezone
from pathlib import Path
import numpy as np
import pandas as pd
import requests
from src.config import ROOT
from src.events.ingestion import COMPANIES, HEADERS

ENERGY_COMPANIES={"XOM":("0000034088","Exxon Mobil","XLE","Energy"),"CVX":("0000093410","Chevron","XLE","Energy"),"COP":("0001163165","ConocoPhillips","XLE","Energy"),"SLB":("0000087347","SLB","XLE","Energy"),"EOG":("0000821189","EOG Resources","XLE","Energy")}
FUNDAMENTAL_COMPANIES={**COMPANIES,**ENERGY_COMPANIES}
CONCEPTS={
 "revenue":["RevenueFromContractWithCustomerExcludingAssessedTax","Revenues","SalesRevenueNet"],
 "operating_income":["OperatingIncomeLoss"],"net_income":["NetIncomeLoss","ProfitLoss"],
 "operating_cash_flow":["NetCashProvidedByUsedInOperatingActivities"],
 "capex":["PaymentsToAcquirePropertyPlantAndEquipment"],"debt":["LongTermDebtAndFinanceLeaseObligationsCurrent","LongTermDebtCurrent","LongTermDebtNoncurrent"],
 "cash":["CashAndCashEquivalentsAtCarryingValue","CashCashEquivalentsRestrictedCashAndRestrictedCashEquivalents"],"assets":["Assets"],
}

def fetch_companyfacts(cik,refresh=False):
    path=ROOT/"data/cache/fundamentals"/f"CIK{cik}.json"; path.parent.mkdir(parents=True,exist_ok=True)
    if path.exists() and not refresh: return json.loads(path.read_text())
    r=requests.get(f"https://data.sec.gov/api/xbrl/companyfacts/CIK{cik}.json",headers=HEADERS,timeout=60); r.raise_for_status(); path.write_text(r.text); return r.json()

def _annual_fact(payload,names,flow=True):
    facts=payload.get("facts",{}).get("us-gaap",{}); entries=[]
    for name in names:
        fact=facts.get(name,{}); units=fact.get("units",{}); candidates=units.get("USD",[])
        for x in candidates:
            if x.get("form") not in ("10-K","10-K/A") or not x.get("fy") or not x.get("filed"): continue
            if flow:
                if not x.get("start") or not x.get("end"): continue
                if (pd.Timestamp(x["end"])-pd.Timestamp(x["start"])).days<300: continue
            entries.append({"fy":int(x["fy"]),"filed":pd.Timestamp(x["filed"]),"value":float(x["val"]),"concept":name,"form":x["form"]})
        if entries: break
    if not entries: return pd.DataFrame(columns=["fy","filed","value"])
    frame=pd.DataFrame(entries).sort_values(["fy","filed"]); return frame.groupby("fy",as_index=False).first()[["fy","filed","value"]]

def company_fundamentals(ticker,meta,payload):
    cik,company,etf,sector=meta; parts=[]
    for field,names in CONCEPTS.items():
        x=_annual_fact(payload,names,flow=field not in ("debt","cash","assets")).rename(columns={"filed":f"{field}_filed","value":field}); parts.append(x)
    years=sorted(set().union(*[set(x.fy) for x in parts])); out=pd.DataFrame({"fy":years})
    for x in parts: out=out.merge(x,on="fy",how="left")
    filed=[c for c in out if c.endswith("_filed")]; out["available_date"]=out[filed].max(axis=1); out["ticker"]=ticker; out["company"]=company; out["etf"]=etf; out["sector"]=sector
    out["revenue_growth"]=out.revenue.pct_change(fill_method=None); out["operating_margin"]=out.operating_income/out.revenue; out["operating_margin_change"]=out.operating_margin.diff(); out["operating_cash_flow_growth"]=out.operating_cash_flow.pct_change(fill_method=None); out["debt_growth"]=out.debt.pct_change(fill_method=None); out["cash_growth"]=out.cash.pct_change(fill_method=None); out["free_cash_flow"]=out.operating_cash_flow-out.capex
    def score(r):
        if pd.isna(r.revenue_growth) or pd.isna(r.operating_margin_change) or pd.isna(r.operating_cash_flow_growth): return np.nan
        s=0
        s += -2 if r.revenue_growth<=-.10 else -1 if r.revenue_growth<=-.03 else 1 if r.revenue_growth>=.05 else 0
        s += -2 if r.operating_margin_change<=-.05 else -1 if r.operating_margin_change<=-.02 else 1 if r.operating_margin_change>=.02 else 0
        s += -2 if r.operating_cash_flow_growth<=-.20 else -1 if r.operating_cash_flow_growth<=-.05 else 1 if r.operating_cash_flow_growth>=.10 else 0
        if pd.notna(r.debt_growth): s += -2 if r.debt_growth>=.25 and r.cash_growth<0 else -1 if r.debt_growth>=.10 else 0
        return s
    out["stability_score"]=out.apply(score,axis=1)
    out["fundamental_state"]=pd.cut(out.stability_score,[-np.inf,-5,-2,1,np.inf],labels=["SEVERELY DETERIORATING","DETERIORATING","STABLE","IMPROVING"]).astype(object); out.loc[out.stability_score.isna(),"fundamental_state"]="INSUFFICIENT DATA"
    return out

def ingest_fundamentals(refresh=False):
    frames=[]; quality=[]
    for ticker,meta in FUNDAMENTAL_COMPANIES.items():
        try:
            frame=company_fundamentals(ticker,meta,fetch_companyfacts(meta[0],refresh)); frames.append(frame); usable=int(frame.stability_score.notna().sum()); quality.append({"ticker":ticker,"company":meta[1],"etf":meta[2],"status":"HIGH" if usable>=10 else "MEDIUM" if usable>=5 else "LOW","usable_annual_observations":usable,"first_available":str(frame.available_date.min().date()) if len(frame) else None,"last_available":str(frame.available_date.max().date()) if len(frame) else None,"source":"SEC companyfacts XBRL","point_in_time":"filing date"})
        except Exception as exc: quality.append({"ticker":ticker,"company":meta[1],"etf":meta[2],"status":"UNAVAILABLE","warning":str(exc)})
    result=pd.concat(frames,ignore_index=True); result.to_parquet(ROOT/"data/processed/company_fundamentals.parquet",index=False); (ROOT/"reports/fundamental_data_quality.json").write_text(json.dumps(quality,indent=2),encoding="utf-8"); return result
