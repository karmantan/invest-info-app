from __future__ import annotations
import html, json, re
from datetime import datetime, timezone
from pathlib import Path
from zoneinfo import ZoneInfo
import pandas as pd
import requests
from src.config import ROOT
from src.data.adapters import YahooChartPrices, FredCsvAdapter

COMPANIES={
 "NVDA":("0001045810","NVIDIA","SMH","Semiconductors"),"AMD":("0000002488","AMD","SMH","Semiconductors"),"AVGO":("0001730168","Broadcom","SMH","Semiconductors"),"INTC":("0000050863","Intel","SMH","Semiconductors"),"QCOM":("0000804328","Qualcomm","SMH","Semiconductors"),"MU":("0000723125","Micron","SMH","Semiconductors"),"AMAT":("0000006951","Applied Materials","SMH","Semiconductors"),"TXN":("0000097476","Texas Instruments","SMH","Semiconductors"),
 "AAPL":("0000320193","Apple","XLK","Technology"),"MSFT":("0000789019","Microsoft","XLK","Technology"),"ORCL":("0001341439","Oracle","XLK","Technology"),"CRM":("0001108524","Salesforce","XLK","Technology"),"ADBE":("0000796343","Adobe","XLK","Technology"),"IBM":("0000051143","IBM","XLK","Technology"),
}
HEADERS={"User-Agent":"invest-info-app/0.1 research-only"}
BLS_HEADERS={"User-Agent":"Mozilla/5.0 invest-info-app-event-research/0.1"}

def _get_json(url,cache,refresh=False):
    cache=Path(cache); cache.parent.mkdir(parents=True,exist_ok=True)
    if cache.exists() and not refresh: return json.loads(cache.read_text())
    r=requests.get(url,headers=HEADERS,timeout=60); r.raise_for_status(); cache.write_text(r.text); return r.json()

def sec_earnings_events(refresh=False):
    rows=[]; cache=ROOT/"data/cache/events/sec"
    for ticker,(cik,company,etf,sector) in COMPANIES.items():
        root=_get_json(f"https://data.sec.gov/submissions/CIK{cik}.json",cache/f"{cik}.json",refresh); filings=[root["filings"]["recent"]]
        for file in root["filings"].get("files",[]): filings.append(_get_json("https://data.sec.gov/submissions/"+file["name"],cache/file["name"],refresh))
        for f in filings:
            count=len(f.get("form",[]))
            for i in range(count):
                if f["form"][i]!="8-K" or "2.02" not in str(f.get("items",[""]*count)[i]): continue
                accepted=pd.to_datetime(f.get("acceptanceDateTime",[None]*count)[i],utc=True,errors="coerce"); filing=pd.to_datetime(f["filingDate"][i])
                timing="unknown"; effective=filing
                if pd.notna(accepted):
                    eastern=accepted.tz_convert(ZoneInfo("America/New_York")); timing="after_market_close" if eastern.hour>=16 else "before_or_during_market"; effective=filing+pd.offsets.BDay(1) if eastern.hour>=16 else filing
                rows.append({"event_type":"major_earnings","company":company,"company_ticker":ticker,"etf":etf,"sector":sector,"announcement_date":filing,"effective_market_date":pd.Timestamp(effective).normalize(),"announcement_timing":timing,"actual_earnings":None,"expected_earnings":None,"earnings_surprise":None,"revenue_surprise":None,"guidance_direction":None,"scheduled":None,"source":"SEC submissions 8-K Item 2.02","point_in_time_quality":"filing timestamp is point-in-time; filing may lag the actual public earnings release","accession_number":f.get("accessionNumber",[None]*count)[i]})
    return pd.DataFrame(rows).drop_duplicates(["company_ticker","announcement_date","accession_number"]).sort_values("effective_market_date")

def _strip(fragment): return re.sub(r"\s+"," ",html.unescape(re.sub("<[^>]+>"," ",fragment))).strip()

def bls_cpi_calendar(start=2005,end=None,refresh=False):
    end=end or datetime.now().year; rows=[]; cache=ROOT/"data/cache/events/bls"; cache.mkdir(parents=True,exist_ok=True)
    for year in range(start,end+1):
        path=cache/f"schedule_{year}.html"
        if not path.exists() or refresh:
            r=requests.get(f"https://www.bls.gov/schedule/{year}/home.htm",headers=BLS_HEADERS,timeout=60)
            if not r.ok: continue
            path.write_text(r.text,encoding="utf-8")
        text=path.read_text(encoding="utf-8",errors="ignore")
        for tr in re.findall(r"<tr[^>]*>(.*?)</tr>",text,re.S|re.I):
            if "Consumer Price Index" not in tr: continue
            cells=[_strip(x) for x in re.findall(r"<td[^>]*>(.*?)</td>",tr,re.S|re.I)]
            if len(cells)<3: continue
            released=pd.to_datetime(cells[0],errors="coerce")
            if pd.isna(released): continue
            reference=re.search(r"for\s+([A-Za-z]+\s+\d{4})",cells[2]); ref=pd.to_datetime(reference.group(1),errors="coerce") if reference else pd.NaT
            rows.append({"event_type":"cpi","company":None,"company_ticker":None,"etf":"ALL","sector":"Macro","announcement_date":released,"effective_market_date":released,"announcement_timing":cells[1],"reference_month":ref,"reported_cpi":None,"prior_cpi":None,"consensus":None,"surprise":None,"scheduled":True,"source":"BLS archived release calendar","point_in_time_quality":"official release date/time; reported values later matched to current non-seasonally-adjusted BLS series, not a vintage release table"})
    return pd.DataFrame(rows).drop_duplicates("announcement_date").sort_values("effective_market_date")

def add_bls_values(events,refresh=False):
    cache=ROOT/"data/cache/events/bls/cpi_values.json"; payload=[]
    if cache.exists() and not refresh: payload=json.loads(cache.read_text())
    else:
        for start,end in ((2004,2014),(2015,2025),(2026,2026)):
            r=requests.post("https://api.bls.gov/publicAPI/v2/timeseries/data/",json={"seriesid":["CUUR0000SA0"],"startyear":str(start),"endyear":str(end)},headers=BLS_HEADERS,timeout=60)
            if r.ok:
                for series in r.json().get("Results",{}).get("series",[]): payload.extend(series.get("data",[]))
        cache.write_text(json.dumps(payload))
    values={pd.Timestamp(int(x["year"]),int(x["period"][1:]),1):float(x["value"]) for x in payload if x.get("period","").startswith("M") and x["period"]!="M13"}
    out=events.copy(); out["reported_cpi"]=out.reference_month.map(values); out["prior_cpi"]=out.reference_month.map(lambda d:values.get(d-pd.offsets.MonthBegin(1)) if pd.notna(d) else None); return out

def fomc_calendar(start=2005,end=None,refresh=False):
    end=end or datetime.now().year; rows=[]; cache=ROOT/"data/cache/events/fed"; cache.mkdir(parents=True,exist_ok=True)
    sources=[]
    for year in range(start,end+1):
        path=cache/f"fomc_{year}.html"
        if not path.exists() or refresh:
            r=requests.get(f"https://www.federalreserve.gov/monetarypolicy/fomchistorical{year}.htm",headers=HEADERS,timeout=60)
            if not r.ok: continue
            path.write_text(r.text,encoding="utf-8")
        if path.exists(): sources.append(path)
    current=cache/"fomccalendars.html"
    if current.exists(): sources.append(current)
    for path in sources:
        text=path.read_text(encoding="utf-8",errors="ignore")
        for match in re.finditer(r'href="[^"]*monetary(\d{8})a\.htm"[^>]*>(?:Statement|HTML)',text,re.I):
            date=pd.to_datetime(match.group(1),format="%Y%m%d"); preceding=_strip(text[max(0,match.start()-1600):match.start()]); unscheduled="unscheduled" in preceding.lower()
            rows.append({"event_type":"fomc","company":None,"company_ticker":None,"etf":"ALL","sector":"Macro","announcement_date":date,"effective_market_date":date,"announcement_timing":"approximately 14:00 ET for scheduled decisions; see source statement","scheduled":not unscheduled,"target_rate_change":None,"source":"Federal Reserve historical FOMC statement archive","point_in_time_quality":"official statement date; release time not parsed per statement"})
    return pd.DataFrame(rows).drop_duplicates("announcement_date").sort_values("effective_market_date")

def add_fomc_rates(events,refresh=False):
    parts=[]
    for sid in ("DFEDTAR","DFEDTARU"):
        try: parts.append(FredCsvAdapter().fetch(sid,refresh)[["date","value"]].rename(columns={"value":sid.lower()}))
        except Exception: pass
    out=events.copy()
    if not parts: return out
    rates=parts[0]
    for p in parts[1:]: rates=rates.merge(p,on="date",how="outer")
    rates=rates.sort_values("date"); rates["target"]=rates.get("dfedtaru").combine_first(rates.get("dfedtar")); rates["target_rate_change"]=rates.target.diff()
    out=out.drop(columns=["target_rate_change"],errors="ignore")
    return pd.merge_asof(out.sort_values("effective_market_date"),rates[["date","target_rate_change"]].sort_values("date"),left_on="effective_market_date",right_on="date",direction="backward",tolerance=pd.Timedelta("3d")).drop(columns="date")

def ingest_event_library(refresh=False):
    earnings=sec_earnings_events(refresh); cpi=add_bls_values(bls_cpi_calendar(refresh=refresh),refresh); fomc=add_fomc_rates(fomc_calendar(refresh=refresh),refresh)
    events=pd.concat([earnings,cpi,fomc],ignore_index=True,sort=False); events["event_id"]=[f"EVT-{i:05d}" for i in range(1,len(events)+1)]; events["ingested_at_utc"]=datetime.now(timezone.utc).isoformat()
    path=ROOT/"data/processed/events.parquet"; events.to_parquet(path,index=False); return events
