from __future__ import annotations

import json, math
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Callable
from zoneinfo import ZoneInfo

import yaml

from src.config import ROOT
from src.data.adapters import YahooBidAsk
from src.storage import Database

UTC=timezone.utc

def utc_iso(value: datetime) -> str:
    if value.tzinfo is None: raise ValueError("timestamp must be timezone-aware")
    return value.astimezone(UTC).isoformat()

def load_spread_config(path: str | Path | None=None) -> dict:
    with Path(path or ROOT/"config/spreads.yaml").open(encoding="utf-8") as handle:
        return yaml.safe_load(handle)

def _number(value):
    if value is None or isinstance(value,bool): return None
    try: result=float(value)
    except (TypeError,ValueError): return None
    return result if math.isfinite(result) else None

def validate_quote(raw: dict, mapping: dict, requested: datetime, collected: datetime, cfg: dict) -> tuple[dict,dict|None]:
    metadata={k:v for k,v in raw.items() if k not in {"provider_fields"}}
    metadata["provider_fields"]=raw.get("provider_fields",[])
    base={"requested_at_utc":utc_iso(requested),"collected_at_utc":utc_iso(collected),
          "available_at_utc":utc_iso(collected),"isin":mapping["isin"],"ticker":mapping["yahoo_ticker"],
          "expected_exchange":mapping["exchange"],"expected_currency":mapping["currency"],
          "provider":cfg["provider"],"http_status":None,"provider_metadata_json":json.dumps(metadata,sort_keys=True,default=str)}
    reported_ticker=str(raw.get("ticker") or "").upper()
    exchange=str(raw.get("exchange") or "").upper()
    exchange_code=str(raw.get("exchange_code") or "").upper()
    allowed={mapping["exchange"].upper(),*(str(x).upper() for x in mapping.get("exchange_aliases",[]))}
    if reported_ticker and reported_ticker != mapping["yahoo_ticker"].upper():
        return {**base,"status":"REJECTED","quality":"UNUSABLE","reason":"Yahoo ticker does not match the verified listing mapping"},None
    if exchange not in allowed and exchange_code not in allowed:
        return {**base,"status":"REJECTED","quality":"UNUSABLE","reason":f"Yahoo exchange {raw.get('exchange') or raw.get('exchange_code') or 'missing'} does not match Xetra mapping"},None
    if str(raw.get("currency") or "").upper()!=mapping["currency"].upper():
        return {**base,"status":"REJECTED","quality":"UNUSABLE","reason":"Yahoo trading currency does not match the verified listing mapping"},None
    bid,ask=_number(raw.get("bid")),_number(raw.get("ask"))
    if bid is None or ask is None:
        return {**base,"status":"REJECTED","quality":"UNUSABLE","reason":"Bid or ask is missing or nonfinite; spread remains unavailable"},None
    if bid<=0 or ask<=0:
        return {**base,"status":"REJECTED","quality":"UNUSABLE","reason":"Bid and ask must be positive; spread remains unavailable"},None
    if bid>ask:
        return {**base,"status":"REJECTED","quality":"UNUSABLE","reason":"Crossed quote (bid above ask); spread remains unavailable"},None
    midpoint=(bid+ask)/2; spread=ask-bid; pct=spread/midpoint*100; bps=spread/midpoint*10000
    flags=[]
    if spread==0: flags.append("zero-width quote")
    if bps>=float(cfg["suspicious_spread_bps"]): flags.append("suspiciously wide quote")
    # Yahoo exposes regularMarketTime (last trade), not a bid/ask timestamp.
    flags.append("quote freshness unverified")
    delay=_number(raw.get("exchange_data_delayed_by"))
    if delay and delay>0: flags.append(f"Yahoo reports data delayed by {delay:g} minutes")
    if str(raw.get("market_state") or "").upper() not in {"REGULAR","OPEN"}: flags.append("Yahoo reports market closed or delayed state")
    quality="USABLE_WITH_FLAGS" if flags else "USABLE"
    reason="; ".join(flags) if flags else "Indicative bid/ask passed validation"
    attempt={**base,"status":"OBSERVED","quality":quality,"reason":reason}
    observation={"requested_at_utc":base["requested_at_utc"],"collected_at_utc":base["collected_at_utc"],
      "available_at_utc":base["available_at_utc"],"isin":mapping["isin"],"ticker":mapping["yahoo_ticker"],
      "exchange":mapping["exchange"],"currency":mapping["currency"],"bid":bid,"ask":ask,
      "bid_size":(_number(raw.get("bid_size")) if (_number(raw.get("bid_size")) or 0)>0 else None),
      "ask_size":(_number(raw.get("ask_size")) if (_number(raw.get("ask_size")) or 0)>0 else None),"midpoint":midpoint,
      "spread_currency":spread,"spread_percent":pct,"spread_bps":bps,"quote_timestamp_utc":None,
      "quote_freshness":"UNVERIFIED — Yahoo supplies no quote-specific timestamp",
      "provider":cfg["provider"],"source_label":cfg["source_label"],"quality":quality,"reason":reason,
      "provider_metadata_json":base["provider_metadata_json"]}
    return attempt,observation

def is_xetra_collection_time(now: datetime, cfg: dict, calendar=None) -> tuple[bool,str]:
    local=now.astimezone(ZoneInfo(cfg["timezone"])); day=local.date()
    if calendar is None:
        import exchange_calendars as xcals
        calendar=xcals.get_calendar(cfg["market_calendar"])
    if not calendar.is_session(str(day)): return False,"not an Xetra trading day"
    open_h,open_m=map(int,cfg["window"]["open"].split(":")); close_h,close_m=map(int,cfg["window"]["close"].split(":"))
    minute=local.hour*60+local.minute
    return (open_h*60+open_m<=minute<=close_h*60+close_m,
            "within Xetra collection window" if open_h*60+open_m<=minute<=close_h*60+close_m else "outside 09:00–17:30 Europe/Berlin")

def collect(db: Database|None=None, cfg: dict|None=None, adapter=None, now: datetime|None=None, force=False) -> dict:
    db=db or Database(); db.initialize(); cfg=cfg or load_spread_config(); adapter=adapter or YahooBidAsk(); now=now or datetime.now(UTC)
    with db.connect() as c:
        row=c.execute("SELECT value FROM spread_collector_state WHERE key='cooldown_until_utc'").fetchone()
    if row and datetime.fromisoformat(row[0])>now:
        return {"status":"COOLDOWN","reason":f"Yahoo cooldown active until {row[0]}","requests":0}
    if not force:
        allowed,reason=is_xetra_collection_time(now,cfg)
        if not allowed: return {"status":"SKIPPED","reason":reason,"requests":0}
    results=[]
    for mapping in cfg["mappings"]:
        requested=datetime.now(UTC)
        try:
            raw=adapter.fetch(mapping["yahoo_ticker"]); collected=datetime.now(UTC)
            attempt,observation=validate_quote(raw,mapping,requested,collected,cfg)
        except Exception as exc:
            collected=datetime.now(UTC); message=str(exc); rate_limited="429" in message or "too many requests" in message.lower()
            attempt={"requested_at_utc":utc_iso(requested),"collected_at_utc":utc_iso(collected),"available_at_utc":utc_iso(collected),
              "isin":mapping["isin"],"ticker":mapping["yahoo_ticker"],"expected_exchange":mapping["exchange"],
              "expected_currency":mapping["currency"],"provider":cfg["provider"],"status":"RATE_LIMITED" if rate_limited else "FAILED",
              "quality":"UNUSABLE","reason":message[:1000],"http_status":429 if rate_limited else None,"provider_metadata_json":"{}"}
            observation=None
            if rate_limited:
                until=now+timedelta(seconds=max(3600,int(cfg["cooldown_seconds"])))
                with db.connect() as c: c.execute("INSERT INTO spread_collector_state(key,value,updated_at_utc) VALUES('cooldown_until_utc',?,?) ON CONFLICT(key) DO UPDATE SET value=excluded.value,updated_at_utc=excluded.updated_at_utc",(utc_iso(until),utc_iso(collected)))
        db.record_spread_attempt(attempt,observation); results.append({"isin":mapping["isin"],"status":attempt["status"],"reason":attempt["reason"]})
        if attempt["status"]=="RATE_LIMITED": break
    return {"status":"COMPLETE","requests":len(results),"results":results}
