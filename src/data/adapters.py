from __future__ import annotations
import io, os, json
from datetime import datetime, timezone
from pathlib import Path
import pandas as pd
import requests
from src.config import ROOT

class YahooBidAsk:
    """Small yfinance boundary. Provider last-trade times are metadata, never quote times."""
    provider = "Yahoo Finance via yfinance"

    def fetch(self, ticker: str) -> dict:
        try:
            import yfinance as yf
        except ImportError as exc:
            raise RuntimeError("yfinance is not installed") from exc
        info = yf.Ticker(ticker).get_info()
        if not isinstance(info, dict) or not info:
            raise RuntimeError(f"Yahoo Finance returned no quote metadata for {ticker}")
        return {
            "ticker": info.get("symbol") or ticker,
            "exchange": info.get("fullExchangeName") or info.get("exchange"),
            "exchange_code": info.get("exchange"),
            "currency": info.get("currency"),
            "bid": info.get("bid"), "ask": info.get("ask"),
            "bid_size": info.get("bidSize"), "ask_size": info.get("askSize"),
            # This is explicitly retained only as last-trade/provider metadata.
            "regular_market_time": info.get("regularMarketTime"),
            "market_state": info.get("marketState"),
            "exchange_data_delayed_by": info.get("exchangeDataDelayedBy"),
            "quote_source_name": info.get("quoteSourceName"),
            "quote_type": info.get("quoteType"),
            "provider_fields": sorted(info.keys()),
        }

class CachedAdapter:
    def __init__(self, cache_dir: str | Path = ROOT / "data/cache"):
        self.cache_dir = Path(cache_dir); self.cache_dir.mkdir(parents=True, exist_ok=True)
    def _read(self, key):
        path = self.cache_dir / f"{key}.parquet"
        return pd.read_parquet(path) if path.exists() else None
    def _write(self, key, frame):
        frame.to_parquet(self.cache_dir / f"{key}.parquet", index=False)

class AlphaVantagePrices(CachedAdapter):
    """Free-tier adjusted daily prices. Never substitutes invented observations."""
    def fetch(self, ticker: str, refresh: bool = False) -> pd.DataFrame:
        key = f"alpha_{ticker.lower()}"
        if not refresh and (cached := self._read(key)) is not None: return cached
        api_key = os.getenv("ALPHA_VANTAGE_API_KEY")
        if not api_key: raise RuntimeError("ALPHA_VANTAGE_API_KEY is not configured")
        response = requests.get("https://www.alphavantage.co/query", params={"function":"TIME_SERIES_DAILY_ADJUSTED","symbol":ticker,"outputsize":"full","datatype":"csv","apikey":api_key}, timeout=30)
        response.raise_for_status(); frame = pd.read_csv(io.StringIO(response.text))
        if "timestamp" not in frame: raise RuntimeError("Price provider returned no observations (free-tier limit may apply)")
        frame = frame.rename(columns={"timestamp":"date","adjusted_close":"close"})[["date","close"]].sort_values("date")
        frame["ticker"] = ticker; frame["source"] = "Alpha Vantage"; self._write(key, frame); return frame

class YahooChartPrices(CachedAdapter):
    """No-key fallback using Yahoo Finance chart data, with adjusted close kept explicit."""
    endpoint = "https://query2.finance.yahoo.com/v8/finance/chart/{ticker}"
    def fetch(self, ticker: str, refresh: bool = False) -> pd.DataFrame:
        safe=ticker.replace("=","_").lower(); key=f"yahoo_{safe}"
        if not refresh and (cached:=self._read(key)) is not None: return cached
        r=requests.get(self.endpoint.format(ticker=ticker),params={"period1":0,"period2":int(datetime.now(timezone.utc).timestamp())+86400,"interval":"1d","events":"div,splits","includeAdjustedClose":"true"},headers={"User-Agent":"Mozilla/5.0 invest-info-app/0.1"},timeout=60)
        if r.status_code==429: raise RuntimeError("Yahoo Finance rate limit reached; cached downloads are preserved and ingestion can be resumed")
        r.raise_for_status(); result=r.json().get("chart",{}).get("result")
        if not result: raise RuntimeError(f"Yahoo Finance returned no history for {ticker}")
        payload=result[0]; quote=payload["indicators"]["quote"][0]; adjusted=payload["indicators"].get("adjclose",[{}])[0].get("adjclose")
        frame=pd.DataFrame({"date":pd.to_datetime(payload["timestamp"],unit="s",utc=True).tz_convert(None).normalize(),"close_unadjusted":quote.get("close"),"close":adjusted or quote.get("close"),"volume":quote.get("volume")})
        frame=frame.dropna(subset=["close"]).drop_duplicates("date").sort_values("date"); frame["ticker"]=ticker; frame["currency"]=payload["meta"].get("currency"); frame["source"]="Yahoo Finance chart"; frame["adjustment"]="adjusted close" if adjusted else "unadjusted close"
        self._write(key,frame); return frame

class FredCsvAdapter(CachedAdapter):
    """Official FRED graph CSV; values reflect the latest available vintage."""
    def fetch(self, series_id: str, refresh: bool=False) -> pd.DataFrame:
        key=f"fredcsv_{series_id.lower()}"
        if not refresh and (cached:=self._read(key)) is not None: return cached
        r=requests.get("https://fred.stlouisfed.org/graph/fredgraph.csv",params={"id":series_id},headers={"User-Agent":"invest-info-app/0.1"},timeout=60); r.raise_for_status()
        frame=pd.read_csv(io.StringIO(r.text)).rename(columns={"observation_date":"date",series_id:"value"}); frame["date"]=pd.to_datetime(frame["date"]); frame["value"]=pd.to_numeric(frame["value"],errors="coerce"); frame["series_id"]=series_id; frame["source"]="FRED latest vintage"
        self._write(key,frame); return frame

def price_quality(frame: pd.DataFrame, ticker: str, proxy: dict | None=None) -> dict:
    x=frame.sort_values("date"); returns=x["close"].pct_change(); weekdays=pd.bdate_range(x.date.min(),x.date.max()); missing=max(0,len(weekdays)-len(x))
    warnings=[]
    if (returns.abs()>.35).any(): warnings.append("one or more adjusted daily moves exceed 35%; inspect splits/provider data")
    if missing/len(weekdays)>.08: warnings.append("more than 8% of weekdays are absent (holidays explain part of this)")
    return {"status":"WORKING" if len(x)>=1000 else "PARTIAL","provider":x.source.iloc[0],"ticker":ticker,"download_timestamp_utc":datetime.now(timezone.utc).isoformat(),"earliest_date":str(x.date.min().date()),"latest_date":str(x.date.max().date()),"rows":len(x),"adjustment":x.adjustment.iloc[0],"currency":x.currency.iloc[0],"missing_weekdays":missing,"suspicious_jump_count":int((returns.abs()>.35).sum()),"proxy":proxy,"warnings":warnings}

class FredAlfredAdapter(CachedAdapter):
    def fetch(self, series_id: str, realtime_start: str | None = None, realtime_end: str | None = None, refresh=False):
        key = f"fred_{series_id}_{realtime_start or 'latest'}_{realtime_end or 'latest'}"
        if not refresh and (cached := self._read(key)) is not None: return cached
        api_key = os.getenv("FRED_API_KEY")
        if not api_key: raise RuntimeError("FRED_API_KEY is required for explicit vintage metadata")
        params = {"series_id":series_id,"api_key":api_key,"file_type":"json"}
        if realtime_start: params["realtime_start"] = realtime_start
        if realtime_end: params["realtime_end"] = realtime_end
        r=requests.get("https://api.stlouisfed.org/fred/series/observations",params=params,timeout=30); r.raise_for_status()
        frame=pd.DataFrame(r.json()["observations"]).rename(columns={"date":"observation_date"})
        frame["value"]=pd.to_numeric(frame["value"],errors="coerce"); frame["series_id"]=series_id; frame["source"]="FRED/ALFRED"
        self._write(key,frame); return frame

class SecEdgarAdapter(CachedAdapter):
    def submissions(self, cik: str, refresh=False):
        cik=str(cik).zfill(10); key=f"sec_submissions_{cik}"
        path=self.cache_dir/f"{key}.json"
        if path.exists() and not refresh: return pd.read_json(path)
        agent=os.getenv("SEC_USER_AGENT")
        if not agent or "@" not in agent: raise RuntimeError("Set SEC_USER_AGENT to an identifying name and contact email")
        r=requests.get(f"https://data.sec.gov/submissions/CIK{cik}.json",headers={"User-Agent":agent},timeout=30); r.raise_for_status()
        recent=pd.DataFrame(r.json()["filings"]["recent"]); recent.to_json(path,orient="records"); return recent
