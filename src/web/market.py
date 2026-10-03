"""Daily prices from Yahoo Finance with an on-disk cache and EUR conversion.

The cache lives in the persistent data directory, so a restart (or a Yahoo
outage) still has the last good prices. Stale data is used rather than none,
and the page shows the date of the latest price.
"""
from __future__ import annotations
import json, os, re, threading, time
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable
import pandas as pd
import requests
import yaml
from src.config import ROOT

# Yahoo answers 429 to non-browser user agents from cloud hosts.
HEADERS = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/126.0 Safari/537.36",
           "Accept": "application/json,text/plain,*/*", "Accept-Language": "en-US,en;q=0.9"}
CHART_HOSTS = ("https://query2.finance.yahoo.com", "https://query1.finance.yahoo.com")


def yahoo_chart(symbol: str) -> tuple[pd.Series, dict]:
    """Full daily adjusted-close history and metadata for one Yahoo symbol."""
    last_error: Exception | None = None
    for host in CHART_HOSTS:
        try:
            r = requests.get(f"{host}/v8/finance/chart/{symbol}", params={"period1": 0, "period2": int(time.time()) + 86400, "interval": "1d", "events": "div,splits", "includeAdjustedClose": "true"}, headers=HEADERS, timeout=30)
            r.raise_for_status()
            result = (r.json().get("chart") or {}).get("result")
            if not result: raise RuntimeError(f"no history for {symbol}")
            payload = result[0]
            stamps = payload.get("timestamp") or []
            close = (payload["indicators"].get("adjclose") or [{}])[0].get("adjclose") or payload["indicators"]["quote"][0].get("close")
            index = pd.to_datetime(stamps, unit="s", utc=True).tz_convert(None).normalize()
            series = pd.Series(close, index=index, dtype="float64").dropna()
            series = series[~series.index.duplicated(keep="last")].sort_index()
            if series.empty: raise RuntimeError(f"no prices for {symbol}")
            meta = payload.get("meta", {})
            return series, {"currency": meta.get("currency"), "name": meta.get("longName") or meta.get("shortName"), "exchange": meta.get("fullExchangeName") or meta.get("exchangeName")}
        except Exception as exc:  # try the other host, then yfinance
            last_error = exc
    try:
        return _yfinance_history(symbol)
    except Exception as exc:
        raise RuntimeError(f"Yahoo Finance unavailable for {symbol}: {last_error}; yfinance: {exc}") from exc


def _yfinance_history(symbol: str) -> tuple[pd.Series, dict]:
    """Fallback: yfinance manages Yahoo's cookie/crumb handshake, which cloud hosts sometimes need."""
    import yfinance as yf
    ticker = yf.Ticker(symbol)
    frame = ticker.history(period="max", interval="1d", auto_adjust=True)
    if frame is None or frame.empty: raise RuntimeError(f"no history for {symbol}")
    index = pd.DatetimeIndex(frame.index).tz_localize(None).normalize()
    series = pd.Series(frame["Close"].to_numpy(dtype=float), index=index).dropna()
    series = series[~series.index.duplicated(keep="last")].sort_index()
    meta = ticker.history_metadata or {}
    return series, {"currency": meta.get("currency"), "name": meta.get("longName") or meta.get("shortName"), "exchange": meta.get("fullExchangeName")}


def yahoo_search_isin(isin: str) -> list[dict]:
    r = requests.get("https://query2.finance.yahoo.com/v1/finance/search", params={"q": isin, "quotesCount": 10, "newsCount": 0}, headers=HEADERS, timeout=20)
    r.raise_for_status()
    return r.json().get("quotes", [])


GERMAN_VENUES = (".DE", ".F", ".SG", ".MU", ".DU", ".BE")


def rank_symbols(quotes: list[dict], isin: str | None = None) -> list[str]:
    """Candidate symbols, Xetra first, then other German venues, then the rest.
    Yahoo's placeholder listings named after the ISIN (e.g. 'IE00….SG') carry no prices and are dropped."""
    symbols = [q.get("symbol") for q in quotes if q.get("symbol")]
    symbols = [s for s in dict.fromkeys(symbols) if not (isin and s.upper().startswith(isin.upper()))]
    order = lambda s: next((i for i, suffix in enumerate(GERMAN_VENUES) if s.endswith(suffix)), len(GERMAN_VENUES))
    return sorted(symbols, key=order)


def pick_symbol(quotes: list[dict], isin: str | None = None) -> str | None:
    ranked = rank_symbols(quotes, isin)
    return ranked[0] if ranked else None


def clean_name(name: str | None) -> str | None:
    """'Semiconductor USD (Acc)' -> 'Semiconductor' for a name search."""
    if not name: return None
    text = re.sub(r"\((Acc|Dist)\)|\b(USD|EUR|GBP)\b", " ", name)
    return re.sub(r"\s+", " ", text).strip() or None


class PriceStore:
    def __init__(self, cache_dir: Path, max_age_hours: float = 12, fetcher: Callable[[str], tuple[pd.Series, dict]] | None = None):
        self.dir = Path(cache_dir); self.dir.mkdir(parents=True, exist_ok=True)
        self.max_age = max_age_hours * 3600
        self.fetcher = fetcher or yahoo_chart
        self.errors: dict[str, str] = {}
        self._locks: dict[str, threading.Lock] = {}
        self._guard = threading.Lock()

    def _lock(self, symbol: str) -> threading.Lock:
        with self._guard: return self._locks.setdefault(symbol, threading.Lock())

    def _paths(self, symbol: str) -> tuple[Path, Path]:
        safe = "".join(c if c.isalnum() or c in ".-" else "_" for c in symbol)
        return self.dir / f"{safe}.parquet", self.dir / f"{safe}.json"

    def raw(self, symbol: str) -> tuple[pd.Series | None, dict]:
        with self._lock(symbol): return self._raw(symbol)

    def _raw(self, symbol: str) -> tuple[pd.Series | None, dict]:
        data_path, meta_path = self._paths(symbol)
        cached = None; meta: dict = {}
        if data_path.exists() and meta_path.exists():
            try:
                meta = json.loads(meta_path.read_text())
                frame = pd.read_parquet(data_path)
                cached = pd.Series(frame["close"].to_numpy(), index=pd.to_datetime(frame["date"]), name=symbol)
                if cached.dropna().empty: cached = None
            except Exception:  # unreadable cache: refetch
                cached, meta = None, {}
            if cached is not None and time.time() - meta.get("fetched_at", 0) < self.max_age: return cached, meta
        try:
            series, meta = self.fetcher(symbol)
            meta = {**meta, "fetched_at": time.time(), "symbol": symbol}
            tmp = data_path.with_suffix(".tmp")
            pd.DataFrame({"date": series.index, "close": series.to_numpy()}).to_parquet(tmp, index=False)
            os.replace(tmp, data_path)
            meta_path.with_suffix(".tmpj").write_text(json.dumps(meta)); os.replace(meta_path.with_suffix(".tmpj"), meta_path)
            self.errors.pop(symbol, None)
            return series.rename(symbol), meta
        except Exception as exc:
            self.errors[symbol] = str(exc)
            return cached, {**meta, "stale": cached is not None}

    def eur(self, symbol: str) -> tuple[pd.Series | None, dict]:
        """Price history converted to EUR (GBp handled)."""
        series, meta = self.raw(symbol)
        if series is None: return None, meta
        currency = meta.get("currency") or "EUR"
        if currency == "GBp": series = series / 100; currency = "GBP"
        if currency != "EUR":
            fx, _ = self.raw(f"EUR{currency}=X")   # units of currency per 1 EUR
            if fx is None: return None, {**meta, "error": f"no EUR/{currency} rate"}
            fx = fx.reindex(series.index.union(fx.index)).ffill().reindex(series.index)
            series = (series / fx).dropna()
        return series.rename(symbol), meta

    def many(self, symbols: list[str], workers: int = 8) -> dict[str, tuple[pd.Series | None, dict]]:
        unique = list(dict.fromkeys(s for s in symbols if s))
        with ThreadPoolExecutor(max_workers=workers) as pool:
            return dict(zip(unique, pool.map(self.eur, unique)))

    def latest_date(self, results: dict) -> str | None:
        dates = [s.index.max() for s, _ in results.values() if s is not None and len(s)]
        return str(max(dates).date()) if dates else None


class IsinResolver:
    """ISIN -> Yahoo symbol via config overrides, the ETF list, verified spread mappings, then Yahoo search.

    Search candidates (by ISIN, then by name) are tried in order of preference and the
    first one that `validate` accepts — i.e. that actually has prices — is kept."""
    def __init__(self, cache_path: Path, overrides: dict | None = None, universe: pd.DataFrame | None = None,
                 search: Callable[[str], list[dict]] | None = None, validate: Callable[[str], bool] | None = None):
        self.cache_path = Path(cache_path)
        self.known: dict[str, str] = {}
        spreads = ROOT / "config/spreads.yaml"
        if spreads.exists():
            for m in (yaml.safe_load(spreads.read_text()) or {}).get("mappings", []):
                if m.get("verified"): self.known[m["isin"]] = m["yahoo_ticker"]
        if universe is not None: self.known.update(dict(zip(universe["isin"], universe["ticker"])))
        self.known.update(overrides or {})
        self.search = search or yahoo_search_isin
        self.validate = validate
        try: self.cache: dict = json.loads(self.cache_path.read_text()) if self.cache_path.exists() else {}
        except (OSError, ValueError): self.cache = {}

    def resolve(self, isin: str | None, name: str | None = None) -> str | None:
        if not isin: return None
        if isin in self.known: return self.known[isin]
        hit = self.cache.get(isin)
        if hit and (hit.get("symbol") or time.time() - hit.get("at", 0) < 7 * 86400): return hit.get("symbol")
        try:
            candidates = rank_symbols(self.search(isin), isin)
            query = clean_name(name)
            if query: candidates += [c for c in rank_symbols(self.search(query), isin) if c not in candidates]
        except Exception:
            return None  # retry on a later visit
        symbol = next((c for c in candidates if self.validate is None or self.validate(c)), None)
        self.cache[isin] = {"symbol": symbol, "at": time.time(), "checked": datetime.now(timezone.utc).isoformat(), "candidates": candidates[:10]}
        try: self.cache_path.write_text(json.dumps(self.cache, indent=1))
        except OSError: pass
        return symbol
