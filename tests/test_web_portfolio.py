import numpy as np
import pandas as pd
import pytest
from src.portfolio.pdf_import import _extract_crypto_holdings
from src.web import portfolio as pf
from src.web.market import IsinResolver, PriceStore, pick_symbol
from src.web.store import WebStore

DATES = pd.bdate_range("2025-01-01", periods=300)


def test_buy_and_hold_return_weights_by_value():
    a = pd.Series(np.linspace(100, 110, 300), index=DATES)        # +10%
    b = pd.Series(np.full(300, 50.0), index=DATES)                  # flat
    mapped = pd.DataFrame({"name": ["A", "B"], "kind": ["ETF / fund"] * 2, "quantity": [10, 20], "statement_value": [1100, 1000], "symbol": ["A", "B"], "isin": [None, None]})
    values = pf.value_history(mapped, {"A": a, "B": b}, DATES[-1])
    r = pf.buy_and_hold_return(values, DATES[0])
    assert r.iloc[0] == 0 and r.iloc[-1] == pytest.approx((1100 + 1000) / (1000 + 1000) - 1)


def test_price_that_disagrees_with_statement_is_rejected():
    mapped = pd.DataFrame({"name": ["A"], "kind": ["ETF / fund"], "quantity": [10], "statement_value": [1000.0], "symbol": ["A"], "isin": [None]})
    wrong = pd.Series(np.full(300, 300.0), index=DATES)   # statement says 100 per unit
    assert pf.value_history(mapped, {"A": wrong}, DATES[-1]).empty
    assert pf.price_check(mapped, {"A": wrong}, DATES[-1]).price_status.iloc[0].startswith("price does not match")
    scaled = pd.Series(np.full(300, 1.1), index=DATES) * 100   # 110 vs 100: same instrument, used relative to statement
    assert pf.value_history(mapped, {"A": scaled}, DATES[-1]).iloc[-1, 0] == pytest.approx(1000)


def test_young_position_joins_when_its_data_starts():
    a = pd.Series(np.full(300, 100.0), index=DATES)
    b = pd.Series(np.linspace(10, 20, 100), index=DATES[200:])
    mapped = pd.DataFrame({"name": ["A", "B"], "kind": ["ETF / fund"] * 2, "quantity": [1, 1], "statement_value": [100.0, 20.0], "symbol": ["A", "B"], "isin": [None, None]})
    values = pf.value_history(mapped, {"A": a, "B": b}, DATES[-1])
    r = pf.buy_and_hold_return(values, DATES[0])
    assert np.isfinite(r).all() and r.iloc[-1] > 0
    assert pf.coverage(values, mapped, DATES[0]) == pytest.approx(100 / 120)


def test_period_start_and_whatif_value():
    end = pd.Timestamp("2026-10-02")
    assert pf.period_start(end, "YTD") == pd.Timestamp("2026-01-01")
    assert pf.period_start(end, "3Y") == pd.Timestamp("2023-10-02")
    prices = pd.Series(np.linspace(100, 120, 300), index=DATES)
    row = pd.Series({"mode": "lump", "amount_eur": 1000.0, "entry_price": 100.0, "entry_date": str(DATES[0].date())})
    assert pf.whatif_value(row, prices)["value"] == pytest.approx(1200)
    plan = pf.whatif_value(pd.Series({"mode": "monthly", "amount_eur": 100.0, "entry_price": 100.0, "entry_date": str(DATES[0].date())}), prices)
    assert plan["paid"] == pytest.approx(100 * 14) and plan["value"] > plan["paid"]


def test_crypto_rows_are_parsed_with_symbols():
    lines = ["CRYPTO WALLET", "Aufstellung …", "STK. / NOMINALE", "0,012625 Stk. Bitcoin 66.345,21 837,61", "BTC 02.09.2026",
             "0,023 Stk. 2.048,29 47,11", "ETH 02.09.2026", "ANZAHL POSITIONEN: 2 884,72 EUR"]
    rows = _extract_crypto_holdings(lines)
    assert [(r["symbol"], r["name"], r["market_value_eur"]) for r in rows] == [("BTC", "Bitcoin", 837.61), ("ETH", "ETH", 47.11)]


def test_store_round_trip(tmp_path):
    store = WebStore(tmp_path / "web.db")
    parsed = {"snapshot_date": "2026-09-02", "cash_eur": 450.0, "crypto_value_eur": 50.0, "source_file": "/x/y/statement.pdf",
              "holdings": [{"security_name": "Fund", "isin": "IE00B4L5Y983", "quantity": 10, "displayed_price": 100, "market_value_eur": 1000.0}],
              "crypto_holdings": [{"name": "Bitcoin", "symbol": "BTC", "quantity": 0.001, "displayed_price": 50000, "market_value_eur": 50.0}]}
    sid = store.save_statement(parsed)
    latest = store.latest()
    assert latest["total"] == 1500 and latest["brokerage"] == 1000 and latest["crypto"].symbol.tolist() == ["BTC"]
    assert store.snapshots().source_file.iloc[0] == "statement.pdf"
    store.save_settings({"tax": {"saver_allowance_eur": 200}}); assert store.settings_overrides()["tax"]["saver_allowance_eur"] == 200
    wid = store.add_whatif(ticker="EUNL.DE", name="World", isin="IE00B4L5Y983", mode="lump", amount_eur=1000, horizon_years=5, entry_date="2026-10-01", entry_price=100)
    assert len(store.whatifs()) == 1; store.close_whatif(wid); assert store.whatifs().empty
    store.delete_snapshot(sid); assert store.latest() is None


def test_price_store_caches_and_converts_currency(tmp_path):
    calls = []
    def fetch(symbol):
        calls.append(symbol)
        if symbol == "EURUSD=X": return pd.Series(2.0, index=DATES), {"currency": "USD"}
        return pd.Series(10.0, index=DATES), {"currency": "USD" if symbol == "TSLA" else "EUR"}
    store = PriceStore(tmp_path, fetcher=fetch)
    eur, _ = store.eur("TSLA")
    assert eur.iloc[-1] == pytest.approx(5.0)
    store.eur("TSLA"); assert calls.count("TSLA") == 1         # served from disk cache
    failing = PriceStore(tmp_path, max_age_hours=0, fetcher=lambda s: (_ for _ in ()).throw(RuntimeError("down")))
    stale, meta = failing.raw("TSLA")
    assert stale is not None and meta["stale"] and "TSLA" in failing.errors


def test_isin_resolution_prefers_xetra(tmp_path):
    assert pick_symbol([{"symbol": "TSLA"}, {"symbol": "TL0.F"}, {"symbol": "TL0.DE"}]) == "TL0.DE"
    r = IsinResolver(tmp_path / "c.json", {"US88160R1014": "TSLA"}, search=lambda isin: [{"symbol": "ABC.DE"}])
    assert r.resolve("US88160R1014") == "TSLA" and r.resolve("IE00B3WJKG14") == "QDVE.DE" and r.resolve("XX0000000000") == "ABC.DE"
