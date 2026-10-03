from __future__ import annotations

import re
from datetime import datetime
from typing import Any

from pypdf import PdfReader

ISIN = re.compile(r"\b[A-Z]{2}[A-Z0-9]{9}[0-9]\b")
DATE = re.compile(r"\b(\d{2}[./]\d{2}[./]\d{4})\b")
NUMBER = r"[-+]?\d(?:[\d.\u00a0\u202f ]*\d)?(?:,\d+)?"
MONEY = re.compile(rf"(?:(€|EUR)\s*)?({NUMBER})\s*(€|EUR)?", re.I)
QUANTITY = re.compile(rf"(?:St(?:ü|u)ck|Stk\.?|Anteile?|Shares?)\s*:?\s*({NUMBER})|({NUMBER})\s*(?:St(?:ü|u)ck|Stk\.?|Anteile?|Shares?)", re.I)
TICKER = re.compile(r"\b(?:Ticker|Symbol)\s*:?\s*([A-Z][A-Z0-9.\-]{0,9})\b", re.I)
TOTAL_LABEL = re.compile(r"(?:Gesamtwert(?:\s+Wertpapiere)?|Depotwert|Wertpapiervermögen|Wertpapierbestand|Securities value|Portfolio value)", re.I)
CASH_LABEL = re.compile(r"(?:Cash balance|Available cash|Cash|Verrechnungskonto|Guthaben|Kontostand|Bargeld|Cashkonto)", re.I)
OVERALL_TOTAL_LABEL = re.compile(r"^(?:GESAMT|TOTAL(?: FINANCIAL ASSETS)?)\b", re.I)
POSITION_ROW = re.compile(
    rf"^\s*({NUMBER})\s*(?:St(?:ü|u)ck|Stk\.?)\s{{2,}}(.+?)\s{{2,}}({NUMBER})\s{{2,}}({NUMBER})(?:\s+EUR)?\s*$",
    re.I,
)


def _number(text: str) -> float:
    """Parse German/European numbers, including NBSP thousands separators."""
    value = re.sub(r"[\u00a0\u202f\s€]", "", text).replace("EUR", "")
    if "," in value:
        value = value.replace(".", "").replace(",", ".")
    elif value.count(".") > 1 or re.fullmatch(r"[-+]?\d{1,3}(?:\.\d{3})+", value):
        value = value.replace(".", "")
    return float(value)


def _money_values(text: str) -> list[float]:
    values = []
    for match in MONEY.finditer(text):
        if match.group(1) or match.group(3):
            try:
                values.append(_number(match.group(2)))
            except ValueError:
                pass
    return values


def _labelled_money(lines: list[str], label: re.Pattern[str]) -> float | None:
    for index, line in enumerate(lines):
        if not label.search(line):
            continue
        for candidate in (line, *lines[index + 1:index + 3]):
            values = _money_values(candidate)
            if values:
                return values[-1]
    return None


def _document_type(text: str) -> tuple[str, bool]:
    lower = text.lower()
    types = (
        ("Trade Republic wealth overview", ("vermögensübersicht",)),
        ("portfolio overview", ("portfolioübersicht", "portfolio overview")),
        ("securities statement", ("depotauszug", "depotübersicht", "securities statement")),
        ("asset statement", ("vermögensaufstellung", "asset statement")),
        ("annual statement", ("jahressteuerbescheinigung", "jahresabrechnung", "annual statement")),
        ("account statement", ("kontoauszug", "account statement")),
        ("transaction statement", ("wertpapierabrechnung", "abrechnung", "transaction statement")),
    )
    for name, markers in types:
        if any(marker in lower for marker in markers):
            return name, name in {"Trade Republic wealth overview", "portfolio overview", "securities statement", "asset statement"}
    snapshot = ((len(ISIN.findall(text)) > 1 and bool(TOTAL_LABEL.search(text)))
                or (bool(ISIN.search(text)) and "portfolio" in lower))
    return ("portfolio/securities statement" if snapshot else "unrecognized Trade Republic document"), snapshot


def _security_name(lines: list[str], isin_index: int, isin: str) -> str | None:
    same = re.sub(r"\bISIN\s*:?", "", ISIN.sub("", lines[isin_index]), flags=re.I).strip(" :-")
    candidates = [same] + list(reversed(lines[max(0, isin_index - 4):isin_index]))
    rejected = re.compile(r"^(?:ISIN|WKN|Ticker|Positionen?|Wertpapiere?|Portfolio|Depot|Seite\s+\d+|Trade Republic)\s*:?$", re.I)
    for candidate in candidates:
        clean = re.sub(r"\s+", " ", candidate).strip(" :-")
        if clean and clean != isin and not rejected.match(clean) and not _money_values(clean) and not QUANTITY.search(clean) and len(clean) > 2:
            return clean
    return None


def _extract_holdings(lines: list[str]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    anchors = [(i, match.group()) for i, line in enumerate(lines) for match in ISIN.finditer(line)]
    holdings, unmatched, seen = [], [], set()
    for anchor_no, (index, isin) in enumerate(anchors):
        if isin in seen:
            continue
        seen.add(isin)
        following = anchors[anchor_no + 1][0] if anchor_no + 1 < len(anchors) else len(lines)
        # Numeric evidence belongs at/on or after its ISIN. Looking behind the
        # anchor can steal the preceding holding's value when columns interleave.
        block = lines[index:min(following, index + 9)]
        total_boundary = next((i for i, line in enumerate(block) if i and TOTAL_LABEL.search(line)), None)
        if total_boundary is not None:
            block = block[:total_boundary]
        name = _security_name(lines, index, isin)
        ticker_match = next((TICKER.search(line) for line in block if TICKER.search(line)), None)
        ticker = ticker_match.group(1).upper() if ticker_match else None
        quantity = None
        for line in block:
            match = QUANTITY.search(line)
            if match:
                quantity = _number(match.group(1) or match.group(2))
                break
        labelled_value = None
        for pos, line in enumerate(block):
            if re.search(r"(?:Marktwert|Positionswert|Gesamtwert|Current value|Market value|Wert)\b", line, re.I):
                labelled_value = _labelled_money(block[pos:pos + 3], re.compile(r".", re.S))
                if labelled_value is not None:
                    break
        amounts = [value for line in block for value in _money_values(line)]
        market_value = labelled_value if labelled_value is not None else (max(amounts) if amounts else None)
        displayed_price = next((v for v in reversed(amounts) if v != market_value), None)
        if not name or market_value is None:
            unmatched.append({"page_line": index + 1, "isin_present": True, "name_present": bool(name), "market_value_present": market_value is not None, "reason": "Could not confidently associate security identity and market value"})
            continue
        field_confidence = {"security_name": "Confident", "isin": "Confident", "ticker": "Confident" if ticker else "Missing", "quantity": "Confident" if quantity is not None else "Missing", "market_value_eur": "Confident" if labelled_value is not None else "Needs review", "displayed_price": "Needs review" if displayed_price is not None else "Missing"}
        confidence = .95 if labelled_value is not None and quantity is not None else .78
        holdings.append({"security_name": name, "isin": isin, "ticker": ticker, "quantity": quantity, "displayed_price": displayed_price, "market_value_eur": market_value, "currency": "EUR", "asset_type": "security", "confidence": confidence, "field_confidence": field_confidence})
    return holdings, unmatched


def _extract_layout_holdings(lines: list[str]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Parse Trade Republic's repeating, column-aligned brokerage position rows.

    A position starts with quantity/name/price/value on one visual row. Its ISIN
    is on one of the following detail rows, before the next position starts.
    This bounded block prevents values from adjacent positions being combined.
    """
    rows = [(index, match) for index, line in enumerate(lines) if (match := POSITION_ROW.match(line))]
    holdings: list[dict[str, Any]] = []
    unmatched: list[dict[str, Any]] = []
    for row_no, (index, match) in enumerate(rows):
        boundary = rows[row_no + 1][0] if row_no + 1 < len(rows) else len(lines)
        block = lines[index:boundary]
        isin_match = next((ISIN.search(line) for line in block if ISIN.search(line)), None)
        if not isin_match:
            # Crypto rows use the same columns but have no ISIN and belong to a
            # separately subtotalled section, not the brokerage securities list.
            continue
        name = re.sub(r"\s+", " ", match.group(2)).strip()
        holdings.append({
            "security_name": name,
            "isin": isin_match.group(),
            "ticker": None,
            "quantity": _number(match.group(1)),
            "displayed_price": _number(match.group(3)),
            "market_value_eur": _number(match.group(4)),
            "currency": "EUR",
            "asset_type": "security",
            "confidence": .99,
            "field_confidence": {
                "security_name": "Confident", "isin": "Confident", "ticker": "Missing",
                "quantity": "Confident", "market_value_eur": "Confident", "displayed_price": "Confident",
            },
        })
    return holdings, unmatched


CRYPTO_ROW = re.compile(r"^\s*(\d[\d.]*(?:,\d+)?)\s+Stk\.\s+(?:(.*?)\s+)?(\d[\d.]*,\d+)\s+(\d[\d.]*,\d+)\s*$")
CRYPTO_SYMBOL = re.compile(r"^\s*([A-Z0-9]{2,10})\b")


def _extract_crypto_holdings(lines: list[str]) -> list[dict[str, Any]]:
    """Crypto Wallet rows: quantity, optional name, price and value, with the coin symbol on the next row."""
    start = next((i + 1 for i, line in enumerate(lines) if re.match(r"^\s*CRYPTO WALLET\s*$", line, re.I)), None)
    if start is None:
        return []
    holdings: list[dict[str, Any]] = []
    for index in range(start, len(lines)):
        line = lines[index]
        if re.search(r"ANZAHL\s+POSITIONEN", line, re.I):
            break
        match = CRYPTO_ROW.match(line)
        if not match:
            continue
        symbol_match = CRYPTO_SYMBOL.match(lines[index + 1]) if index + 1 < len(lines) else None
        symbol = symbol_match.group(1) if symbol_match else None
        holdings.append({"name": (match.group(2) or symbol or "Crypto").strip(), "symbol": symbol, "quantity": _number(match.group(1)),
                         "displayed_price": _number(match.group(3)), "market_value_eur": _number(match.group(4))})
    return holdings


def _layout_labelled_value(lines: list[str], label: re.Pattern[str], lookahead: int = 0) -> float | None:
    for index, line in enumerate(lines):
        if not label.search(line.strip()):
            continue
        for candidate in lines[index:index + lookahead + 1]:
            # Layout spacing separates columns; do not let NUMBER consume the
            # spaces between a position count and its monetary subtotal.
            numbers = re.findall(r"[-+]?\d[\d.]*,\d+|[-+]?\d+", candidate)
            # Ignore the position count in a subtotal such as "ANZAHL POSITIONEN: 20".
            if numbers:
                try:
                    return _number(numbers[-1])
                except ValueError:
                    pass
    return None


def _section_position_subtotal(lines: list[str], section_label: re.Pattern[str] | None = None) -> float | None:
    start = 0
    if section_label is not None:
        start = next((i + 1 for i, line in enumerate(lines) if section_label.search(line.strip())), len(lines))
    return _layout_labelled_value(lines[start:], re.compile(r"ANZAHL\s+POSITIONEN:\s*\d+.*EUR", re.I))


def parse_text(text: str, source_file: str = "upload.pdf", *, page_texts: list[str] | None = None, positioned_lines: list[str] | None = None) -> dict[str, Any]:
    normalized = text.replace("\u00a0", " ").replace("\u202f", " ")
    lines = [re.sub(r"\s+", " ", line).strip() for line in normalized.splitlines() if line.strip()]
    warnings: list[str] = []
    date = DATE.search(normalized)
    snapshot_date = None
    if date:
        try:
            snapshot_date = datetime.strptime(date.group(1).replace("/", "."), "%d.%m.%Y").date().isoformat()
        except ValueError:
            warnings.append("An apparent date could not be parsed")
    if not snapshot_date:
        warnings.append("Snapshot date not confidently identified")
    document_type, contains_snapshot = _document_type(normalized)
    layout_lines = positioned_lines or []
    layout_holdings, layout_unmatched = _extract_layout_holdings(layout_lines)
    holdings, unmatched = (layout_holdings, layout_unmatched) if layout_holdings else _extract_holdings(layout_lines or lines)
    reported_securities_value = _section_position_subtotal(layout_lines) if layout_holdings else _labelled_money(lines, TOTAL_LABEL)
    cash = _layout_labelled_value(layout_lines, re.compile(r"^\s*Cashkonto\b", re.I)) if layout_holdings else _labelled_money(lines, CASH_LABEL)
    crypto_value = _section_position_subtotal(layout_lines, re.compile(r"^\s*CRYPTO WALLET\s*$", re.I)) if layout_holdings else None
    reported_total = _layout_labelled_value(layout_lines, OVERALL_TOTAL_LABEL) if layout_holdings else None
    cash_included = cash is not None
    if cash is None:
        warnings.append("Cash is not included in this statement." if not cash_included else "Cash value needs review.")
    securities_value = round(sum(h["market_value_eur"] for h in holdings), 2)
    difference = round(securities_value - reported_securities_value, 2) if reported_securities_value is not None else None
    # Currency totals are rounded to cents; at most two cents of accumulated
    # line-item rounding is accepted without review.
    tolerance = .02
    material_mismatch = difference is not None and abs(difference) > tolerance
    substantial_failure = not holdings or len(unmatched) > len(holdings) or not contains_snapshot
    if substantial_failure:
        warnings.append("We could read this Trade Republic statement, but could not reliably identify its individual holdings. No portfolio data has been changed.")
        holdings, securities_value = [], None
    elif material_mismatch:
        warnings.append("Extracted holdings do not reconcile to the reported securities value; review is required.")
    pages = page_texts or [normalized]
    calculated_total = (securities_value + (crypto_value or 0) + (cash or 0)) if holdings else None
    total_difference = round(calculated_total - reported_total, 2) if calculated_total is not None and reported_total is not None else None
    if total_difference is not None and abs(total_difference) > tolerance:
        warnings.append("The securities, crypto, and cash subtotals do not reconcile to the reported total financial assets.")
    diagnostics = {"page_count": len(pages), "embedded_text_present": bool(normalized.strip()), "page_character_counts": [len(page) for page in pages], "positioned_extraction_used": positioned_lines is not None, "structural_layout_parser_used": bool(layout_holdings), "isin_candidates": len(set(ISIN.findall(normalized))), "accepted_holdings": len(holdings), "unmatched_candidates": unmatched}
    return {"snapshot_date": snapshot_date, "document_type": document_type, "contains_portfolio_snapshot": contains_snapshot, "cash_eur": cash, "cash_included": cash_included, "cash_source": "extracted" if cash is not None else "absent", "securities_value_eur": securities_value, "reported_securities_value_eur": reported_securities_value, "crypto_value_eur": crypto_value, "reported_total_financial_assets_eur": reported_total, "total_financial_assets_difference_eur": total_difference, "reconciliation_difference_eur": difference, "reconciliation_tolerance_eur": tolerance, "material_reconciliation_mismatch": material_mismatch, "total_value_eur": reported_total if reported_total is not None and total_difference is not None and abs(total_difference) <= tolerance else calculated_total, "holdings": holdings, "crypto_holdings": _extract_crypto_holdings(layout_lines), "unmatched": unmatched, "source_file": source_file, "warnings": warnings, "diagnostics": diagnostics}


def _positioned_page_lines(page: Any) -> list[str]:
    fragments: list[tuple[float, float, str]] = []
    def visitor(text: str, _cm: Any, tm: Any, _font: Any, _size: Any) -> None:
        clean = re.sub(r"\s+", " ", text).strip()
        if clean:
            fragments.append((float(tm[5]), float(tm[4]), clean))
    page.extract_text(visitor_text=visitor)
    rows: list[list[tuple[float, float, str]]] = []
    for y, x, value in sorted(fragments, key=lambda item: (-item[0], item[1])):
        row = next((r for r in rows if abs(r[0][0] - y) <= 2.5), None)
        if row is None:
            row = []
            rows.append(row)
        row.append((y, x, value))
    return [" ".join(value for _y, _x, value in sorted(row, key=lambda item: item[1])) for row in rows]


def parse_pdf(file_obj: Any) -> dict[str, Any]:
    reader = PdfReader(file_obj)
    page_texts = [page.extract_text() or "" for page in reader.pages]
    text = "\n".join(page_texts)
    source_file = getattr(file_obj, "name", "upload.pdf")
    if not text.strip():
        return {"snapshot_date": None, "document_type": "unreadable PDF", "cash_eur": None, "cash_included": False, "securities_value_eur": 0.0, "reported_securities_value_eur": None, "reconciliation_difference_eur": None, "material_reconciliation_mismatch": False, "total_value_eur": None, "holdings": [], "unmatched": [], "source_file": source_file, "warnings": ["PDF contains no extractable text; OCR was not used."], "diagnostics": {"page_count": len(reader.pages), "embedded_text_present": False, "page_character_counts": [0 for _ in reader.pages], "positioned_extraction_used": False}}
    # Layout mode retains Trade Republic's visual columns and repeating row
    # boundaries. It is embedded-text extraction, not OCR.
    positioned = [line for page in reader.pages for line in (page.extract_text(extraction_mode="layout") or "").splitlines()]
    return parse_text(text, source_file, page_texts=page_texts, positioned_lines=positioned)
