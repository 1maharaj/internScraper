"""
run_on_scraper.py
-----------------
Adapts internScraper's internships.json schema → scam_detector pipeline schema,
then runs the full pipeline.

Usage:
    cd c:\\Users\\4dmin\\Downloads\\iFind30\\IpdFetaures
    python -m scam_detector.run_on_scraper

Output:
    c:\\Users\\4dmin\\Downloads\\iFind30\\IpdFetaures\\scam_detector\\scored_internships.json
"""

from __future__ import annotations

import json
import re
import logging
from pathlib import Path
from typing import Any

# ── Paths ──────────────────────────────────────────────────────────────────────
INPUT_JSON  = Path(r"c:\Users\4dmin\Downloads\iFind30\internScraper\scrapers\internships.json")
OUTPUT_JSON = Path(r"c:\Users\4dmin\Downloads\iFind30\IpdFetaures\scam_detector\scored_internships.json")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("run_on_scraper")


# ── Stipend normalizer ─────────────────────────────────────────────────────────

def _parse_stipend(raw: str | None) -> dict[str, Any]:
    """
    Convert scraper's freetext stipend string into the pipeline's Stipend shape.

    Examples handled:
        "₹10,000 – ₹16,000 per month"  → paid, amount=13000 (midpoint), monthly
        "Performance Based"             → performance-based
        "Unpaid"                        → unpaid
        None / ""                       → unpaid
    """
    if not raw:
        return {"type": "unpaid"}

    cleaned = raw.strip()

    if re.search(r"performance.based|performance based", cleaned, re.I):
        return {"type": "performance-based"}

    if re.search(r"unpaid|no stipend|volunteer", cleaned, re.I):
        return {"type": "unpaid"}

    # Extract all numbers (handles ₹10,000 – ₹16,000 or "10000/month")
    amounts = [int(n.replace(",", "")) for n in re.findall(r"[\d,]+", cleaned)]
    if not amounts:
        return {"type": "unpaid"}

    midpoint = sum(amounts) // len(amounts)

    period: str
    if re.search(r"week", cleaned, re.I):
        period = "weekly"
    elif re.search(r"lump|one.time|total", cleaned, re.I):
        period = "lump-sum"
    else:
        period = "monthly"   # default — most scraped listings are monthly

    # Detect currency
    currency = "INR" if "₹" in cleaned else "USD" if "$" in cleaned else "INR"

    return {
        "type": "paid",
        "amount": midpoint,
        "currency": currency,
        "period": period,
    }


# ── Duration normalizer ────────────────────────────────────────────────────────

def _parse_duration(raw: str | None) -> dict[str, Any]:
    """
    "6 Months" → {"value": 6, "unit": "months"}
    "8 Weeks"  → {"value": 8, "unit": "weeks"}
    """
    if not raw:
        return {"value": 1, "unit": "months"}

    m = re.search(r"(\d+)\s*(month|week)", raw, re.I)
    if m:
        unit = "months" if "month" in m.group(2).lower() else "weeks"
        return {"value": int(m.group(1)), "unit": unit}

    # Fallback: just grab any number and assume months
    nums = re.findall(r"\d+", raw)
    return {"value": int(nums[0]) if nums else 1, "unit": "months"}


# ── Location parser ────────────────────────────────────────────────────────────

def _parse_location(raw: str | None) -> dict[str, Any]:
    """
    "Remote / WFH"       → isRemote=True
    "Mumbai, Maharashtra" → isRemote=False, city=Mumbai, state=Maharashtra
    """
    if not raw:
        return {"isRemote": False}

    if re.search(r"remote|wfh|work from home", raw, re.I):
        return {"isRemote": True}

    parts = [p.strip() for p in raw.split(",")]
    result: dict[str, Any] = {"isRemote": False}
    if len(parts) >= 1:
        result["city"] = parts[0]
    if len(parts) >= 2:
        result["state"] = parts[1]
    if len(parts) >= 3:
        result["country"] = parts[2]
    return result


# ── Deadline parser ────────────────────────────────────────────────────────────

_MONTH_MAP = {
    "january": "01", "february": "02", "march": "03", "april": "04",
    "may": "05", "june": "06", "july": "07", "august": "08",
    "september": "09", "october": "10", "november": "11", "december": "12",
}

def _parse_date(raw: str | None) -> str | None:
    """
    "THURSDAY, MAY 7TH, 2026" → "2026-05-07"
    Returns None if unparseable.
    """
    if not raw or raw.upper().startswith("TO APPLY"):
        return None

    # Try to extract month name, day number, and 4-digit year
    month_m = re.search(
        r"(january|february|march|april|may|june|july|august|september|october|november|december)",
        raw, re.I,
    )
    day_m   = re.search(r"\b(\d{1,2})(?:st|nd|rd|th)?\b", raw, re.I)
    year_m  = re.search(r"\b(20\d{2})\b", raw)

    if month_m and day_m and year_m:
        month = _MONTH_MAP[month_m.group(1).lower()]
        day   = day_m.group(1).zfill(2)
        year  = year_m.group(1)
        return f"{year}-{month}-{day}"

    return None


# ── Main record adapter ────────────────────────────────────────────────────────

def adapt_record(raw: dict[str, Any], index: int) -> dict[str, Any]:
    """Map a single scraper record → pipeline schema."""
    stipend   = _parse_stipend(raw.get("stipend"))
    duration  = _parse_duration(raw.get("duration"))
    location  = _parse_location(raw.get("location"))
    deadline  = _parse_date(raw.get("apply_by"))
    published = _parse_date(raw.get("start_date"))

    return {
        "_id":          raw.get("id") or f"scraper-{index}",
        "name":         raw.get("title") or "",
        "company":      raw.get("company") or "",
        "applyLink":    raw.get("link") or "",
        "summary":      raw.get("description") or "",
        "skills":       raw.get("skills") or [],
        "field":        [raw["type"]] if raw.get("type") else None,
        "stipend":      stipend,
        "duration":     duration,
        "isRemote":     location.get("isRemote", False),
        "city":         location.get("city"),
        "state":        location.get("state"),
        "country":      location.get("country"),
        "deadlineDate": deadline,
        "datePublished": published or "2026-01-01",  # fallback so temporal features work
        "source":       "web_scraping",
        "isActive":     True,
    }


def adapt_all(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    adapted = [adapt_record(r, i) for i, r in enumerate(records)]
    log.info("Adapted %d records", len(adapted))
    return adapted


# ── Entry point ────────────────────────────────────────────────────────────────

def main() -> None:
    log.info("Loading %s", INPUT_JSON)
    with INPUT_JSON.open(encoding="utf-8") as fh:
        raw_records: list[dict[str, Any]] = json.load(fh)

    adapted = adapt_all(raw_records)

    log.info("Running scam_detector pipeline on %d records…", len(adapted))
    from scam_detector.pipeline import process_records, write_records

    scored = process_records(adapted)
    write_records(OUTPUT_JSON, scored)
    log.info("Done → %s", OUTPUT_JSON)


if __name__ == "__main__":
    main()
