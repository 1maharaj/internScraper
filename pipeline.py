"""
pipeline.py — Moderation pipeline (Python port of scrape-and-moderate.mjs)

Validates, deduplicates, scores, and inserts internships into MongoDB.
Each scraper's output is normalized to the internship_format schema before insertion.
"""

import re
import hashlib
import logging
from datetime import datetime, timezone, timedelta
from typing import Optional

from pymongo.collection import Collection

log = logging.getLogger("pipeline")


# ─── Fingerprint ──────────────────────────────────────────────────────────────

def generate_fingerprint(company: str, name: str, city: str) -> str:
    raw = f"{company.lower().strip()}:{name.lower().strip()}:{(city or 'remote').lower().strip()}"
    return hashlib.sha256(raw.encode()).hexdigest()


# ─── Normalizers ──────────────────────────────────────────────────────────────

def normalize_duration(raw) -> dict:
    """Accept a dict with value+unit, a duration_string, or fall back to default."""
    if isinstance(raw, dict) and raw.get("value") and raw.get("unit"):
        return raw

    if isinstance(raw, str) and raw:
        m = re.search(r"(\d+)\s*(week|month)", raw, re.IGNORECASE)
        if m:
            unit = "weeks" if "week" in m.group(2).lower() else "months"
            return {"value": int(m.group(1)), "unit": unit}

    return {"value": 3, "unit": "months"}


def normalize_stipend(raw) -> dict:
    """Accept a structured dict or a plain string like '₹ 12,000 /month'."""
    if isinstance(raw, dict) and raw.get("type"):
        return raw

    if isinstance(raw, str) and raw and raw.lower() not in ("n/a", "unpaid", "not disclosed"):
        text = raw.strip()

        if re.search(r"unpaid|volunteer|no stipend", text, re.IGNORECASE):
            return {"type": "unpaid", "amount": None, "currency": "INR", "period": None}

        if re.search(r"performance|incentive", text, re.IGNORECASE):
            return {"type": "performance-based", "amount": None, "currency": "INR", "period": None}

        currency = "INR"
        if "$" in text or "USD" in text:
            currency = "USD"
        elif "£" in text or "GBP" in text:
            currency = "GBP"
        elif "€" in text or "EUR" in text:
            currency = "EUR"

        clean = text.replace(",", "")
        amount_match = re.search(r"(\d+)", clean)
        amount = int(amount_match.group(1)) if amount_match else None

        period = None
        if re.search(r"/month|per month|monthly", text, re.IGNORECASE):
            period = "monthly"
        elif re.search(r"/week|per week|weekly", text, re.IGNORECASE):
            period = "weekly"
        elif re.search(r"lump.?sum|one.?time|total", text, re.IGNORECASE):
            period = "lump-sum"

        return {
            "type": "paid" if amount else "unpaid",
            "amount": amount,
            "currency": currency,
            "period": period,
        }

    return {"type": "unpaid", "amount": None, "currency": "USD", "period": None}


def normalize_location(item: dict) -> tuple[str, str, bool]:
    """Return (city, country, is_remote) from various scraper field layouts."""
    # Scrapers use different field names
    city = (
        item.get("city")
        or item.get("location")
        or "remote"
    )
    if isinstance(city, str):
        city = city.strip()
    else:
        city = "remote"

    is_remote = bool(re.search(r"remote|wfh|work from home|online", city, re.IGNORECASE))

    country = item.get("country") or ""
    if not country:
        # Guess from city
        if re.search(r"remote|wfh|online", city, re.IGNORECASE):
            country = ""
        elif re.search(
            r"mumbai|delhi|bangalore|bengaluru|hyderabad|pune|chennai|kolkata|"
            r"noida|gurgaon|india|jaipur|lucknow|chandigarh|indore|bhopal|kochi",
            city, re.IGNORECASE
        ):
            country = "india"
        else:
            country = ""

    return city, country, is_remote


def normalize_skills(raw) -> list:
    if isinstance(raw, list):
        return [str(s).strip() for s in raw if s]
    return []


def normalize_name(item: dict) -> str:
    """Scrapers use 'name' or 'title'."""
    return (item.get("name") or item.get("title") or "").strip()


def normalize_apply_link(item: dict) -> str:
    return (item.get("apply_link") or item.get("applyLink") or item.get("link") or "").strip()


def normalize_summary(item: dict) -> str:
    """Scrapers use 'summary', 'description', or 'responsibilities'."""
    s = item.get("summary") or item.get("description") or ""
    if not s and isinstance(item.get("responsibilities"), list):
        s = " ".join(item["responsibilities"][:3])
    return str(s).strip()


# ─── Scoring ──────────────────────────────────────────────────────────────────

def score_internship(name: str, company: str, apply_link: str, summary: str, skills: list) -> int:
    """
    Simplified scoring (no live link check for bulk import).
    Max possible = 50.
    """
    score = 0
    if name and company and apply_link and summary:
        score += 25
    if len(skills) >= 1:
        score += 10
    if len(summary) >= 80:
        score += 10
    score += 5  # no deadline penalty
    return score


# ─── Main pipeline function ───────────────────────────────────────────────────

def push_to_pipeline(
    items: list,
    source: str,
    col: Collection,
    label: str = "",
) -> dict:
    """
    Process a list of raw scraped internships through the moderation pipeline
    and insert passing ones into MongoDB.

    Returns stats dict: {saved, duplicate, rejected, errors}
    """
    stats = {"saved": 0, "duplicate": 0, "rejected": 0, "errors": 0}

    for item in items:
        try:
            name       = normalize_name(item)
            company    = (item.get("company") or "").strip()
            apply_link = normalize_apply_link(item)
            summary    = normalize_summary(item)
            city, country, is_remote = normalize_location(item)
            skills     = normalize_skills(item.get("skills"))

            # ── Basic validation ──────────────────────────────────────────
            if not name or not company or not apply_link or not summary:
                log.debug("  ⚠  Skipping '%s' — missing required fields", name or "unnamed")
                stats["rejected"] += 1
                continue

            if not apply_link.startswith("http"):
                log.debug("  ⚠  Skipping '%s' — invalid URL: %s", name, apply_link)
                stats["rejected"] += 1
                continue

            # ── Deduplication ─────────────────────────────────────────────
            fingerprint = generate_fingerprint(company, name, city)
            if col.find_one({"fingerprint": fingerprint}):
                log.debug("  ⏭  Duplicate: '%s' @ %s", name, company)
                stats["duplicate"] += 1
                continue

            # ── Score & gate ──────────────────────────────────────────────
            score = score_internship(name, company, apply_link, summary, skills)
            if score < 40:
                log.debug("  ❌ Auto-rejected: '%s' (score: %d)", name, score)
                stats["rejected"] += 1
                continue

            status = "pending_review"

            # ── Normalize sub-fields ──────────────────────────────────────
            stipend  = normalize_stipend(item.get("stipend") or item.get("stipend_text") or "")
            duration = normalize_duration(item.get("duration") or item.get("duration_string") or "")

            # Deadline
            deadline_date: Optional[datetime] = None
            if item.get("deadline_date"):
                try:
                    deadline_date = datetime.fromisoformat(str(item["deadline_date"]))
                except Exception:
                    pass

            # Degree / field
            raw_degree = item.get("degree")
            degree = (
                raw_degree if isinstance(raw_degree, list)
                else [raw_degree] if raw_degree
                else None
            )
            raw_field = item.get("field")
            field = (
                raw_field if isinstance(raw_field, list)
                else [raw_field] if raw_field
                else None
            )

            next_check_at = datetime.now(timezone.utc) + timedelta(days=7)
            now = datetime.now(timezone.utc)

            doc = {
                "name":       name,
                "company":    company,
                "applyLink":  apply_link,
                "summary":    summary,
                "city":       city or None,
                "country":    country or None,
                "state":      item.get("state") or None,
                "isRemote":   is_remote,
                "skills":     skills,
                "degree":     degree,
                "field":      field,
                "responsibilities": (
                    item["responsibilities"]
                    if isinstance(item.get("responsibilities"), list)
                    else None
                ),
                "perks":    item.get("perks") if isinstance(item.get("perks"), list) else None,
                "tags":     item.get("tags")  if isinstance(item.get("tags"),  list) else None,
                "openings": item.get("openings") or None,
                "source":   source,
                "isActive": True,
                "datePublished": now,
                "deadlineDate":  deadline_date,
                "stipend":   stipend,
                "duration":  duration,
                "experienceRequired": {"unit": "months"},
                "fingerprint": fingerprint,
                "linkVerification": {
                    "reachable":       None,
                    "statusCode":      None,
                    "redirectedTo":    None,
                    "isScamSuspected": None,
                    "isExpired":       None,
                    "scamSignals":     [],
                    "checkedAt":       None,
                    "nextCheckAt":     next_check_at,
                },
                "moderation": {
                    "status":          status,
                    "score":           score,
                    "flags":           ["missing_skills"] if not skills else [],
                    "source":          source,
                    "reviewedBy":      None,
                    "reviewedAt":      None,
                    "rejectionReason": None,
                },
                "createdAt": now,
                "updatedAt": now,
            }

            col.insert_one(doc)
            log.info("  ✅ [%s] score:%d — '%s' @ %s", status, score, name, company)
            stats["saved"] += 1

        except Exception as e:
            log.error("  💥 Error processing '%s': %s", item.get("name") or item.get("title", "?"), e)
            stats["errors"] += 1

    return stats
