"""
pipeline.py — Moderation pipeline integrated with scam_detector

Validates, deduplicates, runs scam detection, normalizes internships to the
canonical Internship schema defined in ifind/types/internship.ts, and stores
them in the MongoDB collection 'internships.mod-unvectorised'.
"""

import re
import hashlib
import logging
from datetime import datetime, timezone, timedelta
from typing import Optional, List, Literal, TypedDict, Any, Dict

from pymongo.collection import Collection

log = logging.getLogger("pipeline")


# ─── Python TypedDict definitions (matching ifind/types/internship.ts) ────────

class StipendDict(TypedDict, total=False):
    type: Literal["paid", "unpaid", "performance-based"]
    amount: Optional[float]
    currency: Optional[str]
    period: Optional[Literal["monthly", "weekly", "lump-sum"]]


class DurationDict(TypedDict, total=False):
    value: int
    unit: Literal["weeks", "months"]


class ExperienceRequiredDict(TypedDict, total=False):
    min: Optional[int]
    max: Optional[int]
    unit: Literal["months", "years"]


class LinkVerificationDict(TypedDict, total=False):
    reachable: Optional[bool]
    statusCode: Optional[int]
    redirectedTo: Optional[str]
    isScamSuspected: Optional[bool]
    isExpired: Optional[bool]
    scamSignals: List[str]
    checkedAt: Optional[str]
    nextCheckAt: Optional[str]


class RiskBreakdownDict(TypedDict, total=False):
    textRisk: Optional[float]
    companyRisk: Optional[float]
    urlRisk: Optional[float]
    stipendRisk: Optional[float]
    anomalyScore: Optional[float]


class ScamDetailsDict(TypedDict, total=False):
    score: float
    decision: Literal["clear", "review", "block"]
    confidence: float
    explanationSummary: str
    scamFlags: List[str]
    evaluatedAt: Optional[str]
    riskBreakdown: Optional[RiskBreakdownDict]


class ModerationDict(TypedDict, total=False):
    status: Literal["auto_approved", "pending_review", "auto_rejected", "manually_approved", "manually_rejected"]
    score: Optional[float]
    flags: List[str]
    source: Literal["web_scraping", "api", "user_contributed", "email_parsing", "rss", "community_bot", "manual"]
    reviewedBy: Optional[str]
    reviewedAt: Optional[str]
    rejectionReason: Optional[str]
    scamDetails: Optional[ScamDetailsDict]


class InternshipDict(TypedDict, total=False):
    _id: Optional[str]
    name: str
    company: str
    applyLink: str
    datePublished: str
    deadlineDate: Optional[str]
    country: Optional[str]
    state: Optional[str]
    city: Optional[str]
    isRemote: bool
    stipend: StipendDict
    duration: DurationDict
    skills: List[str]
    degree: Optional[List[str]]
    field: Optional[List[str]]
    experienceRequired: Optional[ExperienceRequiredDict]
    openings: Optional[int]
    summary: str
    responsibilities: Optional[List[str]]
    perks: Optional[List[str]]
    tags: Optional[List[str]]
    source: Optional[str]
    isActive: bool
    fingerprint: Optional[str]
    linkVerification: Optional[LinkVerificationDict]
    moderation: Optional[ModerationDict]
    createdAt: str
    updatedAt: str


# ─── Fingerprint ──────────────────────────────────────────────────────────────

def generate_fingerprint(company: str, name: str, city: str) -> str:
    raw = f"{company.lower().strip()}:{name.lower().strip()}:{(city or 'remote').lower().strip()}"
    return hashlib.sha256(raw.encode()).hexdigest()


# ─── Normalizers ──────────────────────────────────────────────────────────────

def normalize_duration(raw) -> DurationDict:
    """Accept a dict with value+unit, a duration_string, or fall back to default."""
    if isinstance(raw, dict) and raw.get("value") and raw.get("unit"):
        unit = "weeks" if "week" in str(raw.get("unit")).lower() else "months"
        return {"value": int(raw["value"]), "unit": unit}

    if isinstance(raw, str) and raw:
        m = re.search(r"(\d+)\s*(week|month)", raw, re.IGNORECASE)
        if m:
            unit = "weeks" if "week" in m.group(2).lower() else "months"
            return {"value": int(m.group(1)), "unit": unit}

    return {"value": 3, "unit": "months"}


def normalize_stipend(raw) -> StipendDict:
    """Accept a structured dict or a plain string like '₹ 12,000 /month'."""
    if isinstance(raw, dict) and raw.get("type"):
        stype = str(raw.get("type")).lower().replace("_", "-")
        if stype not in ("paid", "unpaid", "performance-based"):
            stype = "paid" if raw.get("amount") else "unpaid"
        return {
            "type": stype,
            "amount": float(raw.get("amount")) if raw.get("amount") is not None else None,
            "currency": str(raw.get("currency") or "INR"),
            "period": raw.get("period") or None,
        }

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
        amount = float(amount_match.group(1)) if amount_match else None

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

    return {"type": "unpaid", "amount": None, "currency": "INR", "period": None}


def normalize_location(item: dict) -> tuple[str, str, bool]:
    """Return (city, country, is_remote) from various scraper field layouts."""
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


def normalize_link_verification(item: dict) -> LinkVerificationDict:
    """Preserve existing linkVerification / linkVerified or build non-null defaults."""
    raw = item.get("linkVerification") or item.get("linkVerified")
    now_iso = datetime.now(timezone.utc).isoformat()
    next_check_iso = (datetime.now(timezone.utc) + timedelta(days=7)).isoformat()

    if isinstance(raw, dict):
        return {
            "reachable": raw.get("reachable") if raw.get("reachable") is not None else True,
            "statusCode": raw.get("statusCode") if raw.get("statusCode") is not None else 200,
            "redirectedTo": raw.get("redirectedTo") or None,
            "isScamSuspected": raw.get("isScamSuspected") if raw.get("isScamSuspected") is not None else False,
            "isExpired": raw.get("isExpired") if raw.get("isExpired") is not None else False,
            "scamSignals": raw.get("scamSignals") if isinstance(raw.get("scamSignals"), list) else [],
            "checkedAt": raw.get("checkedAt") or now_iso,
            "nextCheckAt": raw.get("nextCheckAt") or next_check_iso,
        }

    return {
        "reachable": True,
        "statusCode": 200,
        "redirectedTo": None,
        "isScamSuspected": False,
        "isExpired": False,
        "scamSignals": [],
        "checkedAt": now_iso,
        "nextCheckAt": next_check_iso,
    }


def score_internship(name: str, company: str, apply_link: str, summary: str, skills: list) -> int:
    """
    Simplified scoring check. Max possible = 50.
    """
    score = 0
    if name and company and apply_link and summary:
        score += 25
    if len(skills) >= 1:
        score += 10
    if len(summary) >= 80:
        score += 10
    score += 5
    return score


# Top-level keys scam_detector.process_records adds; the useful parts are
# folded into doc["moderation"]["scamDetails"], so they must not leak into staging.
_DETECTOR_EXTRA_KEYS = (
    "scam_score", "decision", "confidence", "explanation_summary", "confidence_level",
    "triggered_rules", "top_contributing_features", "risk_breakdown",
    "hard_disqualifying_forced", "low_confidence_forced_review",
    "shared_infrastructure", "duplicate_cluster_network_size",
)


# ─── Main pipeline function ───────────────────────────────────────────────────

def push_to_pipeline(
    items: list,
    source: str,
    col: Collection,
    label: str = "",
) -> dict:
    """
    Process a list of raw scraped internships through basic validation,
    scam detection, and normalization into canonical Internship schema,
    storing passing records in collection 'internships.mod-unvectorised'.

    Returns stats dict: {saved, duplicate, rejected, errors}
    """
    stats = {"saved": 0, "duplicate": 0, "rejected": 0, "errors": 0}

    # Ensure collection points to 'internships.mod-unvectorised'
    if col.name != "internships.mod-unvectorised":
        col = col.database["internships.mod-unvectorised"]

    valid_candidates: list[dict] = []

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
            # Approved listings leave staging, so dedupe against the live collection too.
            if col.find_one({"fingerprint": fingerprint}) or col.database["internships"].find_one({"fingerprint": fingerprint}, {"_id": 1}):
                log.debug("  ⏭  Duplicate: '%s' @ %s", name, company)
                stats["duplicate"] += 1
                continue

            # ── Normalize sub-fields ──────────────────────────────────────
            stipend  = normalize_stipend(item.get("stipend") or item.get("stipend_text") or "")
            duration = normalize_duration(item.get("duration") or item.get("duration_string") or "")

            deadline_date_str: Optional[str] = None
            if item.get("deadline_date"):
                try:
                    dt = datetime.fromisoformat(str(item["deadline_date"]))
                    deadline_date_str = dt.isoformat()
                except Exception:
                    deadline_date_str = str(item["deadline_date"])

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

            now_iso = datetime.now(timezone.utc).isoformat()
            next_check_iso = (datetime.now(timezone.utc) + timedelta(days=7)).isoformat()

            # Construct InternshipDict matching canonical TS schema
            candidate_doc: InternshipDict = {
                "name":               name,
                "company":            company,
                "applyLink":          apply_link,
                "summary":            summary,
                "city":               city or None,
                "country":            country or None,
                "state":              item.get("state") or None,
                "isRemote":           is_remote,
                "skills":             skills,
                "degree":             degree,
                "field":              field,
                "responsibilities":   (
                    item["responsibilities"]
                    if isinstance(item.get("responsibilities"), list)
                    else None
                ),
                "perks":              item.get("perks") if isinstance(item.get("perks"), list) else None,
                "tags":               item.get("tags") if isinstance(item.get("tags"), list) else None,
                "openings":           item.get("openings") or None,
                "source":             source,
                "isActive":           True,
                "datePublished":      now_iso,
                "deadlineDate":       deadline_date_str,
                "stipend":            stipend,
                "duration":           duration,
                "experienceRequired": {"unit": "months"},
                "fingerprint":        fingerprint,
                "linkVerification":   normalize_link_verification(item),
                "createdAt":           now_iso,
                "updatedAt":           now_iso,
            }

            valid_candidates.append(candidate_doc)

        except Exception as e:
            log.error("  💥 Error processing '%s': %s", item.get("name") or item.get("title", "?"), e)
            stats["errors"] += 1

    if not valid_candidates:
        return stats

    # ── Run scam_detector pipeline ──────────────────────────────────────────
    try:
        from scam_detector.pipeline import process_records
        scored_candidates = process_records(valid_candidates)
    except Exception as exc:
        # Fail closed: an unscored posting must never be auto-approved.
        log.warning("Scam detector processing failed (%s) — routing all to manual review", exc)
        scored_candidates = [
            {
                **c,
                "scam_score": 50.0,
                "decision": "review",
                "confidence": 0.0,
                "explanation_summary": "Scam detector unavailable; manual review required.",
            }
            for c in valid_candidates
        ]

    # ── Final Moderation assembly & insertion into MongoDB ──────────────────
    for doc in scored_candidates:
        try:
            scam_score = float(doc.get("scam_score", 0.0))
            decision = doc.get("decision", "review")  # "clear" | "review" | "block"
            confidence = float(doc.get("confidence", 1.0))
            summary_exp = str(doc.get("explanation_summary", ""))

            raw_flags = []
            if isinstance(doc.get("moderation"), dict):
                raw_flags = doc["moderation"].get("flags") or []

            scam_details: ScamDetailsDict = {
                "score":              scam_score,
                "decision":           decision if decision in ("clear", "review", "block") else "review",
                "confidence":         confidence,
                "explanationSummary": summary_exp,
                "scamFlags":          raw_flags,
                "evaluatedAt":        datetime.now(timezone.utc).isoformat(),
                "riskBreakdown":      {"anomalyScore": (doc.get("risk_breakdown") or {}).get("anomaly_score")},
            }

            mod_status = {"clear": "auto_approved", "block": "auto_rejected"}.get(decision, "pending_review")

            doc["moderation"] = {
                "status":          mod_status,
                "score":           scam_score,
                "flags":           raw_flags,
                "source":          source if source in ("web_scraping", "api", "user_contributed", "email_parsing", "rss", "community_bot", "manual") else "web_scraping",
                "reviewedBy":      None,
                "reviewedAt":      datetime.now(timezone.utc).isoformat(),
                "rejectionReason": summary_exp if decision == "block" else None,
                "scamDetails":     scam_details,
            }

            # Staging schema: listing + moderation only (no vectors, no loose detector fields).
            for k in _DETECTOR_EXTRA_KEYS:
                doc.pop(k, None)

            col.insert_one(doc)
            log.info("  ✅ [%s] scam_score:%.1f — '%s' @ %s", mod_status, scam_score, doc["name"], doc["company"])
            stats["saved"] += 1

        except Exception as e:
            log.error("  💥 Error inserting '%s': %s", doc.get("name", "?"), e)
            stats["errors"] += 1

    return stats
