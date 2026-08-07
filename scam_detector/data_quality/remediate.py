"""
remediate.py
------------
Phase 1 data-quality cleanup pass.

Operates on records already in the standardised schema produced by
format_internship.py:

    _id, name, company, applyLink, datePublished, deadlineDate,
    country, state, city, isRemote, stipend, duration, skills,
    degree, field, experienceRequired, openings, summary,
    responsibilities, perks, tags, source, isActive,
    createdAt, updatedAt

Each fix is a pure function with signature:
    fix(record: dict) -> tuple[dict, dict[str, bool | str]]

The returned record is a *shallow copy* (top-level keys replaced as needed).
The flags dict documents every inferred or repaired value so downstream
feature extractors know what they're working with.

Public API
----------
    remediate_record(record)  -> RemediatedRecord
    remediate_batch(records)  -> list[RemediatedRecord]
"""

from __future__ import annotations

import copy
import re
from datetime import date, datetime
from typing import Any

from pydantic import BaseModel, Field

# ---------------------------------------------------------------------------
# Pydantic output model
# ---------------------------------------------------------------------------

Flags = dict[str, bool | str]


class RemediatedRecord(BaseModel):
    """
    Output of a single remediation run.

    ``record`` carries the (potentially modified) internship dict.
    ``flags``  carries every data-quality annotation made during the run.
    """

    record: dict[str, Any] = Field(description="The cleaned internship document")
    flags: Flags = Field(
        default_factory=dict,
        description="Annotations emitted during remediation — never silently mutated",
    )


# ---------------------------------------------------------------------------
# Internal constants
# ---------------------------------------------------------------------------

# Category / skill strings that occasionally appear in the `company` field
# due to scraper regex fallback (observed heavily in Internshala data).
_CATEGORY_TERMS: frozenset[str] = frozenset(
    {
        # Marketing / sales
        "digital marketing",
        "social media marketing",
        "social media",
        "content writing",
        "content",
        "seo",
        "sem",
        "email marketing",
        "affiliate marketing",
        "performance marketing",
        "growth marketing",
        "influencer marketing",
        "brand marketing",
        "marketing",
        "sales",
        "business development",
        "business development (sales)",
        "lead generation",
        "customer service",
        "customer support",
        "telecalling",
        "telesales",
        # Tools / software
        "ms-office",
        "ms office",
        "microsoft office",
        "adobe photoshop",
        "photoshop",
        "canva",
        "tally",
        "autocad",
        "figma",
        # Social / messaging platforms (appear when company is blank and
        # a WhatsApp/Instagram apply-link leaks into the company field)
        "whatsapp",
        "instagram",
        "linkedin",
        "telegram",
        "facebook",
        # Generic role / domain labels mistaken for company names
        "human resources",
        "hr",
        "finance",
        "accounting",
        "data entry",
        "graphic design",
        "video editing",
        "content creation",
        "copywriting",
        "research",
        "teaching",
        "education",
        "healthcare",
        "architecture",
        "civil engineering",
        "mechanical engineering",
        "electrical engineering",
        "studio marketing",
    }
)

# Engineering keywords used to decide whether a M.Tech/B.Tech degree default
# is genuinely plausible for this particular role.
_ENGINEERING_KEYWORDS: frozenset[str] = frozenset(
    {
        "engineer",
        "engineering",
        "technical",
        "developer",
        "development",
        "software",
        "hardware",
        "mechanical",
        "electrical",
        "civil",
        "tech",
        "coding",
        "programmer",
        "programming",
        "devops",
        "cloud",
        "embedded",
        "robotics",
        "automation",
        "firmware",
    }
)

# The two exact degree combinations that are disproportionately common due
# to scraper fallback behaviour.
_SUSPECT_DEGREE_COMBOS: tuple[frozenset[str], ...] = (
    frozenset({"M.Tech"}),
    frozenset({"B.Tech", "M.Tech"}),
)

# Regex: trailing stray numbered-list artifact at end of a responsibility
# string, e.g. "...outreach opportunities   4." or "...results   12."
# We match one-or-two-digit number + period at the very end, optionally
# preceded by whitespace.  We do NOT touch legitimate sentence-ending
# punctuation.
_TRAILING_LIST_ARTIFACT = re.compile(r"\s+\d{1,2}\.\s*$")

# A summary is considered possibly truncated when it ends with "..." or is
# exactly this many characters (Internshala's observed hard-truncation limit).
_TRUNCATION_LENGTH = 300


# ---------------------------------------------------------------------------
# Fix 1 — flag_mislabeled_company
# ---------------------------------------------------------------------------

def flag_mislabeled_company(record: dict[str, Any]) -> tuple[dict[str, Any], Flags]:
    """
    Detect when the ``company`` field contains a category / skill string
    rather than a real organisation name.

    Detection strategy (case-insensitive):
      1. Exact match against ``_CATEGORY_TERMS`` (hardcoded list).
      2. Exact match against any entry in the record's own ``skills`` list.
      3. Exact match against any entry in the record's own ``tags`` list.

    The company value is **never altered** — we only set flags so scoring
    can down-weight or skip company-identity features for this record.

    Flags emitted
    -------------
    ``company_suspect``  : True
    ``company_source``   : "category_leak"
    ``company_match_via``: which check matched (``"hardcoded_list"``,
                           ``"skills_field"``, or ``"tags_field"``)
    """
    flags: Flags = {}
    company_raw: str = record.get("company") or ""
    company_lc = company_raw.strip().lower()

    if not company_lc:
        return record, flags

    # Check 1 — hardcoded category list
    if company_lc in _CATEGORY_TERMS:
        flags["company_suspect"] = True
        flags["company_source"] = "category_leak"
        flags["company_match_via"] = "hardcoded_list"
        return record, flags

    # Check 2 — record's own skills list
    skills: list[str] = record.get("skills") or []
    if any(company_lc == s.strip().lower() for s in skills):
        flags["company_suspect"] = True
        flags["company_source"] = "category_leak"
        flags["company_match_via"] = "skills_field"
        return record, flags

    # Check 3 — record's own tags list
    tags: list[str] = record.get("tags") or []
    if any(company_lc == t.strip().lower() for t in tags):
        flags["company_suspect"] = True
        flags["company_source"] = "category_leak"
        flags["company_match_via"] = "tags_field"
        return record, flags

    return record, flags


# ---------------------------------------------------------------------------
# Fix 2 — flag_degree_default
# ---------------------------------------------------------------------------

def flag_degree_default(record: dict[str, Any]) -> tuple[dict[str, Any], Flags]:
    """
    Detect when the ``degree`` list is exactly one of the two scraper
    fallback combinations **and** the role is non-engineering.

    The degree value is **not altered** — downstream code decides whether
    to ignore it during scoring.

    Flags emitted
    -------------
    ``degree_suspect_default`` : True
    """
    flags: Flags = {}
    degree_raw = record.get("degree")

    if not degree_raw or not isinstance(degree_raw, list):
        return record, flags

    degree_set = frozenset(d.strip() for d in degree_raw if isinstance(d, str))

    if degree_set not in _SUSPECT_DEGREE_COMBOS:
        return record, flags

    # Only flag it when the role title + summary show no engineering keywords
    title: str = (record.get("name") or "").lower()
    summary: str = (record.get("summary") or "").lower()
    combined_text = f"{title} {summary}"

    has_engineering_context = any(kw in combined_text for kw in _ENGINEERING_KEYWORDS)

    if not has_engineering_context:
        flags["degree_suspect_default"] = True

    return record, flags


# ---------------------------------------------------------------------------
# Fix 3 — flag_missing_deadline
# ---------------------------------------------------------------------------

def flag_missing_deadline(record: dict[str, Any]) -> tuple[dict[str, Any], Flags]:
    """
    Flag listings where ``deadlineDate`` is null / missing.

    A missing deadline is semantically different from "no deadline pressure":
    we simply don't know, and urgency-based features should treat it as
    *unknown* rather than *safe*.

    Flags emitted
    -------------
    ``deadline_missing`` : True
    """
    flags: Flags = {}
    deadline = record.get("deadlineDate")

    # Treat None, empty string, and missing key all as absent
    if not deadline:
        flags["deadline_missing"] = True

    return record, flags


# ---------------------------------------------------------------------------
# Fix 4 — clean_responsibilities
# ---------------------------------------------------------------------------

def clean_responsibilities(record: dict[str, Any]) -> tuple[dict[str, Any], Flags]:
    """
    Remove trailing numbered-list artefacts from each string in
    ``responsibilities``.

    Scraped ``<li>`` text sometimes carries a dangling item index from the
    *next* sibling element, e.g.:

        "...identify new outreach opportunities   4."

    Pattern stripped: optional whitespace + one/two digits + period at EOL.
    We never strip a period that is part of a normal sentence (it would have
    to be preceded by whitespace + digits to match).

    The record dict is *replaced* (not mutated) only when at least one
    change is made.

    Flags emitted
    -------------
    ``responsibilities_cleaned`` : True  (only if at least one string changed)
    """
    flags: Flags = {}
    raw: Any = record.get("responsibilities")

    if not isinstance(raw, list) or not raw:
        return record, flags

    cleaned: list[str] = []
    any_changed = False

    for item in raw:
        if not isinstance(item, str):
            cleaned.append(item)
            continue
        new_item = _TRAILING_LIST_ARTIFACT.sub("", item)
        if new_item != item:
            any_changed = True
        cleaned.append(new_item)

    if any_changed:
        record = {**record, "responsibilities": cleaned}
        flags["responsibilities_cleaned"] = True

    return record, flags


# ---------------------------------------------------------------------------
# Fix 5 — flag_truncated_summary
# ---------------------------------------------------------------------------

def flag_truncated_summary(record: dict[str, Any]) -> tuple[dict[str, Any], Flags]:
    """
    Detect hard-truncated summaries.

    Two heuristics:
      • Summary ends with "…" or "..." (explicit ellipsis / unicode ellipsis)
      • Summary is exactly ``_TRUNCATION_LENGTH`` characters (Internshala
        and some other scrapers cap at exactly 300 chars)

    Flags emitted
    -------------
    ``summary_truncated`` : True
    """
    flags: Flags = {}
    summary: str = record.get("summary") or ""

    if not summary:
        return record, flags

    ends_with_ellipsis = summary.rstrip().endswith("...") or summary.rstrip().endswith("\u2026")
    exactly_truncation_length = len(summary) == _TRUNCATION_LENGTH

    if ends_with_ellipsis or exactly_truncation_length:
        flags["summary_truncated"] = True

    return record, flags


# ---------------------------------------------------------------------------
# Fix 6 — flag_inferred_date
# ---------------------------------------------------------------------------

def _as_date(value: Any) -> date | None:
    """
    Coerce a value to a :class:`date`, accepting:
      • ``datetime``  objects
      • ``date``      objects
      • ISO 8601 strings (``"2024-06-05"`` or ``"2024-06-05T00:00:00Z"``)
      • MongoDB Extended JSON ``{"$date": "..."}`` dicts
    Returns ``None`` if the value cannot be parsed.
    """
    if isinstance(value, datetime):
        return value.date()
    if isinstance(value, date):
        return value
    if isinstance(value, dict):
        value = value.get("$date", "")
    if isinstance(value, str) and value:
        try:
            return datetime.fromisoformat(value.replace("Z", "+00:00")).date()
        except ValueError:
            pass
    return None


def flag_inferred_date(record: dict[str, Any]) -> tuple[dict[str, Any], Flags]:
    """
    Detect records where ``datePublished`` was most likely synthesised by
    ``format_internship.py``'s ``datetime.now()`` fallback.

    Heuristic: when ``datePublished`` and ``createdAt`` share the same
    *calendar date*, the posting date was probably inferred rather than
    scraped.  This is a proxy — it also fires for records that genuinely
    were published on the day they were imported, but that is a safe
    false-positive (conservative).

    Flags emitted
    -------------
    ``date_possibly_inferred`` : True
    """
    flags: Flags = {}

    published = _as_date(record.get("datePublished"))
    created = _as_date(record.get("createdAt"))

    if published is not None and created is not None and published == created:
        flags["date_possibly_inferred"] = True

    return record, flags


# ---------------------------------------------------------------------------
# Orchestrator
# ---------------------------------------------------------------------------

# Ordered list of all fix functions applied in sequence.
_FIXES = (
    flag_mislabeled_company,
    flag_degree_default,
    flag_missing_deadline,
    clean_responsibilities,
    flag_truncated_summary,
    flag_inferred_date,
)


def remediate_record(raw: dict[str, Any]) -> RemediatedRecord:
    """
    Run all six remediation passes against a single record.

    ``raw`` is never mutated — each pass works on a copy.
    All flags from every pass are merged into one flat dict.
    In the unlikely event of a key collision the *later* pass wins.

    Parameters
    ----------
    raw:
        A single internship document in the standardised schema.

    Returns
    -------
    RemediatedRecord
        ``.record`` — cleaned document (shallow copy; only changed keys differ)
        ``.flags``  — merged flags from all passes
    """
    record: dict[str, Any] = copy.copy(raw)
    merged_flags: Flags = {}

    for fix in _FIXES:
        record, flags = fix(record)
        merged_flags.update(flags)

    return RemediatedRecord(record=record, flags=merged_flags)


def remediate_batch(records: list[dict[str, Any]]) -> list[RemediatedRecord]:
    """
    Run :func:`remediate_record` over a list of raw internship documents.

    Errors in individual records are caught and re-raised with the record
    index prepended to aid debugging.
    """
    results: list[RemediatedRecord] = []
    for i, raw in enumerate(records):
        try:
            results.append(remediate_record(raw))
        except Exception as exc:
            raise RuntimeError(f"remediate_batch: error at index {i}") from exc
    return results
