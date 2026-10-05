"""
company_features.py
-------------------
Signals derived from the company name and identity.

All public functions are pure and individually testable.  They take plain
Python primitives and return floats, bools, or small dicts.

Design contract
---------------
``is_company_suspect`` is the gate function.  When it returns ``True`` the
remediation stage already flagged the company field as a category-leak or
skill-string (see ``data_quality.remediate.flag_mislabeled_company``).  In
that case ALL downstream company-identity signals are unreliable and must be
short-circuited by the caller.  ``extract_company_url_features`` enforces
this automatically.

The top-level entry point is :func:`extract_company_url_features`, which
accepts a remediated record + full batch and returns a
:class:`CompanyUrlFeatureVector` combining both company and URL signals.
"""

from __future__ import annotations

import math
import re
import sqlite3
import time
from collections import defaultdict
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Any


from pydantic import BaseModel, Field

# ---------------------------------------------------------------------------
# Pydantic output models
# ---------------------------------------------------------------------------


class CompanyFeatures(BaseModel):
    """
    Feature slice for company-identity signals.

    When ``is_suspect`` is True all other fields are set to safe/neutral
    defaults and must be ignored by scoring — the company name is not a
    reliable identity signal for this record.
    """

    is_suspect: bool = Field(
        default=False,
        description="Passthrough of flags['company_suspect'] from remediation",
    )
    has_legal_suffix: bool = Field(default=False)
    posting_count: int = Field(default=1, ge=0)
    posting_date_span_days: float = Field(default=0.0, ge=0.0)
    role_diversity_score: float = Field(
        default=0.0,
        ge=0.0,
        le=1.0,
        description="0 = all same category; 1 = maximally diverse roles",
    )
    typosquat_min_distance: float = Field(
        default=1.0,
        ge=0.0,
        le=1.0,
        description="Normalised edit distance to nearest known brand (0 = exact match)",
    )
    domain_age_days: int | None = Field(
        default=None,
        description="Age of domain in days from WHOIS creation date (None if lookup fails or missing)",
    )



# ---------------------------------------------------------------------------
# Seed data
# ---------------------------------------------------------------------------

# ~20 major Indian/global employer names commonly impersonated in scam postings.
# ⚠  DOCUMENTED AS NEEDING EXPANSION — integrate a full company gazetteer
#    (e.g. Forbes India 500 + LinkedIn top employers) in a later data-enrichment
#    phase.  This seed list covers the most commonly observed impersonation
#    targets in the Internshala / Naukri / Telegram corpus.
_KNOWN_BRANDS: list[str] = [
    # Indian tech / unicorns
    "Razorpay",
    "Zepto",
    "Zomato",
    "Swiggy",
    "Flipkart",
    "Ola",
    "Paytm",
    "PhonePe",
    "BYJU'S",
    "Unacademy",
    "Meesho",
    "ShareChat",
    "Nykaa",
    "Groww",
    "Upstox",
    # Global
    "Google",
    "Microsoft",
    "Amazon",
    "Meta",
    "Apple",
    "Salesforce",
    "Infosys",
    "Wipro",
    "TCS",
    "Accenture",
]

# Regex for legal entity suffixes — ordered longest first to avoid partial
# matches (e.g. "Ltd" before "Ltd Pvt" would not matter, but keeping order
# makes intent clear).
_LEGAL_SUFFIX_RE = re.compile(
    r"""
    \b(
        private\s+limited
        | pvt\.?\s*ltd\.?
        | public\s+limited
        | limited\s+liability\s+partnership
        | llp
        | incorporated
        | inc\.?
        | limited
        | ltd\.?
        | foundation
        | trust
        | society
        | association
        | ngo
        | llc
        | corporation
        | corp\.?
        | gmbh
        | s\.a\.
        | n\.v\.
    )\b
    """,
    re.IGNORECASE | re.VERBOSE,
)


# ---------------------------------------------------------------------------
# Function 1 — is_company_suspect
# ---------------------------------------------------------------------------


def is_company_suspect(flags: dict[str, Any]) -> bool:
    """
    Gate function: returns True when remediation flagged the company field as
    a category-leak or skill-string.

    When this returns ``True`` all downstream company-identity features
    (legal suffix, posting frequency, typosquatting distance) are meaningless
    and must be skipped by the caller.

    Parameters
    ----------
    flags:
        The ``.flags`` dict from :class:`~scam_detector.data_quality.RemediatedRecord`.
    """
    return bool(flags.get("company_suspect", False))


# ---------------------------------------------------------------------------
# Function 2 — has_legal_suffix
# ---------------------------------------------------------------------------


def has_legal_suffix(company: str) -> bool:
    """
    Return True when *company* contains a recognised legal entity suffix.

    Presence of a suffix is a mild positive signal (real companies often
    register formally); absence is neutral.  This is intentionally not a
    strong signal on its own — scammers can also add suffixes.

    Parameters
    ----------
    company:
        The raw company name string.
    """
    if not company or not company.strip():
        return False
    return bool(_LEGAL_SUFFIX_RE.search(company))


# ---------------------------------------------------------------------------
# Function 3 — company_posting_frequency
# ---------------------------------------------------------------------------


def _parse_date(value: Any) -> date | None:
    """Coerce various date representations to a :class:`date`."""
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


def company_posting_frequency(
    company: str,
    all_records: list[dict[str, Any]],
) -> dict[str, Any]:
    """
    Analyse how many times *company* appears across *all_records* and how
    diverse those postings are.

    Returned dict
    -------------
    ``posting_count``        — number of postings by this company in the batch
    ``date_span_days``       — calendar-day span between oldest and newest posting
                               (0.0 if only one posting or no parseable dates)
    ``role_diversity_score`` — Jaccard-based diversity of field/tags across
                               postings; 0.0 = all identical, 1.0 = maximally
                               varied.  High diversity from a single company is a
                               shell-company signal.

    Shell-company heuristic: a company with many postings spanning many
    wildly different role categories (high ``role_diversity_score``) is
    suspicious — real employers hire for related roles.

    Parameters
    ----------
    company:
        Exact company string to look up (case-insensitive).
    all_records:
        Full list of internship dicts in the current batch.
    """
    company_lc = (company or "").strip().lower()
    if not company_lc:
        return {"posting_count": 0, "date_span_days": 0.0, "role_diversity_score": 0.0}

    matches = [
        r for r in all_records
        if (r.get("company") or "").strip().lower() == company_lc
    ]

    posting_count = len(matches)
    if posting_count == 0:
        return {"posting_count": 0, "date_span_days": 0.0, "role_diversity_score": 0.0}

    # Date range
    dates = [d for r in matches if (d := _parse_date(r.get("datePublished") or r.get("date_published")))]
    if len(dates) >= 2:
        date_span_days = float((max(dates) - min(dates)).days)
    else:
        date_span_days = 0.0

    # Role diversity via field + tags
    # Collect all unique category tokens across this company's postings.
    all_tokens: list[set[str]] = []
    for r in matches:
        tokens: set[str] = set()
        field_raw = r.get("field") or []
        if isinstance(field_raw, list):
            tokens.update(f.strip().lower() for f in field_raw if f)
        tags_raw = r.get("tags") or []
        if isinstance(tags_raw, list):
            tokens.update(t.strip().lower() for t in tags_raw if t)
        elif isinstance(tags_raw, str):
            tokens.update(t.strip().lower() for t in tags_raw.split(",") if t.strip())
        if tokens:
            all_tokens.append(tokens)

    if len(all_tokens) < 2:
        role_diversity_score = 0.0
    else:
        # Mean pairwise Jaccard distance across all posting-pairs.
        # Jaccard distance = 1 - |A ∩ B| / |A ∪ B|
        pair_distances: list[float] = []
        for i in range(len(all_tokens)):
            for j in range(i + 1, len(all_tokens)):
                a, b = all_tokens[i], all_tokens[j]
                union = a | b
                inter = a & b
                if union:
                    pair_distances.append(1.0 - len(inter) / len(union))
                else:
                    pair_distances.append(0.0)
        role_diversity_score = sum(pair_distances) / len(pair_distances)

    return {
        "posting_count": posting_count,
        "date_span_days": date_span_days,
        "role_diversity_score": round(role_diversity_score, 4),
    }


# ---------------------------------------------------------------------------
# Function 4 — typosquat_brand_distance
# ---------------------------------------------------------------------------


def typosquat_brand_distance(
    company: str,
    known_brands: list[str] | None = None,
) -> float:
    """
    Minimum normalised edit distance between *company* and each entry in
    *known_brands*.

    Uses ``python-Levenshtein`` (C extension) for speed.  Falls back to a
    pure-Python implementation if the library is not installed.

    The returned value is normalised to [0, 1]:
      0.0 → exact match with a known brand  (maximum impersonation risk)
      1.0 → maximally distant from all brands

    Why normalise by max-length?  A 2-character edit on "Google" → "Goggle"
    is more suspicious than a 2-character edit on "NayePankh Foundation".

    Parameters
    ----------
    company:
        The company name to test.
    known_brands:
        Override the module-level ``_KNOWN_BRANDS`` seed list (useful in
        tests or when a richer gazetteer is available).

    Notes
    -----
    The seed list ``_KNOWN_BRANDS`` covers ~25 commonly-impersonated names.
    It is documented as requiring expansion via a company-gazetteer enrichment
    step — see Phase 1 architecture notes.
    """
    brands = known_brands if known_brands is not None else _KNOWN_BRANDS
    if not company or not company.strip() or not brands:
        return 1.0

    company_clean = company.strip().lower()

    try:
        from Levenshtein import distance as lev_distance  # type: ignore
    except ImportError:
        # Pure-Python fallback (Wagner-Fischer DP)
        def lev_distance(a: str, b: str) -> int:  # type: ignore[misc]
            m, n = len(a), len(b)
            dp = list(range(n + 1))
            for i in range(1, m + 1):
                prev = dp[0]
                dp[0] = i
                for j in range(1, n + 1):
                    temp = dp[j]
                    if a[i - 1] == b[j - 1]:
                        dp[j] = prev
                    else:
                        dp[j] = 1 + min(prev, dp[j], dp[j - 1])
                    prev = temp
            return dp[n]

    min_normalised = 1.0
    for brand in brands:
        brand_lc = brand.strip().lower()
        raw_dist = lev_distance(company_clean, brand_lc)
        max_len = max(len(company_clean), len(brand_lc), 1)
        normalised = raw_dist / max_len
        if normalised < min_normalised:
            min_normalised = normalised
        if min_normalised == 0.0:
            break  # exact match found — stop early

    return round(float(min_normalised), 4)


# ---------------------------------------------------------------------------
# Function 5 — fetch_domain_age_days
# ---------------------------------------------------------------------------

_WHOIS_CACHE_DB_PATH = Path(__file__).parent / "whois_cache.sqlite"


def fetch_domain_age_days(
    domain: str,
    cache_db_path: str | Path | None = None,
    ttl_days: int = 30,
) -> int | None:
    """
    Fetch WHOIS domain creation date and compute domain age in days.
    Uses an SQLite local cache with TTL (default 30 days).

    Lookup failures (socket error, domain missing, no python-whois installed)
    are treated as missing data (returns None, NOT 0).
    """
    if not domain or not domain.strip():
        return None

    clean_domain = domain.strip().lower()
    clean_domain = re.sub(r"^https?://", "", clean_domain).split("/")[0].split(":")[0]
    if not clean_domain:
        return None

    db_path = Path(cache_db_path) if cache_db_path else _WHOIS_CACHE_DB_PATH
    db_path.parent.mkdir(parents=True, exist_ok=True)

    now = time.time()
    ttl_seconds = ttl_days * 86400

    # Query SQLite Cache
    try:
        conn = sqlite3.connect(str(db_path))
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS whois_cache (
                domain TEXT PRIMARY KEY,
                age_days INTEGER,
                fetched_at REAL
            )
            """
        )
        cursor = conn.execute(
            "SELECT age_days, fetched_at FROM whois_cache WHERE domain = ?",
            (clean_domain,),
        )
        row = cursor.fetchone()
        if row is not None:
            cached_age, fetched_at = row
            conn.close()
            if now - fetched_at < ttl_seconds:
                return cached_age
    except Exception:
        conn = None

    # Perform WHOIS Lookup
    age_days: int | None = None
    try:
        import whois  # type: ignore
        w = whois.whois(clean_domain)
        creation_date = w.creation_date
        if isinstance(creation_date, list):
            creation_date = creation_date[0]
        if isinstance(creation_date, (datetime, date)):
            c_date = creation_date.date() if isinstance(creation_date, datetime) else creation_date
            today = datetime.now(timezone.utc).date()
            if c_date <= today:
                age_days = (today - c_date).days
    except Exception:
        age_days = None  # Lookup failure treated as missing data

    # Store Result in Cache
    try:
        if conn is None:
            conn = sqlite3.connect(str(db_path))
        with conn:
            conn.execute(
                "REPLACE INTO whois_cache (domain, age_days, fetched_at) VALUES (?, ?, ?)",
                (clean_domain, age_days, now),
            )
        conn.close()
    except Exception:
        pass

    return age_days



# ---------------------------------------------------------------------------
# Combined output model & top-level extractor
# ---------------------------------------------------------------------------
# Import here (bottom of file) to avoid circular imports between sibling
# modules at collection time.
from scam_detector.features.url_features import (  # noqa: E402
    UrlFeatures,
    parse_url_components,
    is_platform_internal_link,
    url_entropy,
    is_url_shortener,
    domain_company_name_similarity,
    is_known_ats_domain,
    _tld_risk_score,
)


class CompanyUrlFeatureVector(BaseModel):
    """
    Combined company + URL feature vector for a single internship record.
    Passed into the scoring layer.
    """

    company: CompanyFeatures = Field(default_factory=CompanyFeatures)
    url: UrlFeatures = Field(default_factory=UrlFeatures)


def extract_company_url_features(
    record: dict[str, Any],
    all_records: list[dict[str, Any]] | None = None,
    flags: dict[str, Any] | None = None,
) -> CompanyUrlFeatureVector:
    """
    Extract all company and URL features from a remediated internship record.

    Company-identity features are short-circuited when ``flags`` indicates
    the company field is a category-leak (``is_company_suspect`` returns True).
    URL features are always computed — they are independent of company identity.

    Parameters
    ----------
    record:
        Internship document with at minimum ``company`` and ``applyLink`` keys.
    all_records:
        Full batch of internship dicts for frequency analysis.  Defaults to
        a single-element list containing only *record* when omitted.
    flags:
        Remediation flags dict (from ``RemediatedRecord.flags``).  Can also
        be embedded in *record* under the ``"_flags"`` key.
    """
    effective_flags: dict[str, Any] = flags or record.get("_flags") or {}
    batch = all_records if all_records is not None else [record]

    company: str = record.get("company") or ""
    apply_link: str = record.get("applyLink") or record.get("apply_link") or ""

    # ── URL features ──────────────────────────────────────────────────────
    components = parse_url_components(apply_link)
    tld = components["tld"]
    registered_dom = components["registered_domain"] or components["domain"]
    is_internal = is_platform_internal_link(apply_link)
    is_ats = is_known_ats_domain(apply_link)

    # ── Company features ──────────────────────────────────────────────────
    suspect = is_company_suspect(effective_flags)

    if suspect:
        # Short-circuit — all identity signals are unreliable
        company_feats = CompanyFeatures(is_suspect=True)
    else:
        freq = company_posting_frequency(company, batch)
        typo_dist = typosquat_brand_distance(company)
        # Skip the WHOIS lookup entirely for platform-internal / known-ATS
        # links: their domain age is always old and identical across nearly
        # every record on that platform (internshala.com, naukri.com, ...),
        # so it carries zero discriminative value even if looked up — this
        # was previously an unconditional network call (SQLite-cached, but
        # still wasted) on ~every record for a feature that was then never
        # read by any rule or the anomaly model at all. Only look it up for
        # the case where it's actually actionable: a genuine off-platform
        # employer domain.
        domain_age = (
            fetch_domain_age_days(registered_dom)
            if (registered_dom and not is_internal and not is_ats)
            else None
        )
        company_feats = CompanyFeatures(
            is_suspect=False,
            has_legal_suffix=has_legal_suffix(company),
            posting_count=freq["posting_count"],
            posting_date_span_days=freq["date_span_days"],
            role_diversity_score=freq["role_diversity_score"],
            typosquat_min_distance=typo_dist,
            domain_age_days=domain_age,
        )

    url_feats = UrlFeatures(
        domain=components["domain"],
        subdomain=components["subdomain"],
        registered_domain=components["registered_domain"],
        tld=tld,
        path_depth=components["path_depth"],
        query_param_count=components["query_param_count"],
        is_https=components["is_https"],
        is_platform_internal=is_internal,
        is_url_shortener=is_url_shortener(apply_link),
        is_known_ats=is_ats,
        domain_entropy=url_entropy(apply_link),
        domain_company_similarity=domain_company_name_similarity(apply_link, company),
        tld_risk_score=_tld_risk_score(tld),
    )

    return CompanyUrlFeatureVector(company=company_feats, url=url_feats)



# ---------------------------------------------------------------------------
# NGO / fundraising-sector coordinated stipend network detection
# ---------------------------------------------------------------------------
#
# Empirical finding (real Internshala + Unstop data): 7 distinct company
# names — several plausible-sounding "Foundation" entities among them — post
# under DIFFERENT role titles (Business Consultant, Program Assistant,
# Fundraising, Crowdfunding, Social Entrepreneurship) but share the *exact
# same* lump-sum stipend amount (₹15,000). SBERT-based duplicate detection
# (0.92 cosine threshold) does NOT catch this — the posting text is
# independently written per "company", only the underlying stipend template
# is identical. This is a distinct, complementary signal: same operator (or
# same aggregator platform) running multiple NGO-sounding fronts with a
# templated pay structure, rather than a copy-pasted script.
#
# Deliberately scoped to lump_sum paid stipends only (not monthly), since
# monthly stipends legitimately cluster around common round numbers
# (₹5,000/₹10,000 per month) across totally unrelated real companies —
# lump-sum is a rarer, more specific structure where exact-amount collisions
# across "different" companies are far less likely to be coincidental.
#
# Per the same philosophy already documented in duplicate_detection.py's
# NayePankh/Basti Ki Pathshala note: this is a known ambiguity (legitimate
# fundraising-platform aggregators standardizing pay across NGO partners are
# a real, non-fraudulent business model too) — surface it for human review,
# do not treat it as an automatic hard reject.

_NGO_SECTOR_KEYWORDS: list[str] = [
    "ngo", "foundation", "trust", "society", "nonprofit", "non-profit",
    "non profit", "charity", "welfare", "fundraising", "fundraiser",
    "crowdfunding", "social work", "social entrepreneurship", "csr",
    "philanthropy", "ngo internship",
]


def is_ngo_or_fundraising_sector(record: dict[str, Any]) -> bool:
    """
    Return True when the posting's company name, title, or tags indicate an
    NGO / fundraising / social-sector internship.

    Deliberately broad substring matching — this is a SECTOR classifier, not
    a fraud signal on its own. It only matters combined with
    ``ngo_stipend_network_flag``.
    """
    company = str(record.get("company") or "").lower()
    title = str(record.get("name") or record.get("title") or "").lower()
    tags = record.get("tags") or []
    tags_text = " ".join(str(t) for t in tags).lower() if isinstance(tags, list) else str(tags).lower()

    haystack = f"{company} {title} {tags_text}"
    return any(kw in haystack for kw in _NGO_SECTOR_KEYWORDS)


def build_lump_sum_stipend_index(
    remediated_records: list[tuple[dict[str, Any], dict[str, Any]]],
) -> dict[tuple[float, str], set[str]]:
    """
    Corpus-level index: (amount, currency) -> set of distinct company keys
    posting a "paid" / "lump_sum" stipend at exactly that amount.

    Parameters
    ----------
    remediated_records:
        List of (record, flags) tuples — flags used to exclude companies
        already known to be scraper noise (``company_suspect``) from the
        distinct-company count, so category-leak artifacts (e.g. a company
        field literally reading "Social Work") don't inflate the signal.
    """
    index: dict[tuple[float, str], set[str]] = defaultdict(set)
    for rec, flags in remediated_records:
        if flags.get("company_suspect"):
            continue
        stipend = rec.get("stipend") or {}
        if not isinstance(stipend, dict):
            continue
        if stipend.get("type") != "paid" or stipend.get("period") != "lump_sum":
            continue
        amount = stipend.get("amount")
        if not amount or not isinstance(amount, (int, float)) or amount <= 0:
            continue
        currency = str(stipend.get("currency") or "INR")
        company_key = (rec.get("company") or "").strip().lower()
        if not company_key:
            continue
        index[(float(amount), currency)].add(company_key)
    return index


def ngo_stipend_network_flag(
    record: dict[str, Any],
    stipend_index: dict[tuple[float, str], set[str]],
    *,
    min_distinct_companies: int = 3,
) -> tuple[bool, int]:
    """
    Return (flagged, distinct_company_count) for this record's stipend
    template network.

    Flagged when at least ``min_distinct_companies`` distinct (non-suspect)
    companies in the corpus share this record's exact lump-sum stipend
    amount — a coordinated-payout-template signal, independent of text
    similarity.
    """
    stipend = record.get("stipend") or {}
    if not isinstance(stipend, dict) or stipend.get("type") != "paid" or stipend.get("period") != "lump_sum":
        return False, 0
    amount = stipend.get("amount")
    if not amount or not isinstance(amount, (int, float)) or amount <= 0:
        return False, 0
    currency = str(stipend.get("currency") or "INR")

    companies = stipend_index.get((float(amount), currency), set())
    count = len(companies)
    return count >= min_distinct_companies, count
