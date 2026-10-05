"""
url_features.py
---------------
Signals derived from the ``applyLink`` field of an internship posting.

All public functions are pure and individually testable.

Phase 1 analysis note on platform-internal links
-------------------------------------------------
When ``is_platform_internal_link`` returns True the apply link points back to
the same aggregator that sourced the record (e.g. an Internshala posting
whose applyLink is also internshala.com).  Such links carry LESS independent
verification signal than off-platform links — the listing has not been
cross-validated against an independent employer domain.  Downstream scoring
should REDUCE (not increase) the weight given to URL-based trust signals for
these records, not penalise them harder.

Phase 1 analysis note on ATS domains
-------------------------------------
When ``is_known_ats_domain`` returns True the employer is using a legitimate
third-party Applicant Tracking System (greenhouse.io, lever.co, workday, etc.).
These domains must NOT be penalised the same way as an unknown random third-
party domain would be.  Downstream scoring should treat ATS domains as
neutral-to-positive signals.
"""

from __future__ import annotations

from functools import lru_cache
import math
import re
from typing import Any

from pydantic import BaseModel, Field

# ---------------------------------------------------------------------------
# Pydantic output model
# ---------------------------------------------------------------------------


class UrlFeatures(BaseModel):
    """Feature slice for URL / apply-link signals."""

    domain: str = Field(default="")
    subdomain: str = Field(default="")
    registered_domain: str = Field(default="")
    tld: str = Field(default="")
    path_depth: int = Field(default=0, ge=0)
    query_param_count: int = Field(default=0, ge=0)
    is_https: bool = Field(default=True)
    is_platform_internal: bool = Field(default=False)
    is_url_shortener: bool = Field(default=False)
    is_known_ats: bool = Field(default=False)
    domain_entropy: float = Field(default=0.0, ge=0.0)
    domain_company_similarity: float = Field(default=0.0, ge=0.0, le=1.0)
    tld_risk_score: float = Field(default=0.0, ge=0.0, le=1.0)


# ---------------------------------------------------------------------------
# Seed data
# ---------------------------------------------------------------------------

# Aggregator / source-platform domains — links pointing here are
# platform-internal (less independent verification value).
_PLATFORM_DOMAINS: list[str] = [
    "internshala.com",
    "naukri.com",
    "freshersworld.com",
    "letsintern.in",
    "unstop.com",
    "indeed.co.in",
    "indeed.com",
    "foundit.in",          # formerly Monster India
    "shine.com",
    "timesjobs.com",
    "apna.co",
    "hirist.tech",
    "cutshort.io",
    "wellfound.com",       # AngelList Talent
    "internshala.app",
]

# Known Applicant Tracking System domains — do NOT penalise these.
_ATS_DOMAINS: list[str] = [
    "greenhouse.io",
    "boards.greenhouse.io",
    "job-boards.greenhouse.io",
    "lever.co",
    "jobs.lever.co",
    "workday.com",
    "myworkdayjobs.com",
    "wd1.myworkdayjobs.com",
    "wd5.myworkdayjobs.com",
    "wd3.myworkdayjobs.com",
    "zohorecruit.com",
    "recruit.zoho.com",
    "taleo.net",
    "icims.com",
    "smartrecruiters.com",
    "breezy.hr",
    "recruitee.com",
    "keka.com",
    "darwinbox.com",
    "springrecruit.com",
    "freshteam.com",
    "bamboohr.com",
    "ashbyhq.com",
    "rippling.com",
    "jobvite.com",
]

# URL shortener registered domains
_SHORTENER_DOMAINS: list[str] = [
    "bit.ly",
    "tinyurl.com",
    "t.co",
    "ow.ly",
    "buff.ly",
    "goo.gl",
    "short.io",
    "rb.gy",
    "cutt.ly",
    "shorturl.at",
    "tiny.cc",
    "go.acciojob.com",    # recruiter-facing acciojob shortener seen in corpus
    "links.acciojob.com",
    "forms.gle",          # Google Forms (often used as apply-link proxy)
    "docs.google.com",    # Google Docs application form pattern
]

# High-risk TLDs observed in scam postings (probabilistic, not definitive).
_HIGH_RISK_TLDS: frozenset[str] = frozenset(
    {
        "xyz", "top", "click", "link", "online", "site", "website",
        "space", "tech", "store", "live", "pw", "cc", "tk", "ml",
        "ga", "cf", "gq",
    }
)

# Low-risk established TLDs (not penalised, gives tiny positive signal)
_LOW_RISK_TLDS: frozenset[str] = frozenset(
    {
        "com", "org", "net", "edu", "gov", "io", "co", "in",
        "co.in", "ac.in",
    }
)


# ---------------------------------------------------------------------------
# Helper: tldextract wrapper
# ---------------------------------------------------------------------------


@lru_cache(maxsize=4096)
def _extract(url: str) -> tuple[str, str, str]:
    """
    Return (subdomain, domain, suffix/tld) using tldextract.
    Falls back to a simple regex split if tldextract is not installed.
    """
    if not url:
        return "", "", ""
    try:
        import tldextract  # type: ignore
        result = tldextract.extract(url)
        return result.subdomain, result.domain, result.suffix
    except ImportError:
        # Minimal fallback — strips scheme and path
        cleaned = re.sub(r"^https?://", "", url).split("/")[0].split("?")[0]
        parts = cleaned.rsplit(".", 2)
        if len(parts) >= 2:
            return "", parts[-2], parts[-1]
        return "", cleaned, ""


def _registered_domain(subdomain: str, domain: str, suffix: str) -> str:
    """Reconstruct the registered domain (domain + TLD)."""
    if domain and suffix:
        return f"{domain}.{suffix}"
    return domain or ""


# ---------------------------------------------------------------------------
# Function 1 — parse_url_components
# ---------------------------------------------------------------------------


def parse_url_components(url: str) -> dict[str, Any]:
    """
    Decompose *url* into structured components.

    Returns
    -------
    dict with keys:
        ``domain``             — base domain name without TLD
        ``subdomain``          — subdomain string (empty if none)
        ``registered_domain``  — domain + TLD
        ``tld``                — public suffix
        ``path_depth``         — number of non-empty path segments
        ``query_param_count``  — number of distinct query-string parameters
        ``is_https``           — True if scheme is https
    """
    if not url or not url.strip():
        return {
            "domain": "",
            "subdomain": "",
            "registered_domain": "",
            "tld": "",
            "path_depth": 0,
            "query_param_count": 0,
            "is_https": False,
        }

    subdomain, domain, tld = _extract(url)
    reg_domain = _registered_domain(subdomain, domain, tld)

    is_https = url.strip().lower().startswith("https://")

    # Path depth: strip scheme + host, count non-empty segments
    path_part = re.sub(r"^https?://[^/]+", "", url).split("?")[0]
    path_segments = [s for s in path_part.split("/") if s]
    path_depth = len(path_segments)

    # Query params
    qs_match = re.search(r"\?(.+)$", url)
    if qs_match:
        pairs = [p for p in qs_match.group(1).split("&") if p]
        query_param_count = len(pairs)
    else:
        query_param_count = 0

    return {
        "domain": domain,
        "subdomain": subdomain,
        "registered_domain": reg_domain,
        "tld": tld,
        "path_depth": path_depth,
        "query_param_count": query_param_count,
        "is_https": is_https,
    }


# ---------------------------------------------------------------------------
# Function 2 — is_platform_internal_link
# ---------------------------------------------------------------------------


def is_platform_internal_link(
    url: str,
    known_platform_domains: list[str] | None = None,
) -> bool:
    """
    Return True if *url* points to a known aggregator / source platform.

    Platform-internal links carry LESS independent verification signal.
    Downstream scoring should REDUCE the weight of URL-based signals for
    these records, not penalise them.

    Parameters
    ----------
    url:
        The apply link URL.
    known_platform_domains:
        Override the module-level ``_PLATFORM_DOMAINS`` list.
    """
    platforms = known_platform_domains if known_platform_domains is not None else _PLATFORM_DOMAINS
    if not url:
        return False

    subdomain, domain, tld = _extract(url)
    reg_domain = _registered_domain(subdomain, domain, tld)
    full_with_sub = f"{subdomain}.{reg_domain}" if subdomain else reg_domain

    for platform in platforms:
        platform_lc = platform.strip().lower()
        if reg_domain.lower() == platform_lc or full_with_sub.lower() == platform_lc:
            return True
        # Also handle subdomain-of-platform (e.g. careers.internshala.com)
        if reg_domain.lower().endswith(platform_lc):
            return True

    return False


# ---------------------------------------------------------------------------
# Function 3 — url_entropy
# ---------------------------------------------------------------------------


def url_entropy(url: str) -> float:
    """
    Shannon entropy of the registered domain string.

    Randomly-generated or algorithmically-constructed domains
    (common in phishing / scam infrastructure) have higher entropy than
    natural human-readable names.

    Typical ranges (observed):
        google.com       → ~2.75 bits
        razorpay.com     → ~3.0  bits
        xkf3q2mz.xyz    → ~3.5+ bits  (high entropy = suspicious)

    Parameters
    ----------
    url:
        The full URL string.

    Returns
    -------
    float
        Shannon entropy in bits.  Returns 0.0 for empty/unparseable input.
    """
    if not url:
        return 0.0

    _, domain, _ = _extract(url)
    s = domain.strip().lower()
    if not s:
        return 0.0

    freq: dict[str, int] = {}
    for c in s:
        freq[c] = freq.get(c, 0) + 1

    n = len(s)
    entropy = -sum((count / n) * math.log2(count / n) for count in freq.values())
    return round(entropy, 4)


# ---------------------------------------------------------------------------
# Function 4 — is_url_shortener
# ---------------------------------------------------------------------------


def is_url_shortener(url: str) -> bool:
    """
    Return True if *url* uses a known URL-shortening service.

    URL shorteners obscure the destination and are a common signal in
    Telegram-channel job postings (heavily present in the corpus).  They
    should be treated as a mild-to-moderate risk signal, not a hard reject —
    many legitimate aggregators also shorten links.

    Parameters
    ----------
    url:
        The apply link URL.
    """
    if not url:
        return False

    subdomain, domain, tld = _extract(url)
    reg_domain = _registered_domain(subdomain, domain, tld)
    full_with_sub = f"{subdomain}.{reg_domain}" if subdomain else reg_domain

    for shortener in _SHORTENER_DOMAINS:
        shortener_lc = shortener.strip().lower()
        if reg_domain.lower() == shortener_lc or full_with_sub.lower() == shortener_lc:
            return True

    return False


# ---------------------------------------------------------------------------
# Function 5 — domain_company_name_similarity
# ---------------------------------------------------------------------------


def domain_company_name_similarity(url: str, company: str) -> float:
    """
    Fuzzy similarity between the URL's domain name and the stated company name.

    For off-platform apply links, a substantial mismatch suggests the
    listing may be fraudulent or scraped incorrectly.

    Example:
        company = "Razorpay", domain = "razorpay" → similarity ≈ 1.0
        company = "Acme Global Pvt Ltd", domain = "somexyz23" → similarity ≈ 0.1

    Uses ``rapidfuzz.fuzz.token_set_ratio`` when available; falls back to
    simple longest-common-subsequence ratio otherwise.

    Returns
    -------
    float in [0, 1]
        1.0 → domain closely matches company name.
    """
    if not url or not company or not company.strip():
        return 0.0

    _, domain, _ = _extract(url)
    if not domain:
        return 0.0

    # Normalise: remove legal suffixes and non-alpha chars for comparison
    company_clean = re.sub(
        r"\b(pvt|ltd|private|limited|llp|inc|foundation|trust|corp)\b\.?",
        "",
        company.strip().lower(),
        flags=re.IGNORECASE,
    )
    company_clean = re.sub(r"[^a-z0-9]", "", company_clean)
    domain_clean = re.sub(r"[^a-z0-9]", "", domain.strip().lower())

    if not company_clean or not domain_clean:
        return 0.0

    try:
        from rapidfuzz import fuzz  # type: ignore
        ratio = fuzz.token_set_ratio(company_clean, domain_clean) / 100.0
    except ImportError:
        # LCS-based fallback
        m, n = len(company_clean), len(domain_clean)
        dp = [[0] * (n + 1) for _ in range(m + 1)]
        for i in range(1, m + 1):
            for j in range(1, n + 1):
                if company_clean[i - 1] == domain_clean[j - 1]:
                    dp[i][j] = dp[i - 1][j - 1] + 1
                else:
                    dp[i][j] = max(dp[i - 1][j], dp[i][j - 1])
        lcs = dp[m][n]
        ratio = (2 * lcs) / (m + n)

    return round(float(min(ratio, 1.0)), 4)


# ---------------------------------------------------------------------------
# Function 6 — is_known_ats_domain
# ---------------------------------------------------------------------------


def is_known_ats_domain(url: str) -> bool:
    """
    Return True if *url* points to a recognised third-party ATS platform.

    ATS (Applicant Tracking System) domains must NOT be penalised the same
    way as unknown third-party domains.  Legitimate companies commonly post
    apply links via greenhouse.io, lever.co, workday, etc.

    Phase 1 false-positive risk: penalising ATS domains would incorrectly
    flag high-quality postings from well-known companies (Gemini via
    greenhouse.io, Motorola via workday — both confirmed in the corpus).

    Parameters
    ----------
    url:
        The apply link URL.
    """
    if not url:
        return False

    subdomain, domain, tld = _extract(url)
    reg_domain = _registered_domain(subdomain, domain, tld)
    full_with_sub = f"{subdomain}.{reg_domain}" if subdomain else reg_domain

    for ats in _ATS_DOMAINS:
        ats_lc = ats.strip().lower()
        if reg_domain.lower() == ats_lc or full_with_sub.lower() == ats_lc:
            return True
        # Handle branded Workday subdomains: motorolasolutions.wd5.myworkdayjobs.com
        if reg_domain.lower().endswith(ats_lc):
            return True

    return False


# ---------------------------------------------------------------------------
# TLD risk scoring helper (used by CompanyUrlFeatureVector)
# ---------------------------------------------------------------------------


def _tld_risk_score(tld: str) -> float:
    """Return a [0, 1] risk score for the TLD."""
    if not tld:
        return 0.5  # unknown → neutral-ish
    tld_lc = tld.strip().lower()
    if tld_lc in _HIGH_RISK_TLDS:
        return 0.85
    if tld_lc in _LOW_RISK_TLDS:
        return 0.05
    return 0.3  # unfamiliar but not in explicit high-risk set
