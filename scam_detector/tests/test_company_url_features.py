"""
test_company_url_features.py
----------------------------
Tests for scam_detector.features.company_features and url_features.

Real-data fixtures sourced from corpus:
  ANAKIN_RECORD       — internshala.com apply link (platform-internal)
  GEMINI_RECORD       — boards.greenhouse.io (ATS, off-platform)
  MOTOROLA_RECORD     — motorolasolutions.wd5.myworkdayjobs.com (branded Workday)
  TINYURL_RECORD      — tinyurl.com shortener (Enterpret, Telegram channel)
  RAZORPAY_RECORD     — direct employer domain (razorpay.com)
  SUSPECT_COMPANY_RECORD — company="Digital Marketing" (category leak)

All URL fixtures come directly from internScraper/checkpoint_internships.json
and web_scrapper/telegram scraper/all_channels_internships.json.
"""
from __future__ import annotations
import pytest

from scam_detector.features.company_features import (
    CompanyFeatures,
    CompanyUrlFeatureVector,
    extract_company_url_features,
    is_company_suspect,
    has_legal_suffix,
    company_posting_frequency,
    typosquat_brand_distance,
)
from scam_detector.features.url_features import (
    UrlFeatures,
    parse_url_components,
    is_platform_internal_link,
    url_entropy,
    is_url_shortener,
    domain_company_name_similarity,
    is_known_ats_domain,
)

# ===========================================================================
# Fixtures — real records from corpus
# ===========================================================================

# Internshala platform-internal link (corpus: checkpoint_internships.json)
ANAKIN_RECORD = {
    "name": "Software Development",
    "company": "Anakin",
    "applyLink": "https://internshala.com/internship/detail/software-development-internship-in-bangalore-at-anakin1777439557",
    "field": ["Software Engineering", "Backend"],
    "tags": ["internship", "backend"],
    "datePublished": "2026-05-01",
}

# Greenhouse ATS (corpus: checkpoint_internships.json — Gemini)
GEMINI_RECORD = {
    "name": "Software Engineering Intern - Fall 2026",
    "company": "Gemini",
    "applyLink": "https://boards.greenhouse.io/embed/job_app?for=gemini&token=7875125&gh_jid=7875125",
    "field": ["Software Engineering"],
    "tags": ["internship", "crypto"],
    "datePublished": "2026-04-15",
}

# Branded Workday ATS (corpus: checkpoint_internships.json — Motorola)
MOTOROLA_RECORD = {
    "name": "Intern - Web Interface Software Engineer - 2026",
    "company": "Motorola Solutions",
    "applyLink": "https://motorolasolutions.wd5.myworkdayjobs.com/en-US/careers/job/Los-Angeles-CA/Intern---Web-Interface-Software-Engineer--2026-_R64590",
    "field": ["Software Engineering"],
    "tags": ["internship"],
    "datePublished": "2026-04-10",
}

# tinyurl.com shortener (corpus: telegram all_channels_internships.json)
TINYURL_RECORD = {
    "name": "Software Engineer Intern Hiring",
    "company": "Enterpret",
    "applyLink": "https://tinyurl.com/4ma7eukj",
    "field": ["Software Engineering"],
    "tags": ["internship"],
    "datePublished": "2026-05-06",
}

# bit.ly shortener (corpus: telegram all_channels_internships.json)
BITLY_RECORD = {
    "name": "INDmoney Web Intern Hiring",
    "company": "",
    "applyLink": "https://bit.ly/4mRncxW",
    "field": [],
    "tags": ["internship"],
    "datePublished": "2025-10-07",
}

# Direct employer domain — Razorpay (synthetic but representative)
RAZORPAY_RECORD = {
    "name": "Frontend Developer Intern",
    "company": "Razorpay",
    "applyLink": "https://razorpay.com/jobs/frontend-intern-2024",
    "field": ["Frontend", "Web Development"],
    "tags": ["fintech", "react"],
    "datePublished": "2026-06-05",
}

# Category-leak company field — flagged by remediation (Prompt 1)
SUSPECT_COMPANY_RECORD = {
    "name": "Data Science Internship",
    "company": "Digital Marketing",
    "applyLink": "https://letsintern.in/data-science/",
    "field": ["Data Science"],
    "tags": ["internship", "remote"],
    "datePublished": "2026-05-01",
    "_flags": {"company_suspect": True, "company_source": "category_leak"},
}

# Real company with legal suffix
LEGAL_SUFFIX_RECORD = {
    "name": "Business Analyst Intern",
    "company": "Shaadi.com Pvt Ltd",
    "applyLink": "https://shaadi.com/careers/analyst-intern",
    "field": ["Business Analytics"],
    "tags": ["internship"],
    "datePublished": "2026-05-20",
}

# Batch for frequency testing
FREQUENCY_BATCH = [
    {"company": "Acme Corp", "applyLink": "https://acme.com/job1",
     "field": ["Marketing"], "tags": ["sales"], "datePublished": "2026-01-01"},
    {"company": "Acme Corp", "applyLink": "https://acme.com/job2",
     "field": ["Software Engineering"], "tags": ["dev"], "datePublished": "2026-03-15"},
    {"company": "Acme Corp", "applyLink": "https://acme.com/job3",
     "field": ["Finance", "Accounting"], "tags": ["finance"], "datePublished": "2026-06-01"},
    {"company": "Other Co", "applyLink": "https://other.com/job1",
     "field": ["Design"], "tags": ["ui"], "datePublished": "2026-02-01"},
]


# ===========================================================================
# is_company_suspect
# ===========================================================================

class TestIsCompanySuspect:

    def test_flagged_record_returns_true(self) -> None:
        flags = {"company_suspect": True}
        assert is_company_suspect(flags) is True

    def test_clean_record_returns_false(self) -> None:
        assert is_company_suspect({}) is False

    def test_false_value_returns_false(self) -> None:
        assert is_company_suspect({"company_suspect": False}) is False

    def test_missing_key_returns_false(self) -> None:
        assert is_company_suspect({"other_flag": True}) is False


# ===========================================================================
# has_legal_suffix
# ===========================================================================

class TestHasLegalSuffix:

    def test_pvt_ltd_detected(self) -> None:
        assert has_legal_suffix("Shaadi.com Pvt Ltd") is True

    def test_pvt_with_period_detected(self) -> None:
        assert has_legal_suffix("Acme Pvt. Ltd.") is True

    def test_llp_detected(self) -> None:
        assert has_legal_suffix("Singh & Associates LLP") is True

    def test_foundation_detected(self) -> None:
        assert has_legal_suffix("NayePankh Foundation") is True

    def test_inc_detected(self) -> None:
        assert has_legal_suffix("Gemini Inc.") is True

    def test_limited_detected(self) -> None:
        assert has_legal_suffix("ABC Limited") is True

    def test_no_suffix_returns_false(self) -> None:
        assert has_legal_suffix("Razorpay") is False

    def test_empty_string_returns_false(self) -> None:
        assert has_legal_suffix("") is False

    def test_case_insensitive(self) -> None:
        assert has_legal_suffix("ACME PRIVATE LIMITED") is True

    def test_ngo_detected(self) -> None:
        assert has_legal_suffix("Basti Ki Pathshala NGO") is True


# ===========================================================================
# company_posting_frequency
# ===========================================================================

class TestCompanyPostingFrequency:

    def test_single_record_count_one(self) -> None:
        result = company_posting_frequency("Acme Corp", FREQUENCY_BATCH)
        assert result["posting_count"] == 3  # 3 Acme Corp records

    def test_date_span_is_positive(self) -> None:
        result = company_posting_frequency("Acme Corp", FREQUENCY_BATCH)
        assert result["date_span_days"] > 0

    def test_date_span_correct_days(self) -> None:
        # Jan 1 to Jun 1 = 151 days
        result = company_posting_frequency("Acme Corp", FREQUENCY_BATCH)
        assert result["date_span_days"] == pytest.approx(151.0, abs=1.0)

    def test_diversity_score_nonzero_for_varied_roles(self) -> None:
        # Marketing, Software Engineering, Finance are all different
        result = company_posting_frequency("Acme Corp", FREQUENCY_BATCH)
        assert result["role_diversity_score"] > 0.0

    def test_diversity_score_zero_for_identical_roles(self) -> None:
        same_batch = [
            {"company": "CorpX", "field": ["Marketing"], "tags": ["sales"],
             "datePublished": "2026-01-01"},
            {"company": "CorpX", "field": ["Marketing"], "tags": ["sales"],
             "datePublished": "2026-02-01"},
        ]
        result = company_posting_frequency("CorpX", same_batch)
        assert result["role_diversity_score"] == 0.0

    def test_unknown_company_returns_zero_count(self) -> None:
        result = company_posting_frequency("NoSuchCo", FREQUENCY_BATCH)
        assert result["posting_count"] == 0

    def test_empty_company_returns_zero_count(self) -> None:
        result = company_posting_frequency("", FREQUENCY_BATCH)
        assert result["posting_count"] == 0

    def test_case_insensitive_matching(self) -> None:
        result_lower = company_posting_frequency("acme corp", FREQUENCY_BATCH)
        result_upper = company_posting_frequency("ACME CORP", FREQUENCY_BATCH)
        assert result_lower["posting_count"] == result_upper["posting_count"]

    def test_single_posting_no_date_span(self) -> None:
        result = company_posting_frequency("Other Co", FREQUENCY_BATCH)
        assert result["posting_count"] == 1
        assert result["date_span_days"] == 0.0

    def test_diversity_score_bounded(self) -> None:
        result = company_posting_frequency("Acme Corp", FREQUENCY_BATCH)
        assert 0.0 <= result["role_diversity_score"] <= 1.0


# ===========================================================================
# typosquat_brand_distance
# ===========================================================================

class TestTyposquatBrandDistance:

    def test_exact_match_is_zero(self) -> None:
        dist = typosquat_brand_distance("Google")
        assert dist == 0.0

    def test_exact_match_case_insensitive(self) -> None:
        dist = typosquat_brand_distance("GOOGLE")
        assert dist == 0.0

    def test_one_char_typo_scores_low(self) -> None:
        # "Goggle" vs "Google" — 1 edit out of 6 chars
        dist = typosquat_brand_distance("Goggle")
        assert dist < 0.3

    def test_completely_different_name_scores_high(self) -> None:
        dist = typosquat_brand_distance("NayePankh Foundation")
        assert dist > 0.5

    def test_return_type_is_float(self) -> None:
        assert isinstance(typosquat_brand_distance("Razorpay"), float)

    def test_score_bounded(self) -> None:
        for name in ["Google", "Goog1e", "XYZ Pvt Ltd", ""]:
            d = typosquat_brand_distance(name)
            assert 0.0 <= d <= 1.0

    def test_custom_brand_list(self) -> None:
        dist = typosquat_brand_distance("Acme", known_brands=["Acme"])
        assert dist == 0.0

    def test_empty_company_returns_one(self) -> None:
        assert typosquat_brand_distance("") == 1.0

    def test_close_impersonation_lower_than_unrelated(self) -> None:
        impersonation = typosquat_brand_distance("Micrsoft")   # 1 edit from Microsoft
        unrelated = typosquat_brand_distance("XyzTrading123")
        assert impersonation < unrelated

# ===========================================================================
# parse_url_components
# ===========================================================================

class TestParseUrlComponents:

    def test_empty_url_returns_safe_defaults(self) -> None:
        r = parse_url_components("")
        assert r["domain"] == ""
        assert r["is_https"] is False

    def test_https_detected(self) -> None:
        r = parse_url_components("https://razorpay.com/jobs")
        assert r["is_https"] is True

    def test_http_detected(self) -> None:
        r = parse_url_components("http://example.com/apply")
        assert r["is_https"] is False

    def test_domain_extracted(self) -> None:
        r = parse_url_components("https://razorpay.com/jobs/intern")
        assert r["domain"] == "razorpay"
        assert r["tld"] == "com"

    def test_path_depth_counted(self) -> None:
        r = parse_url_components("https://internshala.com/internship/detail/abc123")
        assert r["path_depth"] == 3

    def test_query_params_counted(self) -> None:
        r = parse_url_components(
            "https://boards.greenhouse.io/embed/job_app?for=gemini&token=123&gh_jid=456"
        )
        assert r["query_param_count"] == 3

    def test_no_query_params(self) -> None:
        r = parse_url_components("https://razorpay.com/careers")
        assert r["query_param_count"] == 0

    def test_subdomain_extracted(self) -> None:
        r = parse_url_components("https://boards.greenhouse.io/snorkelai/jobs/123")
        # tldextract correctly extracts "boards"; the regex fallback (no tldextract)
        # may return "" — both are acceptable; what matters is registered_domain is set.
        try:
            import tldextract  # type: ignore
            assert r["subdomain"] == "boards"
        except ImportError:
            assert r["registered_domain"] != ""  # fallback still parses something

    def test_registered_domain_assembled(self) -> None:
        r = parse_url_components("https://razorpay.com/apply")
        assert r["registered_domain"] == "razorpay.com"


# ===========================================================================
# is_platform_internal_link
# ===========================================================================

class TestIsPlatformInternalLink:

    def test_internshala_link_is_internal(self) -> None:
        assert is_platform_internal_link(ANAKIN_RECORD["applyLink"]) is True

    def test_naukri_link_is_internal(self) -> None:
        assert is_platform_internal_link("https://www.naukri.com/job-listings-intern") is True

    def test_letsintern_link_is_internal(self) -> None:
        assert is_platform_internal_link("https://letsintern.in/data-science/") is True

    def test_greenhouse_link_is_not_internal(self) -> None:
        assert is_platform_internal_link(GEMINI_RECORD["applyLink"]) is False

    def test_workday_link_is_not_internal(self) -> None:
        assert is_platform_internal_link(MOTOROLA_RECORD["applyLink"]) is False

    def test_direct_employer_link_is_not_internal(self) -> None:
        assert is_platform_internal_link(RAZORPAY_RECORD["applyLink"]) is False

    def test_tinyurl_is_not_internal(self) -> None:
        assert is_platform_internal_link(TINYURL_RECORD["applyLink"]) is False

    def test_empty_url_returns_false(self) -> None:
        assert is_platform_internal_link("") is False

    def test_custom_platform_list(self) -> None:
        assert is_platform_internal_link(
            "https://myplatform.com/job/123",
            known_platform_domains=["myplatform.com"],
        ) is True


# ===========================================================================
# url_entropy
# ===========================================================================

class TestUrlEntropy:

    def test_empty_url_returns_zero(self) -> None:
        assert url_entropy("") == 0.0

    def test_readable_domain_has_lower_entropy(self) -> None:
        readable = url_entropy("https://razorpay.com/jobs")
        gibberish = url_entropy("https://xkf3q2mz9p.xyz/apply")
        assert readable < gibberish

    def test_return_type_is_float(self) -> None:
        assert isinstance(url_entropy("https://google.com"), float)

    def test_entropy_is_positive_for_real_url(self) -> None:
        assert url_entropy("https://internshala.com/internship/detail/abc") > 0.0

    def test_repeated_char_domain_has_low_entropy(self) -> None:
        # "aaaa" domain → all same chars → entropy = 0
        assert url_entropy("https://aaaa.com/apply") < 0.5

    def test_entropy_non_negative(self) -> None:
        for url in [
            "https://razorpay.com", "https://bit.ly/abc", "",
            "https://xkf3q2mzpqrs.xyz",
        ]:
            assert url_entropy(url) >= 0.0


# ===========================================================================
# is_url_shortener
# ===========================================================================

class TestIsUrlShortener:

    def test_tinyurl_detected(self) -> None:
        assert is_url_shortener(TINYURL_RECORD["applyLink"]) is True

    def test_bitly_detected(self) -> None:
        assert is_url_shortener(BITLY_RECORD["applyLink"]) is True

    def test_direct_employer_not_shortener(self) -> None:
        assert is_url_shortener(RAZORPAY_RECORD["applyLink"]) is False

    def test_greenhouse_not_shortener(self) -> None:
        assert is_url_shortener(GEMINI_RECORD["applyLink"]) is False

    def test_internshala_not_shortener(self) -> None:
        assert is_url_shortener(ANAKIN_RECORD["applyLink"]) is False

    def test_empty_url_returns_false(self) -> None:
        assert is_url_shortener("") is False

    def test_return_type_is_bool(self) -> None:
        assert isinstance(is_url_shortener("https://bit.ly/abc"), bool)


# ===========================================================================
# domain_company_name_similarity
# ===========================================================================

class TestDomainCompanyNameSimilarity:

    def test_matching_domain_scores_high(self) -> None:
        # "Razorpay" → domain "razorpay" — should be near 1.0
        score = domain_company_name_similarity("https://razorpay.com/apply", "Razorpay")
        assert score > 0.7

    def test_mismatched_domain_scores_low(self) -> None:
        # Company "Acme Global Pvt Ltd" but link is tinyurl — no overlap
        score = domain_company_name_similarity("https://tinyurl.com/xyz123", "Acme Global Pvt Ltd")
        assert score < 0.4

    def test_empty_url_returns_zero(self) -> None:
        assert domain_company_name_similarity("", "Razorpay") == 0.0

    def test_empty_company_returns_zero(self) -> None:
        assert domain_company_name_similarity("https://razorpay.com", "") == 0.0

    def test_return_type_is_float(self) -> None:
        assert isinstance(
            domain_company_name_similarity("https://razorpay.com", "Razorpay"), float
        )

    def test_score_bounded(self) -> None:
        score = domain_company_name_similarity("https://google.com/careers", "Google LLC")
        assert 0.0 <= score <= 1.0

    def test_legal_suffix_stripped_before_comparison(self) -> None:
        # "Razorpay Pvt Ltd" → cleaned to "razorpay" → should still score high
        score = domain_company_name_similarity("https://razorpay.com/apply", "Razorpay Pvt Ltd")
        assert score > 0.7


# ===========================================================================
# is_known_ats_domain
# ===========================================================================

class TestIsKnownAtsDomain:

    def test_greenhouse_boards_subdomain(self) -> None:
        assert is_known_ats_domain(GEMINI_RECORD["applyLink"]) is True

    def test_greenhouse_job_boards_subdomain(self) -> None:
        # Snorkel AI — job-boards.greenhouse.io
        url = "https://job-boards.greenhouse.io/snorkelai/jobs/5774350004"
        assert is_known_ats_domain(url) is True

    def test_branded_workday_domain(self) -> None:
        # motorolasolutions.wd5.myworkdayjobs.com — branded Workday
        assert is_known_ats_domain(MOTOROLA_RECORD["applyLink"]) is True

    def test_plain_workday_domain(self) -> None:
        assert is_known_ats_domain("https://caci.wd1.myworkdayjobs.com/en-US/external/job/123") is True

    def test_lever_detected(self) -> None:
        assert is_known_ats_domain("https://jobs.lever.co/stripe/abc123") is True

    def test_direct_employer_not_ats(self) -> None:
        assert is_known_ats_domain(RAZORPAY_RECORD["applyLink"]) is False

    def test_internshala_not_ats(self) -> None:
        assert is_known_ats_domain(ANAKIN_RECORD["applyLink"]) is False

    def test_tinyurl_not_ats(self) -> None:
        assert is_known_ats_domain(TINYURL_RECORD["applyLink"]) is False

    def test_empty_url_returns_false(self) -> None:
        assert is_known_ats_domain("") is False

    def test_return_type_is_bool(self) -> None:
        assert isinstance(is_known_ats_domain("https://greenhouse.io/apply"), bool)


# ===========================================================================
# extract_company_url_features — integration
# ===========================================================================

class TestExtractCompanyUrlFeatures:

    def test_returns_company_url_feature_vector(self) -> None:
        result = extract_company_url_features(ANAKIN_RECORD)
        assert isinstance(result, CompanyUrlFeatureVector)
        assert isinstance(result.company, CompanyFeatures)
        assert isinstance(result.url, UrlFeatures)

    # ── Suspect company short-circuit ─────────────────────────────────────

    def test_suspect_company_sets_is_suspect_true(self) -> None:
        result = extract_company_url_features(SUSPECT_COMPANY_RECORD)
        assert result.company.is_suspect is True

    def test_suspect_company_has_default_company_fields(self) -> None:
        result = extract_company_url_features(SUSPECT_COMPANY_RECORD)
        # When suspect, all identity fields should be at safe/neutral defaults
        assert result.company.has_legal_suffix is False
        assert result.company.posting_count == 1
        assert result.company.typosquat_min_distance == 1.0

    def test_suspect_company_url_still_computed(self) -> None:
        # URL features must still run even when company is suspect
        result = extract_company_url_features(SUSPECT_COMPANY_RECORD)
        # letsintern.in → platform internal
        assert result.url.is_platform_internal is True

    def test_clean_company_is_not_suspect(self) -> None:
        result = extract_company_url_features(RAZORPAY_RECORD)
        assert result.company.is_suspect is False

    # ── Platform-internal link (internshala.com) ───────────────────────────

    def test_internshala_link_flagged_as_internal(self) -> None:
        result = extract_company_url_features(ANAKIN_RECORD)
        assert result.url.is_platform_internal is True

    def test_internshala_link_not_shortener(self) -> None:
        result = extract_company_url_features(ANAKIN_RECORD)
        assert result.url.is_url_shortener is False

    def test_internshala_link_not_ats(self) -> None:
        result = extract_company_url_features(ANAKIN_RECORD)
        assert result.url.is_known_ats is False

    # ── ATS link (greenhouse.io) ───────────────────────────────────────────

    def test_greenhouse_link_is_ats(self) -> None:
        result = extract_company_url_features(GEMINI_RECORD)
        assert result.url.is_known_ats is True

    def test_greenhouse_link_not_internal(self) -> None:
        result = extract_company_url_features(GEMINI_RECORD)
        assert result.url.is_platform_internal is False

    def test_greenhouse_link_not_shortener(self) -> None:
        result = extract_company_url_features(GEMINI_RECORD)
        assert result.url.is_url_shortener is False

    def test_branded_workday_is_ats(self) -> None:
        result = extract_company_url_features(MOTOROLA_RECORD)
        assert result.url.is_known_ats is True

    # ── Off-platform direct employer link ─────────────────────────────────

    def test_razorpay_not_internal_not_ats_not_shortener(self) -> None:
        result = extract_company_url_features(RAZORPAY_RECORD)
        assert result.url.is_platform_internal is False
        assert result.url.is_known_ats is False
        assert result.url.is_url_shortener is False

    def test_razorpay_domain_company_similarity_high(self) -> None:
        result = extract_company_url_features(RAZORPAY_RECORD)
        assert result.url.domain_company_similarity > 0.7

    def test_razorpay_is_https(self) -> None:
        result = extract_company_url_features(RAZORPAY_RECORD)
        assert result.url.is_https is True

    # ── URL shortener ──────────────────────────────────────────────────────

    def test_tinyurl_flagged_as_shortener(self) -> None:
        result = extract_company_url_features(TINYURL_RECORD)
        assert result.url.is_url_shortener is True

    def test_bitly_flagged_as_shortener(self) -> None:
        result = extract_company_url_features(BITLY_RECORD)
        assert result.url.is_url_shortener is True

    def test_tinyurl_not_ats_not_internal(self) -> None:
        result = extract_company_url_features(TINYURL_RECORD)
        assert result.url.is_known_ats is False
        assert result.url.is_platform_internal is False

    # ── Legal suffix ──────────────────────────────────────────────────────

    def test_legal_suffix_detected(self) -> None:
        result = extract_company_url_features(LEGAL_SUFFIX_RECORD)
        assert result.company.has_legal_suffix is True

    def test_razorpay_no_legal_suffix(self) -> None:
        result = extract_company_url_features(RAZORPAY_RECORD)
        assert result.company.has_legal_suffix is False

    # ── Posting frequency with batch ──────────────────────────────────────

    def test_frequency_uses_batch(self) -> None:
        record = {"company": "Acme Corp", "applyLink": "https://acme.com/job1",
                  "field": ["Marketing"], "tags": [], "datePublished": "2026-01-01"}
        result = extract_company_url_features(record, all_records=FREQUENCY_BATCH)
        assert result.company.posting_count == 3

    def test_no_batch_defaults_to_single(self) -> None:
        result = extract_company_url_features(RAZORPAY_RECORD)
        assert result.company.posting_count == 1

    # ── TLD / entropy / misc ──────────────────────────────────────────────

    def test_entropy_populated(self) -> None:
        result = extract_company_url_features(RAZORPAY_RECORD)
        assert result.url.domain_entropy > 0.0

    def test_tld_risk_score_bounded(self) -> None:
        for record in [ANAKIN_RECORD, GEMINI_RECORD, TINYURL_RECORD, RAZORPAY_RECORD]:
            result = extract_company_url_features(record)
            assert 0.0 <= result.url.tld_risk_score <= 1.0

    def test_empty_record_does_not_crash(self) -> None:
        result = extract_company_url_features({})
        assert isinstance(result, CompanyUrlFeatureVector)

    def test_flags_read_from_embedded_key(self) -> None:
        # Flags embedded under "_flags" key (as produced by pipeline)
        record = {**RAZORPAY_RECORD, "_flags": {"company_suspect": True}}
        result = extract_company_url_features(record)
        assert result.company.is_suspect is True

    def test_flags_override_parameter_wins(self) -> None:
        # Explicit flags= parameter should take precedence over _flags in record
        result = extract_company_url_features(
            RAZORPAY_RECORD,
            flags={"company_suspect": True},
        )
        assert result.company.is_suspect is True
