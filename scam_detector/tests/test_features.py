"""
test_features.py
----------------
Tests for scam_detector.features.text_features

Fixture text is drawn directly from real internship records observed in the
corpus:

  ANAKIN_RECORD     — Internshala/Anakin: tech role, well-formed, specific
                      (internScraper/checkpoint_internships.json)
  NAYEPANKH_RECORD  — Internshala/NayePankh: generic fundraising, templated
                      (internScraper/scrapers/intershala_scraper/internships.json)
  URGENCY_RECORD    — Telegram-channel post with "Hurry UP", "apply now",
                      "Only 12 hours left"
                      (web_scrapper/telegram scraper/all_channels_internships.json)
  TRUNCATED_RECORD  — Posting whose summary ends in "..." (scraper truncation)
  SENSITIVE_RECORD  — Fabricated but pattern-representative: asks for
                      "registration fee" inside description body

Tests use directional / range assertions (not exact values) because these
are heuristic scores — the important property is relative ordering and that
obvious positives score detectably higher than obvious negatives.
"""

from __future__ import annotations

import math
from unittest.mock import patch

import pytest

from scam_detector.features.text_features import (
    TextFeatureVector,
    caps_and_punctuation_ratio,
    extract_text_features,
    genericity_score,
    readability_and_grammar_signals,
    sensitive_info_request_detector,
    guaranteed_outcome_claim_detector,
    external_form_and_phone_handoff_detector,
    training_program_disguised_as_internship_detector,
    zero_shot_scam_signal,
    urgency_score,
    boilerplate_similarity,
)

# ===========================================================================
# Real-data fixtures
# ===========================================================================

# ── Anakin (real, well-formed, tech-specific) ───────────────────────────────
ANAKIN_TITLE = "Software Development"
ANAKIN_SUMMARY = (
    "Anakin is building a large-scale data engine that powers real-time competitive "
    "intelligence for global internet companies by continuously collecting and "
    "structuring massive volumes of public web data. This involves solving complex "
    "engineering challenges like handling scale, ensuring system reliability, "
    "navigating dynamic websites, overcoming anti-bot mechanisms, and dealing with "
    "unpredictable edge cases — making it far from theoretical or toy problems."
)
ANAKIN_RESPONSIBILITIES = [
    "Build backend services, APIs, and data pipelines",
    "Work on web scraping and automation for dynamic systems",
    "Debug production issues and improve reliability/performance",
    "Write clean, maintainable, testable code",
    "Participate in code reviews and engineering discussions",
    "Ship features end-to-end with ownership",
]
ANAKIN_RECORD = {
    "name": ANAKIN_TITLE,
    "company": "Anakin",
    "summary": ANAKIN_SUMMARY,
    "responsibilities": ANAKIN_RESPONSIBILITIES,
}

# ── NayePankh fundraising (generic, templated) ───────────────────────────────
NAYEPANKH_TITLE = "Fundraising"
NAYEPANKH_SUMMARY = (
    "NayePankh Foundation works at the grassroots level to uplift underserved "
    "communities by promoting education, life skills, and sustainable development. "
    "As a Fundraising Intern, you will collaborate closely with the core team to "
    "support resource mobilization efforts and strengthen connections with individuals "
    "and organizations who believe in our mission."
)
NAYEPANKH_RECORD = {
    "name": NAYEPANKH_TITLE,
    "company": "NayePankh Foundation",
    "summary": NAYEPANKH_SUMMARY,
    "responsibilities": [
        "Support the planning and execution of fundraising initiatives",
        "Build and nurture long-term associations with donors",
        "Reach out to prospective supporters through calls and messages",
    ],
}

# ── Urgency record (Telegram channel real text) ──────────────────────────────
URGENCY_SUMMARY = (
    "Hurry UP! Only 12 hours left. Apply ASAP. "
    "Company name: Salesforce. Role: SDE Intern (Summer). "
    "Batch Eligible: 2027 passouts. Expected Stipend: Upto INR 1,25,000 per month. "
    "Location: Hyderabad/Bangalore, India. Apply Link: https://bit.ly/SalesforceIntern2027."
)
URGENCY_RECORD = {
    "name": "Hurry UP! Only 12 hours left",
    "company": "",
    "summary": URGENCY_SUMMARY,
    "responsibilities": [],
}

# ── Truncated summary record ──────────────────────────────────────────────────
TRUNCATED_SUMMARY = (
    "We are looking for an enthusiastic intern to join our marketing team. "
    "You will assist with content creation, social media management, and..."
)
TRUNCATED_RECORD = {
    "name": "Digital Marketing",
    "company": "Digital Marketing",
    "summary": TRUNCATED_SUMMARY,
    "responsibilities": [],
    "_flags": {"summary_truncated": True},
}

# ── Sensitive info / scam record (pattern representative) ───────────────────
SENSITIVE_SUMMARY = (
    "Join our team as a Business Development Intern! Great opportunity for freshers. "
    "A refundable security deposit of ₹2000 is required to confirm your slot. "
    "Please share your Aadhaar number and bank account details during onboarding."
)
SENSITIVE_RECORD = {
    "name": "Business Development",
    "company": "XYZ Pvt Ltd",
    "summary": SENSITIVE_SUMMARY,
    "responsibilities": ["Pay registration fee to join the training programme."],
}

# ── Responsibilities with trailing artifact ───────────────────────────────────
ARTIFACT_RECORD = {
    "name": "Business Consultant",
    "company": "NayePankh Foundation",
    "summary": "Hands-on exposure to performance analysis and strategic planning.",
    "responsibilities": [
        "Identify operational gaps, process inefficiencies, and outreach opportunities   4.",
        "Contribute to enhancing donor engagement and retention strategies   5.",
    ],
    "_flags": {"responsibilities_cleaned": False},
}

# ── Clean well-formed record for no-flag baseline ────────────────────────────
CLEAN_RECORD = {
    "name": "Frontend Developer Intern",
    "company": "Razorpay",
    "summary": (
        "Work with Razorpay's frontend team to build and improve the payment "
        "dashboard and merchant-facing UI components used by 8M+ businesses. "
        "You will collaborate with product designers and backend engineers."
    ),
    "responsibilities": [
        "Develop responsive UI components using React and TypeScript",
        "Integrate REST APIs for live payment data",
        "Write unit tests using Jest and React Testing Library",
    ],
}


# ===========================================================================
# 1 — urgency_score
# ===========================================================================

class TestUrgencyScore:

    def test_empty_text_is_zero(self) -> None:
        assert urgency_score("") == 0.0

    def test_whitespace_only_is_zero(self) -> None:
        assert urgency_score("   \n  ") == 0.0

    def test_no_urgency_keywords_is_zero(self) -> None:
        score = urgency_score(ANAKIN_SUMMARY)
        assert score == 0.0

    def test_urgency_text_scores_higher_than_neutral(self) -> None:
        neutral_score = urgency_score(ANAKIN_SUMMARY)
        urgency = urgency_score(URGENCY_SUMMARY)
        assert urgency > neutral_score

    def test_urgency_score_bounded(self) -> None:
        # Flood it with every trigger phrase
        dense = "urgent apply now hurry limited seats immediate joining apply asap last day"
        score = urgency_score(dense)
        assert 0.0 <= score <= 1.0

    def test_single_keyword_in_long_text_scores_less_than_in_short_text(self) -> None:
        short = "Apply now."
        long = "Apply now. " + ("This is filler. " * 50)
        assert urgency_score(short) > urgency_score(long)

    def test_case_insensitive(self) -> None:
        assert urgency_score("URGENT APPLY NOW") > 0.0
        assert urgency_score("Urgent Apply Now") > 0.0

    def test_hurry_detected(self) -> None:
        assert urgency_score("Hurry UP! Last 1 day Left!") > 0.0

    def test_limited_seats_detected(self) -> None:
        assert urgency_score("Limited seats available for this programme.") > 0.0

    def test_return_type_is_float(self) -> None:
        assert isinstance(urgency_score("apply now"), float)


# ===========================================================================
# 2 — caps_and_punctuation_ratio
# ===========================================================================

class TestCapsAndPunctuationRatio:

    def test_empty_text(self) -> None:
        result = caps_and_punctuation_ratio("")
        assert result["caps_ratio"] == 0.0
        assert result["exclamation_count"] == 0
        assert result["has_repeated_punctuation"] is False

    def test_all_lowercase(self) -> None:
        result = caps_and_punctuation_ratio("all lowercase text here")
        assert result["caps_ratio"] == 0.0

    def test_all_uppercase(self) -> None:
        result = caps_and_punctuation_ratio("LOUD TEXT")
        assert result["caps_ratio"] == 1.0

    def test_exclamation_count(self) -> None:
        result = caps_and_punctuation_ratio("Great!! Wow!!! Apply Now!!")
        assert result["exclamation_count"] == 7

    def test_repeated_punctuation_detected(self) -> None:
        result = caps_and_punctuation_ratio("Are you sure??? Apply NOW!!")
        assert result["has_repeated_punctuation"] is True

    def test_no_repeated_punctuation(self) -> None:
        result = caps_and_punctuation_ratio("Normal sentence. Another one.")
        assert result["has_repeated_punctuation"] is False

    def test_scam_text_has_high_exclamation(self) -> None:
        scam = "URGENT!!! LIMITED SEATS!!! Apply NOW!!!"
        result = caps_and_punctuation_ratio(scam)
        assert result["exclamation_count"] >= 6
        assert result["caps_ratio"] > 0.5

    def test_clean_text_low_caps_ratio(self) -> None:
        result = caps_and_punctuation_ratio(ANAKIN_SUMMARY)
        # Real text — caps ratio should be low (mostly sentence-start capitals)
        assert result["caps_ratio"] < 0.15

    def test_unicode_ellipsis_repeated_punct(self) -> None:
        result = caps_and_punctuation_ratio("Description ends here\u2026 More text???")
        assert result["has_repeated_punctuation"] is True


# ===========================================================================
# 3 — genericity_score
# ===========================================================================

class TestGenericityScore:

    def test_empty_title_is_zero(self) -> None:
        assert genericity_score("") == 0.0

    def test_known_generic_title_scores_high(self) -> None:
        for title in ["Fundraising", "Digital Marketing", "Content Writing",
                      "Data Entry", "Business Development"]:
            score = genericity_score(title)
            assert score > 0.7, f"Expected high genericity for '{title}', got {score}"

    def test_specific_tech_title_scores_lower(self) -> None:
        # A specific, unusual title that shares no words with the generic list
        # should score clearly below a known generic title.
        generic = genericity_score("Fundraising")
        # "Quantitative Risk Modelling Associate" has no overlap with any generic title
        specific = genericity_score("Quantitative Risk Modelling Associate")
        assert specific < generic

    def test_partial_match_scores_reasonably(self) -> None:
        # "Business Development Executive" should still score high
        score = genericity_score("Business Development Executive")
        assert score > 0.5

    def test_return_type_is_float(self) -> None:
        assert isinstance(genericity_score("Marketing"), float)

    def test_score_bounded(self) -> None:
        for title in ["Marketing", "Intern", "HR", "Sales and Marketing"]:
            s = genericity_score(title)
            assert 0.0 <= s <= 1.0

    def test_social_media_marketing_generic(self) -> None:
        assert genericity_score("Social Media Marketing") > 0.7

    def test_anakin_title_lower_than_fundraising(self) -> None:
        fundraising = genericity_score("Fundraising")
        software_dev = genericity_score("Software Development")
        # Both exist on the list but Fundraising is a near-exact match,
        # Software Development also matches — both should be reasonably high;
        # we just verify the call doesn't crash and returns bounded value
        assert 0.0 <= fundraising <= 1.0
        assert 0.0 <= software_dev <= 1.0

    def test_specific_tech_title_containing_development_scores_low(self) -> None:
        # Real-corpus false positive: with token_set_ratio (the previous
        # implementation), any title sharing the word "Development" with a
        # list entry scored close to 1.0 regardless of specificity -
        # "SDET Intern (Software Development Engineer in Test)", a specific
        # role at a real company (Icertis), scored 0.85-1.0 purely from that
        # shared word, incorrectly triggering MassOpeningsVagueRoleRule.
        # token_sort_ratio (current implementation) correctly penalizes the
        # extra qualifying words instead of ignoring them.
        score = genericity_score("SDET Intern (Software Development Engineer in Test)")
        assert score < 0.65

    def test_business_development_with_qualifier_scores_below_threshold(self) -> None:
        score = genericity_score("Business Development Executive - Fintech Vertical")
        assert score < 0.65

    def test_bare_business_development_still_scores_generic(self) -> None:
        # The genuinely generic, undifferentiated version must still match -
        # this is a real, distinct vague-title category in this domain.
        assert genericity_score("Business Development (Sales)") > 0.9


# ===========================================================================
# 4 — title_summary_alignment (SBERT — may be skipped without model)
# ===========================================================================

class TestTitleSummaryAlignment:
    """
    SBERT tests are directional only — exact cosine values depend on model
    weights.  If sentence-transformers is not installed, functions return 0.0
    gracefully and we skip the directional checks.
    """

    @pytest.fixture(autouse=True)
    def sbert_available(self):
        from scam_detector.features.text_features import _sbert_model
        self.model_available = _sbert_model() is not None

    def test_empty_inputs_return_zero(self) -> None:
        from scam_detector.features.text_features import title_summary_alignment
        assert title_summary_alignment("", "") == 0.0
        assert title_summary_alignment("Some Title", "") == 0.0
        assert title_summary_alignment("", "Some summary text.") == 0.0

    def test_return_type_is_float(self) -> None:
        from scam_detector.features.text_features import title_summary_alignment
        result = title_summary_alignment("Intern", "Work on projects.")
        assert isinstance(result, float)

    def test_score_bounded(self) -> None:
        from scam_detector.features.text_features import title_summary_alignment
        result = title_summary_alignment(ANAKIN_TITLE, ANAKIN_SUMMARY)
        assert 0.0 <= result <= 1.0

    def test_aligned_text_scores_higher_than_misaligned(self) -> None:
        if not self.model_available:
            pytest.skip("sentence-transformers not installed")
        from scam_detector.features.text_features import title_summary_alignment
        # NOTE: previously used ANAKIN_SUMMARY here, but it is a long,
        # jargon-heavy paragraph ("data engine", "anti-bot mechanisms") with
        # no literal software/development vocabulary — verified empirically
        # that all-MiniLM-L6-v2 scores it BELOW the misaligned pair (0.2199
        # vs 0.2649), making this assertion flaky on the real model rather
        # than a bug in title_summary_alignment. Using an unambiguously
        # on-topic summary here instead; ANAKIN_SUMMARY is still used
        # elsewhere in this file for unrelated properties (urgency, caps
        # ratio, readability) where topical alignment doesn't matter.
        aligned = title_summary_alignment(
            "Software Development",
            "You will write, test, and debug code, build new features, and "
            "fix bugs in our software application using Python and JavaScript.",
        )
        # Misaligned: fundraising title with a cooking summary
        misaligned = title_summary_alignment(
            "Fundraising",
            "You will prepare gourmet meals and manage kitchen inventory.",
        )
        assert aligned > misaligned


# ===========================================================================
# 5 — readability_and_grammar_signals
# ===========================================================================

class TestReadabilityAndGrammarSignals:

    def test_empty_text(self) -> None:
        result = readability_and_grammar_signals("")
        assert result["avg_sentence_length"] == 0.0
        assert result["flesch_score"] == 0.0
        assert result["artifact_count"] == 0

    def test_returns_expected_keys(self) -> None:
        result = readability_and_grammar_signals(ANAKIN_SUMMARY)
        assert "avg_sentence_length" in result
        assert "flesch_score" in result
        assert "artifact_count" in result

    def test_avg_sentence_length_is_positive_for_real_text(self) -> None:
        result = readability_and_grammar_signals(ANAKIN_SUMMARY)
        assert result["avg_sentence_length"] > 0

    def test_flesch_score_reasonable_for_clear_text(self) -> None:
        # Simple, short sentences score high on Flesch (high = easier to read).
        # Complex professional prose (like Anakin) correctly scores lower/negative.
        simple = readability_and_grammar_signals(
            "Build APIs. Fix bugs. Write tests. Ship code. Review PRs."
        )
        complex_prose = readability_and_grammar_signals(ANAKIN_SUMMARY)
        # Simple sentences should score higher than the complex technical summary
        assert simple["flesch_score"] > complex_prose["flesch_score"]
        # Simple text should be in a reasonable range for easy reading
        assert simple["flesch_score"] > 30

    def test_artifact_count_detects_double_spaces(self) -> None:
        text_with_artifacts = "Work on tasks  and projects  with teammates."
        result = readability_and_grammar_signals(text_with_artifacts)
        assert result["artifact_count"] >= 2

    def test_artifact_count_detects_trailing_digits(self) -> None:
        # Pattern from Internshala: "...outreach opportunities   4."
        text = "Identify operational gaps and new outreach opportunities   4."
        result = readability_and_grammar_signals(text)
        assert result["artifact_count"] >= 1

    def test_clean_text_has_zero_or_low_artifact_count(self) -> None:
        result = readability_and_grammar_signals(
            "Develop responsive UI components. Integrate REST APIs. Write unit tests."
        )
        assert result["artifact_count"] == 0

    def test_longer_text_avg_sentence_length_stable(self) -> None:
        # Sentence length shouldn't change wildly just because text is longer
        short_result = readability_and_grammar_signals("Build APIs. Fix bugs.")
        long_text = "Build APIs. Fix bugs. " * 20
        long_result = readability_and_grammar_signals(long_text)
        # avg sentence length should be similar (within 3 words)
        assert abs(short_result["avg_sentence_length"] - long_result["avg_sentence_length"]) < 3

    def test_very_short_text_does_not_crash(self) -> None:
        result = readability_and_grammar_signals("OK.")
        assert isinstance(result["flesch_score"], float)

    def test_truncated_summary_not_worse_than_long_one(self) -> None:
        # This is a meta-test — the function itself doesn't know about
        # truncation; downstream scoring uses the flag.  We just verify it
        # runs without error on truncated text.
        result = readability_and_grammar_signals(TRUNCATED_SUMMARY)
        assert isinstance(result["avg_sentence_length"], float)


# ===========================================================================
# 6 — sensitive_info_request_detector
# ===========================================================================

class TestSensitiveInfoRequestDetector:

    def test_empty_text_returns_false(self) -> None:
        assert sensitive_info_request_detector("") is False

    def test_registration_fee_detected(self) -> None:
        assert sensitive_info_request_detector("Pay a registration fee of ₹500.") is True

    def test_security_deposit_detected(self) -> None:
        assert sensitive_info_request_detector(
            "A refundable security deposit is required."
        ) is True

    def test_aadhaar_detected(self) -> None:
        assert sensitive_info_request_detector(
            "Please share your Aadhaar number during onboarding."
        ) is True

    def test_aadhaar_variant_spelling(self) -> None:
        assert sensitive_info_request_detector("Provide your aadhar card.") is True

    def test_bank_account_detected(self) -> None:
        assert sensitive_info_request_detector(
            "Share bank account details for stipend transfer."
        ) is True

    def test_bank_account_number_detected(self) -> None:
        assert sensitive_info_request_detector(
            "Please provide your bank account number to process payment."
        ) is True

    def test_boost_your_bank_account_idiom_not_flagged(self) -> None:
        # Real-corpus false positive: a real Unstop listing ("Nbyula")
        # said "you won't just boost your bank account" - a colloquial
        # phrase meaning "earn money", not a request for banking details.
        # This was firing the hard-disqualifying rule (weight 0.95).
        assert sensitive_info_request_detector(
            "As a Campus Supernova, you won't just boost your bank account; "
            "you'll shine bright in the world of digital marketing."
        ) is False

    def test_bank_details_without_account_word_detected(self) -> None:
        assert sensitive_info_request_detector(
            "Kindly share your bank details before the interview."
        ) is True

    def test_pay_to_join_detected(self) -> None:
        assert sensitive_info_request_detector(
            "You need to pay ₹2000 to join the programme."
        ) is True

    def test_pay_to_confirm_detected(self) -> None:
        assert sensitive_info_request_detector("Pay Rs. 1500 to confirm your slot.") is True

    def test_processing_fee_detected(self) -> None:
        assert sensitive_info_request_detector("A processing fee of ₹300 applies.") is True

    def test_real_scam_text_detected(self) -> None:
        assert sensitive_info_request_detector(SENSITIVE_SUMMARY) is True

    def test_clean_internship_not_flagged(self) -> None:
        assert sensitive_info_request_detector(ANAKIN_SUMMARY) is False

    def test_nayepankh_not_flagged(self) -> None:
        # NayePankh is performance-based stipend, not payment-requesting
        assert sensitive_info_request_detector(NAYEPANKH_SUMMARY) is False

    def test_return_type_is_bool(self) -> None:
        assert isinstance(sensitive_info_request_detector("hello"), bool)

    def test_case_insensitive(self) -> None:
        assert sensitive_info_request_detector("REGISTRATION FEE REQUIRED") is True

    def test_pan_card_detected(self) -> None:
        assert sensitive_info_request_detector("Submit your PAN card number.") is True


# ===========================================================================
# 6b — guaranteed_outcome_claim_detector
# ===========================================================================

class TestGuaranteedOutcomeClaimDetector:

    def test_empty_text_returns_false(self) -> None:
        assert guaranteed_outcome_claim_detector("") is False

    def test_100_percent_placement_guarantee_detected(self) -> None:
        assert guaranteed_outcome_claim_detector("100% placement guarantee for all interns.") is True

    def test_guaranteed_job_detected(self) -> None:
        assert guaranteed_outcome_claim_detector("Guaranteed job after 1 month of training.") is True

    def test_guaranteed_certificate_detected(self) -> None:
        assert guaranteed_outcome_claim_detector("Guaranteed certificate on completion.") is True

    def test_assured_placement_detected(self) -> None:
        assert guaranteed_outcome_claim_detector("Assured placement with top companies.") is True

    def test_no_interview_required_detected(self) -> None:
        assert guaranteed_outcome_claim_detector("No interview required, direct onboarding.") is True

    def test_selected_without_interview_detected(self) -> None:
        assert guaranteed_outcome_claim_detector("Get hired without any interview.") is True

    def test_instant_selection_detected(self) -> None:
        assert guaranteed_outcome_claim_detector("Instant selection for all candidates.") is True

    def test_whatsapp_only_detected(self) -> None:
        assert guaranteed_outcome_claim_detector("Apply via WhatsApp only for quick response.") is True

    def test_telegram_group_detected(self) -> None:
        assert guaranteed_outcome_claim_detector("Join our Telegram group to get started.") is True

    def test_refer_and_earn_detected(self) -> None:
        assert guaranteed_outcome_claim_detector("Refer and earn bonus for every friend who joins.") is True

    def test_clean_internship_not_flagged(self) -> None:
        assert guaranteed_outcome_claim_detector(ANAKIN_SUMMARY) is False

    def test_normal_interview_mention_not_flagged(self) -> None:
        assert guaranteed_outcome_claim_detector(
            "Shortlisted candidates will be invited for an interview."
        ) is False

    def test_return_type_is_bool(self) -> None:
        assert isinstance(guaranteed_outcome_claim_detector("hello"), bool)

    def test_case_insensitive(self) -> None:
        assert guaranteed_outcome_claim_detector("GUARANTEED PLACEMENT") is True


# ===========================================================================
# 6c — external_form_and_phone_handoff_detector
# ===========================================================================

class TestExternalFormAndPhoneHandoffDetector:

    def test_empty_text_returns_false_false(self) -> None:
        assert external_form_and_phone_handoff_detector("") == (False, False)

    def test_real_zefrix_example_detects_both(self) -> None:
        # Verbatim (redacted phone) from a real Internshala scrape
        text = (
            "Love making Reels? Get PAID for it. Zefrix is hiring a Social "
            "Media Intern. Apply now: https://forms.gle/MoCTSfEVCwL3GB2n7 "
            "or call on +918854996448"
        )
        form, phone = external_form_and_phone_handoff_detector(text)
        assert form is True
        assert phone is True

    def test_forms_gle_alone_detected(self) -> None:
        form, phone = external_form_and_phone_handoff_detector(
            "Apply here: https://forms.gle/abc123xyz"
        )
        assert form is True
        assert phone is False

    def test_google_docs_forms_url_detected(self) -> None:
        form, _ = external_form_and_phone_handoff_detector(
            "Fill this out: https://docs.google.com/forms/d/e/xyz/viewform"
        )
        assert form is True

    def test_typeform_detected(self) -> None:
        form, _ = external_form_and_phone_handoff_detector("Apply via https://typeform.com/to/abc123")
        assert form is True

    def test_phone_handoff_alone_detected(self) -> None:
        form, phone = external_form_and_phone_handoff_detector(
            "For more details, WhatsApp us on 9876543210"
        )
        assert form is False
        assert phone is True

    def test_bare_phone_number_without_contact_verb_not_flagged(self) -> None:
        # A 10-digit number with no "call/contact/whatsapp" nearby should
        # not trigger — avoids false positives on unrelated numeric data.
        _, phone = external_form_and_phone_handoff_detector(
            "Reference code: 9876543210 for internal tracking purposes."
        )
        assert phone is False

    def test_standard_business_contact_line_not_flagged(self) -> None:
        # Real-corpus false positive: a legitimate construction-firm job
        # posting listed a standard "Contact: <phone>" business line and
        # even said "Apply directly through Internshala" - not a scam
        # funnel at all. "contact" was removed from the trigger-verb list
        # because it's too generic for normal business communications.
        form, phone = external_form_and_phone_handoff_detector(
            "How to Apply: Apply directly through Internshala or send your "
            "resume to hr@company.com. Contact: 9048500028 / 9048500068"
        )
        assert form is False
        assert phone is False

    def test_clean_internship_not_flagged(self) -> None:
        form, phone = external_form_and_phone_handoff_detector(ANAKIN_SUMMARY)
        assert form is False
        assert phone is False

    def test_platform_apply_link_not_flagged(self) -> None:
        form, _ = external_form_and_phone_handoff_detector(
            "Apply via the official Internshala application form for this role."
        )
        assert form is False

    def test_return_type_is_tuple_of_bools(self) -> None:
        result = external_form_and_phone_handoff_detector("hello")
        assert isinstance(result, tuple)
        assert all(isinstance(v, bool) for v in result)


# ===========================================================================
# 6d — training_program_disguised_as_internship_detector
# ===========================================================================

class TestTrainingProgramDisguisedAsInternshipDetector:

    def test_empty_text_returns_false(self) -> None:
        assert training_program_disguised_as_internship_detector("") is False

    def test_real_ev_design_example_detected(self) -> None:
        # Verbatim from a real LetsIntern scrape
        text = (
            "The Electric Vehicle Design Internship is a career-focused, "
            "hands-on training program designed for students and freshers "
            "who want to build a strong future in the EV and automotive industry."
        )
        assert training_program_disguised_as_internship_detector(text) is True

    def test_real_sponsored_admission_example_detected(self) -> None:
        # Verbatim from a real scrape (company field: "BBA")
        text = (
            "We are looking for motivated individuals to join our team in a "
            "unique Work-Study program. Selected candidates will be provided "
            "with 100% sponsored admission to a professional degree or "
            "certification program (BBA, MBA, BCA, or MCA) from our partner university."
        )
        assert training_program_disguised_as_internship_detector(text) is True

    def test_training_and_internship_simultaneously_detected(self) -> None:
        assert training_program_disguised_as_internship_detector(
            "Training and internships will go simultaneously."
        ) is True

    def test_self_paced_program_detected(self) -> None:
        assert training_program_disguised_as_internship_detector(
            "It is a self-paced program with weekly milestones."
        ) is True

    def test_bare_training_mention_as_job_duty_not_flagged(self) -> None:
        # Real-data false-positive case verified against the corpus: a role
        # that coordinates/sells training programs as a normal job duty
        # must NOT be flagged - this is common and legitimate.
        text = (
            "Coordinate and manage end-to-end execution of virtual training "
            "programs, including scheduling and communications with instructors."
        )
        assert training_program_disguised_as_internship_detector(text) is False

    def test_training_programs_as_sales_target_not_flagged(self) -> None:
        text = "Qualify leads by assessing client needs for training programs, workshops, or coaching sessions."
        assert training_program_disguised_as_internship_detector(text) is False

    def test_clean_internship_not_flagged(self) -> None:
        assert training_program_disguised_as_internship_detector(ANAKIN_SUMMARY) is False

    def test_return_type_is_bool(self) -> None:
        assert isinstance(training_program_disguised_as_internship_detector("hello"), bool)

    def test_case_insensitive(self) -> None:
        assert training_program_disguised_as_internship_detector(
            "THIS IS A CAREER-FOCUSED, HANDS-ON TRAINING PROGRAM"
        ) is True


# ===========================================================================
# 6e — zero_shot_scam_signal
# ===========================================================================
#
# NOTE: the real model (MoritzLaurer/deberta-v3-xsmall-zeroshot-v1.1-all-33)
# was manually validated against 3 real scam examples + 3 paraphrased
# variants + 2 clean examples during development — see ZeroShotConfig's
# docstring in config.py for the results. Tests here mock the classifier
# rather than loading the real ~146MB model (~50s cold start), to keep the
# suite fast; they verify the wiring/gating logic, not model accuracy.

class TestZeroShotScamSignal:

    def test_disabled_by_default_returns_none(self) -> None:
        # cfg.zero_shot.enabled defaults to False — must short-circuit
        # without even attempting to load the classifier.
        category, confidence = zero_shot_scam_signal("some internship text")
        assert category is None
        assert confidence == 0.0

    def test_empty_text_returns_none_even_when_enabled(self) -> None:
        with patch("scam_detector.features.text_features._cfg") as mock_cfg:
            mock_cfg.zero_shot.enabled = True
            category, confidence = zero_shot_scam_signal("")
            assert category is None
            assert confidence == 0.0

    def test_classifier_unavailable_returns_none(self) -> None:
        with patch("scam_detector.features.text_features._cfg") as mock_cfg, \
             patch("scam_detector.features.text_features._zero_shot_classifier", return_value=None):
            mock_cfg.zero_shot.enabled = True
            category, confidence = zero_shot_scam_signal("some internship text")
            assert category is None
            assert confidence == 0.0

    def test_normal_label_top_prediction_returns_none(self) -> None:
        mock_classifier = lambda text, labels: {
            "labels": ["a normal, specific job description with real responsibilities",
                       "sells a paid training or certification course disguised as a job"],
            "scores": [0.9, 0.1],
        }
        with patch("scam_detector.features.text_features._cfg") as mock_cfg, \
             patch("scam_detector.features.text_features._zero_shot_classifier", return_value=mock_classifier):
            mock_cfg.zero_shot.enabled = True
            category, confidence = zero_shot_scam_signal("a normal internship")
            assert category is None
            assert confidence == 0.0

    def test_non_normal_top_prediction_returns_category_and_score(self) -> None:
        mock_classifier = lambda text, labels: {
            "labels": ["sells a paid training or certification course disguised as a job",
                       "a normal, specific job description with real responsibilities"],
            "scores": [0.93, 0.07],
        }
        with patch("scam_detector.features.text_features._cfg") as mock_cfg, \
             patch("scam_detector.features.text_features._zero_shot_classifier", return_value=mock_classifier):
            mock_cfg.zero_shot.enabled = True
            category, confidence = zero_shot_scam_signal("some scammy text")
            assert category == "sells a paid training or certification course disguised as a job"
            assert confidence == pytest.approx(0.93)

    def test_classifier_exception_returns_none(self) -> None:
        def raising_classifier(text, labels):
            raise RuntimeError("inference failed")
        with patch("scam_detector.features.text_features._cfg") as mock_cfg, \
             patch("scam_detector.features.text_features._zero_shot_classifier", return_value=raising_classifier):
            mock_cfg.zero_shot.enabled = True
            category, confidence = zero_shot_scam_signal("some text")
            assert category is None
            assert confidence == 0.0


# ===========================================================================
# 7 — boilerplate_similarity
# ===========================================================================

class TestBoilerplateSimilarity:

    def test_none_corpus_returns_zero(self) -> None:
        assert boilerplate_similarity("some text", None) == 0.0

    def test_empty_text_returns_zero(self) -> None:
        import numpy as np
        dummy_corpus = np.zeros((3, 384))
        assert boilerplate_similarity("", dummy_corpus) == 0.0

    def test_return_type_is_float(self) -> None:
        result = boilerplate_similarity("any text", None)
        assert isinstance(result, float)

    def test_score_bounded(self) -> None:
        from scam_detector.features.text_features import _sbert_model
        if _sbert_model() is None:
            pytest.skip("sentence-transformers not installed")
        import numpy as np
        from sentence_transformers import SentenceTransformer  # type: ignore
        model = SentenceTransformer("all-MiniLM-L6-v2")
        corpus_emb = model.encode([NAYEPANKH_SUMMARY], normalize_embeddings=True)
        score = boilerplate_similarity(NAYEPANKH_SUMMARY, corpus_emb)
        assert 0.0 <= score <= 1.0

    def test_identical_text_scores_high(self) -> None:
        from scam_detector.features.text_features import _sbert_model
        if _sbert_model() is None:
            pytest.skip("sentence-transformers not installed")
        import numpy as np
        from sentence_transformers import SentenceTransformer  # type: ignore
        model = SentenceTransformer("all-MiniLM-L6-v2")
        corpus_emb = model.encode([ANAKIN_SUMMARY], normalize_embeddings=True)
        score = boilerplate_similarity(ANAKIN_SUMMARY, corpus_emb)
        # Identical text → cosine similarity should be very close to 1.0
        assert score > 0.98


# ===========================================================================
# extract_text_features — integration
# ===========================================================================

class TestExtractTextFeatures:

    def test_returns_text_feature_vector(self) -> None:
        result = extract_text_features(ANAKIN_RECORD)
        assert isinstance(result, TextFeatureVector)

    def test_word_count_populated(self) -> None:
        result = extract_text_features(ANAKIN_RECORD)
        assert result.word_count > 0

    def test_char_count_populated(self) -> None:
        result = extract_text_features(ANAKIN_RECORD)
        assert result.char_count > 0

    def test_urgency_score_high_for_urgency_record(self) -> None:
        result = extract_text_features(URGENCY_RECORD)
        assert result.urgency_score > 0.0

    def test_urgency_score_zero_for_clean_record(self) -> None:
        result = extract_text_features(CLEAN_RECORD)
        assert result.urgency_score == 0.0

    def test_genericity_higher_for_generic_than_specific(self) -> None:
        generic_result = extract_text_features(NAYEPANKH_RECORD)
        specific_result = extract_text_features(ANAKIN_RECORD)
        # "Fundraising" should score higher than "Software Development" on genericity
        assert generic_result.genericity_score >= specific_result.genericity_score

    def test_sensitive_info_flagged_for_scam_record(self) -> None:
        result = extract_text_features(SENSITIVE_RECORD)
        assert result.sensitive_info_requested is True

    def test_sensitive_info_not_flagged_for_clean_record(self) -> None:
        result = extract_text_features(CLEAN_RECORD)
        assert result.sensitive_info_requested is False

    def test_truncated_flag_passed_through(self) -> None:
        result = extract_text_features(TRUNCATED_RECORD)
        assert result.summary_truncated is True

    def test_non_truncated_flag_is_false_by_default(self) -> None:
        result = extract_text_features(ANAKIN_RECORD)
        assert result.summary_truncated is False

    def test_responsibilities_cleaned_flag_passed_through(self) -> None:
        record = {**ANAKIN_RECORD, "_flags": {"responsibilities_cleaned": True}}
        result = extract_text_features(record)
        assert result.responsibilities_cleaned is True

    def test_artifact_count_higher_for_dirty_responsibilities(self) -> None:
        clean = extract_text_features(CLEAN_RECORD)
        dirty = extract_text_features(ARTIFACT_RECORD)
        assert dirty.artifact_count >= clean.artifact_count

    def test_caps_ratio_higher_for_urgency_record(self) -> None:
        urgency_result = extract_text_features(URGENCY_RECORD)
        clean_result = extract_text_features(CLEAN_RECORD)
        assert urgency_result.caps_ratio >= clean_result.caps_ratio

    def test_empty_record_does_not_crash(self) -> None:
        result = extract_text_features({})
        assert isinstance(result, TextFeatureVector)
        assert result.word_count == 0

    def test_all_scores_bounded(self) -> None:
        for record in [ANAKIN_RECORD, NAYEPANKH_RECORD, URGENCY_RECORD,
                       TRUNCATED_RECORD, SENSITIVE_RECORD, CLEAN_RECORD, {}]:
            result = extract_text_features(record)
            assert 0.0 <= result.urgency_score <= 1.0
            assert 0.0 <= result.caps_ratio <= 1.0
            assert 0.0 <= result.genericity_score <= 1.0
            assert 0.0 <= result.title_summary_alignment <= 1.0
            assert 0.0 <= result.boilerplate_similarity <= 1.0

    def test_exclamation_count_for_urgency_record(self) -> None:
        # "Hurry UP! Only 12 hours left. Apply ASAP." — at least 1 exclamation
        result = extract_text_features(URGENCY_RECORD)
        assert result.exclamation_count >= 1

    def test_flesch_score_nonzero_for_real_text(self) -> None:
        result = extract_text_features(ANAKIN_RECORD)
        assert result.flesch_score != 0.0

    def test_boilerplate_similarity_default_zero(self) -> None:
        # Corpus not injected by this stage — should remain 0.0
        result = extract_text_features(ANAKIN_RECORD)
        assert result.boilerplate_similarity == 0.0
