"""
text_features.py
----------------
Text-based signal extraction from an internship posting's name, summary,
and responsibilities fields.

All public functions are pure and individually testable.  Each takes plain
text (str) and returns a float, bool, or small dict — no side effects.

The top-level entry point is :func:`extract_text_features`, which accepts a
remediated record dict (output of ``data_quality.remediate_record``) and
returns a :class:`TextFeatureVector` pydantic model.

Sentence-transformer note
--------------------------
``title_summary_alignment`` and ``boilerplate_similarity`` use the model
named in ``config.cfg.embeddings.sbert_model_name`` (default
``all-MiniLM-L6-v2``, 384-d, ~80 MB).  Prompt 5's ``DuplicateIndex`` loads
the same singleton — change the name only in ``config.py``.  The model is
loaded **lazily** at most once per process; if sentence-transformers is
missing the functions degrade to ``0.0``.
"""

from __future__ import annotations

import math
import re
import unicodedata
from functools import lru_cache
from typing import Any

import numpy as np
from pydantic import BaseModel, Field

from scam_detector.config import cfg as _cfg

# ---------------------------------------------------------------------------
# Lazy sentence-transformers import
# ---------------------------------------------------------------------------


def _get_sbert():
    """Return a cached SentenceTransformer instance, or None if unavailable."""
    try:
        from sentence_transformers import SentenceTransformer  # type: ignore
        return SentenceTransformer(_cfg.embeddings.sbert_model_name)
    except Exception:
        return None


@lru_cache(maxsize=1)
def _sbert_model():
    return _get_sbert()


# ---------------------------------------------------------------------------
# Lazy zero-shot NLI classifier (optional, opt-in via cfg.zero_shot.enabled)
# ---------------------------------------------------------------------------


def _get_zero_shot_classifier():
    """Return a cached zero-shot-classification pipeline, or None if unavailable."""
    try:
        from transformers import pipeline  # type: ignore
        return pipeline(
            "zero-shot-classification",
            model=_cfg.zero_shot.model_name,
            device=-1,
        )
    except Exception:
        return None


@lru_cache(maxsize=1)
def _zero_shot_classifier():
    return _get_zero_shot_classifier()


# ---------------------------------------------------------------------------
# Pydantic output model
# ---------------------------------------------------------------------------

class TextFeatures(BaseModel):
    """Slim backwards-compat alias used by the scoring skeleton stubs."""

    scam_keyword_count: int = Field(default=0, ge=0)
    scam_keyword_density: float = Field(default=0.0, ge=0.0, le=1.0)
    urgency_score: float = Field(default=0.0, ge=0.0, le=1.0)
    vagueness_score: float = Field(default=0.0, ge=0.0, le=1.0)
    embedding_scam_similarity: float = Field(default=0.0, ge=0.0, le=1.0)
    scam_corpus_similarity: float = Field(default=0.0, ge=0.0, le=1.0)
    word_count: int = Field(default=0, ge=0)


class TextFeatureVector(BaseModel):
    """
    All text-derived features for one internship posting.

    ``text_from_scraper_flags`` passes through remediation flags that affect
    how downstream scoring should interpret readability/grammar signals:
      - ``summary_truncated``       → readability scores are unreliable
      - ``responsibilities_cleaned`` → artifact count already corrected upstream
    """

    # 1. Urgency
    urgency_score: float = Field(default=0.0, ge=0.0, le=1.0)

    # 2. Caps / punctuation
    caps_ratio: float = Field(default=0.0, ge=0.0, le=1.0)
    exclamation_count: int = Field(default=0, ge=0)
    has_repeated_punctuation: bool = Field(default=False)

    # 3. Generic title similarity
    genericity_score: float = Field(default=0.0, ge=0.0, le=1.0)

    # 4. Semantic alignment between title and summary
    title_summary_alignment: float = Field(default=0.0, ge=0.0, le=1.0)

    # 5. Readability / grammar signals
    avg_sentence_length: float = Field(default=0.0, ge=0.0)
    flesch_score: float = Field(default=0.0)          # can exceed [0,100] for malformed text
    artifact_count: int = Field(default=0, ge=0)       # stray double-spaces / digit artifacts

    # 6. Sensitive info request (near-hard disqualifying signal)
    sensitive_info_requested: bool = Field(default=False)

    # 6b. Guaranteed-outcome / no-interview marketing claim (near-hard disqualifying signal)
    guaranteed_outcome_claim: bool = Field(default=False)

    # 6c. Off-platform form/phone handoff (near-hard disqualifying signal)
    external_form_detected: bool = Field(default=False)
    personal_contact_handoff_detected: bool = Field(default=False)

    # 6d. Internship framed as a packaged training/certification product
    training_program_disguised_as_internship: bool = Field(default=False)

    # 6e. Optional pretrained zero-shot semantic scam signal (opt-in)
    zero_shot_scam_category: str | None = Field(default=None)
    zero_shot_scam_confidence: float = Field(default=0.0, ge=0.0, le=1.0)

    # 7. Boilerplate / near-duplicate similarity
    boilerplate_similarity: float = Field(default=0.0, ge=0.0, le=1.0)
    scam_corpus_similarity: float = Field(default=0.0, ge=0.0, le=1.0)


    # Pass-through remediation flags — affect score discounting
    summary_truncated: bool = Field(default=False)
    responsibilities_cleaned: bool = Field(default=False)

    # Convenience aggregates
    word_count: int = Field(default=0, ge=0)
    char_count: int = Field(default=0, ge=0)


# ---------------------------------------------------------------------------
# 1. urgency_score
# ---------------------------------------------------------------------------

# Patterns ordered by severity; each match adds its weight to the raw score.
# Using compiled regexes with word-boundary anchors to avoid false positives
# (e.g. "apply" inside "applicant").
_URGENCY_PATTERNS: list[tuple[re.Pattern[str], float]] = [
    (re.compile(r"\burgent\b",                       re.I), 1.0),
    (re.compile(r"\bimmediate\s+joining\b",          re.I), 1.0),
    (re.compile(r"\blimited\s+seats?\b",             re.I), 0.9),
    (re.compile(r"\bonly\s+\d+\s+spots?\s+left\b",  re.I), 0.9),
    (re.compile(r"\bapply\s+now\b",                  re.I), 0.6),
    (re.compile(r"\bhurry\b",                        re.I), 0.8),
    (re.compile(r"\blast\s+(?:day|chance|few)\b",    re.I), 0.7),
    (re.compile(r"\bfill\s+fast\b",                  re.I), 0.7),
    (re.compile(r"\bapply\s+asap\b",                 re.I), 0.8),
    (re.compile(r"\bdeadline\s+(?:today|tonight)\b", re.I), 0.7),
    (re.compile(r"\bseats?\s+filling\b",             re.I), 0.8),
    (re.compile(r"\bhiring\s+now\b",                 re.I), 0.5),
    (re.compile(r"\bdon['']t\s+miss\b",              re.I), 0.5),
]

# Maximum achievable raw score if every pattern fired once — used to
# normalise the output to [0, 1].
_URGENCY_MAX_RAW = sum(w for _, w in _URGENCY_PATTERNS)


def urgency_score(text: str) -> float:
    """
    Keyword-density urgency score, normalised to [0, 1].

    Each matched pattern contributes its weight once regardless of how many
    times it appears (capped per pattern).  The raw weight sum is then
    divided by a length-attenuation factor so that a single "apply now" in a
    three-word posting scores higher than the same phrase buried deep in a
    400-word description.

    Parameters
    ----------
    text:
        Combined text to analyse (title + summary + responsibilities).

    Returns
    -------
    float
        0.0 → no urgency signals; 1.0 → extreme urgency pressure.
    """
    if not text or not text.strip():
        return 0.0

    word_count = max(1, len(text.split()))
    raw = sum(weight for pattern, weight in _URGENCY_PATTERNS if pattern.search(text))
    if raw == 0.0:
        return 0.0

    # Length attenuation: log-scale so scores don't vanish for long texts but
    # short texts don't dominate unfairly either.
    attenuation = 1.0 + math.log(word_count / 10.0 + 1.0)
    score = raw / (_URGENCY_MAX_RAW * attenuation / 10.0)
    return float(min(score, 1.0))


# ---------------------------------------------------------------------------
# 2. caps_and_punctuation_ratio
# ---------------------------------------------------------------------------

_REPEATED_PUNCT = re.compile(r"[!?]{2,}|\.{3,}")


def caps_and_punctuation_ratio(text: str) -> dict[str, float | int | bool]:
    """
    Surface-level orthographic signals.

    Returns
    -------
    dict with keys:
        ``caps_ratio``             — fraction of alphabetic chars that are uppercase
        ``exclamation_count``      — raw count of "!" characters
        ``has_repeated_punctuation`` — True if "!!", "???", "..." appear
    """
    if not text:
        return {"caps_ratio": 0.0, "exclamation_count": 0, "has_repeated_punctuation": False}

    alpha_chars = [c for c in text if c.isalpha()]
    upper_chars = [c for c in alpha_chars if c.isupper()]
    caps_ratio = len(upper_chars) / max(len(alpha_chars), 1)

    return {
        "caps_ratio": round(caps_ratio, 4),
        "exclamation_count": text.count("!"),
        "has_repeated_punctuation": bool(_REPEATED_PUNCT.search(text)),
    }


# ---------------------------------------------------------------------------
# 3. genericity_score
# ---------------------------------------------------------------------------

# Seeded from real corpus titles (Internshala data, Naukri data, LetsIntern,
# Freshersworld) — titles that appear on hundreds of listings with
# negligible company-specific content.
_GENERIC_TITLES: list[str] = [
    # Business / sales
    "Business Development",
    "Business Development (Sales)",
    "Sales and Marketing",
    "Sales",
    "Marketing",
    "Lead Generation",
    "Telecalling",
    "Customer Service",
    "Customer Support",
    # Digital / content
    "Digital Marketing",
    "Social Media Marketing",
    "Social Media",
    "Content Writing",
    "Content Creation",
    "Copywriting",
    "Graphic Design",
    "Video Editing",
    "Video Editing/Making",
    "SEO",
    "Performance Marketing",
    # Finance / ops
    "Finance",
    "Accounting",
    "Data Entry",
    "Back Office",
    "Operations",
    "Human Resources",
    "HR",
    "Recruitment",
    # Social / non-profit (NayePankh / Basti Ki Pathshala pattern)
    "Fundraising",
    "Social Work",
    "Social Entrepreneurship",
    "Crowdfunding",
    "Program Assistant",
    # Generic tech (safe to keep now that genericity_score uses
    # token_sort_ratio, not token_set_ratio — see that function's docstring
    # for the false positive this previously caused on specific titles like
    # "SDET Intern (Software Development Engineer in Test)")
    "Software Development",
    "Web Development",
    "App Development",
    # NOTE: bare "Internship" / "Intern" deliberately excluded — every record
    # in this domain is an internship posting, so those words are present in
    # nearly every title regardless of specificity. With token_set_ratio, a
    # single-word generic entry that appears in almost all titles saturates
    # genericity_score to 1.0 universally (e.g. "Data Science Intern at
    # Google-scale ML team" scored maximally "generic" purely from "Intern"),
    # destroying the signal's ability to discriminate vague vs specific roles.
]

_GENERIC_TITLES_LC: list[str] = [t.lower() for t in _GENERIC_TITLES]


def genericity_score(title: str) -> float:
    """
    Fuzzy similarity of *title* to the curated generic-title list.

    Uses ``rapidfuzz.fuzz.token_sort_ratio`` — NOT ``token_set_ratio``.
    ``token_set_ratio` was tried first and found to be a real bug: it
    deliberately ignores extra/non-shared words, so ANY title sharing even
    one common word with a list entry (e.g. "Development") scored close to
    1.0 regardless of how specific the rest of the title was. Verified on
    real data: "SDET Intern (Software Development Engineer in Test)" — a
    specific technical role — scored 0.71-1.0 against "Business
    Development"/"Web Development" purely from the shared word.
    ``token_sort_ratio`` sorts tokens and compares as a sequence, so it
    naturally penalizes length differences — extra qualifying words correctly
    pull the score down instead of being ignored — while still matching
    "Business Development Executive" reasonably against "Business
    Development" (0.58, below the generic threshold, appropriately: having
    a specific qualifier like "- Fintech Vertical" makes a title LESS
    generic, not equally generic).

    Returns
    -------
    float in [0, 1]
        1.0 → title is a known generic/templated label.
        0.0 → title is specific and unique.
    """
    if not title or not title.strip():
        return 0.0

    try:
        from rapidfuzz import fuzz  # type: ignore
    except ImportError:
        # Fallback: exact-match only
        return 1.0 if title.strip().lower() in _GENERIC_TITLES_LC else 0.0

    title_lc = title.strip().lower()
    best = max(
        fuzz.token_sort_ratio(title_lc, generic) / 100.0
        for generic in _GENERIC_TITLES_LC
    )
    return round(float(min(best, 1.0)), 4)


# ---------------------------------------------------------------------------
# 4. title_summary_alignment
# ---------------------------------------------------------------------------

def title_summary_alignment(title: str, summary: str) -> float:
    """
    Cosine similarity between SBERT embeddings of *title* and *summary*.

    A very low score (< 0.15) suggests the summary describes something
    unrelated to the advertised title — a common pattern in copy-pasted or
    auto-generated scam postings.

    Model: ``cfg.embeddings.sbert_model_name`` (shared with DuplicateIndex).
    Falls back to 0.0 if sentence-transformers is not installed.

    Returns
    -------
    float in [0, 1]
    """
    if not title or not title.strip() or not summary or not summary.strip():
        return 0.0

    model = _sbert_model()
    if model is None:
        return 0.0

    try:
        embeddings = model.encode([title.strip(), summary.strip()], normalize_embeddings=True)
        sim = float(np.dot(embeddings[0], embeddings[1]))
        return round(float(np.clip(sim, 0.0, 1.0)), 4)
    except Exception:
        return 0.0


# ---------------------------------------------------------------------------
# 5. readability_and_grammar_signals
# ---------------------------------------------------------------------------

# Re-use the same artifact pattern from data_quality.remediate but as a
# *detector*, not a cleaner — we count occurrences rather than stripping them.
_ARTIFACT_TRAILING = re.compile(r"\s+\d{1,2}\.\s*$", re.MULTILINE)
_DOUBLE_SPACE = re.compile(r"  +")
_SENTENCE_SPLIT = re.compile(r"(?<=[.!?])\s+")


def _syllable_count(word: str) -> int:
    """
    Lightweight syllable approximation (no NLTK dependency).

    Rules (in order):
      1. Strip non-alpha characters.
      2. Count vowel groups (consecutive vowels = 1 syllable).
      3. Subtract silent trailing 'e'.
      4. Minimum 1 syllable per non-empty word.
    """
    word = re.sub(r"[^a-zA-Z]", "", word).lower()
    if not word:
        return 0
    count = len(re.findall(r"[aeiou]+", word))
    if word.endswith("e") and count > 1:
        count -= 1
    return max(count, 1)


def readability_and_grammar_signals(text: str) -> dict[str, float | int]:
    """
    Lightweight readability and surface-grammar metrics.

    Metrics returned
    ----------------
    ``avg_sentence_length``   — mean words per sentence
    ``flesch_score``          — approximated Flesch Reading Ease score
                                (higher = easier; typical range 0–100)
    ``artifact_count``        — occurrences of double-spaces + trailing
                                digit-period fragments (signals scraped HTML)

    No heavy NLP dependency required.  Uses a simple vowel-group syllable
    approximation rather than CMU Pronouncing Dictionary.

    Notes
    -----
    When ``summary_truncated`` flag is set downstream, readability scores
    should be treated as unreliable and not penalised.
    """
    if not text or not text.strip():
        return {"avg_sentence_length": 0.0, "flesch_score": 0.0, "artifact_count": 0}

    # Split into sentences
    sentences = [s.strip() for s in _SENTENCE_SPLIT.split(text) if s.strip()]
    if not sentences:
        sentences = [text]

    word_lists = [s.split() for s in sentences]
    total_words = sum(len(wl) for wl in word_lists)
    num_sentences = max(len(sentences), 1)
    avg_sentence_length = total_words / num_sentences

    # Syllables
    all_words = text.split()
    total_syllables = sum(_syllable_count(w) for w in all_words)
    total_word_count = max(len(all_words), 1)

    # Flesch Reading Ease
    # = 206.835 - 1.015*(words/sentences) - 84.6*(syllables/words)
    flesch = (
        206.835
        - 1.015 * (total_word_count / num_sentences)
        - 84.6 * (total_syllables / total_word_count)
    )

    # Artifact detection — count, don't clean
    artifact_count = len(_ARTIFACT_TRAILING.findall(text)) + len(_DOUBLE_SPACE.findall(text))

    return {
        "avg_sentence_length": round(avg_sentence_length, 2),
        "flesch_score": round(flesch, 2),
        "artifact_count": artifact_count,
    }


# ---------------------------------------------------------------------------
# 6. sensitive_info_request_detector
# ---------------------------------------------------------------------------

# ⚠️  NEAR-HARD DISQUALIFYING SIGNAL — documented for downstream scoring.
#
# Any match here should be treated as a strong candidate for auto-rejection,
# NOT as a weighted feature to blend with other scores.  Downstream risk
# engine should apply a score floor / hard cap when this flag is True.
#
# Patterns cover:
#   - Upfront payment demands (fee, deposit, charge)
#   - Identity document requests (Aadhaar, PAN) inside posting body
#   - Bank account / UPI / payment detail requests
#   - "Pay to start / join / confirm" construction

_SENSITIVE_PATTERNS: list[re.Pattern[str]] = [
    re.compile(r"\bregistration\s+fee\b",                        re.I),
    re.compile(r"\bsecurity\s+deposit\b",                        re.I),
    re.compile(r"\bprocessing\s+fee\b",                          re.I),
    re.compile(r"\btraining\s+fee\b",                            re.I),
    re.compile(r"\bjoining\s+fee\b",                             re.I),
    re.compile(r"\brefundable\s+deposit\b",                      re.I),
    re.compile(r"\bpay\b.{0,30}\b(to\s+)?(join|start|confirm|register|proceed)\b",
               re.I | re.S),
    # "charge" deliberately excluded from this generic proximity match — it is
    # heavily overloaded in legitimate business text (EV/phone charging, "in
    # charge of"), causing false positives (e.g. "scan, pay, charge experience"
    # in an EV-charging company's product description). registration/processing
    # charge wording is still covered by the explicit patterns below.
    re.compile(r"\bpay\b.{0,20}\b(fee|amount|deposit)\b", re.I | re.S),
    re.compile(r"\b(registration|processing)\s+charges?\b", re.I),
    # NOTE: bare "bank\s+account" (without a following qualifier like
    # "details"/"number") was removed after a real-corpus false positive:
    # "you won't just boost your bank account" — a common colloquial phrase
    # meaning "earn money", not a request for banking information. This is
    # the hard-disqualifying rule (weight 0.95), so a loose match here is
    # more costly than anywhere else in the engine.
    re.compile(r"\bbank\s+account\s+(details|number)\b", re.I),
    re.compile(r"\bbank\s+(details|number|transfer)\b",  re.I),
    re.compile(r"\bupi\s+(id|payment|transfer)\b",               re.I),
    re.compile(r"\baadh?a?ar\b",                                  re.I),
    re.compile(r"\bpan\s*(card|number|no\.?)\b",                 re.I),
    re.compile(r"\bpayment\s+required\b",                        re.I),
    re.compile(r"\bdeposit\s+required\b",                        re.I),
    re.compile(r"\bfee\s+required\b",                            re.I),
    re.compile(r"\bpay\s+(?:rs\.?|inr|₹)\s*\d+\b",              re.I),
]


def sensitive_info_request_detector(text: str) -> bool:
    """
    Binary flag: does the posting body request upfront payment or sensitive
    personal data from the applicant?

    ⚠️  NEAR-HARD DISQUALIFYING SIGNAL
    Downstream scoring must treat a ``True`` return as a near-certain reject,
    not as a weighted soft signal.  Do not blend this into a continuous risk
    score — cap the total score at the scam threshold regardless of other
    features.

    Parameters
    ----------
    text:
        Combined posting text (summary + responsibilities).

    Returns
    -------
    bool
        ``True`` → payment or sensitive-data request detected.
    """
    if not text or not text.strip():
        return False
    return any(pattern.search(text) for pattern in _SENSITIVE_PATTERNS)


# ---------------------------------------------------------------------------
# 6b. guaranteed_outcome_claim_detector
# ---------------------------------------------------------------------------

# ⚠️  NEAR-HARD DISQUALIFYING SIGNAL — documented for downstream scoring.
#
# Legitimate internships never legally guarantee a job/placement/certificate
# outcome, and never skip an interview/screening step entirely. These claims
# are a marketing hook scammers front-load into the title/summary/perks — the
# part of a posting that survives scraper truncation — unlike fee/payment
# details, which are usually further down and get cut off. This makes this
# detector one of the few real-data-effective hard signals available today.
_GUARANTEE_PATTERNS: list[re.Pattern[str]] = [
    re.compile(r"\b(100\s*%|hundred\s+percent)\s*(job\s+)?(placement|guarantee|guaranteed)\b", re.I),
    re.compile(r"\bguaranteed\s+(job|placement|internship|offer|stipend|certificate|ppo)\b", re.I),
    re.compile(r"\bassured\s+(job|placement|internship|offer|certificate|ppo)\b", re.I),
    re.compile(r"\bno\s+interview(s)?\s*(required|needed)?\b", re.I),
    re.compile(r"\bwithout\s+(any\s+)?interview\b", re.I),
    re.compile(r"\binstant\s+(selection|hiring|offer|joining|registration)\b", re.I),
    re.compile(r"\bselected?\s+(instantly|immediately)\b", re.I),
    re.compile(r"\bwhatsapp\s+(only|number|group)\b", re.I),
    re.compile(r"\btelegram\s+(group|channel|link)\b", re.I),
    re.compile(r"\brefer\s+and\s+earn\b", re.I),
    re.compile(r"\bearn\s+(upto|up\s+to)\s*(rs\.?|inr|₹)\s*\d+.{0,20}\bfrom\s+home\b", re.I),
    re.compile(r"\bno\s+(skills?|experience)\s+(required|needed).{0,20}\b(guaranteed|assured|100\s*%)\b", re.I),
    re.compile(r"\bwork\s+2\s*[-–]?\s*3\s+hours?.{0,20}\bearn\b", re.I),
]


def guaranteed_outcome_claim_detector(text: str) -> bool:
    """
    Binary flag: does the posting make a guaranteed-outcome or
    skip-the-interview marketing claim typical of fraudulent postings?

    ⚠️  NEAR-HARD DISQUALIFYING SIGNAL — same treatment as
    ``sensitive_info_request_detector``: downstream scoring must apply a
    score floor / hard cap, not blend this as a soft weighted feature.

    Parameters
    ----------
    text:
        Combined posting text (title + summary + responsibilities + perks +
        tags) — deliberately wider than ``sensitive_info_request_detector``'s
        input, since these claims are usually front-loaded as a hook rather
        than buried in a payment-details paragraph.

    Returns
    -------
    bool
        ``True`` → guaranteed-outcome / no-interview claim detected.
    """
    if not text or not text.strip():
        return False
    return any(pattern.search(text) for pattern in _GUARANTEE_PATTERNS)


# ---------------------------------------------------------------------------
# 6c. external_form_and_phone_handoff_detector
# ---------------------------------------------------------------------------
#
# Real-data basis: a posting for "Zefrix" (company field corrupted to
# "Content" by the scraper) reads, verbatim: "Apply now:
# https://forms.gle/MoCTSfEVCwL3GB2n7 or call on +918854996448" — written in
# informal WhatsApp-forward style (emoji, asterisk/underscore markdown).
#
# This is a distinct, well-known scam funnel: the listing routes the
# applicant AWAY from the platform's own apply flow into a generic
# third-party form (which harvests contact details with no real ATS/company
# behind it), followed by a personal phone call — where the actual pitch
# (a paid "training program" sold as an internship) happens verbally, off
# the platform, invisible to any text-mining of the original posting. The
# Google Form + phone number combination is the only trace this leaves in
# scrapable text.
#
# No legitimate internship on an aggregator platform needs a SECOND,
# generic, unbranded form outside the platform's own application system —
# a real employer's ATS or company careers page would appear instead, not a
# bare forms.gle/typeform.com link.

_EXTERNAL_FORM_PATTERNS: list[re.Pattern[str]] = [
    re.compile(r"forms\.gle/\S+", re.I),
    re.compile(r"docs\.google\.com/forms/\S+", re.I),
    re.compile(r"forms\.office\.com/\S+", re.I),
    re.compile(r"typeform\.com/\S+", re.I),
    re.compile(r"tally\.so/\S+", re.I),
    re.compile(r"jotform\.com/\S+", re.I),
    re.compile(r"forms\.zoho\.\S+", re.I),
    re.compile(r"\bgoogle\s+form\b", re.I),
]

# Indian mobile number (10 digits, starts 6-9, optional +91/91 prefix),
# required to appear near a contact-invitation verb to avoid matching an
# unrelated 10-digit number elsewhere in the text.
#
# NOTE: "contact" was deliberately removed from the trigger-verb list after
# a real-corpus false positive: a legitimate construction-firm job posting
# ("Dharvesh Builders") listed a standard "📞 Contact: 9048500028" business
# line — completely normal, and the posting even said "Apply directly
# through Internshala" (i.e. not funneling away from the platform at all).
# "contact" is too generic a word for legitimate business communications
# ("Contact us at...", "For queries, contact HR..."); the remaining verbs
# ("call", "whatsapp", "dm", "message", "ping") carry a more specific,
# casual personal-outreach connotation closer to the actual scam pattern.
_PERSONAL_CONTACT_PATTERN = re.compile(
    r"\b(?:call|whatsapp|dm|message|ping)\b.{0,20}"
    r"(?:\+?91[\-\s]?)?[6-9]\d{9}\b"
    r"|\b(?:\+?91[\-\s]?)?[6-9]\d{9}\b.{0,20}\b(?:call|whatsapp|dm|message)\b",
    re.I | re.S,
)


def external_form_and_phone_handoff_detector(text: str) -> tuple[bool, bool]:
    """
    Detect a listing that funnels applicants off-platform via a generic
    third-party form link and/or a personal phone-number handoff.

    Returns
    -------
    (external_form_detected, personal_contact_detected)
        Either can be True independently. The combination of both is the
        strongest version of this signal (see
        ``ExternalFormHandoffRule`` docstring) but each is meaningful alone.
    """
    if not text or not text.strip():
        return False, False
    form_detected = any(p.search(text) for p in _EXTERNAL_FORM_PATTERNS)
    contact_detected = bool(_PERSONAL_CONTACT_PATTERN.search(text))
    return form_detected, contact_detected


# ---------------------------------------------------------------------------
# 6d. training_program_disguised_as_internship_detector
# ---------------------------------------------------------------------------
#
# Real-data basis (two examples found on LetsIntern):
#   1. company="Electric Vehicle Design Internship" (identical to its own
#      title), summary: "...is a career-focused, hands-on training program
#      designed for students and freshers..." — an incoherent skill list
#      (AWS/DevOps/WordPress for an "EV Design" role) confirms this is
#      templated course-marketing content, not a real posting.
#   2. company="BBA" (a degree name, not an organisation), summary:
#      "...a unique Work-Study program. Selected candidates will be provided
#      with 100% sponsored admission to a professional degree or
#      certification program (BBA, MBA, BCA, or MCA) from our partner
#      university..." — recruiting students into a paid degree/certification
#      admission, framed as an internship.
#
# IMPORTANT — precision note: the bare word "training" or even "training
# program" is NOT a usable signal alone. Checked against the real corpus:
# of 6 postings mentioning "training program", 4 are ordinary legitimate
# internships where training programs are something the intern coordinates,
# sells, or attends as a normal job duty (e.g. "coordinate training
# programs" at a training company, "leads for training programs" in a sales
# role). Only postings where the phrasing frames the INTERNSHIP ITSELF as a
# packaged training/admission product are the real signal — hence the
# specific, narrow phrases below rather than a generic "training" keyword.

_TRAINING_DISGUISED_AS_INTERNSHIP_PATTERNS: list[re.Pattern[str]] = [
    re.compile(r"\bcareer.?focused,?\s+hands.?on\s+training\s+program\b", re.I),
    re.compile(r"\btraining\s+and\s+internships?\s+will\s+go\s+simultaneously\b", re.I),
    re.compile(r"\b100\s*%\s*sponsored\s+admission\b", re.I),
    re.compile(r"\bsponsored\s+admission\s+to\s+a\b.{0,40}\b(certification|degree)\s+program\b", re.I | re.S),
    re.compile(r"\bself.?paced\s+program\b", re.I),
    re.compile(r"\bthis\s+internship\s+is\s+a\b.{0,30}\btraining\s+program\b", re.I | re.S),
]


def training_program_disguised_as_internship_detector(text: str) -> bool:
    """
    Detect a listing that frames the internship itself as a packaged
    training or paid-admission product, rather than actual work — a
    recruitment funnel for phone-sold training/certification fees rather
    than a real job.

    Deliberately narrow: does NOT match generic mentions of "training" or
    even "training program" as a job duty (e.g. "coordinate training
    programs"), since that pattern is common in ordinary, legitimate
    internships and was verified to produce mostly false positives on real
    data. Only matches phrasing that frames the WHOLE INTERNSHIP as a
    training/certification/admission product.

    Parameters
    ----------
    text:
        Combined posting text (title + summary + responsibilities + perks +
        tags).

    Returns
    -------
    bool
        ``True`` → the posting is framed as a training/certification
        product rather than a real job.
    """
    if not text or not text.strip():
        return False
    return any(pattern.search(text) for pattern in _TRAINING_DISGUISED_AS_INTERNSHIP_PATTERNS)


# ---------------------------------------------------------------------------
# 6e. zero_shot_scam_signal (optional — pretrained model, not hand-written regex)
# ---------------------------------------------------------------------------
#
# Validation history — READ BEFORE ENABLING (cfg.zero_shot.enabled):
#   1. Tested an abstract "legitimate vs fraudulent" label pair first — this
#      performed at chance level, misclassifying all 3 known real scam
#      examples as "legitimate" with high confidence.
#   2. Switched to concrete, specific candidate labels (naming the actual
#      red-flag behavior). On 8 hand-picked examples (3 known scams + 3
#      paraphrases + 2 clean) this looked like genuine generalization past
#      regex phrasing — all 8 classified correctly.
#   3. BUT a follow-up test against a RANDOM 250-record real sample found a
#      ~20% false-positive rate. Nearly every false positive landed on
#      "sells a paid training or certification course disguised as a job",
#      over-triggering on completely ordinary postings (including one from
#      Airbus) — the model cannot distinguish "this internship teaches you
#      skills" (true of almost every legitimate internship) from "this
#      internship IS a fake training product" (the actual red flag).
# Conclusion: 8 hand-picked examples were not enough to validate a
# probabilistic signal. This needs a reworked label set and/or a much
# higher confidence threshold, re-validated on a large random sample,
# before it should ever be enabled. See ZeroShotConfig in config.py.

_ZERO_SHOT_LABELS: list[str] = [
    "asks the applicant to pay money or fill an external form to apply",
    "promises a guaranteed job or admission outcome without proper screening",
    "sells a paid training or certification course disguised as a job",
    "a normal, specific job description with real responsibilities",
]
_ZERO_SHOT_NORMAL_LABEL = _ZERO_SHOT_LABELS[-1]
_ZERO_SHOT_MAX_INPUT_CHARS = 2000  # cap for inference speed


def zero_shot_scam_signal(text: str) -> tuple[str | None, float]:
    """
    Semantic (non-regex) scam-pattern classification via a pretrained
    zero-shot NLI model — opt-in via ``cfg.zero_shot.enabled`` (off by
    default; see ``ZeroShotConfig`` docstring for the latency tradeoff and
    validation results).

    Parameters
    ----------
    text:
        Combined posting text (title + summary + responsibilities + perks +
        tags).

    Returns
    -------
    (category, confidence)
        ``category`` is one of the three non-normal labels in
        ``_ZERO_SHOT_LABELS``, or ``None`` if the classifier is
        disabled/unavailable, the input is empty, or the top prediction is
        the "normal job description" label. ``confidence`` is 0.0 when
        ``category`` is None.
    """
    if not _cfg.zero_shot.enabled:
        return None, 0.0
    if not text or not text.strip():
        return None, 0.0

    classifier = _zero_shot_classifier()
    if classifier is None:
        return None, 0.0

    try:
        result = classifier(text.strip()[:_ZERO_SHOT_MAX_INPUT_CHARS], _ZERO_SHOT_LABELS)
        top_label = result["labels"][0]
        top_score = float(result["scores"][0])
    except Exception:
        return None, 0.0

    if top_label == _ZERO_SHOT_NORMAL_LABEL:
        return None, 0.0
    return top_label, round(top_score, 4)


# ---------------------------------------------------------------------------
# 7. boilerplate_similarity
# ---------------------------------------------------------------------------

def boilerplate_similarity(
    text: str,
    corpus_embeddings: "np.ndarray | list[list[float]] | None",
) -> float:
    """
    Max cosine similarity of *text* against a precomputed corpus embedding
    matrix.

    Used for near-duplicate / heavily-templated posting detection.  A very
    high score (> 0.92) against the existing corpus signals copy-pasted
    boilerplate rather than genuine company-specific content.

    The **corpus embedding cache** is built and maintained by the duplicate-
    detection stage (Prompt 5).  This function only implements the similarity
    lookup — it does not manage the cache.

    Parameters
    ----------
    text:
        The posting text to embed and compare.
    corpus_embeddings:
        2-D array of shape ``(N, embedding_dim)`` where each row is an
        L2-normalised embedding of another posting.  Pass ``None`` to skip
        (returns ``0.0``).

    Returns
    -------
    float in [0, 1]
        Max cosine similarity across all corpus embeddings.
        0.0 → no corpus provided or model unavailable.
    """
    if corpus_embeddings is None or not text or not text.strip():
        return 0.0

    model = _sbert_model()
    if model is None:
        return 0.0

    try:
        emb = model.encode([text.strip()], normalize_embeddings=True)   # shape (1, d)
        corpus = np.array(corpus_embeddings, dtype=np.float32)          # shape (N, d)
        sims = corpus @ emb[0]                                          # shape (N,)
        return round(float(np.clip(float(sims.max()), 0.0, 1.0)), 4)
    except Exception:
        return 0.0


# ---------------------------------------------------------------------------
# 8. scam_corpus_similarity
# ---------------------------------------------------------------------------

def scam_corpus_similarity(
    text: str,
    scam_embeddings: np.ndarray | list[list[float]] | None,
) -> float:
    """
    Max cosine similarity of *text* against a maintained set of confirmed-scam embeddings.

    Parameters
    ----------
    text:
        The posting text to embed and compare.
    scam_embeddings:
        2-D array of shape ``(M, embedding_dim)`` containing L2-normalised
        embeddings of confirmed scam postings from FeedbackStore.

    Returns
    -------
    float in [0, 1]
    """
    if scam_embeddings is None or not text or not text.strip():
        return 0.0

    model = _sbert_model()
    if model is None:
        return 0.0

    try:
        scam_arr = np.array(scam_embeddings, dtype=np.float32)
        if scam_arr.size == 0 or len(scam_arr.shape) != 2:
            return 0.0
        emb = model.encode([text.strip()], normalize_embeddings=True)
        sims = scam_arr @ emb[0]
        return round(float(np.clip(float(sims.max()), 0.0, 1.0)), 4)
    except Exception:
        return 0.0


def get_scam_corpus_embeddings(
    feedback_store: Any | None = None,
    all_records: list[dict[str, Any]] | None = None,
) -> np.ndarray | None:
    """
    Build and return L2-normalised SBERT embeddings matrix of confirmed scam records
    pulled from FeedbackStore.
    """
    if feedback_store is None or all_records is None:
        return None

    try:
        history = feedback_store.load_feedback_history()
        scam_ids = {fb.record_id for fb in history if fb.reviewer_decision == "confirmed_scam"}
        if not scam_ids:
            return None

        from scam_detector.features.duplicate_detection import _record_id, _record_text
        scam_texts = [
            _record_text(r)
            for i, r in enumerate(all_records)
            if _record_id(r, i) in scam_ids
        ]
        if not scam_texts:
            return None

        model = _sbert_model()
        if model is None:
            return None

        return model.encode(scam_texts, normalize_embeddings=True, show_progress_bar=False).astype(np.float32)
    except Exception:
        return None


# ---------------------------------------------------------------------------
# Top-level extractor
# ---------------------------------------------------------------------------

def extract_text_features(
    record: dict[str, Any],
    scam_embeddings: np.ndarray | list[list[float]] | None = None,
) -> TextFeatureVector:
    """
    Extract all text features from a remediated internship record.

    The *record* should be the ``.record`` field from a
    :class:`~scam_detector.data_quality.RemediatedRecord`, but a plain dict
    in the standardised schema also works.

    Remediation flags ``summary_truncated`` and ``responsibilities_cleaned``
    are passed through to the vector so downstream scoring can discount
    readability/grammar signals when the scraper hard-truncated the text
    rather than the employer writing poorly.

    Parameters
    ----------
    record:
        Internship document with keys: name, summary, responsibilities,
        and optionally the remediation flags dict (passed separately or
        embedded as ``_flags``).
    scam_embeddings:
        Optional precomputed embedding matrix of confirmed scam records from FeedbackStore.
    """
    title: str = record.get("name") or ""
    summary: str = record.get("summary") or ""
    responsibilities_raw = record.get("responsibilities") or []
    if isinstance(responsibilities_raw, list):
        resp_text = " ".join(str(r) for r in responsibilities_raw if r)
    else:
        resp_text = str(responsibilities_raw)

    perks_raw = record.get("perks") or []
    perks_text = " ".join(str(p) for p in perks_raw if p) if isinstance(perks_raw, list) else str(perks_raw)
    tags_raw = record.get("tags") or []
    tags_text = " ".join(str(t) for t in tags_raw if t) if isinstance(tags_raw, list) else str(tags_raw)

    # Full body text for multi-field analysers
    full_text = " ".join(filter(None, [title, summary, resp_text]))

    # Wider scan window for hard-disqualifying pattern detectors: scammers
    # front-load hooks (guarantees, WhatsApp/Telegram handoffs) into the
    # title/perks/tags, which survive scraper summary-truncation better than
    # payment details buried mid-paragraph.
    pattern_scan_text = " ".join(filter(None, [title, summary, resp_text, perks_text, tags_text]))

    # Remediation pass-through flags (may come from the flags dict or be
    # embedded directly on the record by the pipeline)
    flags: dict[str, Any] = record.get("_flags") or {}
    summary_truncated: bool = bool(flags.get("summary_truncated", False))
    responsibilities_cleaned: bool = bool(flags.get("responsibilities_cleaned", False))

    # 1. Urgency
    u_score = urgency_score(full_text)

    # 2. Caps / punctuation (applied to full text)
    cp = caps_and_punctuation_ratio(full_text)

    # 3. Genericity
    g_score = genericity_score(title)

    # 4. Alignment (title ↔ summary)
    align = title_summary_alignment(title, summary)

    # 5. Readability (applied to summary + responsibilities)
    body_text = " ".join(filter(None, [summary, resp_text]))
    rg = readability_and_grammar_signals(body_text)

    # 6. Sensitive info request (widened scan window — see pattern_scan_text)
    sensitive = sensitive_info_request_detector(pattern_scan_text)

    # 6b. Guaranteed-outcome / no-interview marketing claim
    guarantee_claim = guaranteed_outcome_claim_detector(pattern_scan_text)

    # 6c. Off-platform external form / personal phone handoff
    external_form_detected, personal_contact_detected = external_form_and_phone_handoff_detector(pattern_scan_text)

    # 6d. Internship framed as a packaged training/certification product
    training_disguise = training_program_disguised_as_internship_detector(pattern_scan_text)

    # 6e. Optional pretrained zero-shot semantic scam signal (opt-in, off by default)
    zero_shot_category, zero_shot_confidence = zero_shot_scam_signal(pattern_scan_text)

    # 7. Scam corpus similarity
    scam_sim = scam_corpus_similarity(full_text, scam_embeddings)

    word_count = len(full_text.split()) if full_text.strip() else 0
    char_count = len(full_text)

    return TextFeatureVector(
        urgency_score=u_score,
        caps_ratio=float(cp["caps_ratio"]),
        exclamation_count=int(cp["exclamation_count"]),
        has_repeated_punctuation=bool(cp["has_repeated_punctuation"]),
        genericity_score=g_score,
        title_summary_alignment=align,
        avg_sentence_length=float(rg["avg_sentence_length"]),
        flesch_score=float(rg["flesch_score"]),
        artifact_count=int(rg["artifact_count"]),
        sensitive_info_requested=sensitive,
        guaranteed_outcome_claim=guarantee_claim,
        external_form_detected=external_form_detected,
        personal_contact_handoff_detected=personal_contact_detected,
        training_program_disguised_as_internship=training_disguise,
        zero_shot_scam_category=zero_shot_category,
        zero_shot_scam_confidence=zero_shot_confidence,
        boilerplate_similarity=0.0,   # injected by duplicate-detection stage
        scam_corpus_similarity=scam_sim,
        summary_truncated=summary_truncated,
        responsibilities_cleaned=responsibilities_cleaned,
        word_count=word_count,
        char_count=char_count,
    )

