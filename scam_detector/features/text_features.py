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
# Pydantic output model
# ---------------------------------------------------------------------------

class TextFeatures(BaseModel):
    """Slim backwards-compat alias used by the scoring skeleton stubs."""

    scam_keyword_count: int = Field(default=0, ge=0)
    scam_keyword_density: float = Field(default=0.0, ge=0.0, le=1.0)
    urgency_score: float = Field(default=0.0, ge=0.0, le=1.0)
    vagueness_score: float = Field(default=0.0, ge=0.0, le=1.0)
    embedding_scam_similarity: float = Field(default=0.0, ge=0.0, le=1.0)
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

    # 7. Boilerplate / near-duplicate similarity
    boilerplate_similarity: float = Field(default=0.0, ge=0.0, le=1.0)

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
    # Generic tech
    "Software Development",
    "Web Development",
    "App Development",
    "Internship",
    "Intern",
]

_GENERIC_TITLES_LC: list[str] = [t.lower() for t in _GENERIC_TITLES]


def genericity_score(title: str) -> float:
    """
    Fuzzy similarity of *title* to the curated generic-title list.

    Uses ``rapidfuzz.fuzz.token_set_ratio`` (handles word-order and partial
    overlaps well) so "Business Development Executive" still scores high
    against "Business Development".

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
        fuzz.token_set_ratio(title_lc, generic) / 100.0
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
    re.compile(r"\bpay\b.{0,20}\b(fee|amount|charge|deposit)\b", re.I | re.S),
    re.compile(r"\bbank\s+(account|details|number|transfer)\b",  re.I),
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
# Top-level extractor
# ---------------------------------------------------------------------------

def extract_text_features(record: dict[str, Any]) -> TextFeatureVector:
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

    Notes
    -----
    ``boilerplate_similarity`` is left at its default (0.0) here — the corpus
    embedding cache is injected by the duplicate-detection stage.
    """
    title: str = record.get("name") or ""
    summary: str = record.get("summary") or ""
    responsibilities_raw = record.get("responsibilities") or []
    if isinstance(responsibilities_raw, list):
        resp_text = " ".join(str(r) for r in responsibilities_raw if r)
    else:
        resp_text = str(responsibilities_raw)

    # Full body text for multi-field analysers
    full_text = " ".join(filter(None, [title, summary, resp_text]))

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

    # 6. Sensitive info request (summary + responsibilities only)
    sensitive = sensitive_info_request_detector(body_text)

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
        boilerplate_similarity=0.0,   # injected by duplicate-detection stage
        summary_truncated=summary_truncated,
        responsibilities_cleaned=responsibilities_cleaned,
        word_count=word_count,
        char_count=char_count,
    )
