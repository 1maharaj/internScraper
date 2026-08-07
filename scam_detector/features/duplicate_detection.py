"""
duplicate_detection.py
-----------------------
Corpus-level near-duplicate and template-posting detection.

This module is intentionally corpus-level, not single-record: it operates on
the full batch of internship records and builds an in-memory index so every
record can be compared against every other.

Architecture
------------
Embeddings: model name from ``config.cfg.embeddings.sbert_model_name``
(same constant as Prompt 2's ``title_summary_alignment``).  Both modules
share one loaded instance via ``text_features._sbert_model`` (``@lru_cache``).

Nearest-neighbour index: ``sklearn.NearestNeighbors`` with cosine metric.

⚠  Scalability note: sklearn's NearestNeighbors is brute-force (O(n²) at
query time) and holds all embeddings in RAM.  This is fine for corpora up to
~50 000 records on a modern machine.  Beyond that, replace the sklearn index
with a proper ANN library:
    - FAISS   (https://github.com/facebookresearch/faiss)  — GPU-optional,
               widely used in production
    - hnswlib (https://github.com/nmslib/hnswlib)          — CPU-optimised
               hierarchical small-world graph, very low latency
See ``DuplicateIndex._build_index`` for the annotated swap-out point.

``cross_company_duplicate_flag``
---------------------------------
Returns True when a record's text is near-identical to another record posted
under a DIFFERENT company name.  This is the strongest signal in the entire
system — the same scam script being posted by multiple shell companies is
far harder to fake than any single-record feature.

NayePankh / Basti Ki Pathshala ambiguity note
----------------------------------------------
Both organisations post structurally similar fundraising internship text
(confirmed in internScraper/scrapers/intershala_scraper/internships.json).
``cross_company_duplicate_flag`` will fire on these records because they have
high text similarity AND different company names.  In production, a
``same_parent_organization_allowlist`` lookup should be run before treating
this flag as a hard reject:
    - These two foundations appear to use the same Internshala posting
      template and may share a parent entity or reseller account.
    - Without an allowlist, legitimate affiliated orgs will generate false
      positives here.
    - This is a known, named ambiguity (Phase 1 flag) — do not silently
      resolve it by lowering the threshold.  Surface it and let a human
      reviewer (or a company-graph lookup) make the call.
"""

from __future__ import annotations

import re
from collections import defaultdict
from typing import Any

import numpy as np
from pydantic import BaseModel, Field

# ---------------------------------------------------------------------------
# Lazy imports for optional heavy dependencies
# ---------------------------------------------------------------------------


def _get_sbert():
    """Return cached SentenceTransformer, or None if unavailable."""
    try:
        # Reuse the singleton from text_features so the model is loaded once.
        from scam_detector.features.text_features import _sbert_model
        return _sbert_model()
    except Exception:
        return None


def _get_nn_class():
    """Return sklearn NearestNeighbors class, or None if unavailable."""
    try:
        from sklearn.neighbors import NearestNeighbors  # type: ignore
        return NearestNeighbors
    except ImportError:
        return None


# ---------------------------------------------------------------------------
# Pydantic models
# ---------------------------------------------------------------------------


class DuplicateMatch(BaseModel):
    """A single near-duplicate pair."""

    record_id: str
    similarity: float = Field(ge=0.0, le=1.0)


class ClusterReport(BaseModel):
    """Summary of near-duplicate clustering across the corpus."""

    total_records: int = Field(ge=0)
    total_clusters: int = Field(ge=0)
    singleton_count: int = Field(ge=0, description="Records with no near-duplicate")
    cluster_size_distribution: dict[int, int] = Field(
        default_factory=dict,
        description="Maps cluster_size → number_of_clusters_of_that_size",
    )
    largest_cluster_size: int = Field(ge=0)
    largest_cluster_ids: list[str] = Field(default_factory=list)


# ---------------------------------------------------------------------------
# Helper: record text for embedding
# ---------------------------------------------------------------------------

_WHITESPACE = re.compile(r"\s+")


def _record_text(record: dict[str, Any]) -> str:
    """
    Concatenate name + summary for embedding.

    Using both fields gives better semantic coverage than either alone:
    - name captures the job title signal
    - summary captures the description / boilerplate signal
    """
    name = (record.get("name") or record.get("title") or "").strip()
    summary = (record.get("summary") or record.get("description") or "").strip()
    raw = f"{name} {summary}"
    return _WHITESPACE.sub(" ", raw).strip()


def _record_id(record: dict[str, Any], idx: int) -> str:
    """
    Extract a stable string ID from a record, falling back to positional index.
    """
    rid = record.get("_id") or record.get("id") or record.get("internship_id")
    if rid is None:
        return str(idx)
    if isinstance(rid, dict):
        # MongoDB Extended JSON: {"$oid": "..."}
        return str(rid.get("$oid") or rid)
    return str(rid)


# ---------------------------------------------------------------------------
# DuplicateIndex
# ---------------------------------------------------------------------------


class DuplicateIndex:
    """
    In-memory near-duplicate index for a corpus of internship records.

    Usage
    -----
        index = DuplicateIndex()
        index.build(records)
        dupes = index.find_near_duplicates("some-record-id", threshold=0.92)
        report = index.duplicate_cluster_report()

    Thread safety: not thread-safe.  Build once, then read concurrently.
    """

    def __init__(self) -> None:
        self._records: list[dict[str, Any]] = []
        self._ids: list[str] = []
        self._embeddings: np.ndarray | None = None   # shape (N, D), L2-normalised
        self._nn_index: Any = None                    # sklearn NearestNeighbors
        self._id_to_idx: dict[str, int] = {}
        self._built: bool = False

    # ── Build ─────────────────────────────────────────────────────────────

    def build(self, records: list[dict[str, Any]]) -> None:
        """
        Embed all records and build the nearest-neighbour index.

        Parameters
        ----------
        records:
            Full list of internship dicts in the standardised schema.
            Each record should have at minimum ``name`` (or ``title``) and
            ``summary`` (or ``description``) fields.

        Raises
        ------
        RuntimeError
            If ``sentence-transformers`` or ``scikit-learn`` is not installed.
        """
        if not records:
            self._records = []
            self._ids = []
            self._embeddings = np.empty((0, 384), dtype=np.float32)
            self._id_to_idx = {}
            self._built = True
            return

        self._records = list(records)
        self._ids = [_record_id(r, i) for i, r in enumerate(records)]
        self._id_to_idx = {rid: i for i, rid in enumerate(self._ids)}

        model = _get_sbert()
        if model is None:
            raise RuntimeError(
                "sentence-transformers is required for DuplicateIndex. "
                "Install it with: pip install sentence-transformers"
            )

        texts = [_record_text(r) for r in records]

        # Encode in one batch; L2-normalise so dot product == cosine similarity.
        self._embeddings = model.encode(
            texts,
            normalize_embeddings=True,
            show_progress_bar=False,
            batch_size=64,
        ).astype(np.float32)  # shape (N, D)

        self._build_index()
        self._built = True

    def _build_index(self) -> None:
        """
        Build the sklearn NearestNeighbors index.

        ⚠  SWAP-OUT POINT for ANN scaling:
        Replace the block below with a FAISS or hnswlib index once the
        corpus grows past ~50 000 records.  The interface contract is:
            - fit on self._embeddings
            - query returns (distances, indices) where distances are cosine
              *similarities* (not distances) in [0, 1]
        For FAISS example:
            import faiss
            d = self._embeddings.shape[1]
            self._nn_index = faiss.IndexFlatIP(d)   # inner product == cosine for L2-normed
            self._nn_index.add(self._embeddings)
        """
        NNClass = _get_nn_class()
        if NNClass is None:
            raise RuntimeError(
                "scikit-learn is required for DuplicateIndex. "
                "Install it with: pip install scikit-learn"
            )

        n = len(self._records)
        k = min(n, 50)  # return at most 50 neighbours per query

        # metric='cosine' returns cosine *distance* (1 - similarity); we
        # convert to similarity in find_near_duplicates.
        self._nn_index = NNClass(
            n_neighbors=k,
            metric="cosine",
            algorithm="brute",
        )
        self._nn_index.fit(self._embeddings)

    # ── Query ─────────────────────────────────────────────────────────────

    def find_near_duplicates(
        self,
        record_id: str,
        threshold: float = 0.92,
    ) -> list[tuple[str, float]]:
        """
        Return other record IDs whose text is above *threshold* cosine
        similarity to the queried record.

        Parameters
        ----------
        record_id:
            The ``_id`` (or positional index string) of the record to query.
        threshold:
            Minimum cosine similarity to include in results.  Default 0.92
            is conservative enough to avoid false positives on topically
            similar (but distinct) postings while catching template clones.

        Returns
        -------
        list of (record_id, similarity) tuples, sorted descending by similarity.
        Excludes the queried record itself.

        Returns empty list if the index has not been built or ``record_id``
        is not found.
        """
        if not self._built or self._embeddings is None or len(self._records) == 0:
            return []

        idx = self._id_to_idx.get(record_id)
        if idx is None:
            return []

        query_vec = self._embeddings[idx : idx + 1]   # shape (1, D)
        n_neighbors = self._nn_index.n_neighbors

        distances, indices = self._nn_index.kneighbors(query_vec, n_neighbors=n_neighbors)
        # sklearn cosine metric returns cosine *distance* = 1 - similarity
        similarities = 1.0 - distances[0]
        neighbor_indices = indices[0]

        results: list[tuple[str, float]] = []
        for ni, sim in zip(neighbor_indices, similarities):
            if ni == idx:
                continue                          # skip self
            if sim >= threshold:
                results.append((self._ids[ni], round(float(sim), 4)))

        results.sort(key=lambda x: x[1], reverse=True)
        return results

    # ── Cluster report ────────────────────────────────────────────────────

    def duplicate_cluster_report(self, threshold: float = 0.92) -> ClusterReport:
        """
        Group the entire corpus into near-duplicate clusters using single-link
        connected-components and report the size distribution.

        This is O(N × k) where k = n_neighbors, making it tractable for the
        sklearn brute-force index.

        Parameters
        ----------
        threshold:
            Cosine similarity threshold to define a duplicate edge.

        Returns
        -------
        ClusterReport
            Useful both as a feature source (which cluster does this record
            belong to?) and as a standalone data-quality audit report.
        """
        n = len(self._records)
        if not self._built or n == 0:
            return ClusterReport(
                total_records=0,
                total_clusters=0,
                singleton_count=0,
                largest_cluster_size=0,
            )

        # Union-Find for connected components
        parent = list(range(n))

        def find(x: int) -> int:
            while parent[x] != x:
                parent[x] = parent[parent[x]]
                x = parent[x]
            return x

        def union(x: int, y: int) -> None:
            parent[find(x)] = find(y)

        # Build edges from pairwise near-duplicate relationships
        k = self._nn_index.n_neighbors
        distances, indices = self._nn_index.kneighbors(self._embeddings, n_neighbors=k)

        for i in range(n):
            for j_pos in range(k):
                j = indices[i][j_pos]
                if j == i:
                    continue
                sim = 1.0 - distances[i][j_pos]
                if sim >= threshold:
                    union(i, j)

        # Collect cluster membership
        clusters: dict[int, list[int]] = defaultdict(list)
        for i in range(n):
            clusters[find(i)].append(i)

        size_dist: dict[int, int] = defaultdict(int)
        for members in clusters.values():
            size_dist[len(members)] += 1

        singleton_count = size_dist.get(1, 0)
        largest = max(clusters.values(), key=len)
        largest_ids = [self._ids[i] for i in largest]

        return ClusterReport(
            total_records=n,
            total_clusters=len(clusters),
            singleton_count=singleton_count,
            cluster_size_distribution=dict(size_dist),
            largest_cluster_size=len(largest),
            largest_cluster_ids=largest_ids,
        )

    # ── Utilities ─────────────────────────────────────────────────────────

    @property
    def record_count(self) -> int:
        return len(self._records)

    def get_embedding(self, record_id: str) -> np.ndarray | None:
        """Return the raw embedding vector for a record (for external use)."""
        idx = self._id_to_idx.get(record_id)
        if idx is None or self._embeddings is None:
            return None
        return self._embeddings[idx]

    def corpus_embeddings(self) -> np.ndarray | None:
        """
        Return the full (N, D) embedding matrix.

        Used by ``text_features.boilerplate_similarity`` to inject the corpus
        cache without re-embedding.
        """
        return self._embeddings


# ---------------------------------------------------------------------------
# cross_company_duplicate_flag
# ---------------------------------------------------------------------------


def cross_company_duplicate_flag(
    record: dict[str, Any],
    duplicate_neighbors: list[tuple[str, float]],
    all_records: list[dict[str, Any]],
) -> bool:
    """
    Return True when a record has near-duplicate text but a DIFFERENT company
    name from at least one of its high-similarity neighbours.

    This is the strongest single signal in the scam-detection system.  The
    same script posted under multiple shell company names is a near-certain
    indicator of coordinated fraudulent posting behaviour.

    ⚠  NayePankh Foundation / Basti Ki Pathshala Foundation false-positive
    warning (Phase 1 ambiguity):
    Both organisations post structurally similar fundraising internship text
    on Internshala.  This function WILL fire on their records because the
    text similarity is genuinely high and they ARE different companies.
    In production, implement a ``same_parent_organization_allowlist`` lookup
    before treating this flag as a hard reject:
        allowlist = {frozenset(["NayePankh Foundation",
                                "the NayePankh Foundation",
                                "Basti Ki Pathshala Foundation"])}
        if any(frozenset([co_a, co_b]) in allowlist ...): skip
    Do NOT silently lower the threshold to suppress this — that would weaken
    detection of genuine cross-company duplicates.  Surface it and let a
    human reviewer or org-graph lookup resolve the ambiguity.

    Parameters
    ----------
    record:
        The record being evaluated.  Must have ``company`` key.
    duplicate_neighbors:
        Output of ``DuplicateIndex.find_near_duplicates`` for this record —
        list of (record_id, similarity) tuples.
    all_records:
        Full corpus list, used to look up the company name of each neighbour.
        Must be the same list (same order) as was passed to
        ``DuplicateIndex.build``.

    Returns
    -------
    bool
        ``True`` → at least one near-duplicate exists under a different
        company name → high-risk cross-company duplicate pattern.
    """
    if not duplicate_neighbors:
        return False

    own_flags = record.get("_flags") or {}
    if own_flags.get("company_suspect"):
        return False

    own_company = (record.get("company") or "").strip().lower()

    # Build a lookup from record_id → (company, suspect_flag) for fast access.
    # If all_records is large this is O(N) but only done once per call.
    id_to_company_info: dict[str, tuple[str, bool]] = {}
    for i, r in enumerate(all_records):
        rid = _record_id(r, i)
        co = (r.get("company") or "").strip().lower()
        r_flags = r.get("_flags") or {}
        suspect = bool(r_flags.get("company_suspect"))
        id_to_company_info[rid] = (co, suspect)

    for neighbour_id, _sim in duplicate_neighbors:
        neighbour_company, neighbour_suspect = id_to_company_info.get(neighbour_id, ("", False))
        if neighbour_suspect:
            continue
        if neighbour_company and neighbour_company != own_company:
            return True

    return False
