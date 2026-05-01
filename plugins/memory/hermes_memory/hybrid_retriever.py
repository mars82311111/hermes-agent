"""
Hybrid Retrieval Pipeline
=========================

Combines vector search (ANN) + BM25 full-text search with:
- RRF (Reciprocal Rank Fusion) for result fusion
- Cross-encoder reranking via Jina Rerank API
- Scoring pipeline: Recency boost, Importance weight,
  Length normalisation, Hard minimum score, Time decay
- MMR (Maximum Marginal Relevance) diversity filtering

Reference: /tmp/memory-lancedb-pro/src/retriever.ts
Reference: /tmp/memory-lancedb-pro/src/embedder.ts
"""

from __future__ import annotations

import asyncio
import json
import logging
import math
import os
import time
from abc import ABC, abstractmethod
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from typing import Any, Optional

import httpx

from .lancedb_storage import LanceDBStorage, MemoryEntry, MemorySearchResult

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Embedder abstraction layer
# ---------------------------------------------------------------------------

class Embedder(ABC):
    """Abstract base for embedding providers."""

    @abstractmethod
    def embed(self, text: str) -> list[float]:
        """Return a single embedding vector for *text*."""
        ...

    @abstractmethod
    def embed_batch(self, texts: list[str]) -> list[list[float]]:
        """Return embedding vectors for a batch of texts."""
        ...

    @property
    @abstractmethod
    def dimensions(self) -> int:
        """Embedding vector dimension."""
        ...


class JinaEmbedder(Embedder):
    """
    Jina AI embeddings (OpenAI-compatible API).
    Supports jina-embeddings-v5 and jina-embeddings-v4 models.
    """

    DEFAULT_BASE_URL = "https://api.jina.ai/v1"
    DEFAULT_MODEL = "jina-embeddings-v5"

    def __init__(
        self,
        api_key: Optional[str] = None,
        model: str = DEFAULT_MODEL,
        base_url: str = DEFAULT_BASE_URL,
        dimensions: int = 1024,
        timeout: float = 30.0,
    ):
        self.api_key = api_key or os.environ.get("JINA_API_KEY", "")
        self.model = model
        self.base_url = base_url
        self._dimensions = dimensions
        self.timeout = timeout

    @property
    def dimensions(self) -> int:
        return self._dimensions

    def _post(self, payload: dict) -> dict:
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}",
        }
        resp = httpx.post(
            f"{self.base_url}/embeddings",
            json=payload,
            headers=headers,
            timeout=self.timeout,
        )
        resp.raise_for_status()
        return resp.json()

    def embed(self, text: str) -> list[float]:
        result = self._post(
            {
                "model": self.model,
                "input": text,
            }
        )
        return result["data"][0]["embedding"]

    def embed_batch(self, texts: list[str]) -> list[list[float]]:
        result = self._post(
            {
                "model": self.model,
                "input": texts,
            }
        )
        # Sort by index to guarantee order
        sorted_data = sorted(result["data"], key=lambda x: x["index"])
        return [item["embedding"] for item in sorted_data]


class LocalEmbedder(Embedder):
    """
    Local embedding model using sentence-transformers + Apple Silicon MPS.
    Defaults to BAAI/bge-m3 (1024d, Chinese-optimized).
    Falls back gracefully if model is not available.
    """

    DEFAULT_MODEL = "BAAI/bge-m3"
    DEFAULT_DIM = 1024

    def __init__(
        self,
        model_name: str = DEFAULT_MODEL,
        device: Optional[str] = None,
        normalize: bool = True,
    ):
        self.model_name = model_name
        self.normalize = normalize
        self._model = None
        self._dimensions = self.DEFAULT_DIM

        # Auto-detect best device: mps > cpu
        if device is None:
            try:
                import torch
                device = "mps" if torch.backends.mps.is_available() else "cpu"
            except ImportError:
                device = "cpu"
        self.device = device

        self._load_model()

    def _load_model(self) -> None:
        """Lazy-load the sentence-transformers model."""
        try:
            from sentence_transformers import SentenceTransformer
            import os
            # Use hf-mirror for China network
            os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
            # CRITICAL FIX: Use offline mode to avoid HuggingFace connection timeout
            # The model is already cached at ~/.cache/huggingface/hub/models--BAAI--bge-m3
            os.environ.setdefault("HF_HUB_OFFLINE", "1")
            os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
            # Also disable telemetry to avoid network calls
            os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")
            # Disable auto_conversion thread that spawns network requests
            os.environ.setdefault("HF_HUB_DISABLE_AUTO_CONVERSION", "1")

            logger.info("Loading local embedding model: %s (device=%s)", self.model_name, self.device)
            self._model = SentenceTransformer(
                self.model_name,
                device=self.device,
                trust_remote_code=False,  # bge-m3 does NOT need remote code; True breaks tokenizer loading
                local_files_only=True,  # Force offline mode - never try to download
            )
            # COMPAT: get_sentence_embedding_dimension() was renamed in newer versions
            try:
                self._dimensions = self._model.get_embedding_dimension()
            except AttributeError:
                self._dimensions = self._model.get_sentence_embedding_dimension()
            logger.info("Local model loaded: %sd on %s", self._dimensions, self.device)
        except Exception as e:
            logger.error("Failed to load local embedding model: %s", e)
            self._model = None
            raise

    @property
    def dimensions(self) -> int:
        return self._dimensions

    def embed(self, text: str) -> list[float]:
        if self._model is None:
            raise RuntimeError("Local embedding model not loaded")
        vec = self._model.encode(text, normalize_embeddings=self.normalize)
        return vec.tolist()

    def embed_batch(self, texts: list[str]) -> list[list[float]]:
        if self._model is None:
            raise RuntimeError("Local embedding model not loaded")
        vecs = self._model.encode(texts, normalize_embeddings=self.normalize)
        return vecs.tolist()


def create_embedder(
    prefer_local: bool = True,
    model_name: Optional[str] = None,
    device: Optional[str] = None,
    provider: Optional[str] = None,
) -> Optional[Embedder]:
    """Factory: try local model first, fallback to Jina if key available.

    OPTIMIZATION: Now respects user config from hermes_memory.json:
      - provider: "local" | "jina"
      - model_name: e.g. "BAAI/bge-m3"
      - device: e.g. "cpu" | "mps"
    """
    # If provider is explicitly set, honour it
    if provider == "jina":
        jina_key = os.environ.get("JINA_API_KEY", "")
        if jina_key:
            try:
                embedder = JinaEmbedder(api_key=jina_key)
                embedder.embed("test")
                return embedder
            except Exception as e:
                logger.warning("Jina embedder failed: %s", e)
        logger.warning("Provider=jina requested but JINA_API_KEY not set")
        return None

    if provider == "local" or prefer_local:
        try:
            return LocalEmbedder(
                model_name=model_name or LocalEmbedder.DEFAULT_MODEL,
                device=device,
            )
        except Exception as e:
            logger.warning("Local embedder failed: %s", e)

    # Fallback to Jina
    jina_key = os.environ.get("JINA_API_KEY", "")
    if jina_key:
        try:
            embedder = JinaEmbedder(api_key=jina_key)
            embedder.embed("test")
            return embedder
        except Exception as e:
            logger.warning("Jina embedder failed: %s", e)

    return None


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

DEFAULT_CONFIG = {
    # Search mode
    "mode": "hybrid",          # "hybrid" | "vector"
    "vector_weight": 0.7,       # RRF weight for vector search
    "bm25_weight": 0.3,        # RRF weight for BM25 search

    # Candidate pool
    "candidate_pool_size": 20,  # Number of results fetched from each search arm

    # Scoring pipeline
    "recency_half_life_days": 14,
    "recency_weight": 0.10,    # Max recency boost factor
    "importance_weight": 1.0,  # Multiplier on entry.importance
    "length_norm_anchor": 500, # Reference length for log normalisation
    "hard_min_score": 0.35,     # Discard results below this after all scoring

    # Time decay
    "time_decay_half_life_days": 60,
    "reinforcement_factor": 0.5,
    "max_half_life_multiplier": 3.0,

    # Diversity (MMR)
    "mmr_diversity_threshold": 0.85,  # Cosine sim above this → penalise

    # Reranking
    "rerank": "cross-encoder",  # "cross-encoder" | "lightweight" | "none"
    "rerank_model": "jina-reranker-v3",
    "rerank_endpoint": "https://api.jina.ai/v1/rerank",
    "rerank_timeout_ms": 5000,
}


# ---------------------------------------------------------------------------
# RRF Fusion
# ---------------------------------------------------------------------------

def rrf_fuse(
    ranked_lists: list[list[tuple[str, float, float]]],
    weights: list[float],
    k: int = 60,
) -> list[tuple[str, float]]:
    """
    Reciprocal Rank Fusion.

    Args:
        ranked_lists: List of ranked result lists.
                     Each item is (entry_id, score, raw_score).
        weights:     Weight for each ranked list (sum not required to be 1).
        k:           RRF smoothing constant (default 60).

    Returns:
        Merged list of (entry_id, fused_score) sorted descending.
    """
    scores: dict[str, float] = {}
    for ranked_list, weight in zip(ranked_lists, weights):
        for rank, (entry_id, _, _) in enumerate(ranked_list, start=1):
            rrf = weight / (k + rank)
            scores[entry_id] = scores.get(entry_id, 0.0) + rrf

    return sorted(scores.items(), key=lambda x: x[1], reverse=True)


# ---------------------------------------------------------------------------
# Cross-encoder reranking via Jina
# ---------------------------------------------------------------------------

def rerank_cross_encoder(
    query: str,
    entries: list[MemoryEntry],
    api_key: Optional[str] = None,
    model: str = "jina-reranker-v3",
    endpoint: str = "https://api.jina.ai/v1/rerank",
    timeout_ms: int = 5000,
) -> list[tuple[MemoryEntry, float]]:
    """
    Re-rank *entries* using the Jina rerank API.

    Returns:
        List of (entry, relevance_score) sorted descending by relevance.
    """
    api_key = api_key or os.environ.get("JINA_API_KEY", "")
    if not api_key:
        logger.warning("No JINA_API_KEY — skipping cross-encoder reranking")
        return [(e, e.importance) for e in entries]

    documents = [e.text for e in entries]
    payload = {
        "model": model,
        "query": query,
        "documents": documents,
        "top_n": len(entries),
    }
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {api_key}",
    }

    try:
        resp = httpx.post(
            endpoint,
            json=payload,
            headers=headers,
            timeout=timeout_ms / 1000.0,
        )
        resp.raise_for_status()
        data = resp.json()
    except Exception as e:
        logger.warning("Rerank API call failed: %s — returning unranked", e)
        return [(e, e.importance) for e in entries]

    # Build index → score map
    # Jina response: { "results": [{ "index": int, "relevance_score": float }] }
    index_scores: dict[int, float] = {}
    for item in data.get("results", []):
        idx = int(item.get("index", -1))
        score = float(item.get("relevance_score", 0.0))
        if idx >= 0:
            index_scores[idx] = score

    reranked = []
    for i, entry in enumerate(entries):
        score = index_scores.get(i, entry.importance)
        reranked.append((entry, score))

    reranked.sort(key=lambda x: x[1], reverse=True)
    return reranked


# ---------------------------------------------------------------------------
# Scoring pipeline
# ---------------------------------------------------------------------------

def apply_recency_boost(
    entries: list[MemoryEntry],
    scores: list[float],
    now: float,
    half_life_days: float = 14.0,
    max_weight: float = 0.10,
) -> list[float]:
    """Boost newer entries additively."""
    boost = max_weight / (1.0 + math.log1p(half_life_days * 86400))
    result = []
    for entry, score in zip(entries, scores):
        age_days = (now - entry.timestamp) / 86400.0
        recency = max(0.0, 1.0 - age_days / half_life_days)
        result.append(score + boost * recency)
    return result


def apply_importance_weight(
    entries: list[MemoryEntry],
    scores: list[float],
    weight: float = 1.0,
) -> list[float]:
    """Multiply scores by entry importance."""
    return [score * (entry.importance ** 0.5) * weight for entry, score in zip(entries, scores)]


def apply_length_norm(
    entries: list[MemoryEntry],
    scores: list[float],
    anchor: int = 500,
) -> list[float]:
    """Penalise very long entries using log normalisation."""
    result = []
    for entry, score in zip(entries, scores):
        char_len = max(1, len(entry.text))
        norm = 1.0 / (1.0 + math.log1p(char_len / anchor))
        result.append(score * norm)
    return result


def apply_time_decay(
    entries: list[MemoryEntry],
    scores: list[float],
    now: float,
    half_life_days: float = 60.0,
) -> list[float]:
    """Apply multiplicative time decay: older entries score lower."""
    if half_life_days <= 0:
        return scores
    half_life_s = half_life_days * 86400.0
    result = []
    for entry, score in zip(entries, scores):
        age_days = (now - entry.timestamp) / 86400.0
        decay = 0.5 + 0.5 * math.exp(-age_days * 86400.0 / half_life_s)
        result.append(score * decay)
    return result


def apply_hard_min_score(
    entries: list[MemoryEntry],
    scores: list[float],
    min_score: float = 0.35,
) -> tuple[list[MemoryEntry], list[float]]:
    """Drop entries with scores below *min_score*."""
    filtered_entries = []
    filtered_scores = []
    for entry, score in zip(entries, scores):
        if score >= min_score:
            filtered_entries.append(entry)
            filtered_scores.append(score)
    return filtered_entries, filtered_scores


# ---------------------------------------------------------------------------
# MMR Diversity
# ---------------------------------------------------------------------------

def _cosine(a: list[float], b: list[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    norm_a = math.sqrt(sum(x * x for x in a))
    norm_b = math.sqrt(sum(x * x for x in b))
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return dot / (norm_a * norm_b)


def apply_mmr_diversity(
    entries: list[MemoryEntry],
    scores: list[float],
    threshold: float = 0.85,
) -> tuple[list[MemoryEntry], list[float]]:
    """
    Penalise entries that are too similar to higher-ranked results.
    Entries with cosine similarity > threshold to any already-kept entry
    receive a score penalty of 0.5.
    """
    if not entries:
        return entries, scores

    kept = []
    kept_vectors = []
    final_entries = []
    final_scores = []

    for entry, score in sorted(zip(entries, scores), key=lambda x: x[1], reverse=True):
        # Penalise if too similar to any kept entry
        if entry.vector:
            sims = [_cosine(entry.vector, v) for v in kept_vectors]
            max_sim = max(sims) if sims else 0.0
            if max_sim > threshold:
                score *= 0.5

        final_entries.append(entry)
        final_scores.append(score)
        kept.append((entry, score))
        if entry.vector:
            kept_vectors.append(entry.vector)

    return final_entries, final_scores


# ---------------------------------------------------------------------------
# Hybrid Retriever
# ---------------------------------------------------------------------------

@dataclass
class RetrievalResult:
    """A single retrieval result with scoring breakdown."""
    entry: MemoryEntry
    fused_score: float
    sources: dict = field(default_factory=dict)


@dataclass
class RetrievalDiagnostics:
    """Diagnostic information from a retrieval pass."""
    vector_result_count: int = 0
    bm25_result_count: int = 0
    fused_result_count: int = 0
    reranked_count: int = 0
    final_count: int = 0
    stage_counts: dict = field(default_factory=dict)


class HybridRetriever:
    """
    Full hybrid retrieval pipeline.

    Usage:
        storage = LanceDBStorage("~/.hermes/lancedb")
        embedder = JinaEmbedder(api_key="...")
        retriever = HybridRetriever(storage, embedder)
        results = retriever.retrieve("城哥的模型偏好", limit=10)
    """

    def __init__(
        self,
        storage: LanceDBStorage,
        embedder: Embedder,
        config: Optional[dict] = None,
    ):
        self.storage = storage
        self.embedder = embedder
        self.config = {**DEFAULT_CONFIG, **(config or {})}
        self._executor = ThreadPoolExecutor(max_workers=4)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def retrieve(
        self,
        query: str,
        limit: int = 10,
        scope_filter: Optional[list[str]] = None,
        category: Optional[str] = None,
    ) -> tuple[list[RetrievalResult], RetrievalDiagnostics]:
        """
        Run the full hybrid retrieval pipeline for *query*.

        Returns:
            (list of RetrievalResult, RetrievalDiagnostics)
        """
        cfg = self.config
        diagnostics = RetrievalDiagnostics()
        now = time.time()

        # ── 1. Parallel vector + BM25 search ─────────────────────────────
        # OPTIMIZATION: Use the class-level executor instead of creating a
        # new ThreadPoolExecutor on every retrieve() call. This eliminates
        # thread creation/teardown overhead and prevents thread leaks.
        vec_future = self._executor.submit(
            self.storage.vector_search,
            self.embedder.embed(query),
            limit=cfg["candidate_pool_size"],
            scope_filter=scope_filter,
            category=category,
            min_score=0.0,
        )
        bm25_future = self._executor.submit(
            self.storage.bm25_search,
            query,
            limit=cfg["candidate_pool_size"],
            scope_filter=scope_filter,
            category=category,
            min_score=0.0,
        )

        vec_results = vec_future.result()
        bm25_results = bm25_future.result()

        diagnostics.vector_result_count = len(vec_results)
        diagnostics.bm25_result_count = len(bm25_results)

        # ── 2. Build ranked lists for RRF ───────────────────────────────
        vec_ranked = [
            (r.entry.id, float(r.score), float(r.score))
            for r in sorted(vec_results, key=lambda x: x.score, reverse=True)
        ]
        bm25_ranked = [
            (r.entry.id, float(r.score), float(r.score))
            for r in sorted(bm25_results, key=lambda x: x.score, reverse=True)
        ]

        # ── 3. RRF fusion ───────────────────────────────────────────────
        fused = rrf_fuse(
            [vec_ranked, bm25_ranked],
            [cfg["vector_weight"], cfg["bm25_weight"]],
        )
        diagnostics.fused_result_count = len(fused)

        # Map id → entry
        id_to_entry = {r.entry.id: r.entry for r in vec_results + bm25_results}

        # ── 4. Collect fused entries ────────────────────────────────────
        fused_entries = []
        fused_scores = []
        for entry_id, fused_score in fused:
            entry = id_to_entry.get(entry_id)
            if entry is None:
                entry = self.storage.get_by_id(entry_id)
            if entry is not None:
                fused_entries.append(entry)
                fused_scores.append(fused_score)

        # ── 5. Cross-encoder rerank ───────────────────────────────────
        if cfg["rerank"] == "cross-encoder":
            reranked = rerank_cross_encoder(
                query=query,
                entries=fused_entries,
                api_key=os.environ.get("JINA_API_KEY"),
                model=cfg["rerank_model"],
                endpoint=cfg["rerank_endpoint"],
                timeout_ms=cfg["rerank_timeout_ms"],
            )
            fused_entries = [e for e, _ in reranked]
            fused_scores = [s for _, s in reranked]
        diagnostics.reranked_count = len(fused_entries)

        # ── 6. Scoring pipeline ────────────────────────────────────────
        n = len(fused_entries)
        diagnostics.stage_counts["afterRerank"] = n

        scores = list(fused_scores)
        scores = apply_recency_boost(
            fused_entries, scores, now,
            half_life_days=cfg["recency_half_life_days"],
            max_weight=cfg["recency_weight"],
        )
        diagnostics.stage_counts["afterRecency"] = len(scores)

        scores = apply_importance_weight(fused_entries, scores, cfg["importance_weight"])
        diagnostics.stage_counts["afterImportance"] = len(scores)

        scores = apply_length_norm(fused_entries, scores, cfg["length_norm_anchor"])
        diagnostics.stage_counts["afterLengthNorm"] = len(scores)

        scores = apply_time_decay(
            fused_entries, scores, now,
            half_life_days=cfg["time_decay_half_life_days"],
        )
        diagnostics.stage_counts["afterTimeDecay"] = len(scores)

        # Adjust hard_min_score for RRF-only mode (no reranker available)
        # RRF scores are naturally small (~0.01-0.07) vs reranked scores (~0.3-0.9)
        hard_min = cfg["hard_min_score"]
        if cfg["rerank"] == "cross-encoder" and not os.environ.get("JINA_API_KEY"):
            hard_min = 0.001

        fused_entries, scores = apply_hard_min_score(
            fused_entries, scores, hard_min
        )
        diagnostics.stage_counts["afterHardMin"] = len(fused_entries)

        # ── 7. MMR diversity ────────────────────────────────────────────
        if cfg["mmr_diversity_threshold"] > 0:
            fused_entries, scores = apply_mmr_diversity(
                fused_entries, scores, cfg["mmr_diversity_threshold"]
            )
        diagnostics.stage_counts["afterDiversity"] = len(fused_entries)

        # ── 8. Limit + build results ───────────────────────────────────
        results = []
        for entry, score in zip(fused_entries[:limit], scores[:limit]):
            results.append(
                RetrievalResult(
                    entry=entry,
                    fused_score=score,
                    sources={},
                )
            )
        diagnostics.final_count = len(results)

        return results, diagnostics

    def close(self) -> None:
        """Shut down the executor."""
        self._executor.shutdown(wait=False)
