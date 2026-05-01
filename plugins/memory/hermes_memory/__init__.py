"""
Hermes Memory — LanceDB Storage + Hybrid Retrieval Plugin
=========================================================

Provides a high-performance memory layer backed by LanceDB with:
- Vector (ANN) search via LanceDB native index
- BM25 full-text search via LanceDB FTS index
- Hybrid retrieval with RRF fusion and cross-encoder reranking
- Scoring pipeline: Recency · Importance · LengthNorm · TimeDecay · MMR

Exports
-------
"""

from .lancedb_storage import (
    LanceDBStorage,
    MemoryEntry,
    MemorySearchResult,
)
from .hybrid_retriever import (
    Embedder,
    JinaEmbedder,
    LocalEmbedder,
    create_embedder,
    HybridRetriever,
    RetrievalResult,
    RetrievalDiagnostics,
    DEFAULT_CONFIG as RETRIEVAL_CONFIG,
)
from .decay_engine import (
    DecayEngine,
    DecayConfig,
    DEFAULT_DECAY_CONFIG,
)
from .access_tracker import AccessTracker
from .tier_manager import (
    TierManager,
    TierConfig,
    DEFAULT_TIER_CONFIG,
)
from .kg_integration import (
    kg_search,
    add_triple_with_entity,
    kg_stats,
    kg_repair_entities,
    KGTriple,
    KGEntity,
)
from .working_memory import WorkingMemory
from .wal_manager import (
    _WALBatcher,
    wal_write_episode,
    wal_replay,
    wal_pending_count,
    wal_truncate,
    _start_wal_batcher,
    _stop_wal_batcher,
)
from .noise_filter import is_noise
from .adaptive_retrieval import should_skip_retrieval
from .scope_isolation import Scope, can_access, ScopeGuard
from .smart_extractor import smart_extract, queue_extraction, ExtractionResult
from .hermes_memory_provider import HermesMemoryProvider


# =============================================================================
# Memory Provider Plugin Registration
# =============================================================================

def register(ctx) -> None:
    """Register this plugin's memory provider with the MemoryManager.

    Called by plugins/memory/__init__.py during provider discovery.
    """
    ctx.register_memory_provider(HermesMemoryProvider())

__all__ = [
    # Storage
    "LanceDBStorage",
    "MemoryEntry",
    "MemorySearchResult",
    # Retriever
    "Embedder",
    "JinaEmbedder",
    "LocalEmbedder",
    "create_embedder",
    "HybridRetriever",
    "RetrievalResult",
    "RetrievalDiagnostics",
    "RETRIEVAL_CONFIG",
    # Decay
    "DecayEngine",
    "DecayConfig",
    "DEFAULT_DECAY_CONFIG",
    # Access
    "AccessTracker",
    # Tier
    "TierManager",
    "TierConfig",
    "DEFAULT_TIER_CONFIG",
    # KG
    "kg_search",
    "add_triple_with_entity",
    "kg_stats",
    "kg_repair_entities",
    "KGTriple",
    "KGEntity",
    # Working Memory
    "WorkingMemory",
    # WAL
    "wal_write_episode",
    "wal_replay",
    "wal_pending_count",
    "wal_truncate",
    # Filters
    "is_noise",
    "should_skip_retrieval",
    # Scope
    "Scope",
    "can_access",
    "ScopeGuard",
    # Smart Extractor
    "smart_extract",
    "queue_extraction",
    "ExtractionResult",
    # Provider
    "HermesMemoryProvider",
]
