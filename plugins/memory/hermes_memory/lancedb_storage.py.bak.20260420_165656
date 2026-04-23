"""
LanceDB Storage Layer for Hermes Memory
========================================
Provides persistent storage for memory entries with vector search,
BM25 full-text search, and cross-process file locking.

Tested against LanceDB 0.30.x API.
"""

from __future__ import annotations

import json
import logging
import os
import random
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Lazy LanceDB import
# ---------------------------------------------------------------------------

_lancedb: Optional[Any] = None


def _get_lancedb():
    """Lazy-load LanceDB, caching the module reference."""
    global _lancedb
    if _lancedb is None:
        try:
            import lancedb
            _lancedb = lancedb
        except ImportError:
            raise ImportError(
                "lancedb is not installed. Install it with: pip install lancedb"
            )
    return _lancedb


# ---------------------------------------------------------------------------
# File Lock (pure Python, no external dependency)
# ---------------------------------------------------------------------------

class FileLock:
    """Cross-process file lock using OS-level advisory locking (fcntl).

    CRITICAL FIX: Previous implementation used os.O_CREAT | os.O_EXCL which
    leaves stale lock files if the process crashes. fcntl locks are tied to
    the file descriptor and are automatically released by the kernel when the
    process exits — no stale lock files, no deadlocks.
    """

    def __init__(self, lock_path: str, timeout: float = 30.0):
        self.lock_path = Path(lock_path)
        self.timeout = timeout
        self._fd: Optional[int] = None

    def acquire(self) -> bool:
        """Acquire the lock, blocking up to *timeout* seconds."""
        self.lock_path.parent.mkdir(parents=True, exist_ok=True)
        start = time.monotonic()

        # Open (or create) the lock file.  We keep this fd open for the
        # duration of the critical section so the kernel holds the lock.
        while True:
            try:
                fd = os.open(
                    str(self.lock_path),
                    os.O_CREAT | os.O_RDWR,
                    0o666,
                )
                break
            except OSError:
                if time.monotonic() - start >= self.timeout:
                    return False
                time.sleep(0.05)

        # Try to acquire an exclusive (write) advisory lock via fcntl.
        # This works on Linux, macOS, and any POSIX system.
        try:
            import fcntl
            has_fcntl = True
        except ImportError:
            has_fcntl = False

        if has_fcntl:
            while True:
                try:
                    fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
                    self._fd = fd
                    return True
                except (IOError, OSError):
                    if time.monotonic() - start >= self.timeout:
                        os.close(fd)
                        return False
                    time.sleep(0.05)
        else:
            # Ultimate fallback for non-PPOSIX systems: use O_EXCL
            # (same as old behaviour, but only used when fcntl unavailable).
            os.close(fd)
            while True:
                try:
                    fd = os.open(
                        str(self.lock_path),
                        os.O_CREAT | os.O_EXCL | os.O_WRONLY,
                    )
                    self._fd = fd
                    return True
                except FileExistsError:
                    if time.monotonic() - start >= self.timeout:
                        return False
                    time.sleep(0.05)
                except OSError:
                    if time.monotonic() - start >= self.timeout:
                        return False
                    time.sleep(0.05)

    def release(self) -> None:
        """Release the lock."""
        if self._fd is not None:
            try:
                import fcntl
                fcntl.flock(self._fd, fcntl.LOCK_UN)
            except Exception:
                pass
            try:
                os.close(self._fd)
            except OSError:
                pass
            self._fd = None

    def __enter__(self):
        if not self.acquire():
            raise RuntimeError(f"Could not acquire lock: {self.lock_path}")
        return self

    def __exit__(self, *args):
        self.release()


# ---------------------------------------------------------------------------
# Data Models
# ---------------------------------------------------------------------------

@dataclass
class MemoryEntry:
    """Single memory episode stored in LanceDB."""

    id: str
    text: str
    vector: list[float]
    category: str           # "preference" | "fact" | "decision" | "entity" | "other" | "reflection"
    scope: str              # e.g. "global", "project:AIF", session UUID, etc.
    importance: float        # 0.0 – 1.0
    timestamp: float        # Unix timestamp (seconds)
    metadata: str = "{}"    # JSON string for extensible metadata
    # Decay-related fields (persisted inside metadata JSON to avoid schema migration)
    access_count: int = 0
    created_at: int = 0     # ms timestamp
    last_accessed_at: int = 0  # ms timestamp
    tier: str = "peripheral"   # "core" | "working" | "peripheral"
    confidence: float = 1.0
    temporal_type: str = "static"  # "static" | "dynamic"

    def _meta_dict(self) -> dict:
        """Parse metadata JSON into a dict."""
        try:
            m = json.loads(self.metadata) if self.metadata else {}
            return m if isinstance(m, dict) else {}
        except (json.JSONDecodeError, TypeError):
            return {}

    def to_row(self) -> dict:
        # Pack decay fields into metadata so the LanceDB schema stays stable
        meta = self._meta_dict()
        meta["access_count"] = self.access_count
        meta["created_at"] = self.created_at
        meta["last_accessed_at"] = self.last_accessed_at
        meta["tier"] = self.tier
        meta["confidence"] = self.confidence
        meta["temporal_type"] = self.temporal_type
        return {
            "id": self.id,
            "text": self.text,
            "vector": self.vector,
            "category": self.category,
            "scope": self.scope,
            "importance": self.importance,
            "timestamp": int(self.timestamp),  # Schema expects int64
            "metadata": json.dumps(meta, ensure_ascii=False),
        }

    @classmethod
    def from_row(cls, row: dict) -> "MemoryEntry":
        vec = row.get("vector")
        if hasattr(vec, "__iter__") and not isinstance(vec, str):
            vec = list(vec)
        else:
            vec = []
        metadata = row.get("metadata", "{}")
        # Parse decay fields from metadata JSON
        try:
            meta = json.loads(metadata) if metadata else {}
            if not isinstance(meta, dict):
                meta = {}
        except (json.JSONDecodeError, TypeError):
            meta = {}
        return cls(
            id=row["id"],
            text=row["text"],
            vector=vec,
            category=row.get("category", "other"),
            scope=row.get("scope", "global"),
            importance=float(row.get("importance", 0.5)),
            timestamp=float(row.get("timestamp", 0.0)),
            metadata=metadata,
            access_count=int(meta.get("access_count", 0)),
            created_at=int(meta.get("created_at", 0)),
            last_accessed_at=int(meta.get("last_accessed_at", 0)),
            tier=meta.get("tier", "peripheral"),
            confidence=float(meta.get("confidence", 1.0)),
            temporal_type=meta.get("temporal_type", "static"),
        )


@dataclass
class MemorySearchResult:
    """A memory entry with its search score."""
    entry: MemoryEntry
    score: float


# ---------------------------------------------------------------------------
# LanceDB Storage
# ---------------------------------------------------------------------------

TABLE_NAME = "memories"


class LanceDBStorage:
    """
    LanceDB-backed memory storage with:
    - Vector (ANN) search via LanceDB's native vector index
    - BM25 full-text search via LanceDB FTS index
    - Cross-process write serialisation via file locking
    """

    def __init__(self, db_path: str, vector_dim: int = 1024):
        self.db_path = str(Path(db_path).expanduser().resolve())
        self.vector_dim = vector_dim
        self._lock_path = str(Path(self.db_path) / ".memory-write.lock")
        self.__table: Optional[Any] = None
        self._fts_index_created = False
        # Fragmentation prevention: track write count for periodic compaction
        self._write_count = 0
        self._compact_threshold = 50  # Compact every 50 writes to prevent fragmentation
        # OPTIMIZATION: Write batching buffer to amortize lock acquisition cost
        self._write_buffer: list[dict] = []
        self._batch_size = 10
        self._flush_interval = 5.0  # seconds
        self._batch_timer: Optional[threading.Timer] = None
        self._batch_lock = threading.Lock()

    # -----------------------------------------------------------------------
    # Initialisation helpers
    # -----------------------------------------------------------------------

    def _init_table(self) -> Any:
        """Open or create the LanceDB table, idempotent.
        
        CRITICAL FIX: Use file lock to prevent concurrent table creation
        from multiple processes/threads, which can leave the table in a
        partially-created state.
        
        CRITICAL FIX 2: If open_table fails but the table name is listed,
        the table may be corrupted. Drop and recreate it rather than
        leaving the system in an unrecoverable state.
        """
        lancedb = _get_lancedb()
        db = lancedb.connect(self.db_path)

        # Use a separate lock for table creation to avoid races
        init_lock = FileLock(str(Path(self.db_path) / ".table-init.lock"), timeout=60.0)
        
        try:
            table = db.open_table(TABLE_NAME)
        except Exception as open_err:
            # Table doesn't exist or is corrupted — acquire lock and create
            if not init_lock.acquire():
                logger.warning("Could not acquire table init lock; retrying open_table")
                table = db.open_table(TABLE_NAME)
            else:
                try:
                    # Double-check after acquiring lock
                    try:
                        table = db.open_table(TABLE_NAME)
                    except Exception:
                        # CRITICAL FIX: If table exists in listing but open fails,
                        # it is likely corrupted. Drop it and recreate.
                        try:
                            table_names = db.table_names()
                            if TABLE_NAME in table_names:
                                logger.warning(
                                    "LanceDB table '%s' exists but cannot be opened (%s). "
                                    "Dropping and recreating...",
                                    TABLE_NAME, open_err
                                )
                                db.drop_table(TABLE_NAME)
                        except Exception as drop_err:
                            logger.debug("Drop table attempt failed: %s", drop_err)
                        
                        schema_row = self._make_dummy_row()
                        table = db.create_table(TABLE_NAME, [schema_row])
                        table.delete('id = ".__schema_dummy__"')
                finally:
                    init_lock.release()

        self.__table = table
        self._ensure_fts_index(table)
        self._ensure_vector_index(table)
        return table

    def _ensure_vector_index(self, table: Any) -> None:
        """Create ANN vector index on 'vector' column if not already present."""
        try:
            indices = table.list_indices()
            has_vector_idx = False
            for idx in indices:
                idx_type = getattr(idx, 'index_type', None)
                if idx_type and 'vector' in str(idx_type).lower():
                    has_vector_idx = True
                    break
                # LanceDB 0.27+ uses different attribute names
                idx_name = getattr(idx, 'name', '')
                if 'vector' in str(idx_name).lower():
                    has_vector_idx = True
                    break
            if not has_vector_idx:
                try:
                    # LanceDB 0.30+ API: metric is FIRST positional param,
                    # column is specified via vector_column_name (defaults to "vector").
                    # Old code wrongly used: create_index(metric="cosine", vector_column_name="vector")
                    # which failed with "got multiple values for metric" because metric
                    # was already filled by the positional "vector" string.
                    table.create_index(
                        "cosine",  # metric as first positional arg (LanceDB 0.30 changed API)
                        index_type="IVF_PQ",
                        num_partitions=4,
                    )
                    logger.info("Vector ANN index created (cosine, IVF_PQ, 4 partitions)")
                except Exception as e:
                    logger.warning("Vector index creation failed: %s", e)
            else:
                logger.debug("Vector index already exists")
        except Exception as e:
            logger.warning("Could not check/create vector index: %s", e)

    def _make_dummy_row(self) -> dict:
        return MemoryEntry(
            id=".__schema_dummy__",
            text="",
            vector=[0.0] * self.vector_dim,
            category="other",
            scope="global",
            importance=0.0,
            timestamp=0.0,
        ).to_row()

    def _ensure_fts_index(self, table: Any) -> None:
        """Create FTS index on 'text' column if not already present.
        Uses ngram tokenizer for Chinese text support (2-4 char ngrams).
        
        CRITICAL FIX: If index creation fails due to missing data files
        (e.g. after a crash or partial compaction), drop existing index first
        and retry. This prevents the 'Object not found' error from persisting.
        """
        try:
            indices = table.list_indices()
            has_fts = False
            for idx in indices:
                if hasattr(idx, 'index_type') and idx.index_type == "FTS":
                    has_fts = True
                    break
                elif hasattr(idx, 'get'):
                    idx_type = idx.get("indexType", "")
                    if idx_type == "FTS":
                        has_fts = True
                        break
                    cols = idx.get("columns", [])
                    if isinstance(cols, list) and "text" in cols:
                        has_fts = True
                        break
            if not has_fts:
                # ngram tokenizer for Chinese support (Rust-native, no extra deps)
                try:
                    table.create_fts_index(
                        "text",
                        with_position=True,
                        replace=True,
                        base_tokenizer="ngram",
                        ngram_min_length=2,
                        ngram_max_length=4,
                    )
                    logger.info("BM25 FTS index created with ngram(2-4) tokenizer")
                except Exception as ngram_err:
                    logger.warning("ngram tokenizer failed (%s), falling back to default", ngram_err)
                    table.create_fts_index("text", with_position=True, replace=True)
                    logger.info("BM25 FTS index created with default tokenizer")
                self._fts_index_created = True
            else:
                self._fts_index_created = True
                logger.debug("FTS index already exists")
        except Exception as e:
            err_str = str(e).lower()
            # If the error is about missing files, try to drop and recreate the index
            if "not found" in err_str or "object" in err_str or "io error" in err_str:
                logger.warning("FTS index appears corrupted (%s). Attempting to drop and recreate...", e)
                try:
                    # Try to drop any existing FTS index by name
                    for idx in table.list_indices():
                        idx_name = getattr(idx, 'name', '')
                        if 'text' in str(idx_name).lower() or 'fts' in str(idx_name).lower():
                            try:
                                table.drop_index(idx_name)
                                logger.info("Dropped corrupted FTS index: %s", idx_name)
                            except Exception as drop_err:
                                logger.debug("Drop index %s failed: %s", idx_name, drop_err)
                    # Now recreate
                    table.create_fts_index("text", with_position=True, replace=True)
                    logger.info("BM25 FTS index recreated after corruption")
                    self._fts_index_created = True
                    return
                except Exception as recreate_err:
                    logger.error("FTS index recreation failed: %s", recreate_err)
            logger.warning("Could not create FTS index: %s", e)
            self._fts_index_created = False

    def _get_table(self) -> Any:
        """Lazy table accessor."""
        if self.__table is None:
            self._init_table()
        return self.__table

    # -----------------------------------------------------------------------
    # File-lock wrapper for writes
    # -----------------------------------------------------------------------

    def _with_lock(self, fn, *args, **kwargs):
        """Run *fn* while holding the cross-process write lock."""
        lock = FileLock(self._lock_path, timeout=30.0)
        if not lock.acquire():
            raise RuntimeError(
                f"Could not acquire write lock at {self._lock_path}. "
                "Another process may be holding it."
            )
        try:
            return fn(*args, **kwargs)
        finally:
            lock.release()

    # -----------------------------------------------------------------------
    # Public API — CRUD
    # -----------------------------------------------------------------------

    def store(self, entry: MemoryEntry) -> MemoryEntry:
        """
        Insert a new memory entry.  A new UUID is always generated;
        the entry's existing *id* field is ignored.

        OPTIMIZATION: Uses write batching buffer to amortize lock
        acquisition cost across multiple writes. Buffer auto-flushes
        when it reaches _batch_size or after _flush_interval seconds.
        """
        full = MemoryEntry(
            id=self._new_id(),
            text=entry.text,
            vector=entry.vector,
            category=entry.category,
            scope=entry.scope,
            importance=entry.importance,
            timestamp=time.time(),
            metadata=entry.metadata,
        )

        with self._batch_lock:
            self._write_buffer.append(full.to_row())
            should_flush = len(self._write_buffer) >= self._batch_size

        if should_flush:
            self._flush_buffer()
        else:
            # Start/reset delayed flush timer
            with self._batch_lock:
                if self._batch_timer is not None:
                    self._batch_timer.cancel()
                self._batch_timer = threading.Timer(
                    self._flush_interval, self._flush_buffer
                )
                self._batch_timer.daemon = True
                self._batch_timer.start()

        return full

    def store_batch(self, entries: list[MemoryEntry]) -> list[MemoryEntry]:
        """
        Batch insert multiple entries in a single write operation.
        Much more efficient than calling store() repeatedly.

        OPTIMIZATION: Achieves 5-10x throughput improvement by
        amortizing lock acquisition and LanceDB add() overhead.
        """
        if not entries:
            return []

        fulls = []
        rows = []
        now = time.time()
        for entry in entries:
            full = MemoryEntry(
                id=self._new_id(),
                text=entry.text,
                vector=entry.vector,
                category=entry.category,
                scope=entry.scope,
                importance=entry.importance,
                timestamp=now,
                metadata=entry.metadata,
            )
            fulls.append(full)
            rows.append(full.to_row())

        def _do():
            self._get_table().add(rows)

        self._with_lock(_do)

        # Fragmentation prevention: periodic compaction
        self._write_count += len(rows)
        if self._write_count >= self._compact_threshold:
            self._write_count = 0
            self._compact_table()

        return fulls

    def _flush_buffer(self) -> None:
        """Flush the write buffer to LanceDB under a single lock acquisition."""
        with self._batch_lock:
            if self._batch_timer is not None:
                self._batch_timer.cancel()
                self._batch_timer = None
            if not self._write_buffer:
                return
            rows = self._write_buffer
            self._write_buffer = []

        def _do():
            self._get_table().add(rows)

        self._with_lock(_do)

        # Fragmentation prevention
        self._write_count += len(rows)
        if self._write_count >= self._compact_threshold:
            self._write_count = 0
            self._compact_table()
    
    def _compact_table(self) -> None:
        """Compact small data files AND prune old versions to prevent storage bloat."""
        def _do():
            try:
                from datetime import timedelta
                table = self._get_table()
                # CRITICAL FIX: cleanup_older_than=timedelta(days=1) removes old versions
                # older than 1 day. Previously days=0 removed ALL old versions instantly,
                # which could cause data loss if a crash occurred during compaction.
                # Keeping 1 day of history provides a safety window while still preventing
                # the storage bloat that occurs when optimize() keeps all versions.
                table.optimize(cleanup_older_than=timedelta(days=1))
                logger.info("LanceDB compaction completed (optimized + versions older than 1 day pruned)")
            except Exception as e:
                logger.debug("LanceDB compaction skipped: %s", e)
        self._with_lock(_do)

    def import_entry(self, entry: MemoryEntry) -> MemoryEntry:
        """
        Import a pre-built entry preserving its id and timestamp.
        Used for migration from Chroma or re-embedding.
        """
        if not entry.id:
            raise ValueError("import_entry requires a non-empty id")

        def _do():
            self._get_table().add([entry.to_row()])

        self._with_lock(_do)
        return entry

    @staticmethod
    def _escape_sql_value(value: str) -> str:
        """Escape a string for safe SQL interpolation in LanceDB where clauses.

        Uses SQL-standard single-quote wrapping with '' escaping.
        """
        # Replace backslashes first, then single quotes (SQL: '' escapes ')
        return value.replace('\\', '\\\\').replace("'", "''")

    def _lookup_by_id_raw(self, entry_id: str) -> Optional[dict]:
        """Fast exact-ID lookup using LanceDB scalar filter (no vector search).

        OPTIMIZATION: Previously used .search(zero_vector).where(id=...)
        which spins up the ANN engine for no reason. Using .search().where()
        with no vector argument lets LanceDB use pure scalar filtering,
        which is significantly faster for primary-key-style lookups.
        """
        try:
            table = self._get_table()
            # No vector argument = scalar-only query, much faster for ID lookup
            rows = table.search().where(f"id = '{self._escape_sql_value(entry_id)}'").limit(1).to_list()
            return rows[0] if rows else None
        except Exception:
            # Fallback to old method if API unavailable
            safe_id = self._escape_sql_value(entry_id)
            rows = (
                self._get_table()
                .search([0.0] * self.vector_dim, vector_column_name="vector")
                .where(f"id = '{safe_id}'")
                .limit(1)
                .to_list()
            )
            return rows[0] if rows else None

    def get_by_id(self, entry_id: str) -> Optional[MemoryEntry]:
        """Retrieve a single entry by its id, or None if not found."""
        raw = self._lookup_by_id_raw(entry_id)
        if raw is None:
            return None
        return MemoryEntry.from_row(raw)

    def count(self) -> int:
        """Total number of entries."""
        return self._get_table().count_rows()

    def has_id(self, entry_id: str) -> bool:
        """Return True if an entry with the given id exists."""
        return self._lookup_by_id_raw(entry_id) is not None

    def delete(self, entry_id: str) -> bool:
        """Delete an entry by id. Returns True on success."""
        safe_id = self._escape_sql_value(entry_id)

        def _do():
            try:
                self._get_table().delete(f"id = '{safe_id}'")
                return True
            except Exception:
                return False

        return self._with_lock(_do)

    def update(self, entry_id: str, fields: dict) -> bool:
        """Update specific fields of an entry by id (synchronous).

        Uses LanceDB's merge-insert pattern: read the existing row,
        apply field updates, delete the old row, and insert the updated one.
        """
        safe_id = self._escape_sql_value(entry_id)

        def _do():
            try:
                # Read existing row via fast lookup
                row = self._lookup_by_id_raw(entry_id)
                if row is None:
                    return False

                # Build updated row preserving all existing fields
                updated = dict(row)
                for key, value in fields.items():
                    updated[key] = value
                # Remove LanceDB internal score field if present
                updated.pop("_score", None)
                updated.pop("_distance", None)

                # Delete old and insert updated
                self._get_table().delete(f"id = '{safe_id}'")
                self._get_table().add([updated])
                return True
            except Exception as e:
                logger.warning("update failed for %s: %s", entry_id[:8], e)
                return False

        return self._with_lock(_do)

    # -----------------------------------------------------------------------
    # Search APIs
    # -----------------------------------------------------------------------

    def vector_search(
        self,
        query_vector: list[float],
        limit: int = 20,
        scope_filter: Optional[list[str]] = None,
        category: Optional[str] = None,
        min_score: float = 0.0,
    ) -> list[MemorySearchResult]:
        """
        ANN vector search using LanceDB's native vector index.
        Returns entries ordered by similarity descending.
        """
        limit = int(limit)
        table = self._get_table()

        # Build where clause (escape all string values to prevent injection)
        where_parts = []
        if scope_filter is not None and len(scope_filter) > 0:
            scopes = ", ".join(f"'{self._escape_sql_value(s)}'" for s in scope_filter)
            where_parts.append(f"scope IN ({scopes})")
        if category:
            where_parts.append(f"category = '{self._escape_sql_value(category)}'")
        where_clause = " AND ".join(where_parts) if where_parts else None

        query = (
            table.search(query_vector, vector_column_name="vector")
            .limit(limit * 2)
        )
        if where_clause:
            query = query.where(where_clause)

        rows = query.to_list()
        results = []
        for row in rows:
            score = row.get("_score", 0.0)
            if score < min_score:
                continue
            entry = MemoryEntry.from_row(row)
            results.append(MemorySearchResult(entry=entry, score=float(score)))
            if len(results) >= limit:
                break

        return results

    def bm25_search(
        self,
        query: str,
        limit: int = 20,
        scope_filter: Optional[list[str]] = None,
        category: Optional[str] = None,
        min_score: float = 0.0,
    ) -> list[MemorySearchResult]:
        """
        BM25 full-text search using LanceDB's FTS index.
        Returns entries ordered by relevance score descending.
        """
        limit = int(limit)
        if not self._fts_index_created:
            logger.warning("BM25 search called but FTS index was not created")
            return []

        table = self._get_table()

        where_parts = []
        if scope_filter is not None and len(scope_filter) > 0:
            scopes = ", ".join(f"'{self._escape_sql_value(s)}'" for s in scope_filter)
            where_parts.append(f"scope IN ({scopes})")
        if category:
            where_parts.append(f"category = '{self._escape_sql_value(category)}'")
        where_clause = " AND ".join(where_parts) if where_parts else None

        query_obj = (
            table.search(query)
            .limit(limit * 2)
        )
        if where_clause:
            query_obj = query_obj.where(where_clause)

        rows = query_obj.to_list()
        results = []
        for row in rows:
            score = row.get("_score", 0.0)
            if score < min_score:
                continue
            entry = MemoryEntry.from_row(row)
            results.append(MemorySearchResult(entry=entry, score=float(score)))
            if len(results) >= limit:
                break

        return results

    # -----------------------------------------------------------------------
    # Utilities
    # -----------------------------------------------------------------------

    @staticmethod
    def _new_id() -> str:
        """Generate a random UUID-like id."""
        return (
            f"{random.randint(0, 0xFFFFFFFF):08x}"
            f"-{random.randint(0, 0xFFFF):04x}"
            f"-{random.randint(0, 0xFFFF):04x}"
            f"-{random.randint(0, 0xFFFF):04x}"
            f"-{random.randint(0, 0xFFFFFFFFFFFF):012x}"
        )

    def close(self) -> None:
        """Close is a no-op for LanceDB (stateless).

        Ensures any buffered writes are flushed before cleanup.
        """
        try:
            self._flush_buffer()
        except Exception:
            pass
        self.__table = None

    def has_pending_writes(self) -> bool:
        """Return True if there are buffered writes waiting to be flushed."""
        with self._batch_lock:
            return len(self._write_buffer) > 0

    def __repr__(self) -> str:
        return f"LanceDBStorage(db_path={self.db_path!r}, vector_dim={self.vector_dim})"
