"""
Access Tracker

Tracks memory access patterns to support reinforcement-based decay.
Frequently accessed memories decay more slowly (longer effective half-life).

Key exports:
- parseAccessMetadata   — extract accessCount/lastAccessedAt from metadata JSON
- buildUpdatedMetadata  — merge access fields into existing metadata JSON
- computeEffectiveHalfLife — compute reinforced half-life from access history
- AccessTracker         — debounced write-back tracker for batch metadata updates
"""

from __future__ import annotations

import json
import threading
import time
from dataclasses import dataclass
from typing import Callable, Dict, List, Optional, Any

# ============================================================================
# Constants
# ============================================================================

MIN_ACCESS_COUNT = 0
MAX_ACCESS_COUNT = 10_000

# Access count itself decays with a 30-day half-life
ACCESS_DECAY_HALF_LIFE_DAYS = 30

# ============================================================================
# Types
# ============================================================================

MetadataLike = Optional[str]


@dataclass
class AccessMetadata:
    accessCount: int
    lastAccessedAt: int


@dataclass
class AccessTrackerOptions:
    debounceMs: int = 5_000


# ============================================================================
# Utility
# ============================================================================


def _clamp_access_count(value: float) -> int:
    """Clamp access count to valid range, with robust type handling."""
    if value is None:
        return MIN_ACCESS_COUNT
    if not isinstance(value, (int, float)):
        return MIN_ACCESS_COUNT
    v = float(value)
    if not math.isfinite(v):
        return MIN_ACCESS_COUNT
    return min(MAX_ACCESS_COUNT, max(MIN_ACCESS_COUNT, int(v)))


import math

# ============================================================================
# Metadata Parsing
# ============================================================================


def parse_access_metadata(metadata: MetadataLike) -> AccessMetadata:
    """
    Parse access-related fields from a metadata JSON string.

    Handles: undefined, empty string, malformed JSON, negative numbers,
    numbers exceeding 10000. Always returns a valid AccessMetadata.
    """
    if metadata is None or metadata == "":
        return AccessMetadata(accessCount=0, lastAccessedAt=0)

    try:
        parsed = json.loads(metadata)
    except (json.JSONDecodeError, TypeError):
        return AccessMetadata(accessCount=0, lastAccessedAt=0)

    if not isinstance(parsed, dict):
        return AccessMetadata(accessCount=0, lastAccessedAt=0)

    raw_count: Any = parsed.get("accessCount", parsed.get("access_count"))
    if isinstance(raw_count, (int, float)) and math.isfinite(raw_count):
        count = _clamp_access_count(raw_count)
    else:
        try:
            count = _clamp_access_count(float(raw_count) if raw_count else 0)
        except (TypeError, ValueError):
            count = 0

    raw_last: Any = parsed.get("lastAccessedAt", parsed.get("last_accessed_at"))
    if isinstance(raw_last, (int, float)) and math.isfinite(raw_last) and raw_last >= 0:
        last_accessed = int(raw_last)
    else:
        try:
            last_accessed = int(float(raw_last)) if raw_last else 0
        except (TypeError, ValueError):
            last_accessed = 0

    return AccessMetadata(accessCount=count, lastAccessedAt=last_accessed)


# ============================================================================
# Metadata Building
# ============================================================================


def build_updated_metadata(existing_metadata: MetadataLike, access_delta: int) -> str:
    """
    Merge an access-count increment into existing metadata JSON.

    Preserves ALL existing fields in the metadata object — only overwrites
    `accessCount` and `lastAccessedAt`. Returns a new JSON string.
    """
    existing: Dict[str, Any] = {}

    if existing_metadata is not None and existing_metadata != "":
        try:
            parsed = json.loads(existing_metadata)
            if isinstance(parsed, dict):
                existing = dict(parsed)
        except (json.JSONDecodeError, TypeError):
            # malformed JSON — start fresh but preserve nothing
            pass

    prev = parse_access_metadata(existing_metadata)
    new_count = _clamp_access_count(prev.accessCount + access_delta)
    now_ms = int(time.time() * 1000)

    existing["accessCount"] = new_count
    existing["lastAccessedAt"] = now_ms
    existing["access_count"] = new_count
    existing["last_accessed_at"] = now_ms

    return json.dumps(existing, ensure_ascii=False)


# ============================================================================
# Effective Half-Life Computation
# ============================================================================


def compute_effective_half_life(
    base_half_life: float,
    access_count: int,
    last_accessed_at: int,
    reinforcement_factor: float,
    max_multiplier: float,
) -> float:
    """
    Compute the effective half-life for a memory based on its access history.

    The access count itself decays over time (30-day half-life for access
    freshness), so stale accesses contribute less reinforcement. The extension
    uses a logarithmic curve (log1p) to provide diminishing returns.

    Args:
        base_half_life: Base half-life in days (e.g. 30)
        access_count: Raw number of times the memory was accessed
        last_accessed_at: Timestamp (ms) of last access
        reinforcement_factor: Scaling factor for reinforcement (0 = disabled)
        max_multiplier: Hard cap: result <= baseHalfLife * maxMultiplier

    Returns:
        Effective half-life in days
    """
    # Short-circuit: no reinforcement or no accesses
    if reinforcement_factor == 0 or access_count <= 0:
        return base_half_life

    now = int(time.time() * 1000)
    days_since_last_access = max(0.0, (now - last_accessed_at) / (1000.0 * 60.0 * 60.0 * 24.0))

    # Access freshness decays exponentially with 30-day half-life
    access_freshness = math.exp(-days_since_last_access * (math.log(2) / ACCESS_DECAY_HALF_LIFE_DAYS))

    # Effective access count after freshness decay
    effective_access_count = access_count * access_freshness

    # Logarithmic extension for diminishing returns
    extension = base_half_life * reinforcement_factor * math.log1p(effective_access_count)

    result = base_half_life + extension

    # Hard cap
    cap = base_half_life * max_multiplier
    return min(result, cap)


# ============================================================================
# AccessTracker Class
# ============================================================================


class AccessTracker:
    """
    Debounced write-back tracker for memory access events.

    recordAccess() is synchronous (Map update only, no I/O). Pending deltas
    accumulate until flush() is called (or by a future scheduled callback).
    On flush, each pending entry is read via store.getById(), its metadata
    is merged with the accumulated access delta, and written back via
    store.update().
    """

    def __init__(
        self,
        store: Any,  # MemoryStore-like object
        logger: Optional[Any] = None,
        debounce_ms: int = 5_000,
    ):
        """
        Args:
            store: Must implement getById(id) -> memory and update(id, fields)
            logger: Object with .warn() and optionally .info() / .error_() methods
            debounce_ms: Milliseconds to wait before flushing (default: 5000)
        """
        self._pending: Dict[str, int] = {}
        self._retry_count: Dict[str, int] = {}
        self._max_retries = 5
        self._debounce_timer: Optional[threading.Timer] = None
        self._flush_lock = threading.Lock()
        self._flush_promise: Optional[threading.Event] = None
        self._debounce_ms = debounce_ms
        self._store = store
        self._logger = logger if logger is not None else _NoOpLogger()
        self._destroyed = False

    def record_access(self, memory_ids: List[str]) -> None:
        """
        Record one access for each of the given memory IDs.
        Synchronous — only updates the in-memory pending map.
        """
        if self._destroyed:
            return
        for id_ in memory_ids:
            self._pending[id_] = self._pending.get(id_, 0) + 1
        self._reset_timer()

    def get_pending_updates(self) -> Dict[str, int]:
        """Return a snapshot of all pending (id -> delta) entries."""
        return dict(self._pending)

    def flush(self) -> None:
        """Flush pending access deltas to the store (synchronous)."""
        self._clear_timer()

        with self._flush_lock:
            if self._flush_promise is not None:
                # Wait for in-flight flush to finish
                self._flush_promise.wait()
                if self._destroyed or len(self._pending) == 0:
                    return
                return self.flush()

            if len(self._pending) == 0:
                return

            self._flush_promise = threading.Event()
            try:
                self._do_flush()
            finally:
                self._flush_promise = None

        # If new data accumulated during flush, schedule a follow-up
        if len(self._pending) > 0:
            self._reset_timer()

    def destroy(self) -> None:
        """Tear down the tracker — cancel timers and clear pending state."""
        self._destroyed = True
        self._clear_timer()
        if len(self._pending) > 0:
            self._logger.warn(
                f"access-tracker: destroying with {len(self._pending)} pending writes"
            )
            # Best-effort synchronous flush
            try:
                self._do_flush()
            except Exception:
                pass
            self._pending.clear()
            self._retry_count.clear()
        else:
            self._pending.clear()
            self._retry_count.clear()

    # -------------------------------------------------------------------------
    # Internal helpers
    # -------------------------------------------------------------------------

    def _do_flush(self) -> None:
        """Perform the actual flush to the store (synchronous)."""
        batch = dict(self._pending)
        self._pending.clear()

        for id_, delta in batch.items():
            try:
                current = self._store.get_by_id(id_)
                if current is None:
                    self._retry_count.pop(id_, None)
                    continue

                updated_meta = build_updated_metadata(current.metadata, delta)
                self._store.update(id_, {"metadata": updated_meta})
                self._retry_count.pop(id_, None)  # success
            except Exception as err:
                retry_count = self._retry_count.get(id_, 0) + 1
                if retry_count > self._max_retries:
                    self._retry_count.pop(id_, None)
                    self._logger.error_(
                        f"access-tracker: dropping {id_[:8]} after {retry_count} failed retries"
                    )
                else:
                    self._retry_count[id_] = retry_count
                    self._pending[id_] = self._pending.get(id_, 0) + delta
                    self._logger.warn(
                        f"access-tracker: write-back failed for {id_[:8]} (attempt {retry_count}/{self._max_retries})",
                        err,
                    )

    def _reset_timer(self) -> None:
        self._clear_timer()
        if self._debounce_ms > 0:
            self._debounce_timer = threading.Timer(
                self._debounce_ms / 1000.0,
                self._scheduled_flush,
            )
            self._debounce_timer.daemon = True
            self._debounce_timer.start()

    def _clear_timer(self) -> None:
        if self._debounce_timer is not None:
            self._debounce_timer.cancel()
            self._debounce_timer = None

    def _scheduled_flush(self) -> None:
        try:
            self.flush()
        except Exception:
            pass


class _NoOpLogger:
    def warn(self, *args, **kwargs):
        pass

    def info(self, *args, **kwargs):
        pass

    def error_(self, *args, **kwargs):
        pass

    def error_(self, *args, **kwargs):
        pass
