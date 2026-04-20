"""WAL (Write-Ahead Log) Manager — episodic memory durability.

Episodic writes are first appended to an NDJSON WAL file (O_APPEND,
atomic, <1ms). A singleton background thread periodically flushes
buffered episodes to ChromaDB in batches. On startup any un-flushed
WAL entries are replayed automatically.

Reference: MemPalace WAL batcher design.

WAL file: /Users/mars/.mempalace_hermes/episodes.wal.ndjson
"""

from __future__ import annotations

import json
import logging
import os
import subprocess
import sys
import threading
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

# Global singleton batcher state (shared across all provider instances)
# CRITICAL FIX: Use RLock instead of Lock to prevent deadlock when the WAL
# batcher thread (or any caller) holds the lock and calls _read_wal() or
# _truncate_wal() internally.
_WAL_LOCK = threading.RLock()
_WAL_BATCHER: Optional[threading.Thread] = None
_WAL_BATCHER_STOP = threading.Event()

# Default palace path
_DEFAULT_PALACE_PATH = Path.home() / ".mempalace_hermes"


def _wal_path() -> Path:
    """Return the path to the episodic WAL file."""
    return _DEFAULT_PALACE_PATH / "episodes.wal.ndjson"


def _ensure_mempalace_dirs() -> None:
    """Create palace / logs directories if missing."""
    (_DEFAULT_PALACE_PATH / "logs").mkdir(parents=True, exist_ok=True)


def _append_episode_wal(entry: Dict[str, Any]) -> None:
    """Append a single episode entry to the WAL file (thread-safe)."""
    _ensure_mempalace_dirs()
    path = _wal_path()
    line = json.dumps(entry, ensure_ascii=False, default=str) + "\n"
    with _WAL_LOCK:
        with open(path, "a", encoding="utf-8") as f:
            f.write(line)


def _read_wal() -> List[Dict[str, Any]]:
    """Read all entries from the WAL file (thread-safe)."""
    path = _wal_path()
    if not path.exists():
        return []
    entries: List[Dict[str, Any]] = []
    with _WAL_LOCK:
        try:
            with open(path, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        entries.append(json.loads(line))
                    except json.JSONDecodeError:
                        logger.warning("Corrupt WAL line skipped: %s", line[:200])
        except Exception as e:
            logger.warning("Failed to read WAL: %s", e)
    return entries


def _truncate_wal() -> None:
    """Clear the WAL file after successful flush."""
    path = _wal_path()
    with _WAL_LOCK:
        try:
            if path.exists():
                path.write_text("")
        except Exception as e:
            logger.warning("Failed to truncate WAL: %s", e)


class _WALBatcher(threading.Thread):
    """
    Singleton background thread that periodically rotates the episodic WAL.

    CRITICAL FIX: After memory system consolidation, ChromaDB is no longer
    used as a redundant store. All episodic data is persisted through:
      1. LanceDB (primary storage, via HermesMemoryProvider.store)
      2. This WAL file (text backup, human-readable)

    The batcher now simply rotates the WAL file to prevent unbounded growth,
    instead of flushing to ChromaDB. Archived WAL files are kept for
    manual inspection / disaster recovery.
    """

    _FLUSH_INTERVAL = 30.0   # seconds
    _MAX_BATCH = 500         # episodes per batch

    def __init__(self):
        super().__init__(daemon=True, name="hermes-wal-batcher")
        self._stop_event = threading.Event()

    def request_stop(self):
        self._stop_event.set()

    def run(self):
        logger.info("WAL batcher started (rotation interval=%ss, ChromaDB flush DISABLED)", self._FLUSH_INTERVAL)
        while not self._stop_event.is_set():
            # Wait in small slices so we react quickly to stop()
            for _ in range(int(self._FLUSH_INTERVAL)):
                if self._stop_event.is_set():
                    break
                time.sleep(1.0)

            entries = _read_wal()
            if not entries:
                continue

            # CRITICAL FIX: No longer flush to ChromaDB.
            # LanceDB is the single source of truth for episodic storage.
            # WAL is kept as a human-readable text backup only.
            # Just truncate after a reasonable retention window.
            if len(entries) > self._MAX_BATCH:
                _ensure_mempalace_dirs()
                with _WAL_LOCK:
                    try:
                        # Archive old entries instead of deleting
                        archive_path = _wal_path().parent / f"episodes.wal.archive.{datetime.now().strftime('%Y%m%d_%H%M%S')}.ndjson"
                        with open(archive_path, "w", encoding="utf-8") as f:
                            for e in entries:
                                f.write(json.dumps(e, ensure_ascii=False, default=str) + "\n")
                        _truncate_wal()
                        logger.info("WAL rotated: %d entries archived to %s", len(entries), archive_path.name)
                    except Exception as e:
                        logger.warning("WAL rotation failed: %s", e)
        logger.info("WAL batcher stopped")


def _start_wal_batcher() -> None:
    """Start the singleton WAL batcher thread if it isn't already running."""
    global _WAL_BATCHER
    with _WAL_LOCK:
        if _WAL_BATCHER is None or not _WAL_BATCHER.is_alive():
            _WAL_BATCHER = _WALBatcher()
            _WAL_BATCHER.start()


def _stop_wal_batcher(timeout: float = 5.0) -> None:
    """Signal the WAL batcher to stop and wait for clean shutdown."""
    global _WAL_BATCHER
    batcher = None
    with _WAL_LOCK:
        if _WAL_BATCHER and _WAL_BATCHER.is_alive():
            batcher = _WAL_BATCHER
            batcher.request_stop()
    # CRITICAL FIX: join() must happen OUTSIDE the lock to avoid deadlock
    # with the batcher thread which acquires _WAL_LOCK during its run cycle.
    if batcher:
        batcher.join(timeout=timeout)


def wal_write_episode(
    role: str,
    content: str,
    speaker: str = "hermes",
    topic: str = "general",
    language: str = "unknown",
    importance: float = 0.5,
    session_id: str = "global",
    scope: str = "global",
) -> str:
    """
    Write an episode to WAL and return the episode ID.

    This is the main entry point for episodic persistence.
    The episode is immediately appended to WAL (fast, atomic)
    and the WAL batcher will flush to ChromaDB asynchronously.
    """
    timestamp = datetime.now().isoformat()
    episode_id = f"{session_id}:{timestamp}"

    entry = {
        "episode_id": episode_id,
        "role": role,
        "speaker": speaker,
        "content": content[:5000],
        "timestamp": timestamp,
        "topic": topic,
        "language": language,
        "importance": importance,
        "session_id": session_id,
        "scope": scope,
    }

    _append_episode_wal(entry)
    return episode_id


def wal_replay(working_memory_instance=None) -> List[Dict[str, Any]]:
    """
    Replay WAL entries on startup for crash recovery.

    If working_memory_instance is provided, episodes are also
    restored to working memory (using _add_turn_no_persist to avoid
    re-persistence).

    Returns the list of replayed entries.
    """
    entries = _read_wal()
    if not entries:
        logger.info("WAL replay: no entries to replay")
        return []

    logger.info("WAL replay: found %d entries", len(entries))

    if working_memory_instance is not None:
        # Restore to working memory without re-persistence
        for entry in entries:
            try:
                working_memory_instance._add_turn_no_persist(
                    role=entry.get("role", "user"),
                    content=entry.get("content", ""),
                    speaker=entry.get("speaker", "unknown"),
                    topic=entry.get("topic", ""),
                    importance=entry.get("importance", 0.5),
                    session_id=entry.get("session_id", "global"),
                    scope=entry.get("scope", "global"),
                    timestamp=entry.get("timestamp", ""),
                )
            except Exception as e:
                logger.warning("WAL replay: failed to restore entry %s: %s",
                              entry.get("episode_id", "?"), e)

    return entries


def wal_pending_count() -> int:
    """Return the number of pending (un-flushed) WAL entries."""
    return len(_read_wal())


def wal_truncate() -> None:
    """Manually truncate WAL (after full ChromaDB sync)."""
    _truncate_wal()


# =============================================================================
# WALManager Class — wraps WAL functions for HermesMemoryProvider
# =============================================================================

class WALManager:
    """
    WAL (Write-Ahead Log) Manager that wraps WAL functions.

    Used by HermesMemoryProvider to provide a clean OOP interface to WAL operations.
    
    CRITICAL FIX: Previously flush() was a no-op and shutdown() did not stop the
    background batcher, so HermesMemoryProvider.shutdown() left the batcher running
    which could truncate WAL entries after the provider had shut down.
    """

    def __init__(
        self,
        storage=None,
        flush_interval: float = 30.0,
        wal_path: Optional[str] = None,
        chroma_dir: Optional[str] = None,
    ):
        """Initialize WAL manager.

        Args:
            storage: Storage instance (LanceDBStorage)
            flush_interval: Flush interval in seconds
            wal_path: Optional path to WAL file
            chroma_dir: Optional path to ChromaDB directory
        """
        self._storage = storage
        self._flush_interval = flush_interval
        self._wal_path = wal_path
        self._chroma_dir = chroma_dir
        # Ensure batcher is running (idempotent)
        _start_wal_batcher()
    
    def append(
        self,
        content: str,
        speaker: str = "hermes",
        role: str = "user",
        topic: str = "general",
        importance: float = 0.5,
        scope: str = "global",
    ) -> str:
        """Append an episode to WAL.
        
        Args:
            content: Episode content
            speaker: Speaker identifier
            role: Role (user/assistant)
            topic: Topic category
            importance: Importance score (0-1)
            scope: Scope string
            
        Returns:
            Episode ID
        """
        return wal_write_episode(
            role=role,
            content=content,
            speaker=speaker,
            topic=topic,
            importance=importance,
            scope=scope,
        )
    
    def flush(self) -> None:
        """Flush WAL entries: if WAL is large, trigger an immediate archive rotation."""
        entries = _read_wal()
        if len(entries) > _WALBatcher._MAX_BATCH:
            _ensure_mempalace_dirs()
            with _WAL_LOCK:
                try:
                    archive_path = _wal_path().parent / f"episodes.wal.archive.{datetime.now().strftime('%Y%m%d_%H%M%S')}.ndjson"
                    with open(archive_path, "w", encoding="utf-8") as f:
                        for e in entries:
                            f.write(json.dumps(e, ensure_ascii=False, default=str) + "\n")
                    _truncate_wal()
                    logger.info("WAL flushed: %d entries archived to %s", len(entries), archive_path.name)
                except Exception as e:
                    logger.warning("WAL flush failed: %s", e)
    
    def replay(self, working_memory_instance=None) -> List[Dict[str, Any]]:
        """Replay WAL entries for crash recovery."""
        return wal_replay(working_memory_instance=working_memory_instance)
    
    def pending_count(self) -> int:
        """Return number of pending (un-flushed) entries."""
        return wal_pending_count()
    
    def truncate(self) -> None:
        """Truncate WAL after full sync."""
        wal_truncate()
    
    def shutdown(self) -> None:
        """Stop the WAL batcher thread cleanly."""
        _stop_wal_batcher()


# Auto-start WAL batcher when module is imported
_start_wal_batcher()
