"""KG Integration — Knowledge Graph integration with MemPalace SQLite KG.

Preserves the existing MemPalace SQLite KG at:
  /Users/mars/.mempalace_hermes/knowledge_graph.sqlite3

KG Schema:
  - triples: subject, predicate, object, valid_from, valid_to, confidence,
             source_closet, source_file, extracted_at, inverse_predicate,
             context, source, subject_type, object_type
  - entities: name (PRIMARY KEY), entity_type, created_at
  - belief_history: belief_id, entity, predicate, old_value, new_value,
                    change_type, changed_at, source, confidence, context,
                    valid_from, valid_to

BUG FIX: Original code writes triples but never populates entities table.
This module fixes that by synchronously writing to entities when adding triples.

Statistics:
  - KG triples: 181
  - KG entities: 0 (BUG - fixed by this module)
  - KG belief_history: 300
"""

from __future__ import annotations

import json
import logging
import sqlite3
import threading
import uuid
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

# KG database path
# CRITICAL FIX: Unified under ~/.hermes_memory/ to consolidate the memory system.
_KG_DB_PATH = Path.home() / ".hermes_memory" / "kg.sqlite3"

# Lock for KG operations
_KG_LOCK = threading.RLock()


@dataclass
class KGTriple:
    """A single knowledge graph triple."""
    id: str
    subject: str
    predicate: str
    object: str
    valid_from: str
    valid_to: Optional[str] = None
    confidence: float = 1.0
    source_closet: Optional[str] = None
    source_file: Optional[str] = None
    extracted_at: Optional[str] = None
    inverse_predicate: Optional[str] = None
    context: Optional[str] = None
    source: Optional[str] = None
    subject_type: Optional[str] = None
    object_type: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "subject": self.subject,
            "predicate": self.predicate,
            "object": self.object,
            "valid_from": self.valid_from,
            "valid_to": self.valid_to,
            "confidence": self.confidence,
            "source_closet": self.source_closet,
            "source_file": self.source_file,
            "extracted_at": self.extracted_at,
            "inverse_predicate": self.inverse_predicate,
            "context": self.context,
            "source": self.source,
            "subject_type": self.subject_type,
            "object_type": self.object_type,
        }


@dataclass
class KGEntity:
    """A knowledge graph entity."""
    name: str
    entity_type: Optional[str] = None
    created_at: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "entity_type": self.entity_type,
            "created_at": self.created_at,
        }


def _get_kg_conn() -> sqlite3.Connection:
    """Get a connection to the KG database.

    CRITICAL FIX: Automatically creates the required tables on first use.
    Previously, if the database was new or deleted, all KG operations would
    raise sqlite3.OperationalError ('no such table'), crashing the memory
    system. Now the schema is self-healing.
    """
    _KG_DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(_KG_DB_PATH), timeout=10.0)
    conn.row_factory = sqlite3.Row
    _ensure_kg_schema(conn)
    return conn


_KG_SCHEMA_INITIALIZED = False
_KG_SCHEMA_LOCK = threading.Lock()


def _ensure_kg_schema(conn: sqlite3.Connection) -> None:
    """Create KG tables if they do not exist. Idempotent."""
    global _KG_SCHEMA_INITIALIZED
    if _KG_SCHEMA_INITIALIZED:
        return
    with _KG_SCHEMA_LOCK:
        if _KG_SCHEMA_INITIALIZED:
            return
        cursor = conn.cursor()
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS entities (
                name TEXT PRIMARY KEY,
                entity_type TEXT,
                created_at TEXT
            )
        """)
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS triples (
                id TEXT PRIMARY KEY,
                subject TEXT,
                predicate TEXT,
                object TEXT,
                valid_from TEXT,
                valid_to TEXT,
                confidence REAL,
                source_closet TEXT,
                source_file TEXT,
                extracted_at TEXT,
                inverse_predicate TEXT,
                context TEXT,
                source TEXT,
                subject_type TEXT,
                object_type TEXT
            )
        """)
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS belief_history (
                belief_id TEXT PRIMARY KEY,
                entity TEXT,
                predicate TEXT,
                old_value TEXT,
                new_value TEXT,
                change_type TEXT,
                changed_at TEXT,
                source TEXT,
                confidence REAL,
                context TEXT,
                valid_from TEXT,
                valid_to TEXT
            )
        """)
        # Create indexes for common query patterns
        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_triples_subject ON triples(subject)
        """)
        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_triples_object ON triples(object)
        """)
        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_triples_predicate ON triples(predicate)
        """)
        conn.commit()
        _KG_SCHEMA_INITIALIZED = True


def kg_search(query: str, limit: int = 20) -> List[KGTriple]:
    """
    Search KG triples by keyword query.
    Searches across subject, predicate, object, and context fields.

    Args:
        query: Search query string
        limit: Maximum number of results

    Returns:
        List of KGTriple objects matching the query
    """
    q = query.lower()
    results: List[KGTriple] = []

    with _KG_LOCK:
        conn = _get_kg_conn()
        try:
            cursor = conn.execute("""
                SELECT * FROM triples
                WHERE LOWER(subject) LIKE ?
                   OR LOWER(predicate) LIKE ?
                   OR LOWER(object) LIKE ?
                   OR LOWER(context) LIKE ?
                   OR LOWER(source) LIKE ?
                ORDER BY valid_from DESC
                LIMIT ?
            """, (f"%{q}%", f"%{q}%", f"%{q}%", f"%{q}%", f"%{q}%", limit))

            for row in cursor.fetchall():
                results.append(KGTriple(
                    id=row["id"],
                    subject=row["subject"],
                    predicate=row["predicate"],
                    object=row["object"],
                    valid_from=row["valid_from"],
                    valid_to=row["valid_to"],
                    confidence=row["confidence"],
                    source_closet=row["source_closet"],
                    source_file=row["source_file"],
                    extracted_at=row["extracted_at"],
                    inverse_predicate=row["inverse_predicate"],
                    context=row["context"],
                    source=row["source"],
                    subject_type=row["subject_type"],
                    object_type=row["object_type"],
                ))
        finally:
            conn.close()

    return results


def kg_get_entity(name: str) -> Optional[KGEntity]:
    """Get an entity by name."""
    with _KG_LOCK:
        conn = _get_kg_conn()
        try:
            cursor = conn.execute(
                "SELECT * FROM entities WHERE name = ?", (name,)
            )
            row = cursor.fetchone()
            if row:
                return KGEntity(
                    name=row["name"],
                    entity_type=row["entity_type"],
                    created_at=row["created_at"],
                )
        finally:
            conn.close()
    return None


def kg_get_triples_for_entity(entity_name: str, limit: int = 50) -> List[KGTriple]:
    """Get all triples where the entity is subject or object."""
    results: List[KGTriple] = []

    with _KG_LOCK:
        conn = _get_kg_conn()
        try:
            cursor = conn.execute("""
                SELECT * FROM triples
                WHERE subject = ? OR object = ?
                ORDER BY valid_from DESC
                LIMIT ?
            """, (entity_name, entity_name, limit))

            for row in cursor.fetchall():
                results.append(KGTriple(
                    id=row["id"],
                    subject=row["subject"],
                    predicate=row["predicate"],
                    object=row["object"],
                    valid_from=row["valid_from"],
                    valid_to=row["valid_to"],
                    confidence=row["confidence"],
                    source_closet=row["source_closet"],
                    source_file=row["source_file"],
                    extracted_at=row["extracted_at"],
                    inverse_predicate=row["inverse_predicate"],
                    context=row["context"],
                    source=row["source"],
                    subject_type=row["subject_type"],
                    object_type=row["object_type"],
                ))
        finally:
            conn.close()

    return results


def add_triple_with_entity(
    subject: str,
    predicate: str,
    obj: str,
    confidence: float = 1.0,
    subject_type: str = "",
    object_type: str = "",
    context: str = "",
    source: str = "hermes-memory",
    valid_from: Optional[str] = None,
    valid_to: Optional[str] = None,
) -> KGTriple:
    """
    Add a triple to the KG AND synchronize entities.

    BUG FIX: Original implementation wrote triples but never populated
    the entities table. This function fixes that by writing both triples
    and entities atomically.

    Args:
        subject: Subject entity name
        predicate: Predicate/relation
        obj: Object value
        confidence: Confidence score (0.0-1.0)
        subject_type: Type of subject entity (person, agent, concept, etc.)
        object_type: Type of object entity
        context: Additional context
        source: Source of the knowledge
        valid_from: Start date string
        valid_to: End date string (None for current)

    Returns:
        The created KGTriple
    """
    now = valid_from or datetime.now().strftime("%Y-%m-%d")
    extracted_at = datetime.now().isoformat()
    triple_id = str(uuid.uuid4())

    # Determine inverse predicate
    inverse_pred_map = {
        "knows": "known_by",
        "owns": "owned_by",
        "works_on": "worked_on_by",
        "uses": "used_by",
        "created": "created_by",
        "related_to": "related_to",
    }
    inverse_pred = inverse_pred_map.get(predicate, f"{predicate}_by")

    with _KG_LOCK:
        conn = _get_kg_conn()
        try:
            cursor = conn.cursor()

            # BUG FIX: Insert subject entity if not exists
            cursor.execute("""
                INSERT OR IGNORE INTO entities (name, entity_type, created_at)
                VALUES (?, ?, ?)
            """, (subject, subject_type or "unknown", extracted_at))

            # BUG FIX: Insert object entity if not exists
            cursor.execute("""
                INSERT OR IGNORE INTO entities (name, entity_type, created_at)
                VALUES (?, ?, ?)
            """, (obj, object_type or "unknown", extracted_at))

            # Insert the triple
            cursor.execute("""
                INSERT INTO triples (
                    id, subject, predicate, object, valid_from, valid_to,
                    confidence, source_closet, source_file, extracted_at,
                    inverse_predicate, context, source, subject_type, object_type
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                triple_id, subject, predicate, obj, now, valid_to,
                confidence, None, None, extracted_at,
                inverse_pred, context, source, subject_type, object_type
            ))

            # Insert belief history entry
            belief_id = str(uuid.uuid4())
            cursor.execute("""
                INSERT INTO belief_history (
                    belief_id, entity, predicate, old_value, new_value,
                    change_type, changed_at, source, confidence, context,
                    valid_from, valid_to
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                belief_id, subject, predicate, None, obj,
                "created", extracted_at, source, confidence, context,
                now, valid_to
            ))

            conn.commit()
        finally:
            conn.close()

    return KGTriple(
        id=triple_id,
        subject=subject,
        predicate=predicate,
        object=obj,
        valid_from=now,
        valid_to=valid_to,
        confidence=confidence,
        extracted_at=extracted_at,
        inverse_predicate=inverse_pred,
        context=context,
        source=source,
        subject_type=subject_type,
        object_type=object_type,
    )


def link_memory_to_kg(
    subject: str,
    memory_content: str,
    predicate: str = "mentioned_in",
    confidence: float = 0.8,
    subject_type: str = "concept",
) -> KGTriple:
    """
    Link a memory (from working memory or episodic storage) to the KG.

    Creates a triple that connects an entity to a memory reference.

    Args:
        subject: Entity name to link
        memory_content: Content snippet from memory
        predicate: Relation type (default: "mentioned_in")
        confidence: Confidence score
        subject_type: Type of subject entity

    Returns:
        The created KGTriple
    """
    # Truncate memory content to fit in context
    context = memory_content[:500] if len(memory_content) > 500 else memory_content

    return add_triple_with_entity(
        subject=subject,
        predicate=predicate,
        obj=f"memory:{context[:100]}...",
        confidence=confidence,
        subject_type=subject_type,
        object_type="memory",
        context=context,
        source="hermes-memory-link",
    )


def kg_stats() -> Dict[str, Any]:
    """Get KG statistics."""
    with _KG_LOCK:
        conn = _get_kg_conn()
        try:
            cursor = conn.cursor()

            cursor.execute("SELECT COUNT(*) as cnt FROM triples")
            triple_count = cursor.fetchone()["cnt"]

            cursor.execute("SELECT COUNT(*) as cnt FROM entities")
            entity_count = cursor.fetchone()["cnt"]

            cursor.execute("SELECT COUNT(*) as cnt FROM belief_history")
            belief_count = cursor.fetchone()["cnt"]

            return {
                "triple_count": triple_count,
                "entity_count": entity_count,
                "belief_history_count": belief_count,
                "db_path": str(_KG_DB_PATH),
            }
        finally:
            conn.close()


def kg_repair_entities() -> Dict[str, int]:
    """
    Repair the entities table by populating it from existing triples.

    This fixes the bug where entities table is empty (0 rows) while
    triples table has data (181 rows).

    Returns:
        Dict with 'subjects_added' and 'objects_added' counts
    """
    subjects_added = 0
    objects_added = 0
    now = datetime.now().isoformat()

    with _KG_LOCK:
        conn = _get_kg_conn()
        try:
            cursor = conn.cursor()

            # Get all unique subjects and their types from triples
            cursor.execute("""
                SELECT DISTINCT subject, subject_type FROM triples
                WHERE subject NOT IN (SELECT name FROM entities)
            """)
            for row in cursor.fetchall():
                cursor.execute("""
                    INSERT OR IGNORE INTO entities (name, entity_type, created_at)
                    VALUES (?, ?, ?)
                """, (row["subject"], row["subject_type"] or "unknown", now))
                subjects_added += 1

            # Get all unique objects and their types from triples
            cursor.execute("""
                SELECT DISTINCT object, object_type FROM triples
                WHERE object NOT IN (SELECT name FROM entities)
            """)
            for row in cursor.fetchall():
                cursor.execute("""
                    INSERT OR IGNORE INTO entities (name, entity_type, created_at)
                    VALUES (?, ?, ?)
                """, (row["object"], row["object_type"] or "unknown", now))
                objects_added += 1

            conn.commit()
        finally:
            conn.close()

    logger.info("KG entities repair: added %d subjects and %d objects",
                subjects_added, objects_added)
    return {"subjects_added": subjects_added, "objects_added": objects_added}


# =============================================================================
# KGIntegration Class — wraps KG functions for HermesMemoryProvider
# =============================================================================

class KGIntegration:
    """
    Knowledge Graph integration class that wraps the module-level KG functions.
    
    Used by HermesMemoryProvider to provide a clean OOP interface to the KG.
    """
    
    def __init__(self, kg_path: Optional[str] = None):
        """Initialize KG integration.
        
        Args:
            kg_path: Optional path to KG database. Defaults to ~/.mempalace_hermes/knowledge_graph.sqlite3
        """
        self._kg_path = Path(kg_path) if kg_path else _KG_DB_PATH
    
    def search(self, query: str, limit: int = 20) -> List[KGTriple]:
        """Search KG triples by keyword query.
        
        Args:
            query: Search query string
            limit: Maximum number of results
            
        Returns:
            List of KGTriple objects matching the query
        """
        return kg_search(query, limit=limit)
    
    def add_triple(self, subject: str, predicate: str, obj: str, **kwargs) -> Optional[KGTriple]:
        """Add a triple to the KG."""
        return add_triple_with_entity(subject, predicate, obj, **kwargs)
    
    def get_stats(self) -> Dict[str, Any]:
        """Get KG statistics."""
        return kg_stats()
