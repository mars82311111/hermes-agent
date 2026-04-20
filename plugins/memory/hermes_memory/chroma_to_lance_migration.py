"""
ChromaDB → LanceDB Migration Script

Uses ChromaDB Python client to read embeddings (not raw SQL),
then imports to LanceDB.

Usage:
    python -m hermes_memory.chroma_to_lance_migration
"""

from __future__ import annotations

import json
import logging
import os
import sqlite3
import sys
import uuid
from pathlib import Path
from typing import Any, List, Optional

import lancedb

logger = logging.getLogger(__name__)


def _get_table_names(db) -> list:
    """COMPAT: LanceDB list_tables() returns ListTablesResponse in 0.30+.
    table_names() returns list but is deprecated."""
    try:
        resp = db.list_tables()
        if hasattr(resp, "tables"):
            return resp.tables
        return list(resp)
    except Exception:
        return db.table_names()


# Constants
CHROMA_PATH = "/Users/mars/.mempalace_hermes/palace"
LANCE_PATH = os.path.expanduser("~/.hermes_memory/lance")
VECTOR_DIM = 384  # ChromaDB default embedding dimension

# MemoryEntry schema fields
MEMORY_SCHEMA = {
    "id": "string",
    "text": "string",
    "vector": f"fixed_size_list<float, {VECTOR_DIM}>",
    "category": "string",
    "scope": "string",
    "importance": "float",
    "timestamp": "int64",
    "metadata": "string",  # JSON
}


# ============================================================================
# LanceDB Storage
# ============================================================================

class LanceDBStorage:
    """LanceDB storage layer with FTS support."""

    def __init__(self, db_path: str, vector_dim: int = 384):
        self.db_path = db_path
        self.vector_dim = vector_dim
        self._db: Optional[Any] = None
        self._table: Optional[Any] = None
        self._fts_indexed = False
        self._ensure_db()

    def _ensure_db(self):
        """Ensure database and table exist."""
        os.makedirs(self.db_path, exist_ok=True)
        self._db = lancedb.connect(self.db_path)

        # Try to open existing table or create new
        table_names = _get_table_names(self._db)
        if "memories" in table_names:
            self._table = self._db.open_table("memories")
        else:
            self._create_table()

    def _create_table(self):
        """Create memories table with schema."""
        import pyarrow as pa

        schema = pa.schema([
            ("id", pa.string()),
            ("text", pa.string()),
            ("vector", pa.list_(pa.float32(), self.vector_dim)),
            ("category", pa.string()),
            ("scope", pa.string()),
            ("importance", pa.float32()),
            ("timestamp", pa.int64()),
            ("metadata", pa.string()),
        ])

        # Create table with dummy row (idempotent)
        import numpy as np
        dummy = {
            "id": "__dummy__",
            "text": "",
            "vector": np.zeros(self.vector_dim, dtype=np.float32).tolist(),
            "category": "other",
            "scope": "global",
            "importance": 0.0,
            "timestamp": 0,
            "metadata": "{}",
        }
        self._table = self._db.create_table("memories", schema=schema, data=[dummy])
        # Delete dummy row
        self._table.delete('id = "__dummy__"')
        logger.info("Created LanceDB table with %d-dim vectors", self.vector_dim)

    @property
    def table(self):
        return self._table

    def count(self) -> int:
        if self._table is None:
            return 0
        try:
            return self._table.count_rows()
        except Exception:
            return 0

    def store(self, entry: dict) -> None:
        """Store a single memory entry."""
        import numpy as np
        row = {
            "id": entry["id"],
            "text": entry["text"],
            "vector": np.array(entry["vector"], dtype=np.float32).tolist() if entry["vector"] else np.zeros(self.vector_dim, dtype=np.float32).tolist(),
            "category": entry.get("category", "other"),
            "scope": entry.get("scope", "global"),
            "importance": float(entry.get("importance", 0.5)),
            "timestamp": int(entry.get("timestamp", 0)),
            "metadata": json.dumps(entry.get("metadata", {}), ensure_ascii=False),
        }
        self._table.add([row])

    def search(self, query_vector: List[float], limit: int = 10) -> List[dict]:
        """Vector search."""
        result = (
            self._table.search(query_vector, vector_column_name="vector")
            .limit(limit)
            .to_list()
        )
        return result

    def create_fts_index(self) -> None:
        """Create FTS index for BM25 search."""
        if self._fts_indexed:
            return
        try:
            self._table.create_fts_index("text")
            self._fts_indexed = True
            logger.info("Created FTS index")
        except Exception as e:
            logger.warning("Failed to create FTS index: %s", e)


# ============================================================================
# ChromaDB Reader
# ============================================================================

def read_chroma_data() -> tuple:
    """
    Read all data from ChromaDB using the Python client.
    Returns (documents, embeddings, metadatas).
    """
    import chromadb
    from chromadb.config import Settings

    client = chromadb.PersistentClient(
        path=CHROMA_PATH,
        settings=Settings(anonymized_telemetry=False)
    )

    col = client.get_collection("episodes")
    data = col.get(include=["documents", "embeddings", "metadatas"])

    docs = list(data["documents"]) if data["documents"] is not None else []
    embs = list(data["embeddings"]) if data["embeddings"] is not None else []
    metas = list(data["metadatas"]) if data["metadatas"] is not None else []

    return docs, embs, metas


# ============================================================================
# Migration
# ============================================================================

def migrate_all():
    """Migrate all data from ChromaDB to LanceDB."""
    print("=" * 60)
    print("ChromaDB → LanceDB Migration")
    print("=" * 60)

    # 1. Read from ChromaDB
    print("\n1. Reading from ChromaDB...")
    docs, embs, metas = read_chroma_data()
    print(f"   Total records: {len(docs)}")
    if len(docs) == 0:
        print("   No data to migrate!")
        return {"status": "no_data"}

    # Show sample
    print(f"\n   Sample doc: {docs[0][:80]}...")
    print(f"   Sample embedding shape: {len(embs[0]) if embs else 'None'}")

    # 2. Initialize LanceDB
    print("\n2. Initializing LanceDB...")
    lance = LanceDBStorage(db_path=LANCE_PATH, vector_dim=VECTOR_DIM)
    current_count = lance.count()
    print(f"   Current LanceDB records: {current_count}")

    # 3. Import data
    print("\n3. Importing data...")
    imported = 0
    skipped = 0

    for i, (doc, emb, meta) in enumerate(zip(docs, embs, metas)):
        if not doc or not doc.strip():
            skipped += 1
            continue

        # Generate stable ID
        entry_id = str(uuid.uuid5(uuid.NAMESPACE_DNS, doc[:200]))

        # Parse metadata
        category = "other"
        scope = "global"
        importance = 0.5
        timestamp = 0
        speaker = ""
        topic = "general"

        if meta:
            cat = meta.get("hall", "")
            if cat and cat in ("preference", "fact", "decision", "entity", "reflection", "other"):
                category = cat
            scope = meta.get("room", "global") or "global"
            imp = meta.get("importance")
            if imp is not None:
                try:
                    importance = max(0.0, min(1.0, float(imp)))
                except (TypeError, ValueError):
                    pass
            ts = meta.get("timestamp")
            if ts:
                try:
                    timestamp = int(ts)
                except (TypeError, ValueError):
                    pass
            speaker = meta.get("speaker", "")
            topic = meta.get("topic", "general")

        entry = {
            "id": entry_id,
            "text": doc,
            "vector": list(emb) if emb is not None else None,
            "category": category,
            "scope": scope,
            "importance": importance,
            "timestamp": timestamp,
            "metadata": json.dumps({
                "speaker": speaker,
                "topic": topic,
                "importance": importance,
                "chroma_index": i,
            }, ensure_ascii=False),
        }

        try:
            lance.store(entry)
            imported += 1
        except Exception as e:
            logger.warning("Failed to store entry %d: %s", i, e)
            skipped += 1

        if (i + 1) % 100 == 0:
            print(f"   Progress: {i + 1}/{len(docs)}")

    print(f"\n   Imported: {imported}")
    print(f"   Skipped: {skipped}")

    # 4. Create FTS index
    print("\n4. Creating FTS index...")
    lance.create_fts_index()

    # 5. Verify
    print("\n5. Verification...")
    final_count = lance.count()
    print(f"   Final LanceDB records: {final_count}")

    # Test search
    if final_count > 0 and embs and embs[0] is not None:
        test_results = lance.search(embs[0], limit=3)
        print(f"   Test search: {len(test_results)} results")
        if test_results:
            print(f"   Top result: {test_results[0]['text'][:60]}...")

    print("\n" + "=" * 60)
    print("Migration complete!")
    print("=" * 60)

    return {
        "status": "success",
        "imported": imported,
        "skipped": skipped,
        "total": len(docs),
        "lance_records": final_count,
    }


if __name__ == "__main__":
    import time
    start = time.time()
    result = migrate_all()
    elapsed = time.time() - start
    print(f"\nElapsed: {elapsed:.1f}s")
    print(f"Result: {result}")
