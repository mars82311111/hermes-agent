"""Working Memory — In-process LRU cache of recent conversation turns.

Thread-safe, zero-latency in-memory cache with 50-turn capacity.
Used for immediate context during active conversations.

Reference: MemPalace L-WM (Working Memory) layer.
"""

from __future__ import annotations

import threading
from collections import OrderedDict
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional


MAX_WORKING_MEMORY_TURNS = 50


@dataclass
class Turn:
    """A single conversation turn in working memory."""
    role: str          # 'user' or 'assistant'
    speaker: str       # 'mars', 'hermes', 'feishu', etc.
    content: str
    timestamp: str     # ISO format
    topic: str = ""    # auto-detected topic tag
    importance: float = 0.5  # 0.0-1.0, default medium
    session_id: str = "global"  # session identifier
    scope: str = "global"  # memory scope for access control

    def to_dict(self) -> Dict[str, Any]:
        return {
            "role": self.role,
            "speaker": self.speaker,
            "content": self.content,
            "timestamp": self.timestamp,
            "topic": self.topic,
            "importance": self.importance,
            "session_id": self.session_id,
            "scope": self.scope,
        }


class WorkingMemory:
    """
    In-memory LRU cache of the most recent conversation turns.
    Zero latency — no disk access.

    Thread-safe for concurrent access from the agent loop.
    """

    def __init__(self, max_turns: int = MAX_WORKING_MEMORY_TURNS):
        self._max = max_turns
        self._turns: OrderedDict[str, Turn] = OrderedDict()
        self._lock = threading.RLock()
        self._session_id: Optional[str] = None

    def set_session(self, session_id: str) -> None:
        """Clear working memory when starting a new session."""
        with self._lock:
            if session_id != self._session_id:
                self._turns.clear()
                self._session_id = session_id

    def add_turn(
        self,
        role: str,
        content: str,
        speaker: str = "hermes",
        topic: str = "",
        importance: float = 0.5,
        session_id: str = "global",
        scope: str = "global",
    ) -> str:
        """Add a turn to working memory.

        Returns the turn_id.
        """
        timestamp = datetime.now().isoformat()
        detected_topic = topic or self._detect_topic(content) or "general"

        with self._lock:
            turn_id = f"{session_id}:{timestamp}"
            turn = Turn(
                role=role,
                speaker=speaker,
                content=content[:5000],  # cap long content
                timestamp=timestamp,
                topic=detected_topic,
                importance=importance,
                session_id=session_id,
                scope=scope,
            )
            self._turns[turn_id] = turn
            # Evict oldest if over capacity
            while len(self._turns) > self._max:
                self._turns.popitem(last=False)

        return turn_id

    def _add_turn_no_persist(
        self,
        role: str,
        content: str,
        speaker: str = "hermes",
        topic: str = "",
        importance: float = 0.5,
        session_id: str = "global",
        scope: str = "global",
        timestamp: str = "",
    ) -> str:
        """
        Internal add_turn without persistence.
        Used by WAL replay to rebuild working memory from episodic storage
        without triggering re-persistence (which would cause infinite loop).
        """
        ts = timestamp or datetime.now().isoformat()
        detected_topic = topic or self._detect_topic(content) or "general"

        with self._lock:
            turn_id = f"{session_id}:{ts}"
            turn = Turn(
                role=role,
                speaker=speaker,
                content=content[:5000],
                timestamp=ts,
                topic=detected_topic,
                importance=importance,
                session_id=session_id,
                scope=scope,
            )
            self._turns[turn_id] = turn
            while len(self._turns) > self._max:
                self._turns.popitem(last=False)
            return turn_id

    def get_recent(self, n: int = 10, session_id: Optional[str] = None) -> List[Turn]:
        """Return the n most recent turns (newest last).

        If session_id is provided, only return turns from that session.
        """
        with self._lock:
            items = list(self._turns.values())

        if session_id is not None:
            items = [t for t in items if t.session_id == session_id]

        return items[-n:] if n < len(items) else items

    def search(self, query: str, n: int = 5, session_id: Optional[str] = None) -> List[Turn]:
        """Full-text search within working memory turns."""
        q = query.lower()
        with self._lock:
            # Score by relevance (simple keyword match)
            scored = []
            for turn in self._turns.values():
                if session_id is not None and turn.session_id != session_id:
                    continue
                score = 0
                text = turn.content.lower()
                for word in q.split():
                    if word in text:
                        score += 1
                if score > 0:
                    scored.append((score, turn))
            scored.sort(key=lambda x: x[0], reverse=True)
            return [t for _, t in scored[:n]]

    def get_context_for_wakeup(self, n: int = 20) -> str:
        """Format recent turns as a string for injection at wakeup."""
        turns = self.get_recent(n)
        if not turns:
            return ""
        lines = ["## Working Memory (recent turns)\n"]
        for t in turns:
            lines.append(f"[{t.speaker}/{t.role}] {t.content[:300]}")
            if len(t.content) > 300:
                lines[-1] += "..."
        return "\n".join(lines)

    def stats(self) -> Dict[str, Any]:
        """Return working memory stats."""
        with self._lock:
            return {
                "turn_count": len(self._turns),
                "max_turns": self._max,
                "session_id": self._session_id,
            }

    def clear(self) -> None:
        """Clear all turns from working memory."""
        with self._lock:
            self._turns.clear()

    @staticmethod
    def _detect_topic(content: str) -> str:
        """Simple topic detection from content keywords."""
        content_lower = content.lower()
        topics = {
            "代码/编程": ["code", "function", "class", "import", "def ", "bug", "api", "git"],
            "记忆系统": ["memory", "mempalace", "chroma", "drawer", "kg", "knowledge graph"],
            "OpenClaw": ["openclaw", "agent", "chen", "ying", "wei", "cron", "task"],
            "项目/任务": ["project", "task", "implement", "build", "feature", "pr", "repo"],
            "系统配置": ["config", "setup", "install", "python", "path", "env", "api key"],
            "飞书": ["feishu", "lark", "飞书", "bot", "message", "dm"],
        }
        for topic, keywords in topics.items():
            if any(kw in content_lower for kw in keywords):
                return topic
        return ""


# Global singleton instance
_working_memory_instance: Optional[WorkingMemory] = None
_working_memory_lock = threading.Lock()


def get_working_memory() -> WorkingMemory:
    """Get the global WorkingMemory singleton instance."""
    global _working_memory_instance
    with _working_memory_lock:
        if _working_memory_instance is None:
            _working_memory_instance = WorkingMemory()
        return _working_memory_instance
