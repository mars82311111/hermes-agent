"""
Tier Manager — Three-tier memory promotion/demotion system

Tiers:
- Core (decay floor 0.9): Identity-level facts, almost never forgotten
- Working (decay floor 0.7): Active context, ages out without reinforcement
- Peripheral (decay floor 0.5): Low-priority or aging memories

Promotion: Peripheral → Working → Core (based on access, composite score, importance)
Demotion: Core → Working → Peripheral (based on decay, age)
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import List, Optional

# CRITICAL FIX: Import MemoryTier from a single source of truth to prevent
# drift between decay_engine.py and tier_manager.py.
from .decay_engine import MemoryTier

MS_PER_DAY = 86_400_000


@dataclass
class TierConfig:
    """Minimum access count for Core promotion (default: 10)"""
    coreAccessThreshold: int = 10
    """Minimum composite decay score for Core promotion (default: 0.7)"""
    coreCompositeThreshold: float = 0.7
    """Minimum importance for Core promotion (default: 0.8)"""
    coreImportanceThreshold: float = 0.8
    """Composite threshold below which to demote to Peripheral (default: 0.15)"""
    peripheralCompositeThreshold: float = 0.15
    """Age in days after which infrequent memories demote to Peripheral (default: 60)"""
    peripheralAgeDays: int = 60
    """Minimum access count for Working promotion from Peripheral (default: 3)"""
    workingAccessThreshold: int = 3
    """Minimum composite for Working promotion from Peripheral (default: 0.4)"""
    workingCompositeThreshold: float = 0.4


DEFAULT_TIER_CONFIG = TierConfig()


@dataclass
class TierTransition:
    memoryId: str
    fromTier: MemoryTier
    toTier: MemoryTier
    reason: str


class TierableMemory:
    """Minimal memory fields needed for tier evaluation."""
    id: str
    tier: MemoryTier
    importance: float
    accessCount: int
    createdAt: int

    def __init__(
        self,
        id: str,
        tier: MemoryTier,
        importance: float,
        accessCount: int,
        createdAt: int,
    ):
        self.id = id
        self.tier = tier
        self.importance = importance
        self.accessCount = accessCount
        self.createdAt = createdAt


class TierManager:
    """
    Evaluate whether a memory should change tiers.

    Returns the transition if a change is needed, None otherwise.
    """

    def __init__(self, config: TierConfig = DEFAULT_TIER_CONFIG):
        self.config = config

    def evaluate(
        self,
        memory: TierableMemory,
        decayScore: "DecayScore",  # DecayScore from decay_engine
        now: Optional[int] = None,
    ) -> Optional[TierTransition]:
        """
        Evaluate whether a memory should change tiers.
        Returns the transition if a change is needed, None otherwise.
        """
        if now is None:
            now = int(time.time() * 1000)
        age_days = (now - memory.createdAt) / MS_PER_DAY
        cfg = self.config

        if memory.tier == MemoryTier.PERIPHERAL:
            # Promote to Working?
            if (
                memory.accessCount >= cfg.workingAccessThreshold
                and decayScore.composite >= cfg.workingCompositeThreshold
            ):
                return TierTransition(
                    memoryId=memory.id,
                    fromTier=MemoryTier.PERIPHERAL,
                    toTier=MemoryTier.WORKING,
                    reason=(
                        f"Access count ({memory.accessCount}) >= {cfg.workingAccessThreshold} "
                        f"and composite ({decayScore.composite:.2f}) >= {cfg.workingCompositeThreshold}"
                    ),
                )

        elif memory.tier == MemoryTier.WORKING:
            # Promote to Core?
            if (
                memory.accessCount >= cfg.coreAccessThreshold
                and decayScore.composite >= cfg.coreCompositeThreshold
                and memory.importance >= cfg.coreImportanceThreshold
            ):
                return TierTransition(
                    memoryId=memory.id,
                    fromTier=MemoryTier.WORKING,
                    toTier=MemoryTier.CORE,
                    reason=(
                        f"High access ({memory.accessCount}), "
                        f"composite ({decayScore.composite:.2f}), "
                        f"importance ({memory.importance})"
                    ),
                )
            # Demote to Peripheral?
            if (
                decayScore.composite < cfg.peripheralCompositeThreshold
                or (
                    age_days > cfg.peripheralAgeDays
                    and memory.accessCount < cfg.workingAccessThreshold
                )
            ):
                return TierTransition(
                    memoryId=memory.id,
                    fromTier=MemoryTier.WORKING,
                    toTier=MemoryTier.PERIPHERAL,
                    reason=(
                        f"Low composite ({decayScore.composite:.2f}) or "
                        f"aged {age_days:.0f} days with low access ({memory.accessCount})"
                    ),
                )

        elif memory.tier == MemoryTier.CORE:
            # Demote to Working? (Core rarely demotes, but it can)
            if (
                decayScore.composite < cfg.peripheralCompositeThreshold
                and memory.accessCount < cfg.workingAccessThreshold
            ):
                return TierTransition(
                    memoryId=memory.id,
                    fromTier=MemoryTier.CORE,
                    toTier=MemoryTier.WORKING,
                    reason=(
                        f"Severely low composite ({decayScore.composite:.2f}) "
                        f"and access ({memory.accessCount})"
                    ),
                )

        return None

    def evaluate_all(
        self,
        memories: List[TierableMemory],
        decayScores: List["DecayScore"],  # DecayScore from decay_engine
        now: Optional[int] = None,
    ) -> List[TierTransition]:
        """Evaluate multiple memories and return all transitions."""
        if now is None:
            now = int(time.time() * 1000)
        score_map = {s.memoryId: s for s in decayScores}
        transitions: List[TierTransition] = []

        for memory in memories:
            score = score_map.get(memory.id)
            if score is None:
                continue
            transition = self.evaluate(memory, score, now)
            if transition is not None:
                transitions.append(transition)

        return transitions
