"""
Decay Engine — Weibull stretched-exponential decay model

Composite score = recencyWeight * recency + frequencyWeight * frequency + intrinsicWeight * intrinsic

- Recency: Weibull decay with importance-modulated half-life and tier-specific beta
- Frequency: Logarithmic saturation with time-weighted access pattern bonus
- Intrinsic: importance × confidence
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from enum import Enum
from typing import Optional

MS_PER_DAY = 86_400_000


class MemoryTier(str, Enum):
    CORE = "core"
    WORKING = "working"
    PERIPHERAL = "peripheral"


@dataclass
class DecayConfig:
    """Days until recency score halves (default: 14 for recency, 60 for timeDecay)"""
    recencyHalfLifeDays: float = 14.0
    timeDecayHalfLifeDays: float = 60.0
    recencyWeight: float = 0.4
    frequencyWeight: float = 0.3
    intrinsicWeight: float = 0.3
    staleThreshold: float = 0.3
    searchBoostMin: float = 0.3
    importanceModulation: float = 1.5
    betaCore: float = 0.8
    betaWorking: float = 1.0
    betaPeripheral: float = 1.3
    coreDecayFloor: float = 0.9
    workingDecayFloor: float = 0.7
    peripheralDecayFloor: float = 0.5


DEFAULT_DECAY_CONFIG = DecayConfig()


@dataclass
class DecayScore:
    memoryId: str
    recency: float
    frequency: float
    intrinsic: float
    composite: float


class DecayableMemory:
    """Minimal memory fields needed for decay calculation."""
    id: str
    importance: float
    confidence: float
    tier: MemoryTier
    accessCount: int
    createdAt: int
    lastAccessedAt: int
    temporalType: str = "static"  # "static" | "dynamic"

    def __init__(
        self,
        id: str,
        importance: float,
        confidence: float,
        tier: MemoryTier,
        accessCount: int,
        createdAt: int,
        lastAccessedAt: int,
        temporalType: str = "static",
    ):
        self.id = id
        self.importance = importance
        self.confidence = confidence
        self.tier = tier
        self.accessCount = accessCount
        self.createdAt = createdAt
        self.lastAccessedAt = lastAccessedAt
        self.temporalType = temporalType


class DecayEngine:
    """Calculate decay scores for memories using Weibull stretched-exponential model."""

    def __init__(self, config: DecayConfig = DEFAULT_DECAY_CONFIG):
        self.config = config
        self._half_life = config.recencyHalfLifeDays
        self._rw = config.recencyWeight
        self._fw = config.frequencyWeight
        self._iw = config.intrinsicWeight
        self._stale_threshold = config.staleThreshold
        self._boost_min = config.searchBoostMin
        self._mu = config.importanceModulation
        self._beta_core = config.betaCore
        self._beta_working = config.betaWorking
        self._beta_peripheral = config.betaPeripheral
        self._core_floor = config.coreDecayFloor
        self._working_floor = config.workingDecayFloor
        self._peripheral_floor = config.peripheralDecayFloor

    def _get_tier_beta(self, tier: MemoryTier) -> float:
        if tier == MemoryTier.CORE:
            return self._beta_core
        elif tier == MemoryTier.WORKING:
            return self._beta_working
        else:
            return self._beta_peripheral

    def _get_tier_floor(self, tier: MemoryTier) -> float:
        if tier == MemoryTier.CORE:
            return self._core_floor
        elif tier == MemoryTier.WORKING:
            return self._working_floor
        else:
            return self._peripheral_floor

    def _recency(self, memory: DecayableMemory, now: int) -> float:
        """Recency: Weibull stretched-exponential decay with importance-modulated half-life."""
        last_active = memory.lastAccessedAt if memory.accessCount > 0 else memory.createdAt
        days_since = max(0.0, (now - last_active) / MS_PER_DAY)
        # Dynamic memories decay 3x faster (1/3 half-life)
        base_hl = self._half_life / 3 if memory.temporalType == "dynamic" else self._half_life
        effective_hl = base_hl * math.exp(self._mu * memory.importance)
        lam = math.log(2) / effective_hl
        beta = self._get_tier_beta(memory.tier)
        return math.exp(-lam * math.pow(days_since, beta))

    def _frequency(self, memory: DecayableMemory) -> float:
        """Frequency: logarithmic saturation curve with time-weighted access pattern bonus."""
        base = 1.0 - math.exp(-memory.accessCount / 5.0)
        if memory.accessCount <= 1:
            return base

        last_active = memory.lastAccessedAt if memory.accessCount > 0 else memory.createdAt
        access_span_days = max(1.0, (last_active - memory.createdAt) / MS_PER_DAY)
        avg_gap_days = access_span_days / max(memory.accessCount - 1, 1)
        recentness_bonus = math.exp(-avg_gap_days / 30.0)
        return base * (0.5 + 0.5 * recentness_bonus)

    def _intrinsic(self, memory: DecayableMemory) -> float:
        """Intrinsic value: importance × confidence."""
        return memory.importance * memory.confidence

    def _score_one(self, memory: DecayableMemory, now: int) -> DecayScore:
        r = self._recency(memory, now)
        f = self._frequency(memory)
        i = self._intrinsic(memory)
        composite = self._rw * r + self._fw * f + self._iw * i
        return DecayScore(
            memoryId=memory.id,
            recency=r,
            frequency=f,
            intrinsic=i,
            composite=composite,
        )

    def score(self, memory: DecayableMemory, now: Optional[int] = None) -> DecayScore:
        """Calculate decay score for a single memory."""
        if now is None:
            now = _current_time_ms()
        return self._score_one(memory, now)

    def score_all(self, memories: list[DecayableMemory], now: Optional[int] = None) -> list[DecayScore]:
        """Calculate decay scores for multiple memories."""
        if now is None:
            now = _current_time_ms()
        return [self._score_one(m, now) for m in memories]

    def apply_search_boost(
        self,
        results: list[tuple[DecayableMemory, float]],
        now: Optional[int] = None,
    ) -> None:
        """Apply decay boost to search results (multiplies each score by boost)."""
        if now is None:
            now = _current_time_ms()
        for idx, (memory, score) in enumerate(results):
            ds = self._score_one(memory, now)
            tier_floor = max(self._get_tier_floor(memory.tier), ds.composite)
            multiplier = self._boost_min + ((1.0 - self._boost_min) * tier_floor)
            # Mutate score in place using index to avoid ambiguous .index() lookup
            results[idx] = (memory, score * min(1.0, max(self._boost_min, multiplier)))

    def get_stale_memories(
        self,
        memories: list[DecayableMemory],
        now: Optional[int] = None,
    ) -> list[DecayScore]:
        """Find stale memories (composite below threshold)."""
        if now is None:
            now = _current_time_ms()
        scores = [self._score_one(m, now) for m in memories]
        return sorted([s for s in scores if s.composite < self._stale_threshold], key=lambda s: s.composite)


def _current_time_ms() -> int:
    import time
    return int(time.time() * 1000)
