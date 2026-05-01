"""
Adaptive Retrieval
Determines whether a query needs memory retrieval at all.
Skips retrieval for greetings, commands, simple instructions, and system messages.
Saves embedding API calls and reduces noise injection.
"""

from __future__ import annotations

import re
from typing import List, Pattern, Optional

# ============================================================================
# Pattern Sets
# ============================================================================

# Queries that are clearly NOT memory-retrieval candidates
SKIP_PATTERNS: List[Pattern[str]] = [
    # Greetings & pleasantries
    re.compile(r"^(hi|hello|hey|good\s*(morning|afternoon|evening|night)|greetings|yo|sup|howdy|what'?s up)\b", re.IGNORECASE),
    # System/bot commands
    re.compile(r"^/"),  # slash commands
    re.compile(r"^(run|build|test|ls|cd|git|npm|pip|docker|curl|cat|grep|find|make|sudo)\b", re.IGNORECASE),
    # Simple affirmations/negations
    re.compile(r"^(yes|no|yep|nope|ok|okay|sure|fine|thanks|thank you|thx|ty|got it|understood|cool|nice|great|good|perfect|awesome|👍|👎|✅|❌)\s*[.!]?$", re.IGNORECASE),
    # Continuation prompts
    re.compile(r"^(go ahead|continue|proceed|do it|start|begin|next|实施|執行|开始|開始|继续|繼續|好的|可以|行)\s*[.!]?$", re.IGNORECASE),
    # Pure emoji - Python 3.9 doesn't support \p{Emoji}, use simple range check instead
    # Match strings that are only emoji/whitespace
    re.compile(r"^[!-/:-@[-`{-~\s]+$"),  # Printable ASCII symbols + whitespace
    # Heartbeat/system (match anywhere, not just at start)
    re.compile(r"HEARTBEAT", re.IGNORECASE),
    re.compile(r"^\[System", re.IGNORECASE),
    # Single-word utility pings
    re.compile(r"^(ping|pong|test|debug)\s*[.!?]?$", re.IGNORECASE),
]

# Queries that SHOULD trigger retrieval even if short
FORCE_RETRIEVE_PATTERNS: List[Pattern[str]] = [
    re.compile(r"\b(remember|recall|forgot|memory|memories)\b", re.IGNORECASE),
    re.compile(r"\b(last time|before|previously|earlier|yesterday|ago)\b", re.IGNORECASE),
    re.compile(r"\b(my (name|email|phone|address|birthday|preference))\b", re.IGNORECASE),
    re.compile(r"\b(what did (i|we)|did i (tell|say|mention))\b", re.IGNORECASE),
    re.compile(r"(你记得|[你妳]記得|之前|上次|以前|还记得|還記得|提到过|提到過|说过|說過)", re.IGNORECASE),
]

# CJK character ranges for detection
CJK_RE = re.compile(r"[\u4e00-\u9fff\u3040-\u309f\u30a0-\u30ff\uac00-\ud7af]")


# ============================================================================
# Query Normalization
# ============================================================================


def _normalize_query(query: str) -> str:
    """
    Normalize the raw prompt before applying skip/force rules.

    Strips OpenClaw injected metadata headers, cron wrapper prefixes,
    and timestamp prefixes.
    """
    s = query.strip()

    # 1. Strip OpenClaw injected metadata headers (Conversation info or Sender).
    metadata_pattern = re.compile(
        r"^(Conversation info|Sender) \(untrusted metadata\):////s*S?//n//s*//n",
        re.MULTILINE | re.IGNORECASE,
    )
    s = metadata_pattern.sub("", s)

    # 2. Strip OpenClaw cron wrapper prefix: [cron:<jobId> <jobName>]
    s = re.sub(r"^\[cron:[^\]]+\]\s*", "", s, flags=re.IGNORECASE).strip()

    # 3. Strip OpenClaw timestamp prefix: [Mon 2026-03-02 04:21 GMT+8]
    s = re.sub(
        r"^\[[A-Za-z]{3}\s\d{4}-\d{2}-\d{2}\s\d{2}:\d{2}\s[^\]]+\]\s*",
        "",
        s,
    ).strip()

    return s


# ============================================================================
# Core API
# ============================================================================


def should_skip_retrieval(query: str, min_length: Optional[int] = None) -> bool:
    """
    Determine if a query should skip memory retrieval.
    Returns True if retrieval should be skipped.

    Args:
        query: The raw prompt text
        min_length: Optional minimum length override (if set, overrides built-in thresholds)

    Returns:
        True if retrieval should be skipped, False otherwise
    """
    trimmed = _normalize_query(query)

    # Force retrieve if query has memory-related intent (checked FIRST,
    # before length check, so short CJK queries like "你记得吗" aren't skipped)
    for pattern in FORCE_RETRIEVE_PATTERNS:
        if pattern.search(trimmed):
            return False

    # Skip if matches any skip pattern
    for pattern in SKIP_PATTERNS:
        if pattern.search(trimmed):
            return True

    # If caller provides a custom minimum length, use it
    if min_length is not None and min_length > 0:
        if len(trimmed) < min_length and "?" not in trimmed and "？" not in trimmed:
            return True
        return False

    # Skip very short non-question messages (likely commands or affirmations)
    # CJK characters carry more meaning per character, so use a lower threshold
    has_cjk = bool(CJK_RE.search(trimmed))
    # CJK: 2 chars is minimum for meaningful search (e.g. "记忆", "问题")
    default_min_length = 2 if has_cjk else 15
    if len(trimmed) < default_min_length and "?" not in trimmed and "？" not in trimmed:
        return True

    # Default: do retrieve
    return False
