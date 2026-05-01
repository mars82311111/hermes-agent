"""
Noise Filter
Filters out low-quality memories (meta-questions, agent denials, session boilerplate)
Inspired by openclaw-plugin-continuity's noise filtering approach.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import List, Pattern, Tuple

# ============================================================================
# Constants — Pattern Sets
# ============================================================================

# Agent-side denial patterns
DENIAL_PATTERNS: List[Pattern[str]] = [
    re.compile(r"i don't? have (any )?(information|data|memory|record)", re.IGNORECASE),
    re.compile(r"i'?m not sure about", re.IGNORECASE),
    re.compile(r"i don't? recall", re.IGNORECASE),
    re.compile(r"i don't? remember", re.IGNORECASE),
    re.compile(r"it looks like i don't?", re.IGNORECASE),
    re.compile(r"i wasn't? able to find", re.IGNORECASE),
    re.compile(r"no (relevant )?memories found", re.IGNORECASE),
    re.compile(r"i don't? have access to", re.IGNORECASE),
]

# User-side meta-question patterns (about memory itself, not content)
META_QUESTION_PATTERNS: List[Pattern[str]] = [
    re.compile(r"\bdo you (remember|recall|know about)\b", re.IGNORECASE),
    re.compile(r"\bcan you (remember|recall)\b", re.IGNORECASE),
    re.compile(r"\bdid i (tell|mention|say|share)\b", re.IGNORECASE),
    re.compile(r"\bhave i (told|mentioned|said)\b", re.IGNORECASE),
    re.compile(r"\bwhat did i (tell|say|mention)\b", re.IGNORECASE),
    re.compile(r"如果你知道.+只回复", re.IGNORECASE),
    re.compile(r"如果不知道.+只回复\s*none", re.IGNORECASE),
    re.compile(r"只回复精确代号", re.IGNORECASE),
    re.compile(r"只回复\s*none", re.IGNORECASE),
    # Chinese recall / meta-question patterns
    re.compile(r"你还?记得", re.IGNORECASE),
    re.compile(r"记不记得", re.IGNORECASE),
    re.compile(r"还记得.*吗", re.IGNORECASE),
    re.compile(r"你[知晓]道.+吗", re.IGNORECASE),
    re.compile(r"我(?:之前|上次|以前)(?:说|提|讲).*(?:吗|呢|？|\?)", re.IGNORECASE),
]

# Session boilerplate
BOILERPLATE_PATTERNS: List[Pattern[str]] = [
    re.compile(r"^(hi|hello|hey|good morning|good evening|greetings)", re.IGNORECASE),
    re.compile(r"^fresh session", re.IGNORECASE),
    re.compile(r"^new session", re.IGNORECASE),
    re.compile(r"^HEARTBEAT", re.IGNORECASE),
]

# Extractor artifacts from validation prompts / synthetic summaries
DIAGNOSTIC_ARTIFACT_PATTERNS: List[Pattern[str]] = [
    re.compile(r"\bquery\s*->\s*(none|no explicit solution|unknown|not found)", re.IGNORECASE),
    re.compile(r"\buser asked for\b.*\b(none|no explicit solution|unknown|not found)\b", re.IGNORECASE),
    re.compile(r"\bno explicit solution\b", re.IGNORECASE),
]

# Envelope noise patterns — Discord/channel metadata headers
ENVELOPE_NOISE_PATTERNS: List[Pattern[str]] = [
    re.compile(r"^<<<EXTERNAL_UNTRUSTED_CONTENT\b", re.MULTILINE | re.IGNORECASE),
    re.compile(r"^<<<END_EXTERNAL_UNTRUSTED_CONTENT\b", re.MULTILINE | re.IGNORECASE),
    re.compile(r"^Sender\s*\(untrusted metadata\):", re.MULTILINE | re.IGNORECASE),
    re.compile(r"^Conversation info\s*\(untrusted metadata\):", re.MULTILINE | re.IGNORECASE),
    re.compile(r"^Thread starter\s*\(untrusted, for context\):", re.MULTILINE | re.IGNORECASE),
    re.compile(r"^Forwarded message context\s*\(untrusted metadata\):", re.MULTILINE | re.IGNORECASE),
    re.compile(r"^\[Queued messages while agent was busy\]", re.MULTILINE | re.IGNORECASE),
    re.compile(r"^System:\s*\[[\d\-: +GMT]+\]", re.MULTILINE | re.IGNORECASE),
]

# ============================================================================
# Options
# ============================================================================


@dataclass
class NoiseFilterOptions:
    """Filter configuration options."""
    filterDenials: bool = True
    filterMetaQuestions: bool = True
    filterBoilerplate: bool = True


DEFAULT_OPTIONS = NoiseFilterOptions(
    filterDenials=True,
    filterMetaQuestions=True,
    filterBoilerplate=True,
)


# ============================================================================
# Core API
# ============================================================================


def is_noise(text: str, options: NoiseFilterOptions = DEFAULT_OPTIONS) -> bool:
    """
    Check if a memory text is noise that should be filtered out.

    Returns True if the text is noise.

    Args:
        text: The text to check
        options: Filtering options (default: all filters enabled)

    Returns:
        True if the text is noise and should be filtered, False otherwise
    """
    trimmed = text.strip()

    if len(trimmed) < 5:
        return True

    if options.filterDenials:
        for pattern in DENIAL_PATTERNS:
            if pattern.search(trimmed):
                return True

    if options.filterMetaQuestions:
        for pattern in META_QUESTION_PATTERNS:
            if pattern.search(trimmed):
                return True

    if options.filterBoilerplate:
        for pattern in BOILERPLATE_PATTERNS:
            if pattern.search(trimmed):
                return True

    for pattern in DIAGNOSTIC_ARTIFACT_PATTERNS:
        if pattern.search(trimmed):
            return True

    # OPTIMIZATION: Envelope noise — Discord/飞书等平台注入的元数据头
    # These were defined but never checked in the original code.
    for pattern in ENVELOPE_NOISE_PATTERNS:
        if pattern.search(trimmed):
            return True

    return False


def filter_noise(
    items: List[T],
    get_text: "callable",  # callable[T -> str]
    options: NoiseFilterOptions = DEFAULT_OPTIONS,
) -> List[T]:
    """
    Filter an array of items, removing noise entries.

    Args:
        items: List of items to filter
        get_text: Function to extract text from each item
        options: Filtering options

    Returns:
        Filtered list with noise entries removed
    """
    opts = options if options is not None else DEFAULT_OPTIONS
    return [item for item in items if not is_noise(get_text(item), opts)]
