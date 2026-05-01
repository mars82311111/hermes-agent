"""
Adaptive Retrieval
Determines whether a query needs memory retrieval at all.
Skips retrieval for greetings, commands, simple instructions, and system messages.
Saves embedding API calls and reduces noise injection.

Enhancement (2026-05-01): Intent Gate
- Detects ambiguous keywords that could match multiple memory contexts
- When detected, requires user confirmation BEFORE performing retrieval
- This prevents "context pollution" where wrong memories are retrieved
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

# ============================================================================
# Intent Gate: Ambiguous Keywords (2026-05-01)
# These keywords have multiple meanings across different contexts.
# When detected, we must confirm intent BEFORE retrieval.
# ============================================================================

# Ambiguous keywords that could trigger wrong memory retrieval
# Note: Don't use \b with CJK characters - it doesn't work as expected
AMBIGUOUS_KEYWORDS: List[Pattern[str]] = [
    # "更新" can mean: system update, project update, Douyin content update, progress update
    re.compile(r"(更新|upgrade|update)", re.IGNORECASE),
    # "看一下" / "看看" - could refer to checking anything
    re.compile(r"(看|看看|看一下|look|check|review)", re.IGNORECASE),
    # "进展" - could be project progress, task progress, market progress
    re.compile(r"(进展|进度|progress|status)", re.IGNORECASE),
    # "测试" - could be system test, software test, A/B test
    re.compile(r"(测试|test|testing)", re.IGNORECASE),
    # "配置" - could be system config, model config, project config
    re.compile(r"(配置|config|setting)", re.IGNORECASE),
    # "项目" - specific project vs general project work
    re.compile(r"(项目|project)", re.IGNORECASE),
    # "账号" / "账户" - could be financial account, social media account
    re.compile(r"(账号|账户|account)", re.IGNORECASE),
    # "收入" - could be income, revenue, earnings
    re.compile(r"(收入|earning|income|revenue)", re.IGNORECASE),
    # "风险" - could be investment risk, project risk, market risk
    re.compile(r"(风险|risk)", re.IGNORECASE),
    # "分析" - could be financial analysis, data analysis, project analysis
    re.compile(r"(分析|analysis|analyze)", re.IGNORECASE),
]

# High-confidence single-meaning contexts that should NOT trigger confirmation
# If the query contains these patterns, the intent is clear and we skip confirmation
CLEAR_CONTEXT_PATTERNS: List[Pattern[str]] = [
    # "系统更新" / "system update" - clear intent
    re.compile(r"(系统更新|system update|hermes.*update|openclaw.*update)", re.IGNORECASE),
    # "项目进展" - clear intent about project progress
    re.compile(r"(项目进展|project.*progress|项目.*状态)", re.IGNORECASE),
    # "收入分析" - clear intent about income analysis
    re.compile(r"(收入分析|earnings.*analysis|income.*analyze)", re.IGNORECASE),
    # "配置更新" - clear intent about config update
    re.compile(r"(配置更新|config.*update)", re.IGNORECASE),
    # Explicit context markers
    re.compile(r"(抖音|tiktok|douyin).*(更新|content|作品)", re.IGNORECASE),
    re.compile(r"(系统|system).*(更新|upgrade)", re.IGNORECASE),
    re.compile(r"(项目|project).*(进展|进度|状态)", re.IGNORECASE),
]

# CJK character ranges for detection
CJK_RE = re.compile(r"[\u4e00-\u9fff\u3040-\u309f\u30a0-\u30ff\uac00-\ud7af]")


# ============================================================================
# Intent Gate API (2026-05-01)
# ============================================================================


def should_confirm_intent(query: str) -> bool:
    """
    Determine if a query contains ambiguous keywords that require
    user confirmation BEFORE memory retrieval.
    
    Returns True if we should ask the user to clarify their intent.
    Returns False if intent is clear (skip confirmation).
    
    Args:
        query: The raw prompt text
        
    Returns:
        True if confirmation is needed, False otherwise
    """
    trimmed = _normalize_query(query)
    
    # If query has clear context, skip confirmation
    for pattern in CLEAR_CONTEXT_PATTERNS:
        if pattern.search(trimmed):
            return False
    
    # If query has any ambiguous keyword, confirm intent
    for pattern in AMBIGUOUS_KEYWORDS:
        if pattern.search(trimmed):
            return True
    
    return False


def get_intent_options(query: str) -> List[dict]:
    """
    Return a list of likely intent options for user confirmation.
    Each option is a dict with 'id', 'label', and 'description'.
    
    Returns an empty list if intent is clear or no options available.
    """
    trimmed = _normalize_query(query).lower()
    
    options = []

    # Detect which ambiguous keywords are present (no \b for CJK)
    has_update = bool(re.compile(r"(更新|upgrade|update)").search(trimmed))
    has_look = bool(re.compile(r"(看|看看|look|check|review)").search(trimmed))
    has_progress = bool(re.compile(r"(进展|进度|progress|status)").search(trimmed))
    has_test = bool(re.compile(r"(测试|test)").search(trimmed))
    has_config = bool(re.compile(r"(配置|config)").search(trimmed))
    has_project = bool(re.compile(r"(项目|project)").search(trimmed))
    has_risk = bool(re.compile(r"(风险|risk)").search(trimmed))
    has_analysis = bool(re.compile(r"(分析|analysis)").search(trimmed))
    
    # Generate options based on detected keywords
    if has_update or has_look:
        options.extend([
            {"id": "hermes_update", "label": "Hermes系统更新", "description": "检查Hermes系统是否有新版本/更新"},
            {"id": "project_update", "label": "项目更新进展", "description": "查看当前项目的最新进展状态"},
            {"id": "douyin_update", "label": "抖音账号更新", "description": "检查抖音账号的内容/运营更新"},
            {"id": "general_check", "label": "其他/随便看看", "description": "不是以上选项"},
        ])
    
    if has_progress:
        options.append({"id": "project_progress", "label": "项目进展", "description": "当前项目的进度和状态"})
    
    if has_test:
        options.extend([
            {"id": "system_test", "label": "系统测试", "description": "测试Hermes或其他系统的功能"},
            {"id": "data_test", "label": "数据/模型测试", "description": "测试数据分析或AI模型效果"},
        ])
    
    if has_config:
        options.extend([
            {"id": "model_config", "label": "模型配置", "description": "检查AI模型的配置和参数"},
            {"id": "system_config", "label": "系统配置", "description": "检查系统级配置"},
        ])
    
    if has_project:
        options.append({"id": "project_status", "label": "项目整体状态", "description": "某个项目的综合状态"})
    
    if has_risk:
        options.extend([
            {"id": "investment_risk", "label": "投资风险", "description": "金融市场/投资相关的风险分析"},
            {"id": "project_risk", "label": "项目风险", "description": "项目执行层面的风险"},
        ])
    
    if has_analysis:
        options.extend([
            {"id": "financial_analysis", "label": "财务/投资分析", "description": "华尔街级别的金融分析"},
            {"id": "project_analysis", "label": "项目分析", "description": "某个项目的深度分析"},
        ])
    
    # Remove duplicates by id
    seen = set()
    unique_options = []
    for opt in options:
        if opt["id"] not in seen:
            seen.add(opt["id"])
            unique_options.append(opt)
    
    return unique_options[:5]  # Max 5 options to avoid overwhelming


def build_intent_confirmation_card(query: str) -> str:
    """
    Build a Feishu-compatible confirmation card for intent clarification.
    """
    options = get_intent_options(query)
    if not options:
        return ""
    
    # Build options list for card
    options_md = ""
    for i, opt in enumerate(options, 1):
        options_md += f"**{i}. {opt['label']}** — {opt['description']}\n"
    
    card = f"""**🤔 城哥，请确认你的意图：**

检测到「{query.strip()}」可能有多个含义：

{options_md}

请回复数字或直接说明你想问什么，我会精准检索对应的记忆。"""
    
    return card


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
