"""Cognitive State Manager — deterministic continuity layer for AIAgent.

This module provides persistent cognitive state that survives AIAgent
instance destruction (gateway cache eviction, session timeouts, etc.).

Unlike probabilistic memory retrieval, cognitive state is:
- Deterministic: always loaded, never "maybe"
- Structured: typed fields with clear semantics
- Actionable: directly drives pre-output guardrails

State file: ~/.hermes/cognitive_state.json
"""

import json
import logging
import re
import threading
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

from hermes_constants import get_hermes_home

logger = logging.getLogger(__name__)

STATE_VERSION = 1
STATE_FILENAME = "cognitive_state.json"
_MAX_STATE_AGE_DAYS = 30  # Auto-expire old session states
_MAX_CORRECTIONS_PER_SESSION = 20
_MAX_IMMUNITY_RULES = 50
_MAX_FOCUS_HISTORY = 10

_lock = threading.RLock()


def _get_state_path() -> Path:
    return Path(get_hermes_home()) / STATE_FILENAME


def load_cognitive_state(session_id: str) -> Dict[str, Any]:
    """Load cognitive state for a session. Returns empty dict if none exists."""
    path = _get_state_path()
    if not path.exists():
        return {}

    try:
        with _lock:
            data = json.loads(path.read_text(encoding="utf-8"))

        # Validate version
        if data.get("version") != STATE_VERSION:
            logger.warning("Cognitive state version mismatch: %s vs %s", data.get("version"), STATE_VERSION)
            return {}

        # Return session-specific state merged with global state
        session_state = data.get("session_states", {}).get(session_id, {})
        global_state = data.get("global_state", {})

        # Merge: session state takes precedence, but inherit global immunity
        merged = dict(global_state)
        merged.update(session_state)

        # Inherit persistent immunity from global
        global_immunity = global_state.get("persistent_immunity", [])
        session_immunity = session_state.get("error_immunity", [])
        merged["error_immunity"] = merge_immunity_rules(global_immunity, session_immunity)

        return merged

    except Exception as e:
        logger.warning("Failed to load cognitive state: %s", e)
        return {}


def save_cognitive_state(session_id: str, state: Dict[str, Any]) -> None:
    """Save cognitive state for a session."""
    path = _get_state_path()

    try:
        with _lock:
            # Load existing or create new
            if path.exists():
                try:
                    data = json.loads(path.read_text(encoding="utf-8"))
                except Exception:
                    data = {"version": STATE_VERSION, "session_states": {}, "global_state": {}}
            else:
                data = {"version": STATE_VERSION, "session_states": {}, "global_state": {}}

            # Ensure structure
            if "session_states" not in data:
                data["session_states"] = {}
            if "global_state" not in data:
                data["global_state"] = {}

            # Prune old sessions
            _prune_old_sessions(data["session_states"])

            # Update session state
            data["session_states"][session_id] = state

            # Promote high-value corrections to global state
            _promote_to_global(data, state)

            # Write atomically
            tmp_path = path.with_suffix(".tmp")
            tmp_path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
            tmp_path.replace(path)

    except Exception as e:
        logger.error("Failed to save cognitive state: %s", e)


def _prune_old_sessions(session_states: Dict[str, Any]) -> None:
    """Remove session states older than _MAX_STATE_AGE_DAYS."""
    cutoff = time.time() - (_MAX_STATE_AGE_DAYS * 86400)
    to_remove = []
    for sid, sstate in session_states.items():
        last_updated = sstate.get("last_updated", "")
        if last_updated:
            try:
                ts = datetime.fromisoformat(last_updated).timestamp()
                if ts < cutoff:
                    to_remove.append(sid)
            except Exception:
                pass
    for sid in to_remove:
        del session_states[sid]
        logger.info("Pruned old cognitive state for session %s", sid)


def _promote_to_global(data: Dict[str, Any], session_state: Dict[str, Any]) -> None:
    """Promote high-importance corrections to global state for cross-session immunity."""
    global_state = data["global_state"]
    if "persistent_immunity" not in global_state:
        global_state["persistent_immunity"] = []

    corrections = session_state.get("recent_corrections", [])
    for corr in corrections:
        # Promote if importance >= 0.9 or repeated >= 2 times
        importance = corr.get("importance", 0.5)
        repeated = corr.get("repeated_count", 1)
        if importance >= 0.9 or repeated >= 2:
            pattern = corr.get("trigger_pattern", "")
            if pattern and not any(r.get("pattern") == pattern for r in global_state["persistent_immunity"]):
                global_state["persistent_immunity"].append({
                    "pattern": pattern,
                    "correction": corr.get("correct_action", ""),
                    "created_at": corr.get("timestamp", datetime.now().isoformat()),
                    "source": "promoted_from_session",
                })

    # Cap global immunity
    if len(global_state["persistent_immunity"]) > _MAX_IMMUNITY_RULES:
        global_state["persistent_immunity"] = global_state["persistent_immunity"][-_MAX_IMMUNITY_RULES:]


def merge_immunity_rules(global_rules: List[Dict], session_rules: List[Dict]) -> List[Dict]:
    """Merge immunity rules, deduplicating by pattern."""
    seen = set()
    merged = []
    for rule in session_rules + global_rules:
        pattern = rule.get("pattern", "")
        if pattern and pattern not in seen:
            seen.add(pattern)
            merged.append(rule)
    return merged


def create_default_state() -> Dict[str, Any]:
    """Create a fresh cognitive state."""
    return {
        "current_focus": "",
        "focus_history": [],
        "recent_corrections": [],
        "error_immunity": [],
        "recent_decisions": [],
        "active_project": "",
        "conversation_stage": "init",
        "stage_history": [],
        "last_updated": datetime.now().isoformat(),
        "turn_count": 0,
    }


# ============================================================================
# State update helpers
# ============================================================================

def update_focus(state: Dict[str, Any], new_focus: str) -> Dict[str, Any]:
    """Update current focus, tracking history."""
    if not new_focus or new_focus == state.get("current_focus"):
        return state

    old_focus = state.get("current_focus", "")
    if old_focus:
        history = state.get("focus_history", [])
        history.insert(0, old_focus)
        state["focus_history"] = history[:_MAX_FOCUS_HISTORY]

    state["current_focus"] = new_focus
    state["last_updated"] = datetime.now().isoformat()
    return state


def add_correction(state: Dict[str, Any], error_type: str, description: str,
                   trigger_pattern: str, correct_action: str,
                   importance: float = 0.9) -> Dict[str, Any]:
    """Add a correction to the state. If same pattern exists, increment repeat count."""
    corrections = state.get("recent_corrections", [])

    # Check if same pattern already exists
    for corr in corrections:
        if corr.get("trigger_pattern") == trigger_pattern:
            corr["repeated_count"] = corr.get("repeated_count", 1) + 1
            corr["timestamp"] = datetime.now().isoformat()
            corr["description"] = description  # Update description
            state["recent_corrections"] = corrections
            state["last_updated"] = datetime.now().isoformat()
            return state

    # New correction
    corrections.insert(0, {
        "timestamp": datetime.now().isoformat(),
        "error_type": error_type,
        "description": description,
        "trigger_pattern": trigger_pattern,
        "correct_action": correct_action,
        "importance": importance,
        "repeated_count": 1,
    })

    # Cap corrections
    state["recent_corrections"] = corrections[:_MAX_CORRECTIONS_PER_SESSION]

    # Auto-add to error_immunity
    immunity = state.get("error_immunity", [])
    if not any(r.get("pattern") == trigger_pattern for r in immunity):
        immunity.insert(0, {
            "pattern": trigger_pattern,
            "correction": correct_action,
            "created_at": datetime.now().isoformat(),
            "source": "auto_from_correction",
        })
        state["error_immunity"] = immunity[:_MAX_IMMUNITY_RULES]

    state["last_updated"] = datetime.now().isoformat()
    return state


def add_decision(state: Dict[str, Any], decision: str, context: str = "") -> Dict[str, Any]:
    """Record a decision."""
    decisions = state.get("recent_decisions", [])
    decisions.insert(0, {
        "timestamp": datetime.now().isoformat(),
        "decision": decision,
        "context": context,
    })
    state["recent_decisions"] = decisions[:10]
    state["last_updated"] = datetime.now().isoformat()
    return state


def set_stage(state: Dict[str, Any], stage: str) -> Dict[str, Any]:
    """Set conversation stage, tracking history."""
    valid_stages = {"init", "clarify", "plan", "execute", "verify", "done", "error"}
    if stage not in valid_stages:
        stage = "execute"

    state["conversation_stage"] = stage
    history = state.get("stage_history", [])
    history.append({
        "stage": stage,
        "timestamp": datetime.now().isoformat(),
    })
    state["stage_history"] = history[-20:]  # Keep last 20
    state["last_updated"] = datetime.now().isoformat()
    return state


def increment_turn(state: Dict[str, Any]) -> Dict[str, Any]:
    """Increment turn counter."""
    state["turn_count"] = state.get("turn_count", 0) + 1
    state["last_updated"] = datetime.now().isoformat()
    return state


# ============================================================================
# Pre-output guardrail
# ============================================================================

def check_output_violations(output: str, error_immunity: List[Dict]) -> List[Dict]:
    """Check if output violates any error immunity rules.

    Returns list of violated rules. Empty list = safe.
    """
    violations = []
    if not output or not error_immunity:
        return violations

    for rule in error_immunity:
        pattern = rule.get("pattern", "")
        if not pattern:
            continue
        try:
            if re.search(pattern, output, re.IGNORECASE):
                violations.append(rule)
        except re.error:
            # Fallback to literal match for invalid regex patterns
            if pattern in output:
                violations.append(rule)
            else:
                logger.warning("Invalid immunity pattern (fallback literal): %s", pattern)

    return violations


def build_correction_prompt(violations: List[Dict], original_output: str) -> str:
    """Build a prompt to send back to LLM to correct the output."""
    lines = [
        "⚠️ OUTPUT GUARDRAIL TRIGGERED ⚠️",
        "",
        "Your previous response violated the following error immunity rules:",
    ]
    for i, v in enumerate(violations, 1):
        lines.append(f"  {i}. Pattern: {v.get('pattern', 'unknown')}")
        lines.append(f"     Correction: {v.get('correction', 'fix this')}")
    lines.extend([
        "",
        "You must regenerate your response, ensuring you do NOT repeat this error.",
        "Your original response was:",
        "---",
        original_output[:500],  # Truncate to avoid token bloat
        "---",
        "Now provide the corrected response.",
    ])
    return "\n".join(lines)


# ============================================================================
# Intent prediction + proactive memory loading
# ============================================================================

def generate_predictive_queries(user_message: str, state: Dict[str, Any]) -> List[str]:
    """Generate additional search queries based on cognitive state.

    This helps recall relevant context even when the user's message
    doesn't contain explicit keywords.
    """
    queries = [user_message]

    # If we have a current focus, search for related context
    focus = state.get("current_focus", "")
    if focus and focus not in user_message:
        queries.append(focus)

    # If we're in a project, search for project context
    project = state.get("active_project", "")
    if project:
        queries.append(project)

    # Search for recent corrections related to the message
    for corr in state.get("recent_corrections", [])[:3]:
        pattern = corr.get("trigger_pattern", "")
        if pattern and any(kw in user_message.lower() for kw in _extract_keywords(pattern)):
            queries.append(corr.get("correct_action", ""))

    return list(dict.fromkeys(q for q in queries if q))  # Deduplicate, preserve order


def _extract_keywords(pattern: str) -> List[str]:
    """Extract meaningful keywords from a regex pattern."""
    # Simple heuristic: remove regex special chars, split by non-word
    cleaned = re.sub(r'[.*?+^$\\|()[\]{}]', ' ', pattern)
    words = [w for w in cleaned.split() if len(w) >= 2]
    return words


def detect_corrections_from_turn(messages: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Analyze a completed turn to detect if the user corrected the assistant.

    Heuristic: if the user's message contains correction indicators
    (不是, 错了, 纠正, etc.) and the assistant's previous response
    is in the history, extract the correction.
    """
    corrections = []
    if len(messages) < 2:
        return corrections

    # Look at the last user message
    last_user = None
    last_assistant = None
    for msg in reversed(messages):
        if msg.get("role") == "user" and last_user is None:
            last_user = msg.get("content", "")
        elif msg.get("role") == "assistant" and last_assistant is None:
            last_assistant = msg.get("content", "")
        if last_user and last_assistant:
            break

    if not last_user or not last_assistant:
        return corrections

    # Correction indicators
    correction_indicators = [
        r'不是\s*(?:这个|那样|A|B|那样|这样)',
        r'错了|错了|纠正|应该|不应该|记住.*不要|别再|以后.*要|以后.*不要',
        r'跑偏|走偏|偏离|不是.*而是',
        r'你(?:搞|弄|做)错',
        r'(?:重新|再).*想',
    ]

    user_text = str(last_user)
    for indicator in correction_indicators:
        if re.search(indicator, user_text, re.IGNORECASE):
            # Extract what was wrong
            # Simple heuristic: look for "不是X而是Y" or "错了，应该是"
            desc = user_text[:200]
            corrections.append({
                "error_type": "user_correction",
                "description": desc,
                "trigger_pattern": _infer_trigger_pattern(last_assistant),
                "correct_action": _extract_correct_action(user_text),
                "importance": 0.9,
            })
            break

    return corrections


def _infer_trigger_pattern(assistant_output: str) -> str:
    """Infer a pattern that would match the assistant's wrong output."""
    # Simple: take first 30 chars as pattern, escape regex
    sample = str(assistant_output)[:50]
    # Extract key phrases (3-10 chars)
    words = re.findall(r'\b[\u4e00-\u9fff]{2,8}\b', sample)
    if words:
        return "|".join(words[:3])
    # Fallback: extract English keywords
    words = re.findall(r'\b[a-zA-Z_]{3,15}\b', sample)
    if words:
        return "|".join(words[:3])
    return sample[:20]


def _extract_correct_action(user_text: str) -> str:
    """Extract what the user wants the assistant to do instead."""
    # Look for patterns like "应该是", "而是", "要"
    for pattern in [r'(?:应该是|而是|要|应该|用)\s*([^。！？\n]{3,100})']:
        match = re.search(pattern, user_text)
        if match:
            return match.group(1).strip()
    return user_text[:100]


def format_state_for_system_prompt(state: Dict[str, Any]) -> str:
    """Format cognitive state as text to inject into system prompt."""
    if not state:
        return ""

    lines = ["╔══════════════════════════════════════════════════════════════╗"]
    lines.append("║  COGNITIVE STATE — Deterministic context (NOT probabilistic) ║")
    lines.append("╚══════════════════════════════════════════════════════════════╝")

    focus = state.get("current_focus", "")
    if focus:
        lines.append(f"\n【当前任务】{focus}")

    project = state.get("active_project", "")
    if project:
        lines.append(f"【活跃项目】{project}")

    stage = state.get("conversation_stage", "")
    if stage:
        lines.append(f"【对话阶段】{stage}")

    corrections = state.get("recent_corrections", [])
    if corrections:
        lines.append("\n【近期纠正】")
        for c in corrections[:3]:
            desc = c.get("description", "")[:60]
            lines.append(f"  • {desc}")

    immunity = state.get("error_immunity", [])
    if immunity:
        lines.append("\n【错误免疫】以下模式必须避免：")
        for i in immunity[:5]:
            corr = i.get("correction", "")[:50]
            if corr:
                lines.append(f"  • {corr}")

    decisions = state.get("recent_decisions", [])
    if decisions:
        lines.append("\n【近期决策】")
        for d in decisions[:3]:
            dec = d.get("decision", "")[:60]
            lines.append(f"  • {dec}")

    turn_count = state.get("turn_count", 0)
    if turn_count > 0:
        lines.append(f"\n【对话轮数】{turn_count}")

    return "\n".join(lines)
