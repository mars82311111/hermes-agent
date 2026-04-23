#!/usr/bin/env python3
"""
Skill Search Tool - Unified search for local and external skills.

Searches local skills by name, description, and category.
Falls back to external ClawHub search if local results are insufficient.
"""

import json
import logging
import re
import subprocess
from typing import Any, Dict, List

from tools.registry import registry, tool_error
from tools.skills_tool import _find_all_skills

logger = logging.getLogger(__name__)


def _tokenize(text: str) -> List[str]:
    """Split text into lowercase tokens."""
    if not text:
        return []
    # Keep alphanumeric and spaces, split on non-alphanumeric
    cleaned = re.sub(r"[^\w\s]", " ", text.lower())
    return [t for t in cleaned.split() if len(t) > 1]


def _score_skill(skill: Dict[str, Any], query_tokens: List[str]) -> int:
    """Score a skill based on how many query tokens match."""
    score = 0
    name = (skill.get("name") or "").lower()
    description = (skill.get("description") or "").lower()
    category = (skill.get("category") or "").lower()

    for token in query_tokens:
        if token in name:
            score += 3
        if token in description:
            score += 2
        if token in category:
            score += 1

    return score


def _search_local_skills(query: str, top_n: int = 5) -> List[Dict[str, Any]]:
    """Search local skills by query. Returns top N matches."""
    try:
        all_skills = _find_all_skills()
    except Exception as e:
        logger.warning("Failed to load local skills: %s", e)
        return []

    if not all_skills:
        return []

    query_tokens = _tokenize(query)
    if not query_tokens:
        # If query has no meaningful tokens, return first N skills
        return all_skills[:top_n]

    scored = []
    for skill in all_skills:
        score = _score_skill(skill, query_tokens)
        if score > 0:
            scored.append((score, skill))

    # Sort by score descending, then by name
    scored.sort(key=lambda x: (-x[0], x[1].get("name", "")))

    return [s[1] for s in scored[:top_n]]


def _search_external_skills(query: str, top_n: int = 3) -> List[Dict[str, Any]]:
    """Search external ClawHub for skills. Returns top N matches."""
    try:
        # Run npx clawhub search
        result = subprocess.run(
            ["npx", "clawhub", "search", query, "--json"],
            capture_output=True,
            text=True,
            timeout=15,
        )
        if result.returncode != 0 or not result.stdout.strip():
            return []

        data = json.loads(result.stdout)
        if not isinstance(data, list):
            return []

        external = []
        for item in data[:top_n]:
            if isinstance(item, dict):
                external.append({
                    "name": item.get("name", "unknown"),
                    "description": item.get("description", "")[:200],
                    "category": item.get("category", "external"),
                    "source": "clawhub",
                })

        return external
    except subprocess.TimeoutExpired:
        logger.debug("External skill search timed out")
        return []
    except Exception as e:
        logger.debug("External skill search failed: %s", e)
        return []


def skill_search(
    query: str,
    include_external: bool = True,
    task_id: str = None,
) -> str:
    """
    Search for relevant skills by task description or keywords.

    Searches local skills first. If fewer than 3 local matches are found
    and include_external is True, also searches the ClawHub repository.

    Args:
        query: Task description or keywords (e.g., "post to twitter",
               "debug memory issue", "deploy server")
        include_external: Whether to search ClawHub if local results are
                         insufficient (default: True)
        task_id: Optional task identifier

    Returns:
        JSON string with matched skills (local + external)
    """
    if not query or not query.strip():
        return tool_error("query is required. Provide a task description or keywords.")

    query = query.strip()

    # Search local skills
    local_results = _search_local_skills(query, top_n=5)

    # Search external if needed
    external_results = []
    if include_external and len(local_results) < 3:
        external_results = _search_external_skills(query, top_n=3)

    # Build response
    response = {
        "success": True,
        "query": query,
        "local_skills": [
            {
                "name": s.get("name", ""),
                "description": s.get("description", "")[:200],
                "category": s.get("category", ""),
                "source": "local",
            }
            for s in local_results
        ],
        "external_skills": external_results,
        "hint": (
            "Load any skill with skill_view(name). "
            "External skills can be installed with skill_manage(action='create', ...)."
        ),
    }

    return json.dumps(response, ensure_ascii=False, indent=2)


# Register the tool
registry.register(
    name="skill_search",
    toolset="hermes-cli",
    schema={
        "type": "function",
        "function": {
            "name": "skill_search",
            "description": (
                "Search for relevant skills by task description or keywords. "
                "Searches both local skills and external ClawHub repository. "
                "Use this BEFORE starting any task to find the best skill for the job. "
                "Example queries: 'post to twitter', 'debug memory crash', "
                "'scrape website', 'send feishu message'."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": (
                            "Task description or keywords to search for. "
                            "Be specific about what you want to do."
                        ),
                    },
                    "include_external": {
                        "type": "boolean",
                        "description": "Whether to search ClawHub if local results are insufficient",
                        "default": True,
                    },
                },
                "required": ["query"],
            },
        },
    },
    handler=skill_search,
)
