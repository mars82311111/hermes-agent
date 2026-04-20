"""Scope Isolation — Access control for memory systems.

Scopes define who/what can access memory content:
  - global       — accessible to all agents/projects/users
  - agent:xxx    — accessible only to agent xxx
  - project:xxx  — accessible only to project xxx
  - user:xxx     — accessible only to user xxx

Rule: A requester can access memory if:
  1. memory_scope == 'global', OR
  2. memory_scope matches requester_scope (prefix match), OR
  3. requester_scope == 'global'
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from enum import Enum
from typing import Optional


class ScopeType(Enum):
    GLOBAL = "global"
    AGENT = "agent"
    PROJECT = "project"
    USER = "user"


@dataclass
class Scope:
    """Represents a memory access scope."""
    type: ScopeType
    value: str  # e.g., "hermes", "my-project", "mars"

    @classmethod
    def parse(cls, scope_str: str) -> "Scope":
        """Parse a scope string like 'global', 'agent:hermes', 'project:my-app'."""
        if scope_str == "global":
            return cls(type=ScopeType.GLOBAL, value="global")
        if ":" in scope_str:
            scope_type_str, _, value = scope_str.partition(":")
            try:
                scope_type = ScopeType(scope_type_str)
                return cls(type=scope_type, value=value)
            except ValueError:
                pass
        # Default to user scope for backwards compat
        return cls(type=ScopeType.USER, value=scope_str)

    @classmethod
    def global_scope(cls) -> "Scope":
        return cls(type=ScopeType.GLOBAL, value="global")

    @classmethod
    def agent_scope(cls, agent_id: str) -> "Scope":
        return cls(type=ScopeType.AGENT, value=agent_id)

    @classmethod
    def project_scope(cls, project_id: str) -> "Scope":
        return cls(type=ScopeType.PROJECT, value=project_id)

    @classmethod
    def user_scope(cls, user_id: str) -> "Scope":
        return cls(type=ScopeType.USER, value=user_id)

    def to_string(self) -> str:
        """Convert scope to string representation."""
        if self.type == ScopeType.GLOBAL:
            return "global"
        return f"{self.type.value}:{self.value}"

    def matches(self, other: "Scope") -> bool:
        """Check if this scope matches or is compatible with another scope."""
        # Global matches everything
        if self.type == ScopeType.GLOBAL or other.type == ScopeType.GLOBAL:
            return True
        # Same type and value
        if self.type == other.type and self.value == other.value:
            return True
        return False

    def __str__(self) -> str:
        return self.to_string()

    def __repr__(self) -> str:
        return f"Scope({self.to_string()})"


def can_access(memory_scope: str, requester_scope: str) -> bool:
    """Check if a requester can access memory with the given scope.

    Args:
        memory_scope: The scope of the memory resource (e.g., 'global', 'agent:hermes')
        requester_scope: The scope of the requester

    Returns:
        True if access is allowed, False otherwise
    """
    mem = Scope.parse(memory_scope)
    req = Scope.parse(requester_scope)

    # Rule 1: Memory is global
    if mem.type == ScopeType.GLOBAL:
        return True

    # Rule 2: Requester is global (can access anything)
    if req.type == ScopeType.GLOBAL:
        return True

    # Rule 3: Exact match
    if mem.type == req.type and mem.value == req.value:
        return True

    return False


def filter_by_scope(items: list, scope_attr: str, requester_scope: str) -> list:
    """Filter a list of items by scope compatibility.

    Args:
        items: List of dicts or objects with scope attribute
        scope_attr: Attribute name containing the scope string
        requester_scope: The requester's scope

    Returns:
        Filtered list of items the requester can access
    """
    req = Scope.parse(requester_scope)

    filtered = []
    for item in items:
        if hasattr(item, scope_attr):
            item_scope = getattr(item, scope_attr)
        elif isinstance(item, dict):
            item_scope = item.get(scope_attr, "global")
        else:
            continue

        if can_access(item_scope, requester_scope):
            filtered.append(item)

    return filtered


class ScopeGuard:
    """Context guard for enforcing scope in code blocks."""

    def __init__(self, requester_scope: str):
        self.requester = Scope.parse(requester_scope)
        self._original_scope: Optional[str] = None

    def check(self, memory_scope: str) -> None:
        """Raise PermissionError if access is denied."""
        if not can_access(memory_scope, self.requester.to_string()):
            raise PermissionError(
                f"Access denied: scope '{self.requester.to_string()}' "
                f"cannot access memory with scope '{memory_scope}'"
            )

    def filter_list(self, items: list, scope_attr: str = "scope") -> list:
        """Filter list by scope compatibility."""
        return filter_by_scope(items, scope_attr, self.requester.to_string())

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        return False
