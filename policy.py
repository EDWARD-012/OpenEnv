"""Policy helpers for deterministic review grading."""

from __future__ import annotations

from typing import Iterable

try:
    from .task_fixtures import TaskConfig
except ImportError:  # pragma: no cover
    from task_fixtures import TaskConfig


def format_policy(task: TaskConfig) -> str:
    """Return the review policy as a newline separated string."""
    lines = task.get("policy_text", [])
    return "\n".join(f"- {line}" for line in lines)


def expected_decision(task: TaskConfig) -> str:
    """Return the expected final decision for a task."""
    return str(task.get("expected_decision", "reject"))


def required_comment_ids(task: TaskConfig) -> set[str]:
    """Return finding IDs that must be covered by review comments."""
    return set(task.get("required_comment_ids", []))


def false_positive_ids(task: TaskConfig) -> set[str]:
    """Return finding IDs that should not be treated as blockers."""
    return {
        finding["finding_id"]
        for finding in task.get("findings", [])
        if finding.get("false_positive", False)
    }


def covered_required_ids(task: TaskConfig, comments: Iterable[object]) -> set[str]:
    """Collect required finding IDs covered by posted comments."""
    required = required_comment_ids(task)
    covered: set[str] = set()
    for comment in comments:
        finding_ids = getattr(comment, "finding_ids", []) or []
        for finding_id in finding_ids:
            if finding_id in required:
                covered.add(finding_id)
    return covered


def missing_required_ids(task: TaskConfig, comments: Iterable[object]) -> list[str]:
    """Return required finding IDs that still need coverage."""
    missing = required_comment_ids(task) - covered_required_ids(task, comments)
    return sorted(missing)


def decision_is_correct(task: TaskConfig, decision: str | None) -> bool:
    """Return whether the submitted decision matches the task expectation."""
    return decision is not None and decision == expected_decision(task)
