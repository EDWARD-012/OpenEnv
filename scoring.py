"""Reward shaping helpers for deterministic grading."""

from __future__ import annotations

try:
    from .policy import decision_is_correct, required_comment_ids
    from .task_fixtures import TaskConfig
except ImportError:  # pragma: no cover
    from policy import decision_is_correct, required_comment_ids
    from task_fixtures import TaskConfig


def weight(task: TaskConfig, key: str) -> float:
    """Return a configured reward weight rounded for stable scoring."""
    reward_weights = task.get("reward_weights", {})
    return round(float(reward_weights.get(key, 0.0)), 4)


def clamp_reward(value: float) -> float:
    """Clamp reward values into the [0.0, 1.0] range for stable step-level scoring."""
    return round(min(max(value, 0.0), 1.0), 4)


def comment_reward(task: TaskConfig, newly_covered_required: int) -> float:
    """Compute reward for newly covered required findings."""
    required_count = len(required_comment_ids(task))
    if required_count == 0 or newly_covered_required <= 0:
        return 0.0
    total_weight = weight(task, "comment_required_finding")
    return clamp_reward(total_weight * (newly_covered_required / required_count))


def decision_reward(
    task: TaskConfig, decision: str | None, required_coverage_ratio: float
) -> float:
    """Compute reward for the final decision with partial credit."""
    if not decision_is_correct(task, decision):
        return 0.0
    if not required_comment_ids(task):
        return weight(task, "decision")
    coverage_ratio = min(max(required_coverage_ratio, 0.0), 1.0)
    return clamp_reward(weight(task, "decision") * (0.35 + (0.65 * coverage_ratio)))


def remaining_budget(current_total: float) -> float:
    """Return how much cumulative reward remains before reaching 1.0."""
    return clamp_reward(1.0 - current_total)
