"""Deterministic static analysis reports."""

from __future__ import annotations

from typing import Any

try:
    from ..task_fixtures import TaskConfig, findings_for_ids
except ImportError:  # pragma: no cover
    from task_fixtures import TaskConfig, findings_for_ids


def build_report(task: TaskConfig) -> dict[str, Any]:
    """Return the static analysis report for a task."""
    report = task.get("static_analysis", {})
    return {
        "summary": report.get("summary", "No static analysis findings."),
        "findings": findings_for_ids(task, report.get("finding_ids", [])),
    }
