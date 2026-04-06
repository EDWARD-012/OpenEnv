"""Deterministic CI and test runner output."""

from __future__ import annotations

from typing import Any

try:
    from ..task_fixtures import TaskConfig, findings_for_ids
except ImportError:  # pragma: no cover
    from task_fixtures import TaskConfig, findings_for_ids



def build_report(task: TaskConfig) -> dict[str, Any]:
    """Return the fixed CI report for a task."""
    report = task.get("test_results", {})
    return {
        "status": report.get("status", "unknown"),
        "summary": report.get("summary", "No test data available."),
        "checks": report.get("checks", []),
        "findings": findings_for_ids(task, report.get("finding_ids", [])),
    }
