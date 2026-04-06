"""Helpers for loading deterministic task fixtures."""

from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path
from typing import Any


ROOT_DIR = Path(__file__).resolve().parent
TASKS_DIR = ROOT_DIR / "fixtures" / "tasks"
TASK_ORDER = [
    "clean_refactor_approve",
    "secret_leak_reject",
    "auth_policy_reject",
]


TaskConfig = dict[str, Any]


@lru_cache(maxsize=None)
def load_task(task_id: str) -> TaskConfig:
    """Load one task fixture by identifier."""
    fixture_path = TASKS_DIR / f"{task_id}.json"
    if not fixture_path.exists():
        raise KeyError(f"Unknown task_id: {task_id}")
    with fixture_path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def list_task_ids() -> list[str]:
    """Return task ids in evaluation order."""
    return list(TASK_ORDER)


def list_tasks() -> list[TaskConfig]:
    """Return all task fixtures in evaluation order."""
    return [load_task(task_id) for task_id in TASK_ORDER]


def default_task() -> TaskConfig:
    """Return the default task used by reset()."""
    return load_task(TASK_ORDER[0])


def finding_map(task: TaskConfig) -> dict[str, TaskConfig]:
    """Index task findings by finding id."""
    return {finding["finding_id"]: finding for finding in task.get("findings", [])}


def findings_for_ids(task: TaskConfig, finding_ids: list[str]) -> list[TaskConfig]:
    """Resolve finding payloads in the order requested."""
    indexed = finding_map(task)
    return [indexed[finding_id] for finding_id in finding_ids if finding_id in indexed]


def file_map(task: TaskConfig) -> dict[str, TaskConfig]:
    """Index changed files by file path."""
    changed_files = task.get("pull_request", {}).get("changed_files", [])
    return {changed_file["path"]: changed_file for changed_file in changed_files}

