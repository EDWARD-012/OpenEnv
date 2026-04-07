"""Inference helper tests."""

from __future__ import annotations

import asyncio

import inference
from inference import fallback_action, parse_model_action


class _FakeObservation:
    def __init__(self) -> None:
        self.objective = "Review"
        self.pr_summary = "Use view_pr to reveal the full pull request summary."
        self.changed_files = ["config/settings.py"]
        self.discovered_findings = []
        self.revealed_artifacts = []
        self.comments_posted = []
        self.remaining_steps = 8
        self.last_action_error = None


def test_parse_model_action_accepts_embedded_json() -> None:
    parsed = parse_model_action("next action -> {\"action_type\":\"view_pr\"}")
    assert parsed == {"action_type": "view_pr"}


def test_fallback_action_starts_with_view_pr() -> None:
    action = fallback_action("secret_leak_reject", _FakeObservation())
    assert action.action_type == "view_pr"


def test_run_task_handles_env_start_failure(monkeypatch, capsys) -> None:
    class _BrokenEnvClient:
        @classmethod
        async def from_docker_image(cls, image: str) -> None:
            raise RuntimeError(f"missing docker image: {image}")

    monkeypatch.setattr(inference, "CodeReviewEnv", _BrokenEnvClient)

    asyncio.run(inference.run_task("clean_refactor_approve"))

    output = capsys.readouterr().out.strip().splitlines()
    assert len(output) == 2
    assert output[0].startswith("[START] task=clean_refactor_approve ")
    assert output[1] == "[END] success=false steps=0 score=0.000 rewards="
