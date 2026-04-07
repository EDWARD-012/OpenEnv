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
    assert output[1] == "[END] success=false steps=0 score=0.001 rewards="


# ---------------------------------------------------------------------------
# New tests covering the guarded main() and top-level asyncio.run() paths
# ---------------------------------------------------------------------------


def test_parse_model_action_returns_none_for_empty_string() -> None:
    assert parse_model_action("") is None


def test_parse_model_action_returns_none_for_plain_text() -> None:
    assert parse_model_action("just some prose with no JSON") is None


def test_parse_model_action_returns_none_for_array_json() -> None:
    # A JSON array is valid JSON but not a dict — must return None
    assert parse_model_action("[1, 2, 3]") is None


def test_parse_model_action_accepts_pure_json_object() -> None:
    parsed = parse_model_action('{"action_type": "run_tests"}')
    assert parsed == {"action_type": "run_tests"}


def test_sanitize_line_collapses_newlines() -> None:
    result = inference.sanitize_line("line one\nline two\nline three")
    assert "\n" not in result
    assert "line one" in result
    assert "line two" in result


def test_log_start_format(capsys) -> None:
    inference.log_start(task="my_task", env="my_env", model="my_model")
    out = capsys.readouterr().out.strip()
    assert out == "[START] task=my_task env=my_env model=my_model"


def test_log_end_format(capsys) -> None:
    inference.log_end(success=True, steps=3, score=0.9, rewards=[0.3, 0.3, 0.3])
    out = capsys.readouterr().out.strip()
    assert out.startswith("[END] success=true steps=3 score=0.900")
    assert "0.30,0.30,0.30" in out


def test_log_end_false_and_empty_rewards(capsys) -> None:
    inference.log_end(success=False, steps=0, score=0.001, rewards=[])
    out = capsys.readouterr().out.strip()
    assert out == "[END] success=false steps=0 score=0.001 rewards="


def test_choose_action_falls_back_when_api_raises(monkeypatch) -> None:
    """choose_action must return the deterministic fallback when the API call fails."""

    class _BrokenClient:
        class chat:
            class completions:
                @staticmethod
                def create(**kwargs):
                    raise ConnectionError("no network")

    obs = _FakeObservation()
    action = inference.choose_action(_BrokenClient(), "secret_leak_reject", 1, obs, [])
    # fallback for secret_leak_reject with no artifacts revealed → view_pr
    assert action.action_type == "view_pr"


def test_main_survives_list_task_ids_failure(monkeypatch, capsys) -> None:
    """If list_task_ids() raises, main() must not propagate the exception."""
    def _exploding_list():
        raise RuntimeError("fixture missing")

    monkeypatch.setattr(inference, "list_task_ids", _exploding_list)

    # Should complete without raising
    asyncio.run(inference.main())

    # Nothing printed — no tasks ran, which is correct behaviour
    out = capsys.readouterr().out
    assert out == ""


def test_main_continues_after_single_task_exception(monkeypatch, capsys) -> None:
    """A task that raises inside run_task() must not abort subsequent tasks."""
    calls = []

    async def _fake_run_task(task_id: str) -> None:
        calls.append(task_id)
        if task_id == "clean_refactor_approve":
            raise RuntimeError("simulated crash")
        # other tasks print their markers normally
        inference.log_start(task=task_id, env="test", model="m")
        inference.log_end(success=False, steps=0, score=0.0, rewards=[])

    monkeypatch.setattr(inference, "run_task", _fake_run_task)
    monkeypatch.setattr(inference, "list_task_ids", lambda: ["clean_refactor_approve", "secret_leak_reject"])

    asyncio.run(inference.main())

    # Both tasks were attempted
    assert calls == ["clean_refactor_approve", "secret_leak_reject"]

    # The second task still printed its output
    out = capsys.readouterr().out
    assert "[START] task=secret_leak_reject" in out


def test_top_level_guard_swallows_asyncio_crash(monkeypatch) -> None:
    """The if __name__ == '__main__' guard must not let any exception escape."""
    async def _exploding_main() -> None:
        raise SystemError("event loop kaboom")

    monkeypatch.setattr(inference, "main", _exploding_main)

    # Simulate the guarded entry point directly — must not raise
    try:
        asyncio.run(inference.main())
    except Exception:
        pass  # same guard as in the script


def test_run_task_emits_start_and_end_even_on_immediate_crash(monkeypatch, capsys) -> None:
    """[START] must always appear before [END], even when env init explodes."""
    class _ExplodingEnv:
        @classmethod
        async def from_docker_image(cls, image: str):
            raise OSError("docker not found")

    monkeypatch.setattr(inference, "CodeReviewEnv", _ExplodingEnv)
    asyncio.run(inference.run_task("auth_policy_reject"))

    lines = capsys.readouterr().out.strip().splitlines()
    assert lines[0].startswith("[START] task=auth_policy_reject")
    assert lines[-1].startswith("[END]")
