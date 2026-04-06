"""Environment lifecycle and reward-bound tests."""

from __future__ import annotations

from server.code_review_environment import CodeReviewEnvironment
from models import CodeReviewAction


def test_reset_and_state_lifecycle() -> None:
    env = CodeReviewEnvironment()
    observation = env.reset(task_id="clean_refactor_approve")
    assert observation.task_id == "clean_refactor_approve"
    assert env.state.step_count == 0

    observation = env.step(CodeReviewAction(action_type="view_pr"))
    assert observation.pr_summary.startswith("A low-risk refactor")
    assert env.state.step_count == 1
    assert env.state.viewed_pr is True


def test_max_step_termination() -> None:
    env = CodeReviewEnvironment()
    env.reset(task_id="clean_refactor_approve")

    observation = None
    for _ in range(8):
        observation = env.step(CodeReviewAction(action_type="view_pr"))

    assert observation is not None
    assert observation.done is True
    assert observation.last_action_error == "Step budget exhausted before final decision"


def test_cumulative_reward_stays_normalized() -> None:
    env = CodeReviewEnvironment()
    env.reset(task_id="secret_leak_reject")

    actions = [
        CodeReviewAction(action_type="view_pr"),
        CodeReviewAction(action_type="inspect_file", file_path="config/settings.py"),
        CodeReviewAction(action_type="run_static_analysis"),
        CodeReviewAction(action_type="read_policy"),
        CodeReviewAction(
            action_type="submit_comment",
            file_path="config/settings.py",
            finding_ids=["hardcoded_secret"],
            comment_text="Blocking issue.",
        ),
        CodeReviewAction(action_type="set_decision", decision="reject"),
    ]

    for action in actions:
        env.step(action)

    assert 0.0 <= env.state.cumulative_reward <= 1.0

