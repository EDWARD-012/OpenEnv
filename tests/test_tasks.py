"""Task fixture regression tests."""

from __future__ import annotations

from server.code_review_environment import CodeReviewEnvironment
from models import CodeReviewAction


def test_clean_refactor_optimal_path_scores_one() -> None:
    env = CodeReviewEnvironment()
    env.reset(task_id="clean_refactor_approve")

    actions = [
        CodeReviewAction(action_type="view_pr"),
        CodeReviewAction(action_type="inspect_file", file_path="src/formatting/formatter.py"),
        CodeReviewAction(action_type="run_static_analysis"),
        CodeReviewAction(action_type="run_tests"),
        CodeReviewAction(action_type="read_policy"),
        CodeReviewAction(action_type="request_ai_review"),
        CodeReviewAction(action_type="set_decision", decision="approve"),
    ]

    observation = None
    for action in actions:
        observation = env.step(action)

    assert observation is not None
    assert observation.final_decision == "approve"
    assert env.state.cumulative_reward == 1.0


def test_secret_leak_requires_blocking_comment() -> None:
    env = CodeReviewEnvironment()
    env.reset(task_id="secret_leak_reject")

    env.step(CodeReviewAction(action_type="view_pr"))
    env.step(CodeReviewAction(action_type="inspect_file", file_path="config/settings.py"))
    env.step(CodeReviewAction(action_type="run_static_analysis"))
    env.step(CodeReviewAction(action_type="run_tests"))
    env.step(CodeReviewAction(action_type="read_policy"))
    env.step(CodeReviewAction(action_type="request_ai_review"))
    env.step(
        CodeReviewAction(
            action_type="submit_comment",
            file_path="config/settings.py",
            finding_ids=["hardcoded_secret"],
            comment_text="Blocking: hardcoded secret must be removed.",
        )
    )
    observation = env.step(CodeReviewAction(action_type="set_decision", decision="reject"))

    assert observation.done is True
    assert observation.final_decision == "reject"
    assert env.state.cumulative_reward == 1.0


def test_hard_task_ignores_false_positive_and_covers_required_findings() -> None:
    env = CodeReviewEnvironment()
    env.reset(task_id="auth_policy_reject")

    actions = [
        CodeReviewAction(action_type="view_pr"),
        CodeReviewAction(action_type="inspect_file", file_path="auth/middleware.py"),
        CodeReviewAction(action_type="run_static_analysis"),
        CodeReviewAction(action_type="run_tests"),
        CodeReviewAction(action_type="read_policy"),
        CodeReviewAction(action_type="request_ai_review"),
        CodeReviewAction(
            action_type="submit_comment",
            file_path="auth/middleware.py",
            finding_ids=["auth_scope_gap", "missing_auth_tests"],
            comment_text="Blocking: auth scope gate removal and missing denied-path tests.",
        ),
        CodeReviewAction(action_type="set_decision", decision="reject"),
    ]

    observation = None
    for action in actions:
        observation = env.step(action)

    assert observation is not None
    assert observation.done is True
    assert "unused_session_variable" in {
        finding.finding_id for finding in observation.discovered_findings
    }
    assert env.state.cumulative_reward == 1.0


def test_dependency_confusion_optimal_path_scores_one() -> None:
    env = CodeReviewEnvironment()
    env.reset(task_id="dependency_confusion_reject")

    actions = [
        CodeReviewAction(action_type="view_pr"),
        CodeReviewAction(action_type="inspect_file", file_path="requirements.txt"),
        CodeReviewAction(action_type="run_static_analysis"),
        CodeReviewAction(action_type="run_tests"),
        CodeReviewAction(action_type="read_policy"),
        CodeReviewAction(action_type="request_ai_review"),
        CodeReviewAction(
            action_type="submit_comment",
            file_path="requirements.txt",
            finding_ids=["shadowed_internal_package"],
            comment_text="Blocking: shadowed internal package resolves to unknown public PyPI owner.",
        ),
        CodeReviewAction(action_type="set_decision", decision="reject"),
    ]

    observation = None
    for action in actions:
        observation = env.step(action)

    assert observation is not None
    assert observation.done is True
    assert observation.final_decision == "reject"
    assert "shadowed_internal_package" in {
        finding.finding_id for finding in observation.discovered_findings
    }
    assert env.state.cumulative_reward == 1.0


def test_dependency_confusion_wrong_decision_scores_zero_decision() -> None:
    """Approving a PR with a shadowed package should not award decision reward."""
    env = CodeReviewEnvironment()
    env.reset(task_id="dependency_confusion_reject")

    env.step(CodeReviewAction(action_type="view_pr"))
    env.step(CodeReviewAction(action_type="run_static_analysis"))
    observation = env.step(CodeReviewAction(action_type="set_decision", decision="approve"))

    assert observation.done is True
    assert observation.final_decision == "approve"
    # Decision reward should be 0 for wrong decision
    assert env.state.cumulative_reward < 0.5
