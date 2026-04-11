"""Unit tests for scoring helpers."""

from __future__ import annotations

import pytest

from scoring import clamp_reward, comment_reward, decision_reward, remaining_budget, weight


# Minimal task fixture stubs ------------------------------------------------

def _task(
    *,
    required_ids: list[str] | None = None,
    decision_weight: float = 0.3,
    comment_weight: float = 0.2,
    view_pr_weight: float = 0.05,
) -> dict:
    task: dict = {
        "reward_weights": {
            "decision": decision_weight,
            "comment_required_finding": comment_weight,
            "view_pr": view_pr_weight,
        },
        "required_comment_ids": required_ids if required_ids is not None else [],
        "expected_decision": "reject",
        "findings": [],
    }
    return task


class _Comment:
    """Minimal comment stub."""

    def __init__(self, finding_ids: list[str]) -> None:
        self.finding_ids = finding_ids


# ---------------------------------------------------------------------------
# clamp_reward
# ---------------------------------------------------------------------------


def test_clamp_reward_clamps_below_zero() -> None:
    assert clamp_reward(-5.0) == 0.0


def test_clamp_reward_clamps_above_one() -> None:
    assert clamp_reward(2.5) == 1.0


def test_clamp_reward_passes_through_midrange() -> None:
    assert clamp_reward(0.55) == pytest.approx(0.55)


def test_clamp_reward_exact_boundaries() -> None:
    assert clamp_reward(0.0) == 0.0
    assert clamp_reward(1.0) == 1.0


# ---------------------------------------------------------------------------
# weight
# ---------------------------------------------------------------------------


def test_weight_returns_configured_float() -> None:
    task = _task(view_pr_weight=0.05)
    assert weight(task, "view_pr") == pytest.approx(0.05)


def test_weight_returns_zero_for_missing_key() -> None:
    task = _task()
    assert weight(task, "nonexistent_key") == 0.0


# ---------------------------------------------------------------------------
# remaining_budget
# ---------------------------------------------------------------------------


def test_remaining_budget_at_zero() -> None:
    assert remaining_budget(0.0) == pytest.approx(1.0)


def test_remaining_budget_at_half() -> None:
    assert remaining_budget(0.5) == pytest.approx(0.5)


def test_remaining_budget_at_one() -> None:
    assert remaining_budget(1.0) == 0.0


# ---------------------------------------------------------------------------
# comment_reward
# ---------------------------------------------------------------------------


def test_comment_reward_zero_when_no_required_ids() -> None:
    task = _task(required_ids=[])
    assert comment_reward(task, newly_covered_required=1) == 0.0


def test_comment_reward_zero_when_newly_covered_is_zero() -> None:
    task = _task(required_ids=["hardcoded_secret"])
    assert comment_reward(task, newly_covered_required=0) == 0.0


def test_comment_reward_full_when_all_covered() -> None:
    task = _task(required_ids=["hardcoded_secret"], comment_weight=0.2)
    reward = comment_reward(task, newly_covered_required=1)
    assert reward == pytest.approx(0.2)


def test_comment_reward_partial_when_half_covered() -> None:
    task = _task(required_ids=["a", "b"], comment_weight=0.2)
    reward = comment_reward(task, newly_covered_required=1)
    assert reward == pytest.approx(0.1)


# ---------------------------------------------------------------------------
# decision_reward
# ---------------------------------------------------------------------------


def test_decision_reward_zero_for_wrong_decision() -> None:
    task = _task(required_ids=[], decision_weight=0.3)
    # expected_decision is "reject", so "approve" is wrong
    assert decision_reward(task, "approve", required_coverage_ratio=1.0) == 0.0


def test_decision_reward_full_when_no_required_ids() -> None:
    task = _task(required_ids=[], decision_weight=0.35)
    reward = decision_reward(task, "reject", required_coverage_ratio=1.0)
    assert reward == pytest.approx(0.35)


def test_decision_reward_partial_when_no_comments_posted() -> None:
    task = _task(required_ids=["auth_scope_gap", "missing_auth_tests"], decision_weight=0.3)
    # 0% coverage → partial: 0.3 * (0.35 + 0.65 * 0.0) = 0.3 * 0.35 = 0.105
    reward = decision_reward(task, "reject", required_coverage_ratio=0.0)
    assert reward == pytest.approx(0.105)


def test_decision_reward_full_when_all_comments_posted() -> None:
    task = _task(required_ids=["auth_scope_gap", "missing_auth_tests"], decision_weight=0.3)
    # 100% coverage → 0.3 * (0.35 + 0.65 * 1.0) = 0.3 * 1.0 = 0.3
    reward = decision_reward(task, "reject", required_coverage_ratio=1.0)
    assert reward == pytest.approx(0.3)


def test_decision_reward_none_decision_returns_zero() -> None:
    task = _task(required_ids=[], decision_weight=0.3)
    assert decision_reward(task, None, required_coverage_ratio=1.0) == 0.0
