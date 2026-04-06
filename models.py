"""Pydantic models for the automated code review environment."""

from __future__ import annotations

from typing import Literal

from openenv.core.env_server.types import Action, Observation, State
from pydantic import BaseModel, Field


ActionType = Literal[
    "view_pr",
    "inspect_file",
    "run_static_analysis",
    "run_tests",
    "read_policy",
    "request_ai_review",
    "submit_comment",
    "set_decision",
]

DecisionType = Literal["approve", "reject"]
SeverityType = Literal["info", "low", "medium", "high", "critical"]


class FindingRecord(BaseModel):
    """A structured issue discovered during review."""

    finding_id: str = Field(..., description="Stable finding identifier.")
    source: str = Field(..., description="Subsystem that surfaced the finding.")
    title: str = Field(..., description="Short finding title.")
    severity: SeverityType = Field(..., description="Finding severity.")
    file_path: str | None = Field(default=None, description="Related file path.")
    line: int | None = Field(default=None, description="Related line number.")
    description: str = Field(..., description="Detailed finding description.")
    required: bool = Field(default=False, description="Whether this finding must be raised.")
    false_positive: bool = Field(
        default=False, description="Whether this finding should be ignored."
    )


class ReviewArtifact(BaseModel):
    """A surfaced artifact such as a diff, policy note, or tool report."""

    artifact_type: str = Field(..., description="Artifact category.")
    title: str = Field(..., description="Human readable artifact title.")
    content: str = Field(..., description="Artifact body content.")


class ReviewComment(BaseModel):
    """A submitted review comment."""

    file_path: str | None = Field(default=None, description="Target file path.")
    finding_ids: list[str] = Field(
        default_factory=list, description="Finding IDs referenced by the comment."
    )
    comment_text: str = Field(..., description="Comment content posted by the agent.")
    blocker: bool = Field(default=False, description="Whether the comment blocks merge.")


class ActionTrace(BaseModel):
    """Compact trace of one action taken by the agent."""

    action_type: ActionType = Field(..., description="Executed action type.")
    summary: str = Field(..., description="Short action summary.")
    reward: float = Field(..., ge=0.0, le=1.0, description="Step reward awarded.")


class CodeReviewAction(Action):
    """Action model accepted by the code review environment."""

    action_type: ActionType = Field(..., description="The next review action to execute.")
    file_path: str | None = Field(default=None, description="File to inspect or annotate.")
    finding_ids: list[str] = Field(
        default_factory=list, description="Finding IDs referenced by the action."
    )
    comment_text: str | None = Field(default=None, description="Review comment text.")
    decision: DecisionType | None = Field(
        default=None, description="Final PR decision for set_decision."
    )


class CodeReviewObservation(Observation):
    """Observation returned after reset and each environment step."""

    task_id: str = Field(..., description="Current task identifier.")
    task_title: str = Field(..., description="Current task title.")
    difficulty: str = Field(..., description="Task difficulty.")
    objective: str = Field(..., description="Concrete objective for the agent.")
    pr_summary: str = Field(..., description="Pull request summary visible to the agent.")
    changed_files: list[str] = Field(
        default_factory=list, description="Changed file paths in the pull request."
    )
    discovered_findings: list[FindingRecord] = Field(
        default_factory=list, description="Findings surfaced so far."
    )
    revealed_artifacts: list[ReviewArtifact] = Field(
        default_factory=list, description="Artifacts unlocked so far."
    )
    comments_posted: list[ReviewComment] = Field(
        default_factory=list, description="Review comments posted so far."
    )
    remaining_steps: int = Field(
        default=0, ge=0, description="Number of steps left in the episode."
    )
    final_decision: DecisionType | None = Field(
        default=None, description="Final decision if already submitted."
    )
    last_action_error: str | None = Field(
        default=None, description="Validation or execution error from the last action."
    )


class CodeReviewState(State):
    """Internal environment state exposed through the state() API."""

    task_id: str = Field(..., description="Current task identifier.")
    task_title: str = Field(..., description="Current task title.")
    cumulative_reward: float = Field(
        default=0.0, ge=0.0, le=1.0, description="Normalized cumulative reward."
    )
    discovered_finding_ids: list[str] = Field(
        default_factory=list, description="Finding IDs discovered during the episode."
    )
    checks_run: list[str] = Field(
        default_factory=list, description="Review subsystems executed so far."
    )
    viewed_pr: bool = Field(default=False, description="Whether the PR summary was opened.")
    inspected_files: list[str] = Field(
        default_factory=list, description="Files inspected so far."
    )
    policy_read: bool = Field(default=False, description="Whether review policy was opened.")
    static_analysis_ran: bool = Field(
        default=False, description="Whether static analysis was executed."
    )
    tests_ran: bool = Field(default=False, description="Whether tests were executed.")
    ai_review_requested: bool = Field(
        default=False, description="Whether AI review was requested."
    )
    final_decision: DecisionType | None = Field(
        default=None, description="Final merge decision if present."
    )
    comments_posted: list[ReviewComment] = Field(
        default_factory=list, description="Submitted comments."
    )
    action_history: list[ActionTrace] = Field(
        default_factory=list, description="Compact action history."
    )

