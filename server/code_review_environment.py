"""Deterministic OpenEnv environment for automated code review."""

from __future__ import annotations

from typing import Any
from uuid import uuid4

from openenv.core.env_server.interfaces import Environment
from openenv.core.env_server.types import EnvironmentMetadata

try:
    from ..engines.ai_reviewer import build_report as build_ai_review_report
    from ..engines.static_analysis import build_report as build_static_analysis_report
    from ..engines.test_runner import build_report as build_test_report
    from ..models import (
        ActionTrace,
        CodeReviewAction,
        CodeReviewObservation,
        CodeReviewState,
        FindingRecord,
        ReviewArtifact,
        ReviewComment,
    )
    from ..policy import (
        covered_required_ids,
        decision_is_correct,
        format_policy,
        missing_required_ids,
        required_comment_ids,
    )
    from ..scoring import clamp_reward, comment_reward, decision_reward, remaining_budget, step_efficiency_score, weight
    from ..task_fixtures import default_task, file_map, findings_for_ids, load_task
except ImportError:  # pragma: no cover
    from engines.ai_reviewer import build_report as build_ai_review_report
    from engines.static_analysis import build_report as build_static_analysis_report
    from engines.test_runner import build_report as build_test_report
    from models import (
        ActionTrace,
        CodeReviewAction,
        CodeReviewObservation,
        CodeReviewState,
        FindingRecord,
        ReviewArtifact,
        ReviewComment,
    )
    from policy import (
        covered_required_ids,
        decision_is_correct,
        format_policy,
        missing_required_ids,
        required_comment_ids,
    )
    from scoring import clamp_reward, comment_reward, decision_reward, remaining_budget, step_efficiency_score, weight
    from task_fixtures import default_task, file_map, findings_for_ids, load_task


class CodeReviewEnvironment(Environment[CodeReviewAction, CodeReviewObservation, CodeReviewState]):
    """A deterministic pull request review simulation with structured rewards."""

    SUPPORTS_CONCURRENT_SESSIONS: bool = True

    def __init__(self) -> None:
        super().__init__()
        self._task = default_task()
        self._revealed_artifacts: list[ReviewArtifact] = []
        self._state = self._new_state(self._task, episode_id=str(uuid4()))

    def _new_state(self, task: dict[str, Any], episode_id: str | None = None) -> CodeReviewState:
        return CodeReviewState(
            episode_id=episode_id or str(uuid4()),
            step_count=0,
            task_id=task["task_id"],
            task_title=task["title"],
            cumulative_reward=0.0,
        )

    def _task_changed_files(self) -> list[str]:
        changed_files = self._task.get("pull_request", {}).get("changed_files", [])
        return [changed_file["path"] for changed_file in changed_files]

    def _current_findings(self) -> list[FindingRecord]:
        if not self._state.discovered_finding_ids:
            return []
        raw_findings = findings_for_ids(self._task, self._state.discovered_finding_ids)
        return [FindingRecord.model_validate(raw_finding) for raw_finding in raw_findings]

    def _observation(self, reward: float = 0.0, done: bool = False, error: str | None = None) -> CodeReviewObservation:
        pr_summary = self._task.get("pull_request", {}).get("summary", "")
        if not self._state.viewed_pr:
            pr_summary = "Use view_pr to reveal the full pull request summary."

        return CodeReviewObservation(
            task_id=self._task["task_id"],
            task_title=self._task["title"],
            difficulty=self._task["difficulty"],
            objective=self._task["objective"],
            pr_summary=pr_summary,
            changed_files=self._task_changed_files(),
            discovered_findings=self._current_findings(),
            revealed_artifacts=list(self._revealed_artifacts),
            comments_posted=list(self._state.comments_posted),
            remaining_steps=max(self._task["max_steps"] - self._state.step_count, 0),
            final_decision=self._state.final_decision,
            last_action_error=error,
            reward=reward,
            done=done,
        )

    def _add_artifact(self, artifact_type: str, title: str, content: str) -> None:
        self._revealed_artifacts.append(
            ReviewArtifact(
                artifact_type=artifact_type,
                title=title,
                content=content,
            )
        )

    def _apply_reward(self, reward: float) -> float:
        bounded_reward = min(clamp_reward(reward), remaining_budget(self._state.cumulative_reward))
        self._state.cumulative_reward = clamp_reward(self._state.cumulative_reward + bounded_reward)
        return bounded_reward

    def _append_trace(self, action: CodeReviewAction, reward: float, summary: str) -> None:
        self._state.action_history.append(
            ActionTrace(action_type=action.action_type, summary=summary, reward=reward)
        )

    def _discover_findings(self, finding_ids: list[str]) -> None:
        for finding_id in finding_ids:
            if finding_id not in self._state.discovered_finding_ids:
                self._state.discovered_finding_ids.append(finding_id)

    def _mark_check(self, check_name: str) -> None:
        if check_name not in self._state.checks_run:
            self._state.checks_run.append(check_name)

    def reset(
        self,
        seed: int | None = None,
        episode_id: str | None = None,
        task_id: str | None = None,
        **kwargs: Any,
    ) -> CodeReviewObservation:
        del seed, kwargs
        self._task = load_task(task_id) if task_id else default_task()
        self._revealed_artifacts = []
        self._state = self._new_state(self._task, episode_id=episode_id)
        return self._observation()

    def _run_view_pr(self) -> tuple[float, str]:
        if self._state.viewed_pr:
            return 0.0, "view_pr already executed"
        self._state.viewed_pr = True
        pull_request = self._task["pull_request"]
        content = (
            f"PR #{pull_request['pr_number']}: {pull_request['title']}\n"
            f"Author: {pull_request['author']}\n"
            f"Target branch: {pull_request['target_branch']}\n"
            f"Description: {pull_request['description']}"
        )
        self._add_artifact("pull_request", "Pull request overview", content)
        return weight(self._task, "view_pr"), "Revealed pull request overview"

    def _run_inspect_file(self, file_path: str | None) -> tuple[float, str, str | None]:
        if not file_path:
            return 0.0, "inspect_file missing file_path", "file_path is required for inspect_file"
        changed_files = file_map(self._task)
        changed_file = changed_files.get(file_path)
        if changed_file is None:
            return 0.0, "inspect_file target not found", f"Unknown changed file: {file_path}"
        if file_path in self._state.inspected_files:
            return 0.0, f"{file_path} already inspected", None
        self._state.inspected_files.append(file_path)
        self._add_artifact(
            "file_diff",
            f"Diff for {file_path}",
            f"{changed_file['summary']}\n\n{changed_file['diff']}",
        )
        inspect_reward = float(self._task.get("inspect_rewards", {}).get(file_path, 0.0))
        return inspect_reward, f"Inspected {file_path}", None

    def _run_static_analysis(self) -> tuple[float, str]:
        if self._state.static_analysis_ran:
            return 0.0, "static analysis already executed"
        self._state.static_analysis_ran = True
        self._mark_check("static_analysis")
        report = build_static_analysis_report(self._task)
        finding_ids = [finding["finding_id"] for finding in report["findings"]]
        self._discover_findings(finding_ids)
        content = report["summary"]
        if report["findings"]:
            lines = [
                f"[{finding['severity']}] {finding['finding_id']} - {finding['description']}"
                for finding in report["findings"]
            ]
            content = f"{content}\n" + "\n".join(lines)
        self._add_artifact("static_analysis", "Static analysis report", content)
        return weight(self._task, "run_static_analysis"), "Executed static analysis"

    def _run_tests(self) -> tuple[float, str]:
        if self._state.tests_ran:
            return 0.0, "tests already executed"
        self._state.tests_ran = True
        self._mark_check("tests")
        report = build_test_report(self._task)
        finding_ids = [finding["finding_id"] for finding in report["findings"]]
        self._discover_findings(finding_ids)
        checks_text = ", ".join(report["checks"]) if report["checks"] else "no checks"
        findings_text = "\n".join(
            f"[{finding['severity']}] {finding['finding_id']} - {finding['description']}"
            for finding in report["findings"]
        )
        content = (
            f"Status: {report['status']}\n"
            f"Checks: {checks_text}\n"
            f"Summary: {report['summary']}"
        )
        if findings_text:
            content += f"\n{findings_text}"
        self._add_artifact("tests", "CI and test report", content)
        return weight(self._task, "run_tests"), "Executed CI and tests"

    def _run_policy(self) -> tuple[float, str]:
        if self._state.policy_read:
            return 0.0, "policy already read"
        self._state.policy_read = True
        self._mark_check("policy")
        self._add_artifact("policy", "Review policy", format_policy(self._task))
        return weight(self._task, "read_policy"), "Read repository review policy"

    def _run_ai_review(self) -> tuple[float, str]:
        if self._state.ai_review_requested:
            return 0.0, "AI review already requested"
        self._state.ai_review_requested = True
        self._mark_check("ai_review")
        report = build_ai_review_report(self._task)
        finding_ids = [finding["finding_id"] for finding in report["findings"]]
        self._discover_findings(finding_ids)
        content = report["summary"]
        if report["findings"]:
            lines = [
                f"[{finding['severity']}] {finding['finding_id']} - {finding['description']}"
                for finding in report["findings"]
            ]
            content = f"{content}\n" + "\n".join(lines)
        self._add_artifact("ai_review", "AI reviewer notes", content)
        return weight(self._task, "request_ai_review"), "Requested AI reviewer notes"

    def _run_submit_comment(
        self, action: CodeReviewAction
    ) -> tuple[float, str, str | None]:
        if not action.comment_text:
            return 0.0, "submit_comment missing comment_text", "comment_text is required for submit_comment"
        if not action.finding_ids:
            return 0.0, "submit_comment missing finding_ids", "finding_ids are required for submit_comment"

        undiscovered = [
            finding_id
            for finding_id in action.finding_ids
            if finding_id not in self._state.discovered_finding_ids
        ]
        if undiscovered:
            return (
                0.0,
                "submit_comment referenced undiscovered findings",
                f"Undiscovered findings: {', '.join(sorted(undiscovered))}",
            )

        existing_covered = covered_required_ids(self._task, self._state.comments_posted)
        comment = ReviewComment(
            file_path=action.file_path,
            finding_ids=list(dict.fromkeys(action.finding_ids)),
            comment_text=action.comment_text,
            blocker=any(
                finding_id in required_comment_ids(self._task)
                for finding_id in action.finding_ids
            ),
        )
        self._state.comments_posted.append(comment)
        updated_covered = covered_required_ids(self._task, self._state.comments_posted)
        newly_covered_count = len(updated_covered - existing_covered)
        reward = comment_reward(self._task, newly_covered_count)
        return reward, "Submitted review comment", None

    def _run_set_decision(
        self, action: CodeReviewAction
    ) -> tuple[float, str, bool, str | None]:
        if action.decision is None:
            return 0.0, "set_decision missing decision", True, "decision is required for set_decision"
        self._state.final_decision = action.decision
        covered = covered_required_ids(self._task, self._state.comments_posted)
        required = required_comment_ids(self._task)
        coverage_ratio = 1.0 if not required else len(covered) / len(required)
        reward = decision_reward(self._task, action.decision, coverage_ratio)
        if decision_is_correct(self._task, action.decision):
            if required and len(covered) < len(required):
                missing = ", ".join(missing_required_ids(self._task, self._state.comments_posted))
                summary = f"Submitted correct decision but missed required findings: {missing}"
            else:
                summary = f"Submitted correct final decision: {action.decision}"
        else:
            summary = f"Submitted incorrect final decision: {action.decision}"
        return reward, summary, True, None

    def step(self, action: CodeReviewAction, timeout_s: float | None = None, **kwargs: Any) -> CodeReviewObservation:
        del timeout_s, kwargs
        if self._state.final_decision is not None:
            return self._observation(done=True, error="Episode already finished")

        self._state.step_count += 1

        raw_reward = 0.0
        done = False
        error: str | None = None
        summary = f"No-op action: {action.action_type}"

        if action.action_type == "view_pr":
            raw_reward, summary = self._run_view_pr()
        elif action.action_type == "inspect_file":
            raw_reward, summary, error = self._run_inspect_file(action.file_path)
        elif action.action_type == "run_static_analysis":
            raw_reward, summary = self._run_static_analysis()
        elif action.action_type == "run_tests":
            raw_reward, summary = self._run_tests()
        elif action.action_type == "read_policy":
            raw_reward, summary = self._run_policy()
        elif action.action_type == "request_ai_review":
            raw_reward, summary = self._run_ai_review()
        elif action.action_type == "submit_comment":
            raw_reward, summary, error = self._run_submit_comment(action)
        elif action.action_type == "set_decision":
            raw_reward, summary, done, error = self._run_set_decision(action)

        reward = self._apply_reward(raw_reward)
        self._append_trace(action, reward, summary)

        if self._state.step_count >= self._task["max_steps"] and self._state.final_decision is None:
            done = True
            if error is None:
                error = "Step budget exhausted before final decision"

        if done and self._state.efficiency_score == 0.0:
            self._state.efficiency_score = step_efficiency_score(
                self._state.step_count, self._task["max_steps"]
            )

        return self._observation(reward=reward, done=done, error=error)

    @property
    def state(self) -> CodeReviewState:
        return self._state

    def get_metadata(self) -> EnvironmentMetadata:
        return EnvironmentMetadata(
            name="Automated Code Review Environment",
            description=(
                "A deterministic pull-request review simulator covering five task categories: "
                "safe refactors, hardcoded secret leaks, authorization regressions, "
                "supply-chain dependency confusion, and license compliance violations. "
                "Rewards are partial-progress and normalized to [0, 1]. "
                "The state() API exposes cumulative_reward, discovered findings, "
                "checks run, comments posted, action history, and a step-efficiency score."
            ),
            version="0.2.0",
            author="Ravi Kumar Gupta",
        )

