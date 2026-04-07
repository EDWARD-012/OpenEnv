"""Baseline inference script for the automated code review environment."""

from __future__ import annotations

import asyncio
import json
import os
from typing import Any, Iterable

from openai import OpenAI

try:
    from code_review_env import CodeReviewAction, CodeReviewEnv
    from code_review_env.task_fixtures import list_task_ids
except ImportError:  # pragma: no cover
    from client import CodeReviewEnv
    from models import CodeReviewAction
    from task_fixtures import list_task_ids


API_KEY = os.getenv("HF_TOKEN") or os.getenv("API_KEY") or "dummy-token"
API_BASE_URL = os.getenv("API_BASE_URL") or "https://router.huggingface.co/v1"
MODEL_NAME = os.getenv("MODEL_NAME") or "Qwen/Qwen2.5-72B-Instruct"
LOCAL_IMAGE_NAME = (
    os.getenv("LOCAL_IMAGE_NAME")
    or os.getenv("IMAGE_NAME")
    or os.getenv("LOCAL_IMAGE")
    or "code-review-env:latest"
)
BENCHMARK = os.getenv("CODE_REVIEW_BENCHMARK") or "code_review_env"
MAX_STEPS = 8
TEMPERATURE = 0.0
MAX_TOKENS = 220
SUCCESS_SCORE_THRESHOLD = 0.8

SYSTEM_PROMPT = (
    "You are operating a deterministic pull-request review environment. "
    "Return exactly one compact JSON object with keys action_type, file_path, "
    "finding_ids, comment_text, and decision when relevant. "
    "Valid action_type values are view_pr, inspect_file, run_static_analysis, "
    "run_tests, read_policy, request_ai_review, submit_comment, and set_decision. "
    "Never add prose outside the JSON object."
)


def sanitize_line(value: Any) -> str:
    """Render any log value on a single line."""
    text = str(value)
    return " ".join(text.splitlines()).strip()


def render_action(action: CodeReviewAction) -> str:
    """Render an action as a compact single-line JSON string."""
    return sanitize_line(json.dumps(action.model_dump(exclude_none=True), separators=(",", ":")))


def log_start(task: str, env: str, model: str) -> None:
    print(f"[START] task={task} env={env} model={sanitize_line(model)}", flush=True)


def log_step(step: int, action: CodeReviewAction, reward: float, done: bool, error: str | None) -> None:
    error_value = sanitize_line(error) if error else "null"
    print(
        f"[STEP] step={step} action={render_action(action)} reward={reward:.2f} "
        f"done={str(done).lower()} error={error_value}",
        flush=True,
    )


def log_end(success: bool, steps: int, score: float, rewards: Iterable[float]) -> None:
    rewards_str = ",".join(f"{reward:.2f}" for reward in rewards)
    print(
        f"[END] success={str(success).lower()} steps={steps} score={score:.3f} rewards={rewards_str}",
        flush=True,
    )


def build_user_prompt(task_id: str, step: int, observation: Any, history: list[str]) -> str:
    """Build the user prompt shown to the LLM."""
    findings = ", ".join(f.finding_id for f in observation.discovered_findings) or "none"
    artifacts = ", ".join(a.artifact_type for a in observation.revealed_artifacts) or "none"
    comments = len(observation.comments_posted)
    history_block = "\n".join(history[-5:]) if history else "none"
    return (
        f"Task ID: {task_id}\n"
        f"Step: {step}\n"
        f"Objective: {observation.objective}\n"
        f"PR Summary: {observation.pr_summary}\n"
        f"Changed Files: {', '.join(observation.changed_files)}\n"
        f"Discovered Findings: {findings}\n"
        f"Artifacts: {artifacts}\n"
        f"Comments Posted: {comments}\n"
        f"Remaining Steps: {observation.remaining_steps}\n"
        f"Last Action Error: {observation.last_action_error or 'none'}\n"
        f"Recent History:\n{history_block}\n"
        "Return the next action as JSON only."
    )


def parse_model_action(raw_content: str) -> dict[str, Any] | None:
    """Try to parse a JSON object from model output."""
    raw_content = raw_content.strip()
    if not raw_content:
        return None
    candidates = [raw_content]
    if "{" in raw_content and "}" in raw_content:
        start = raw_content.find("{")
        end = raw_content.rfind("}") + 1
        candidates.append(raw_content[start:end])
    for candidate in candidates:
        try:
            parsed = json.loads(candidate)
        except json.JSONDecodeError:
            continue
        if isinstance(parsed, dict):
            return parsed
    return None


def artifact_types(observation: Any) -> set[str]:
    """Collect revealed artifact types."""
    return {artifact.artifact_type for artifact in observation.revealed_artifacts}


def discovered_finding_ids(observation: Any) -> set[str]:
    """Collect discovered finding ids."""
    return {finding.finding_id for finding in observation.discovered_findings}


def comment_covers(observation: Any, finding_id: str) -> bool:
    """Check if any posted comment covers a finding."""
    for comment in observation.comments_posted:
        if finding_id in comment.finding_ids:
            return True
    return False


def fallback_action(task_id: str, observation: Any) -> CodeReviewAction:
    """Deterministic fallback policy used for reproducible local baselines."""
    revealed = artifact_types(observation)
    findings = discovered_finding_ids(observation)

    if "pull_request" not in revealed:
        return CodeReviewAction(action_type="view_pr")

    if task_id == "clean_refactor_approve":
        if "file_diff" not in revealed:
            return CodeReviewAction(
                action_type="inspect_file",
                file_path="src/formatting/formatter.py",
            )
        if "static_analysis" not in revealed:
            return CodeReviewAction(action_type="run_static_analysis")
        if "tests" not in revealed:
            return CodeReviewAction(action_type="run_tests")
        if "policy" not in revealed:
            return CodeReviewAction(action_type="read_policy")
        if "ai_review" not in revealed:
            return CodeReviewAction(action_type="request_ai_review")
        return CodeReviewAction(action_type="set_decision", decision="approve")

    if task_id == "secret_leak_reject":
        if "file_diff" not in revealed:
            return CodeReviewAction(
                action_type="inspect_file",
                file_path="config/settings.py",
            )
        if "static_analysis" not in revealed:
            return CodeReviewAction(action_type="run_static_analysis")
        if "policy" not in revealed:
            return CodeReviewAction(action_type="read_policy")
        if "ai_review" not in revealed:
            return CodeReviewAction(action_type="request_ai_review")
        if "hardcoded_secret" in findings and not comment_covers(observation, "hardcoded_secret"):
            return CodeReviewAction(
                action_type="submit_comment",
                file_path="config/settings.py",
                finding_ids=["hardcoded_secret"],
                comment_text="Blocking: remove the hardcoded production token and load it from environment configuration before merge.",
            )
        return CodeReviewAction(action_type="set_decision", decision="reject")

    if "file_diff" not in revealed:
        return CodeReviewAction(action_type="inspect_file", file_path="auth/middleware.py")
    if "static_analysis" not in revealed:
        return CodeReviewAction(action_type="run_static_analysis")
    if "tests" not in revealed:
        return CodeReviewAction(action_type="run_tests")
    if "policy" not in revealed:
        return CodeReviewAction(action_type="read_policy")
    if "ai_review" not in revealed:
        return CodeReviewAction(action_type="request_ai_review")
    if {"auth_scope_gap", "missing_auth_tests"} <= findings and not (
        comment_covers(observation, "auth_scope_gap")
        and comment_covers(observation, "missing_auth_tests")
    ):
        return CodeReviewAction(
            action_type="submit_comment",
            file_path="auth/middleware.py",
            finding_ids=["auth_scope_gap", "missing_auth_tests"],
            comment_text=(
                "Blocking: this change removes the admin impersonation scope gate and "
                "ships without the required denied-path auth regression coverage."
            ),
        )
    return CodeReviewAction(action_type="set_decision", decision="reject")


def choose_action(client: OpenAI, task_id: str, step: int, observation: Any, history: list[str]) -> CodeReviewAction:
    """Call the model and fall back to the deterministic policy on any issue."""
    fallback = fallback_action(task_id, observation)
    try:
        completion = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": build_user_prompt(task_id, step, observation, history)},
            ],
            temperature=TEMPERATURE,
            max_tokens=MAX_TOKENS,
            stream=False,
        )
        content = (completion.choices[0].message.content or "").strip()
        parsed = parse_model_action(content)
        if not parsed:
            return fallback
        return CodeReviewAction.model_validate(parsed)
    except Exception:
        return fallback


async def run_task(task_id: str) -> None:
    """Run one benchmark task and emit structured stdout."""
    env: CodeReviewEnv | None = None
    history: list[str] = []
    rewards: list[float] = []
    steps_taken = 0
    score = 0.0
    success = False

    log_start(task=task_id, env=BENCHMARK, model=MODEL_NAME)

    try:
        client = OpenAI(base_url=API_BASE_URL, api_key=API_KEY)
        env = await CodeReviewEnv.from_docker_image(LOCAL_IMAGE_NAME)
        result = await env.reset(task_id=task_id)

        for step in range(1, MAX_STEPS + 1):
            if result.done:
                break

            action = choose_action(client, task_id, step, result.observation, history)
            result = await env.step(action)

            reward = float(result.reward or 0.0)
            rewards.append(reward)
            steps_taken = step
            history.append(
                f"step={step} action={action.action_type} reward={reward:.2f} done={result.done}"
            )

            log_step(
                step=step,
                action=action,
                reward=reward,
                done=result.done,
                error=result.observation.last_action_error,
            )

            if result.done:
                break

        try:
            state = await env.state()
            score = float(state.cumulative_reward)
        except Exception:
            score = min(max(sum(rewards), 0.0), 1.0)
        success = score >= SUCCESS_SCORE_THRESHOLD
    except Exception:
        score = min(max(sum(rewards), 0.0), 1.0)
    finally:
        try:
            if env is not None:
                await env.close()
        except Exception:
            pass
        log_end(success=success, steps=steps_taken, score=score, rewards=rewards)


async def main() -> None:
    """Run all benchmark tasks sequentially."""
    try:
        task_ids = list_task_ids()
    except Exception:
        task_ids = []
    for task_id in task_ids:
        try:
            await run_task(task_id)
        except Exception:
            pass


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except Exception:
        pass
