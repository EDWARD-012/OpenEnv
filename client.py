"""Async OpenEnv client for the code review environment."""

from __future__ import annotations

from typing import Any, Dict

from openenv.core import EnvClient
from openenv.core.client_types import StepResult

try:
    from .models import CodeReviewAction, CodeReviewObservation, CodeReviewState
except ImportError:  # pragma: no cover
    from models import CodeReviewAction, CodeReviewObservation, CodeReviewState


class CodeReviewEnv(EnvClient[CodeReviewAction, CodeReviewObservation, CodeReviewState]):
    """Client for the automated code review environment."""

    def _step_payload(self, action: CodeReviewAction) -> Dict[str, Any]:
        return action.model_dump(exclude_none=True)

    def _parse_result(self, payload: Dict[str, Any]) -> StepResult[CodeReviewObservation]:
        observation_data = payload.get("observation", {})
        observation = CodeReviewObservation.model_validate(
            {
                **observation_data,
                "done": payload.get("done", False),
                "reward": payload.get("reward"),
            }
        )
        return StepResult(
            observation=observation,
            reward=payload.get("reward"),
            done=payload.get("done", False),
        )

    def _parse_state(self, payload: Dict[str, Any]) -> CodeReviewState:
        return CodeReviewState.model_validate(payload)

