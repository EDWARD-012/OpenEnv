"""FastAPI application entrypoint for the code review environment."""

from __future__ import annotations

try:
    from openenv.core.env_server.http_server import create_app
except Exception as exc:  # pragma: no cover
    raise ImportError(
        "openenv is required to run the environment. Install project dependencies first."
    ) from exc

try:
    from ..models import CodeReviewAction, CodeReviewObservation
    from .code_review_environment import CodeReviewEnvironment
except ImportError:  # pragma: no cover
    from models import CodeReviewAction, CodeReviewObservation
    from server.code_review_environment import CodeReviewEnvironment


app = create_app(
    CodeReviewEnvironment,
    CodeReviewAction,
    CodeReviewObservation,
    env_name="code_review_env",
    max_concurrent_envs=2,
)


def main(host: str = "0.0.0.0", port: int = 8000) -> None:
    """Run the FastAPI app directly."""
    import uvicorn

    uvicorn.run(app, host=host, port=port)


if __name__ == "__main__":
    main()

