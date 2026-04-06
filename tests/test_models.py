"""Model validation tests."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from models import CodeReviewAction


def test_invalid_action_type_is_rejected() -> None:
    with pytest.raises(ValidationError):
        CodeReviewAction(action_type="ship_it")  # type: ignore[arg-type]


def test_submit_comment_accepts_structured_fields() -> None:
    action = CodeReviewAction(
        action_type="submit_comment",
        file_path="config/settings.py",
        finding_ids=["hardcoded_secret"],
        comment_text="Blocking issue.",
    )
    assert action.finding_ids == ["hardcoded_secret"]
    assert action.file_path == "config/settings.py"

