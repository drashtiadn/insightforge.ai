"""Extract and validate JSON objects from LLM text."""

from __future__ import annotations

import json
from typing import Any, TypeVar

from pydantic import BaseModel, ValidationError

from insightforge.core.exceptions import ExternalServiceError

T = TypeVar("T", bound=BaseModel)


def extract_json_object(text: str) -> dict[str, Any]:
    """Parse the first JSON object in ``text``.

    Accepts raw objects and objects wrapped in markdown fences or prose.
    """

    cleaned = text.strip()
    if cleaned.startswith("```"):
        lines = cleaned.split("\n")
        # Drop opening fence (``` or ```json) and optional closing fence.
        if lines and lines[0].startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]
        cleaned = "\n".join(lines).strip()

    try:
        parsed: object = json.loads(cleaned)
    except json.JSONDecodeError:
        start = cleaned.find("{")
        end = cleaned.rfind("}")
        if start < 0 or end <= start:
            raise ExternalServiceError(
                "llm output is not a JSON object",
                details={"field": "text"},
            ) from None
        try:
            parsed = json.loads(cleaned[start : end + 1])
        except json.JSONDecodeError as exc:
            raise ExternalServiceError(
                "llm output is not valid JSON",
                details={"field": "text"},
            ) from exc

    if not isinstance(parsed, dict):
        raise ExternalServiceError(
            "llm JSON must be an object",
            details={"type": type(parsed).__name__},
        )
    return parsed


def parse_model(text: str, model_type: type[T]) -> T:
    """Extract JSON from ``text`` and validate it as ``model_type``."""

    payload = extract_json_object(text)
    try:
        return model_type.model_validate(payload)
    except ValidationError as exc:
        raise ExternalServiceError(
            "llm JSON failed schema validation",
            details={"errors": exc.errors()},
        ) from exc
