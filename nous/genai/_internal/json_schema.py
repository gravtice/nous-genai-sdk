from __future__ import annotations

from typing import Any

from .errors import invalid_request_error, not_supported_error

_GEMINI_RESPONSE_SCHEMA_TYPES = frozenset(
    {
        "TYPE_UNSPECIFIED",
        "STRING",
        "NUMBER",
        "INTEGER",
        "BOOLEAN",
        "ARRAY",
        "OBJECT",
    }
)


def reject_gemini_response_schema_dict(schema: dict[str, Any]) -> None:
    t = schema.get("type")
    if isinstance(t, str) and t in _GEMINI_RESPONSE_SCHEMA_TYPES:
        raise invalid_request_error(
            "output.text.json_schema must be JSON Schema (not Gemini responseSchema); "
            "pass a Python type/model or use provider_options.google.generationConfig.responseSchema"
        )


def python_type_to_json_schema(schema: Any) -> dict[str, Any]:
    try:
        from pydantic import TypeAdapter
    except ModuleNotFoundError as e:  # pragma: no cover
        raise not_supported_error("pydantic is required for python-type output.text.json_schema") from e

    try:
        return TypeAdapter(schema).json_schema()
    except Exception:
        pass

    try:
        return TypeAdapter(type(schema)).json_schema()
    except Exception as e:
        raise invalid_request_error(
            "output.text.json_schema must be a JSON Schema object or a Python type supported by pydantic TypeAdapter"
        ) from e


def normalize_json_schema(schema: Any) -> dict[str, Any]:
    if isinstance(schema, dict):
        reject_gemini_response_schema_dict(schema)
        return schema
    return python_type_to_json_schema(schema)

