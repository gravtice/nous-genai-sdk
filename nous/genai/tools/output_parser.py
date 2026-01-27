from __future__ import annotations

from typing import Any

from .._internal.errors import invalid_request_error, not_supported_error
from .._internal.json_schema import normalize_json_schema
from ..client import Client
from ..types import (
    GenerateRequest,
    GenerateResponse,
    Message,
    OutputSpec,
    Part,
    Tool,
    ToolChoice,
)

DEFAULT_OUTPUT_PARSER_TOOL_NAME = "nous_output_parser"

_DEFAULT_WRAPPER_KEY = "output"


def build_output_parser_tool(
    json_schema: Any,
    *,
    name: str = DEFAULT_OUTPUT_PARSER_TOOL_NAME,
    description: str | None = None,
) -> Tool:
    """
    Build a provider-agnostic function tool used for "structured outputs".

    The model is expected to return the final answer by calling this tool with
    a single argument key `"output"` matching the given JSON Schema.
    """
    tool_name = name.strip()
    if not tool_name:
        raise invalid_request_error("output parser tool name must be non-empty")
    schema = normalize_json_schema(json_schema)
    parameters = {
        "type": "object",
        "properties": {_DEFAULT_WRAPPER_KEY: schema},
        "required": [_DEFAULT_WRAPPER_KEY],
        "additionalProperties": False,
    }
    desc = description
    if not isinstance(desc, str) or not desc.strip():
        desc = (
            "Return the final result by calling this tool with a single JSON argument "
            f"`{_DEFAULT_WRAPPER_KEY}` matching the schema. Do not output extra text."
        )
    return Tool(name=tool_name, description=desc, parameters=parameters, strict=True)


def extract_output_from_response(
    response: GenerateResponse,
    *,
    tool_name: str = DEFAULT_OUTPUT_PARSER_TOOL_NAME,
) -> Any:
    """
    Extract the structured output object from a provider response that contains
    a tool_call part for `tool_name`.
    """
    name = tool_name.strip()
    if not name:
        raise invalid_request_error("tool_name must be non-empty")

    for msg in response.output:
        for part in msg.content:
            if part.type != "tool_call":
                continue
            if part.meta.get("name") != name:
                continue
            arguments = part.meta.get("arguments")
            if not isinstance(arguments, dict):
                raise invalid_request_error(
                    "output parser tool_call arguments must be an object"
                )
            if _DEFAULT_WRAPPER_KEY not in arguments:
                raise invalid_request_error(
                    f"output parser tool_call missing '{_DEFAULT_WRAPPER_KEY}'"
                )
            return arguments[_DEFAULT_WRAPPER_KEY]

    raise invalid_request_error(f"missing output parser tool_call: {name}")


def parse_output(
    client: Client,
    *,
    model: str,
    text: str,
    json_schema: Any,
    tool_name: str = DEFAULT_OUTPUT_PARSER_TOOL_NAME,
) -> Any:
    """
    Parse plain text into a structured object by forcing a single tool call.
    """
    if not isinstance(text, str) or not text.strip():
        raise invalid_request_error("text must be a non-empty string")
    cap = client.capabilities(model)
    if not cap.supports_tools:
        raise not_supported_error("this model does not support tools")

    tool = build_output_parser_tool(json_schema, name=tool_name)
    prompt = (
        "Convert the following text into structured output by calling the tool. "
        "Call the tool only; do not output any other text.\n\n" + text.strip()
    )
    req = GenerateRequest(
        model=model,
        input=[Message(role="user", content=[Part.from_text(prompt)])],
        output=OutputSpec(modalities=["text"]),
        tools=[tool],
        tool_choice=ToolChoice(mode="tool", name=tool.name),
    )
    resp = client.generate(req)
    if not isinstance(resp, GenerateResponse):
        raise not_supported_error("streaming responses are not supported")
    return extract_output_from_response(resp, tool_name=tool.name)
