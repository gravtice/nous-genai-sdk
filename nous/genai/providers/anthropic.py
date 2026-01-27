from __future__ import annotations

import json
import os
from dataclasses import dataclass
from typing import Any, Iterator, Literal
from uuid import uuid4

from .._internal.capability_rules import claude_input_modalities
from .._internal.errors import invalid_request_error, not_supported_error, provider_error
from .._internal.http import download_to_tempfile, request_json, request_stream_json_sse
from ..types import (
    Capability,
    GenerateEvent,
    GenerateRequest,
    GenerateResponse,
    Message,
    Part,
    PartSourceBytes,
    PartSourcePath,
    PartSourceUrl,
    Usage,
    bytes_to_base64,
    detect_mime_type,
    file_to_bytes,
    normalize_reasoning_effort,
)


_ANTHROPIC_DEFAULT_BASE_URL = "https://api.anthropic.com"

_INLINE_BYTES_LIMIT = 20 * 1024 * 1024

_DEFAULT_VERSION = "2023-06-01"


_EFFORT_TO_THINKING_BUDGET_TOKENS: dict[str, int] = {
    "minimal": 1_024,
    "low": 1_024,
    "medium": 2_048,
    "high": 4_096,
    "xhigh": 8_192,
}


@dataclass(frozen=True, slots=True)
class AnthropicAdapter:
    api_key: str
    base_url: str = _ANTHROPIC_DEFAULT_BASE_URL
    provider_name: str = "anthropic"
    auth_mode: Literal["x-api-key", "bearer"] = "x-api-key"
    version: str = _DEFAULT_VERSION
    proxy_url: str | None = None

    def capabilities(self, model_id: str) -> Capability:
        mid = model_id.strip()
        if not mid:
            raise invalid_request_error("model_id must not be empty")
        return Capability(
            input_modalities=claude_input_modalities(mid),
            output_modalities={"text"},
            supports_stream=True,
            supports_job=False,
            supports_tools=True,
            supports_json_schema=False,
        )

    def list_models(self, *, timeout_ms: int | None = None) -> list[str]:
        """
        Fetch remote model ids via Anthropic GET /v1/models.
        """
        url = f"{self.base_url.rstrip('/')}/v1/models"
        obj = request_json(
            method="GET",
            url=url,
            headers=self._headers(),
            timeout_ms=timeout_ms,
            proxy_url=self.proxy_url,
        )
        data = obj.get("data")
        if not isinstance(data, list):
            return []
        out: list[str] = []
        for item in data:
            if not isinstance(item, dict):
                continue
            mid = item.get("id")
            if isinstance(mid, str) and mid:
                out.append(mid)
        return sorted(set(out))

    def generate(self, request: GenerateRequest, *, stream: bool) -> GenerateResponse | Iterator[GenerateEvent]:
        if set(request.output.modalities) != {"text"}:
            raise not_supported_error("Anthropic only supports text output in this SDK")
        if request.output.text and (request.output.text.format != "text" or request.output.text.json_schema is not None):
            raise not_supported_error("Anthropic json output is not supported in this SDK")
        if request.params.seed is not None:
            raise not_supported_error("Anthropic does not support seed in this SDK")
        if stream:
            return self._messages_stream(request)
        return self._messages(request)

    def _headers(self, request: GenerateRequest | None = None) -> dict[str, str]:
        headers: dict[str, str] = {
            "Content-Type": "application/json",
            "anthropic-version": self.version,
        }
        if self.auth_mode == "bearer":
            headers["Authorization"] = f"Bearer {self.api_key}"
        else:
            headers["x-api-key"] = self.api_key
        if request and request.params.idempotency_key:
            headers["Idempotency-Key"] = request.params.idempotency_key
        return headers

    def _messages(self, request: GenerateRequest) -> GenerateResponse:
        url = f"{self.base_url.rstrip('/')}/v1/messages"
        body = self._messages_body(request, stream=False)
        obj = request_json(
            method="POST",
            url=url,
            headers=self._headers(request),
            json_body=body,
            timeout_ms=request.params.timeout_ms,
            proxy_url=self.proxy_url,
        )
        return self._parse_message(obj, model_id=request.model_id())

    def _messages_stream(self, request: GenerateRequest) -> Iterator[GenerateEvent]:
        url = f"{self.base_url.rstrip('/')}/v1/messages"
        body = self._messages_body(request, stream=True)
        events = request_stream_json_sse(
            method="POST",
            url=url,
            headers=self._headers(request),
            json_body=body,
            timeout_ms=request.params.timeout_ms,
            proxy_url=self.proxy_url,
        )

        def _iter() -> Iterator[GenerateEvent]:
            for obj in events:
                if isinstance(obj.get("data"), dict):
                    obj = obj["data"]
                if not isinstance(obj, dict):
                    continue
                if obj.get("type") == "content_block_delta":
                    delta = obj.get("delta")
                    if isinstance(delta, dict) and delta.get("type") == "text_delta":
                        text = delta.get("text")
                        if isinstance(text, str) and text:
                            yield GenerateEvent(type="output.text.delta", data={"delta": text})
            yield GenerateEvent(type="done", data={})

        return _iter()

    def _messages_body(self, request: GenerateRequest, *, stream: bool) -> dict[str, Any]:
        model_id = request.model_id()
        max_tokens = _max_tokens(request)
        system = _extract_system_text(request)

        messages: list[dict[str, Any]] = []
        for m in request.input:
            if m.role == "system":
                continue
            if m.role not in {"user", "assistant", "tool"}:
                raise not_supported_error(f"Anthropic does not support role: {m.role}")
            if m.role == "user" and any(p.type == "tool_result" for p in m.content):
                raise invalid_request_error("tool_result parts must be sent as role='tool' for Anthropic")
            if m.role == "user" and any(p.type == "tool_call" for p in m.content):
                raise invalid_request_error("tool_call parts are only allowed in assistant messages")
            if m.role == "assistant" and any(p.type == "tool_result" for p in m.content):
                raise invalid_request_error("tool_result parts must be sent as role='tool' for Anthropic")
            if m.role == "tool" and any(p.type != "tool_result" for p in m.content):
                raise invalid_request_error("tool messages may only contain tool_result parts")
            blocks = [
                _part_to_block(p, timeout_ms=request.params.timeout_ms, proxy_url=self.proxy_url) for p in m.content
            ]
            role = "user" if m.role == "tool" else m.role
            messages.append({"role": role, "content": blocks})

        if not messages:
            raise invalid_request_error("request.input must contain at least one non-system message")

        body: dict[str, Any] = {"model": model_id, "max_tokens": max_tokens, "messages": messages, "stream": stream}
        if system:
            body["system"] = system

        params = request.params
        if params.temperature is not None:
            body["temperature"] = params.temperature
        if params.top_p is not None:
            body["top_p"] = params.top_p
        if params.stop is not None:
            body["stop_sequences"] = params.stop
        thinking = _thinking_param(request, max_tokens=max_tokens)
        if thinking is not None:
            body["thinking"] = thinking

        if request.tools:
            tools: list[dict[str, Any]] = []
            for t in request.tools:
                name = t.name.strip()
                if not name:
                    raise invalid_request_error("tool.name must be non-empty")
                tool_obj: dict[str, Any] = {
                    "name": name,
                    "input_schema": t.parameters if t.parameters is not None else {"type": "object"},
                }
                if isinstance(t.description, str) and t.description.strip():
                    tool_obj["description"] = t.description.strip()
                tools.append(tool_obj)
            body["tools"] = tools

        if request.tool_choice is not None:
            choice = request.tool_choice.normalized()
            if choice.mode in {"required", "tool"} and not request.tools:
                raise invalid_request_error("tool_choice requires request.tools")
            if choice.mode == "required":
                body["tool_choice"] = {"type": "any"}
            elif choice.mode == "tool":
                body["tool_choice"] = {"type": "tool", "name": choice.name}
            else:
                body["tool_choice"] = {"type": choice.mode}

        opts = request.provider_options.get(self.provider_name)
        if isinstance(opts, dict):
            for k, v in opts.items():
                if k in body:
                    raise invalid_request_error(f"provider_options cannot override body.{k}")
                body[k] = v
        return body

    def _parse_message(self, obj: dict[str, Any], *, model_id: str) -> GenerateResponse:
        if isinstance(obj.get("data"), dict):
            obj = obj["data"]
        content = obj.get("content")
        if not isinstance(content, list):
            raise provider_error("anthropic response missing content")
        parts: list[Part] = []
        for item in content:
            if not isinstance(item, dict):
                continue
            typ = item.get("type")
            if typ == "text":
                t = item.get("text")
                if isinstance(t, str):
                    parts.append(Part.from_text(t))
                continue
            if typ == "tool_use":
                tool_use_id = item.get("id")
                name = item.get("name")
                tool_input = item.get("input")
                if isinstance(tool_use_id, str) and tool_use_id and isinstance(name, str) and name and isinstance(tool_input, dict):
                    parts.append(Part.tool_call(tool_call_id=tool_use_id, name=name, arguments=tool_input))

        usage_obj = obj.get("usage")
        usage = None
        if isinstance(usage_obj, dict):
            usage = Usage(
                input_tokens=usage_obj.get("input_tokens"),
                output_tokens=usage_obj.get("output_tokens"),
                total_tokens=usage_obj.get("input_tokens", 0) + usage_obj.get("output_tokens", 0)
                if isinstance(usage_obj.get("input_tokens"), int) and isinstance(usage_obj.get("output_tokens"), int)
                else None,
            )

        return GenerateResponse(
            id=obj.get("id") if isinstance(obj.get("id"), str) else f"sdk_{uuid4().hex}",
            provider=self.provider_name,
            model=f"{self.provider_name}:{model_id}",
            status="completed",
            output=[Message(role="assistant", content=parts if parts else [Part.from_text("")])],
            usage=usage,
        )


def _max_tokens(request: GenerateRequest) -> int:
    spec = request.output.text
    if spec and spec.max_output_tokens is not None:
        return max(1, int(spec.max_output_tokens))
    if request.params.max_output_tokens is not None:
        return max(1, int(request.params.max_output_tokens))
    return 1024


def _thinking_param(request: GenerateRequest, *, max_tokens: int) -> dict[str, Any] | None:
    reasoning = request.params.reasoning
    if reasoning is None:
        return None
    if reasoning.effort is None:
        return None
    effort = normalize_reasoning_effort(reasoning.effort)
    if effort == "none":
        return None
    budget = _EFFORT_TO_THINKING_BUDGET_TOKENS[effort]
    if budget >= max_tokens:
        raise invalid_request_error(
            f"Anthropic thinking budget_tokens must be < max_tokens ({budget} >= {max_tokens}); "
            "increase output.text.max_output_tokens or params.max_output_tokens"
        )
    return {"type": "enabled", "budget_tokens": budget}


def _extract_system_text(request: GenerateRequest) -> str | None:
    chunks: list[str] = []
    for m in request.input:
        if m.role != "system":
            continue
        for p in m.content:
            if p.type != "text":
                raise not_supported_error("Anthropic system messages only support text")
            t = p.require_text().strip()
            if t:
                chunks.append(t)
    if not chunks:
        return None
    return "\n\n".join(chunks)


def _part_to_block(part: Part, *, timeout_ms: int | None, proxy_url: str | None) -> dict[str, Any]:
    if part.type == "text":
        return {"type": "text", "text": part.require_text()}
    if part.type == "tool_call":
        tool_call_id = part.meta.get("tool_call_id")
        name = part.meta.get("name")
        arguments = part.meta.get("arguments")
        if not isinstance(tool_call_id, str) or not tool_call_id:
            raise invalid_request_error("tool_call.meta.tool_call_id required for Anthropic tool_use")
        if not isinstance(name, str) or not name.strip():
            raise invalid_request_error("tool_call.meta.name must be a non-empty string")
        if not isinstance(arguments, dict):
            raise invalid_request_error("Anthropic tool_call.meta.arguments must be an object")
        return {"type": "tool_use", "id": tool_call_id, "name": name.strip(), "input": arguments}
    if part.type == "tool_result":
        tool_call_id = part.meta.get("tool_call_id")
        name = part.meta.get("name")
        result = part.meta.get("result")
        is_error = part.meta.get("is_error")
        if not isinstance(tool_call_id, str) or not tool_call_id:
            raise invalid_request_error("tool_result.meta.tool_call_id required for Anthropic tool_result")
        if not isinstance(name, str) or not name.strip():
            raise invalid_request_error("tool_result.meta.name must be a non-empty string")
        if is_error is not None and not isinstance(is_error, bool):
            raise invalid_request_error("tool_result.meta.is_error must be a bool")
        out = result if isinstance(result, str) else json.dumps(result, ensure_ascii=False, separators=(",", ":"))
        block: dict[str, Any] = {"type": "tool_result", "tool_use_id": tool_call_id, "content": out}
        if is_error is not None:
            block["is_error"] = is_error
        return block
    if part.type == "image":
        source = part.require_source()
        mime_type = part.mime_type
        if mime_type is None and isinstance(source, PartSourcePath):
            mime_type = detect_mime_type(source.path)
        if not mime_type or not mime_type.startswith("image/"):
            raise invalid_request_error("anthropic image requires image/* mime_type")

        if isinstance(source, PartSourceUrl):
            tmp = download_to_tempfile(
                url=source.url,
                timeout_ms=timeout_ms,
                max_bytes=_INLINE_BYTES_LIMIT,
                proxy_url=proxy_url,
            )
            try:
                data = file_to_bytes(tmp, _INLINE_BYTES_LIMIT)
            finally:
                try:
                    os.unlink(tmp)
                except OSError:
                    pass
            data_b64 = bytes_to_base64(data)
        elif isinstance(source, PartSourcePath):
            data = file_to_bytes(source.path, _INLINE_BYTES_LIMIT)
            data_b64 = bytes_to_base64(data)
        elif isinstance(source, PartSourceBytes) and source.encoding == "base64":
            data_b64 = source.data
            if not isinstance(data_b64, str) or not data_b64:
                raise invalid_request_error("image base64 data must be non-empty")
        else:
            assert isinstance(source, PartSourceBytes)
            data = source.data
            if not isinstance(data, bytes):
                raise invalid_request_error("image bytes data must be bytes")
            if len(data) > _INLINE_BYTES_LIMIT:
                raise not_supported_error(f"inline bytes too large ({len(data)} > {_INLINE_BYTES_LIMIT})")
            data_b64 = bytes_to_base64(data)

        return {
            "type": "image",
            "source": {"type": "base64", "media_type": mime_type, "data": data_b64},
        }
    raise not_supported_error(f"Anthropic does not support part type: {part.type}")
