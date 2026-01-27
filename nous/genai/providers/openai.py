from __future__ import annotations

import base64
import json
import os
import tempfile
import time
import urllib.parse
from dataclasses import dataclass
from typing import Any, Iterator
from uuid import uuid4

from .._internal.capability_rules import (
    chat_input_modalities,
    chat_output_modalities,
    image_input_modalities,
    infer_model_kind,
    is_transcribe_model,
    output_modalities_for_kind,
    transcribe_input_modalities,
    video_input_modalities,
)
from .._internal.errors import invalid_request_error, not_supported_error, provider_error
from .._internal.http import (
    download_to_tempfile,
    multipart_form_data,
    multipart_form_data_fields,
    request_bytes,
    request_json,
    request_stream_json_sse,
    request_streaming_body_json,
)
from ..types import (
    Capability,
    GenerateEvent,
    GenerateRequest,
    GenerateResponse,
    JobInfo,
    Message,
    Part,
    PartSourceBytes,
    PartSourcePath,
    PartSourceRef,
    PartSourceUrl,
    Usage,
    bytes_to_base64,
    detect_mime_type,
    file_to_bytes,
    normalize_reasoning_effort,
    sniff_image_mime_type,
)


_OPENAI_DEFAULT_BASE_URL = "https://api.openai.com/v1"

_INLINE_BYTES_LIMIT = 20 * 1024 * 1024


def _is_mcp_transport() -> bool:
    value = os.environ.get("NOUS_GENAI_TRANSPORT", "").strip().lower()
    return value in {"mcp", "sse", "streamable", "streamable-http", "streamable_http"}


def _download_image_url_as_data_url(
    url: str,
    *,
    mime_type: str | None,
    timeout_ms: int | None,
    proxy_url: str | None,
) -> str:
    suffix = urllib.parse.urlparse(url).path
    ext = os.path.splitext(suffix)[1]
    tmp = download_to_tempfile(
        url=url,
        timeout_ms=timeout_ms,
        max_bytes=_INLINE_BYTES_LIMIT,
        suffix=ext if ext else "",
        proxy_url=proxy_url,
    )
    try:
        data = file_to_bytes(tmp, _INLINE_BYTES_LIMIT)
        if not mime_type:
            mime_type = detect_mime_type(tmp) or sniff_image_mime_type(data)
        if not mime_type:
            raise invalid_request_error("could not infer image mime_type from url content")
        b64 = bytes_to_base64(data)
        return f"data:{mime_type};base64,{b64}"
    finally:
        try:
            os.unlink(tmp)
        except OSError:
            pass


def _audio_format_from_mime(mime_type: str | None) -> str | None:
    if mime_type is None:
        return None
    mime_type = mime_type.lower()
    if mime_type in {"audio/wav", "audio/wave"}:
        return "wav"
    if mime_type in {"audio/mpeg", "audio/mp3"}:
        return "mp3"
    if mime_type in {"audio/mp4", "audio/m4a"}:
        return "m4a"
    return None


def _audio_mime_from_format(fmt: str) -> str:
    f = fmt.strip().lower()
    if f == "mp3":
        return "audio/mpeg"
    if f == "wav":
        return "audio/wav"
    if f in {"m4a", "mp4"}:
        return "audio/mp4"
    if f == "aac":
        return "audio/aac"
    if f == "flac":
        return "audio/flac"
    if f == "opus":
        return "audio/opus"
    if f == "pcm":
        return "audio/pcm"
    return f"audio/{f}" if f else "application/octet-stream"


def _download_to_temp(
    url: str, *, timeout_ms: int | None, max_bytes: int | None, proxy_url: str | None
) -> str:
    return download_to_tempfile(url=url, timeout_ms=timeout_ms, max_bytes=max_bytes, proxy_url=proxy_url)


def _part_to_chat_content(
    part: Part, *, timeout_ms: int | None, provider_name: str, proxy_url: str | None
) -> dict[str, Any]:
    if part.type == "text":
        return {"type": "text", "text": part.require_text()}
    if part.type == "image":
        source = part.require_source()
        mime_type = part.mime_type
        if mime_type is None and isinstance(source, PartSourcePath):
            mime_type = detect_mime_type(source.path)
        if isinstance(source, PartSourceUrl):
            if not _is_mcp_transport():
                return {"type": "image_url", "image_url": {"url": source.url}}
            data_url = _download_image_url_as_data_url(
                source.url,
                mime_type=mime_type,
                timeout_ms=timeout_ms,
                proxy_url=proxy_url,
            )
            return {"type": "image_url", "image_url": {"url": data_url}}
        if isinstance(source, PartSourceRef):
            raise not_supported_error("openai does not support image ref in chat input; use url/bytes/path")
        if isinstance(source, PartSourceBytes) and source.encoding == "base64":
            if not mime_type:
                raise invalid_request_error("image mime_type required for base64 input")
            b64 = source.data
            if not isinstance(b64, str) or not b64:
                raise invalid_request_error("image base64 data must be non-empty")
            return {"type": "image_url", "image_url": {"url": f"data:{mime_type};base64,{b64}"}}
        if isinstance(source, PartSourcePath):
            data = file_to_bytes(source.path, _INLINE_BYTES_LIMIT)
        else:
            assert isinstance(source, PartSourceBytes)
            data = source.data
            if not isinstance(data, bytes):
                raise invalid_request_error("image bytes data must be bytes")
            if len(data) > _INLINE_BYTES_LIMIT:
                raise not_supported_error(f"inline bytes too large ({len(data)} > {_INLINE_BYTES_LIMIT})")
        if not mime_type:
            raise invalid_request_error("image mime_type required for bytes/path input")
        b64 = bytes_to_base64(data)
        return {"type": "image_url", "image_url": {"url": f"data:{mime_type};base64,{b64}"}}
    if part.type == "audio":
        source = part.require_source()
        fmt = _audio_format_from_mime(part.mime_type)
        if fmt is None and isinstance(source, PartSourcePath):
            fmt = _audio_format_from_mime(detect_mime_type(source.path))
        if fmt is None:
            raise invalid_request_error("audio format (wav/mp3/m4a) required via mime_type or extension")
        if provider_name == "aliyun":
            if isinstance(source, PartSourceUrl):
                return {"type": "input_audio", "input_audio": {"data": source.url}}
            if isinstance(source, PartSourceRef):
                raise not_supported_error("aliyun does not support audio ref in chat input; use url/bytes/path")
            if isinstance(source, PartSourceBytes) and source.encoding == "base64":
                mime_type = part.mime_type or _audio_mime_from_format(fmt)
                b64 = source.data
                if not isinstance(b64, str) or not b64:
                    raise invalid_request_error("audio base64 data must be non-empty")
                return {"type": "input_audio", "input_audio": {"data": f"data:{mime_type};base64,{b64}"}}
            if isinstance(source, PartSourcePath):
                data = file_to_bytes(source.path, _INLINE_BYTES_LIMIT)
                mime_type = detect_mime_type(source.path) or part.mime_type or _audio_mime_from_format(fmt)
            else:
                assert isinstance(source, PartSourceBytes)
                data = source.data
                if not isinstance(data, bytes):
                    raise invalid_request_error("audio bytes data must be bytes")
                if len(data) > _INLINE_BYTES_LIMIT:
                    raise not_supported_error(f"inline bytes too large ({len(data)} > {_INLINE_BYTES_LIMIT})")
                mime_type = part.mime_type or _audio_mime_from_format(fmt)
            b64 = bytes_to_base64(data)
            return {"type": "input_audio", "input_audio": {"data": f"data:{mime_type};base64,{b64}"}}

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
        elif isinstance(source, PartSourceBytes) and source.encoding == "base64":
            b64 = source.data
            if not isinstance(b64, str) or not b64:
                raise invalid_request_error("audio base64 data must be non-empty")
            return {"type": "input_audio", "input_audio": {"data": b64, "format": fmt}}
        elif isinstance(source, PartSourcePath):
            data = file_to_bytes(source.path, _INLINE_BYTES_LIMIT)
        elif isinstance(source, PartSourceRef):
            raise not_supported_error("openai does not support audio ref in chat input; use url/bytes/path")
        else:
            assert isinstance(source, PartSourceBytes)
            data = source.data
            if not isinstance(data, bytes):
                raise invalid_request_error("audio bytes data must be bytes")
            if len(data) > _INLINE_BYTES_LIMIT:
                raise not_supported_error(f"inline bytes too large ({len(data)} > {_INLINE_BYTES_LIMIT})")
        return {"type": "input_audio", "input_audio": {"data": bytes_to_base64(data), "format": fmt}}
    if part.type in {"video", "embedding"}:
        raise not_supported_error(f"openai chat input does not support part type: {part.type}")
    raise not_supported_error(f"unsupported part type: {part.type}")


def _tool_result_to_string(result: Any) -> str:
    if isinstance(result, str):
        return result
    return json.dumps(result, ensure_ascii=False, separators=(",", ":"))


def _tool_call_to_json_arguments(arguments: Any) -> str:
    if isinstance(arguments, str):
        return arguments
    return json.dumps(arguments, ensure_ascii=False, separators=(",", ":"))


def _parse_tool_call_arguments(value: Any) -> Any:
    if not isinstance(value, str):
        return value
    try:
        return json.loads(value)
    except Exception:
        return value


def _require_tool_call_meta(part: Part) -> tuple[str | None, str, Any]:
    tool_call_id = part.meta.get("tool_call_id")
    if tool_call_id is not None and not isinstance(tool_call_id, str):
        raise invalid_request_error("tool_call.meta.tool_call_id must be a string")
    name = part.meta.get("name")
    if not isinstance(name, str) or not name.strip():
        raise invalid_request_error("tool_call.meta.name must be a non-empty string")
    arguments = part.meta.get("arguments")
    return (tool_call_id, name.strip(), arguments)


def _require_tool_result_meta(part: Part) -> tuple[str | None, str, Any, bool | None]:
    tool_call_id = part.meta.get("tool_call_id")
    if tool_call_id is not None and not isinstance(tool_call_id, str):
        raise invalid_request_error("tool_result.meta.tool_call_id must be a string")
    name = part.meta.get("name")
    if not isinstance(name, str) or not name.strip():
        raise invalid_request_error("tool_result.meta.name must be a non-empty string")
    result = part.meta.get("result")
    is_error = part.meta.get("is_error")
    if is_error is not None and not isinstance(is_error, bool):
        raise invalid_request_error("tool_result.meta.is_error must be a bool")
    return (tool_call_id, name.strip(), result, is_error)


def _part_to_responses_image_content(part: Part, *, timeout_ms: int | None, proxy_url: str | None) -> dict[str, Any]:
    if part.type != "image":
        raise not_supported_error(f"responses protocol does not support part type: {part.type}")
    source = part.require_source()
    mime_type = part.mime_type
    if mime_type is None and isinstance(source, PartSourcePath):
        mime_type = detect_mime_type(source.path)
    if isinstance(source, PartSourceUrl):
        if not _is_mcp_transport():
            return {"type": "input_image", "image_url": source.url}
        data_url = _download_image_url_as_data_url(
            source.url,
            mime_type=mime_type,
            timeout_ms=timeout_ms,
            proxy_url=proxy_url,
        )
        return {"type": "input_image", "image_url": data_url}
    if isinstance(source, PartSourceRef):
        raise not_supported_error("responses protocol does not support image ref in input; use url/bytes/path")
    if isinstance(source, PartSourcePath):
        data = file_to_bytes(source.path, _INLINE_BYTES_LIMIT)
    else:
        assert isinstance(source, PartSourceBytes)
        data = source.data
        if len(data) > _INLINE_BYTES_LIMIT:
            raise not_supported_error(f"inline bytes too large ({len(data)} > {_INLINE_BYTES_LIMIT})")
    if not mime_type:
        raise invalid_request_error("image mime_type required for bytes/path input")
    b64 = bytes_to_base64(data)
    return {"type": "input_image", "image_url": f"data:{mime_type};base64,{b64}"}


def _gather_text_inputs(request: GenerateRequest) -> list[str]:
    texts: list[str] = []
    for message in request.input:
        for part in message.content:
            if part.type != "text":
                raise invalid_request_error("embedding requires text-only input")
            texts.append(part.require_text())
    if not texts:
        raise invalid_request_error("embedding requires at least one text part")
    return texts


def _usage_from_openai(obj: dict[str, Any]) -> Usage | None:
    usage = obj.get("usage")
    if not isinstance(usage, dict):
        return None
    return Usage(
        input_tokens=usage.get("prompt_tokens"),
        output_tokens=usage.get("completion_tokens"),
        total_tokens=usage.get("total_tokens"),
    )


def _usage_from_openai_responses(obj: dict[str, Any]) -> Usage | None:
    usage = obj.get("usage")
    if not isinstance(usage, dict):
        return None
    return Usage(
        input_tokens=usage.get("input_tokens"),
        output_tokens=usage.get("output_tokens"),
        total_tokens=usage.get("total_tokens"),
    )


@dataclass(frozen=True, slots=True)
class OpenAIAdapter:
    api_key: str
    base_url: str = _OPENAI_DEFAULT_BASE_URL
    provider_name: str = "openai"
    chat_api: str = "chat_completions"
    proxy_url: str | None = None

    def capabilities(self, model_id: str) -> Capability:
        kind = infer_model_kind(model_id)
        kind_out_mods = output_modalities_for_kind(kind)

        if kind == "video":
            return Capability(
                input_modalities=video_input_modalities(model_id),
                output_modalities=kind_out_mods or {"video"},
                supports_stream=False,
                supports_job=True,
                supports_tools=False,
                supports_json_schema=False,
            )
        if kind == "image":
            return Capability(
                input_modalities=image_input_modalities(model_id),
                output_modalities=kind_out_mods or {"image"},
                supports_stream=False,
                supports_job=False,
                supports_tools=False,
                supports_json_schema=False,
            )
        if kind == "embedding":
            return Capability(
                input_modalities={"text"},
                output_modalities=kind_out_mods or {"embedding"},
                supports_stream=False,
                supports_job=False,
                supports_tools=False,
                supports_json_schema=False,
            )
        if kind == "tts":
            return Capability(
                input_modalities={"text"},
                output_modalities=kind_out_mods or {"audio"},
                supports_stream=False,
                supports_job=False,
                supports_tools=False,
                supports_json_schema=False,
            )
        if kind == "transcribe":
            return Capability(
                input_modalities=transcribe_input_modalities(model_id),
                output_modalities=kind_out_mods or {"text"},
                supports_stream=False,
                supports_job=False,
                supports_tools=False,
                supports_json_schema=False,
            )
        assert kind == "chat"
        if self.chat_api == "responses":
            in_mods = chat_input_modalities(model_id) & {"text", "image"}
            return Capability(
                input_modalities=in_mods,
                output_modalities={"text"},
                supports_stream=True,
                supports_job=False,
                supports_tools=True,
                supports_json_schema=True,
            )
        in_mods = chat_input_modalities(model_id)
        out_mods = chat_output_modalities(model_id)
        return Capability(
            input_modalities=in_mods,
            output_modalities=out_mods,
            supports_stream=True,
            supports_job=False,
            supports_tools=True,
            supports_json_schema=True,
        )

    def list_models(self, *, timeout_ms: int | None = None) -> list[str]:
        """
        Fetch remote model ids via OpenAI-compatible GET /models.
        """
        url = f"{self.base_url.rstrip('/')}/models"
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
        model_id = request.model_id()
        modalities = set(request.output.modalities)
        if "embedding" in modalities:
            if modalities != {"embedding"}:
                raise not_supported_error("embedding cannot be combined with other output modalities")
            if stream:
                raise not_supported_error("embedding does not support streaming")
            return self._embed(request, model_id=model_id)

        if modalities == {"video"}:
            if stream:
                raise not_supported_error("openai video generation does not support streaming")
            return self._video(request, model_id=model_id)

        if modalities == {"image"}:
            if stream:
                raise not_supported_error("openai image generation does not support streaming")
            return self._images(request, model_id=model_id)

        if modalities == {"audio"}:
            if stream:
                raise not_supported_error("openai TTS does not support streaming")
            return self._tts(request, model_id=model_id)

        if modalities == {"text"} and self._is_transcribe_model(model_id) and self._has_audio_input(request):
            if stream:
                raise not_supported_error("openai transcription does not support streaming")
            return self._transcribe(request, model_id=model_id)

        if self.chat_api == "responses":
            if stream:
                if "audio" in modalities:
                    raise not_supported_error("responses protocol does not support audio output in this SDK yet")
                return self._responses_stream(request, model_id=model_id)
            if "audio" in modalities:
                raise not_supported_error("responses protocol does not support audio output in this SDK yet")
            return self._responses(request, model_id=model_id)

        if stream:
            if "audio" in modalities:
                raise not_supported_error("streaming audio output is not supported in this SDK yet")
            return self._chat_stream(request, model_id=model_id)
        return self._chat(request, model_id=model_id)

    def _headers(self, request: GenerateRequest | None = None) -> dict[str, str]:
        headers = {"Authorization": f"Bearer {self.api_key}"}
        if request and request.params.idempotency_key:
            headers["Idempotency-Key"] = request.params.idempotency_key
        return headers

    def _apply_provider_options(self, body: dict[str, Any], request: GenerateRequest) -> None:
        opts = request.provider_options.get(self.provider_name)
        if not isinstance(opts, dict):
            return
        for k, v in opts.items():
            if k in body:
                raise invalid_request_error(f"provider_options cannot override body.{k}")
            body[k] = v

    def _apply_provider_options_form_fields(self, fields: dict[str, str], request: GenerateRequest) -> None:
        opts = request.provider_options.get(self.provider_name)
        if not isinstance(opts, dict):
            return
        for k, v in opts.items():
            if v is None:
                continue
            if k in fields:
                raise invalid_request_error(f"provider_options cannot override fields.{k}")
            if isinstance(v, bool):
                fields[k] = "true" if v else "false"
            elif isinstance(v, (int, float, str)):
                fields[k] = str(v)
            else:
                fields[k] = json.dumps(v, separators=(",", ":"))

    def _is_transcribe_model(self, model_id: str) -> bool:
        return is_transcribe_model(model_id)

    def _has_audio_input(self, request: GenerateRequest) -> bool:
        for m in request.input:
            for p in m.content:
                if p.type == "audio":
                    return True
        return False

    def _text_max_output_tokens(self, request: GenerateRequest) -> int | None:
        spec = request.output.text
        if spec and spec.max_output_tokens is not None:
            return spec.max_output_tokens
        return request.params.max_output_tokens

    def _chat_response_format(self, request: GenerateRequest) -> dict[str, Any] | None:
        spec = request.output.text
        if spec is None:
            return None
        if spec.format == "text" and spec.json_schema is None:
            return None
        if set(request.output.modalities) != {"text"}:
            raise invalid_request_error("json output requires text-only modality")
        if spec.json_schema is not None:
            return {
                "type": "json_schema",
                "json_schema": {
                    "name": "output",
                    "schema": spec.json_schema,
                    "strict": True,
                },
            }
        return {"type": "json_object"}

    def _responses_text_format(self, request: GenerateRequest) -> dict[str, Any] | None:
        spec = request.output.text
        if spec is None:
            return None
        if spec.format == "text" and spec.json_schema is None:
            return None
        if set(request.output.modalities) != {"text"}:
            raise invalid_request_error("json output requires text-only modality")
        if spec.json_schema is not None:
            return {
                "type": "json_schema",
                "name": "output",
                "schema": spec.json_schema,
                "strict": True,
            }
        return {"type": "json_object"}

    def _chat_body(self, request: GenerateRequest, *, model_id: str) -> dict[str, Any]:
        messages: list[dict[str, Any]] = []
        for m in request.input:
            if m.role == "tool":
                tool_parts = [p for p in m.content if p.type == "tool_result"]
                if len(tool_parts) != 1 or len(m.content) != 1:
                    raise invalid_request_error("tool messages must contain exactly one tool_result part")
                tool_call_id, _, result, _ = _require_tool_result_meta(tool_parts[0])
                if not tool_call_id:
                    raise invalid_request_error("tool_result.meta.tool_call_id required for OpenAI tool messages")
                messages.append({"role": "tool", "tool_call_id": tool_call_id, "content": _tool_result_to_string(result)})
                continue

            tool_calls: list[dict[str, Any]] = []
            content: list[dict[str, Any]] = []
            for p in m.content:
                if p.type == "tool_call":
                    if m.role != "assistant":
                        raise invalid_request_error("tool_call parts are only allowed in assistant messages")
                    tool_call_id, name, arguments = _require_tool_call_meta(p)
                    if not tool_call_id:
                        raise invalid_request_error("tool_call.meta.tool_call_id required for OpenAI tool calls")
                    tool_calls.append(
                        {
                            "id": tool_call_id,
                            "type": "function",
                            "function": {"name": name, "arguments": _tool_call_to_json_arguments(arguments)},
                        }
                    )
                    continue
                if p.type == "tool_result":
                    raise invalid_request_error("tool_result parts must be sent as role='tool'")
                content.append(
                    _part_to_chat_content(
                        p,
                        timeout_ms=request.params.timeout_ms,
                        provider_name=self.provider_name,
                        proxy_url=self.proxy_url,
                    )
                )

            msg: dict[str, Any] = {"role": m.role, "content": content if content else None}
            if tool_calls:
                msg["tool_calls"] = tool_calls
            messages.append(msg)

        body: dict[str, Any] = {"model": model_id, "messages": messages}
        params = request.params
        if params.temperature is not None:
            body["temperature"] = params.temperature
        if params.top_p is not None:
            body["top_p"] = params.top_p
        if params.seed is not None:
            body["seed"] = params.seed
        if params.reasoning is not None:
            if params.reasoning.effort is not None:
                body["reasoning_effort"] = normalize_reasoning_effort(params.reasoning.effort)
        max_out = self._text_max_output_tokens(request)
        if max_out is not None:
            body["max_completion_tokens"] = max_out
        if params.stop is not None:
            body["stop"] = params.stop
        resp_fmt = self._chat_response_format(request)
        if resp_fmt is not None:
            body["response_format"] = resp_fmt

        if request.tools:
            tools: list[dict[str, Any]] = []
            for t in request.tools:
                name = t.name.strip()
                if not name:
                    raise invalid_request_error("tool.name must be non-empty")
                fn: dict[str, Any] = {"name": name}
                if isinstance(t.description, str) and t.description.strip():
                    fn["description"] = t.description.strip()
                if t.parameters is not None:
                    fn["parameters"] = t.parameters
                if t.strict is not None:
                    fn["strict"] = bool(t.strict)
                tools.append({"type": "function", "function": fn})
            body["tools"] = tools

        if request.tool_choice is not None:
            choice = request.tool_choice.normalized()
            if choice.mode in {"required", "tool"} and not request.tools:
                raise invalid_request_error("tool_choice requires request.tools")
            if choice.mode == "tool":
                body["tool_choice"] = {"type": "function", "function": {"name": choice.name}}
            else:
                body["tool_choice"] = choice.mode

        modalities = request.output.modalities
        if "audio" in modalities:
            audio = request.output.audio
            if audio is None or not audio.voice:
                raise invalid_request_error("output.audio.voice required for audio output")
            fmt = audio.format or "wav"
            body["modalities"] = ["audio"] if modalities == ["audio"] else ["text", "audio"]
            body["audio"] = {"voice": audio.voice, "format": fmt}

        self._apply_provider_options(body, request)
        return body

    def _chat(self, request: GenerateRequest, *, model_id: str) -> GenerateResponse:
        url = f"{self.base_url}/chat/completions"
        obj = request_json(
            method="POST",
            url=url,
            headers=self._headers(request),
            json_body=self._chat_body(request, model_id=model_id),
            timeout_ms=request.params.timeout_ms,
            proxy_url=self.proxy_url,
        )
        return self._parse_chat_response(
            obj, provider=self.provider_name, model=f"{self.provider_name}:{model_id}", request=request
        )

    def _responses(self, request: GenerateRequest, *, model_id: str) -> GenerateResponse:
        url = f"{self.base_url}/responses"
        body: dict[str, Any] = {"model": model_id, "input": self._responses_input(request)}
        params = request.params
        if params.temperature is not None:
            body["temperature"] = params.temperature
        if params.top_p is not None:
            body["top_p"] = params.top_p
        if params.reasoning is not None:
            if params.reasoning.effort is not None:
                body["reasoning"] = {"effort": normalize_reasoning_effort(params.reasoning.effort)}
        max_out = self._text_max_output_tokens(request)
        if max_out is not None:
            body["max_output_tokens"] = max_out
        text_fmt = self._responses_text_format(request)
        if text_fmt is not None:
            body["text"] = {"format": text_fmt}

        if request.tools:
            tools: list[dict[str, Any]] = []
            for t in request.tools:
                name = t.name.strip()
                if not name:
                    raise invalid_request_error("tool.name must be non-empty")
                tool_obj: dict[str, Any] = {"type": "function", "name": name}
                if isinstance(t.description, str) and t.description.strip():
                    tool_obj["description"] = t.description.strip()
                tool_obj["parameters"] = t.parameters if t.parameters is not None else {"type": "object"}
                if t.strict is not None:
                    tool_obj["strict"] = bool(t.strict)
                tools.append(tool_obj)
            body["tools"] = tools

        if request.tool_choice is not None:
            choice = request.tool_choice.normalized()
            if choice.mode in {"required", "tool"} and not request.tools:
                raise invalid_request_error("tool_choice requires request.tools")
            if choice.mode == "tool":
                if self.provider_name.startswith("tuzi"):
                    if len(request.tools or []) == 1 and request.tools[0].name.strip() == choice.name:
                        body["tool_choice"] = "required"
                    else:
                        raise not_supported_error(
                            "tuzi responses protocol does not support tool_choice by name; "
                            "use tool_choice.mode='required' with a single tool"
                        )
                else:
                    body["tool_choice"] = {"type": "function", "name": choice.name}
            else:
                body["tool_choice"] = choice.mode

        self._apply_provider_options(body, request)

        obj = request_json(
            method="POST",
            url=url,
            headers=self._headers(request),
            json_body=body,
            timeout_ms=request.params.timeout_ms,
            proxy_url=self.proxy_url,
        )
        return self._parse_responses_response(
            obj, provider=self.provider_name, model=f"{self.provider_name}:{model_id}"
        )

    def _responses_stream(self, request: GenerateRequest, *, model_id: str) -> Iterator[GenerateEvent]:
        url = f"{self.base_url}/responses"
        body: dict[str, Any] = {"model": model_id, "input": self._responses_input(request), "stream": True}
        params = request.params
        if params.temperature is not None:
            body["temperature"] = params.temperature
        if params.top_p is not None:
            body["top_p"] = params.top_p
        if params.reasoning is not None:
            if params.reasoning.effort is not None:
                body["reasoning"] = {"effort": normalize_reasoning_effort(params.reasoning.effort)}
        max_out = self._text_max_output_tokens(request)
        if max_out is not None:
            body["max_output_tokens"] = max_out
        text_fmt = self._responses_text_format(request)
        if text_fmt is not None:
            body["text"] = {"format": text_fmt}

        if request.tools:
            tools: list[dict[str, Any]] = []
            for t in request.tools:
                name = t.name.strip()
                if not name:
                    raise invalid_request_error("tool.name must be non-empty")
                tool_obj: dict[str, Any] = {"type": "function", "name": name}
                if isinstance(t.description, str) and t.description.strip():
                    tool_obj["description"] = t.description.strip()
                tool_obj["parameters"] = t.parameters if t.parameters is not None else {"type": "object"}
                if t.strict is not None:
                    tool_obj["strict"] = bool(t.strict)
                tools.append(tool_obj)
            body["tools"] = tools

        if request.tool_choice is not None:
            choice = request.tool_choice.normalized()
            if choice.mode in {"required", "tool"} and not request.tools:
                raise invalid_request_error("tool_choice requires request.tools")
            if choice.mode == "tool":
                if self.provider_name.startswith("tuzi"):
                    if len(request.tools or []) == 1 and request.tools[0].name.strip() == choice.name:
                        body["tool_choice"] = "required"
                    else:
                        raise not_supported_error(
                            "tuzi responses protocol does not support tool_choice by name; "
                            "use tool_choice.mode='required' with a single tool"
                        )
                else:
                    body["tool_choice"] = {"type": "function", "name": choice.name}
            else:
                body["tool_choice"] = choice.mode

        self._apply_provider_options(body, request)

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
                typ = obj.get("type")
                if typ == "response.output_text.delta":
                    delta = obj.get("delta")
                    if isinstance(delta, str) and delta:
                        yield GenerateEvent(type="output.text.delta", data={"delta": delta})
                    continue
                if typ == "response.completed":
                    break
                if typ == "response.incomplete":
                    resp = obj.get("response")
                    reason = None
                    if isinstance(resp, dict):
                        details = resp.get("incomplete_details")
                        if isinstance(details, dict) and isinstance(details.get("reason"), str):
                            reason = details["reason"]
                    msg = "responses returned status: incomplete"
                    if reason:
                        msg = f"{msg} ({reason})"
                    raise provider_error(msg, retryable=False)
                if typ == "response.failed":
                    resp = obj.get("response")
                    code = None
                    msg = None
                    if isinstance(resp, dict):
                        err = resp.get("error")
                        if isinstance(err, dict):
                            if isinstance(err.get("code"), str):
                                code = err["code"]
                            if isinstance(err.get("message"), str):
                                msg = err["message"]
                    msg = msg or "responses returned status: failed"
                    raise provider_error(msg[:2_000], provider_code=code, retryable=False)
                if typ == "error":
                    code = obj.get("code") if isinstance(obj.get("code"), str) else None
                    msg = obj.get("message") if isinstance(obj.get("message"), str) else None
                    msg = msg or "responses stream error"
                    raise provider_error(msg[:2_000], provider_code=code, retryable=False)
            yield GenerateEvent(type="done", data={})

        return _iter()

    def _responses_input(self, request: GenerateRequest) -> list[dict[str, Any]]:
        items: list[dict[str, Any]] = []
        for m in request.input:
            if m.role == "tool":
                tool_parts = [p for p in m.content if p.type == "tool_result"]
                if len(tool_parts) != 1 or len(m.content) != 1:
                    raise invalid_request_error("tool messages must contain exactly one tool_result part")
                tool_call_id, _, result, _ = _require_tool_result_meta(tool_parts[0])
                if not tool_call_id:
                    raise invalid_request_error("tool_result.meta.tool_call_id required for responses protocol")
                items.append(
                    {
                        "type": "function_call_output",
                        "call_id": tool_call_id,
                        "output": _tool_result_to_string(result),
                    }
                )
                continue
            role = m.role
            content: list[dict[str, Any]] = []
            for p in m.content:
                if p.type == "text":
                    content.append({"type": "input_text", "text": p.require_text()})
                    continue
                if p.type == "image":
                    content.append(
                        _part_to_responses_image_content(p, timeout_ms=request.params.timeout_ms, proxy_url=self.proxy_url)
                    )
                    continue
                if p.type in {"tool_call", "tool_result"}:
                    raise not_supported_error("responses protocol does not support tool parts in message input")
                raise not_supported_error(f"responses protocol does not support input part: {p.type}")
            items.append({"role": role, "content": content})
        return items

    def _parse_responses_response(self, obj: dict[str, Any], *, provider: str, model: str) -> GenerateResponse:
        resp_id = obj.get("id") or f"sdk_{uuid4().hex}"
        status = obj.get("status")
        if status != "completed":
            raise provider_error(f"responses returned status: {status}")
        output = obj.get("output")
        if not isinstance(output, list):
            raise provider_error("responses missing output")

        parts: list[Part] = []
        for item in output:
            if not isinstance(item, dict):
                continue
            typ = item.get("type")
            if typ == "message":
                content = item.get("content")
                if not isinstance(content, list):
                    continue
                for c in content:
                    if not isinstance(c, dict):
                        continue
                    if c.get("type") == "output_text" and isinstance(c.get("text"), str):
                        parts.append(Part.from_text(c["text"]))
                continue
            if typ == "function_call":
                call_id = item.get("call_id")
                name = item.get("name")
                arguments = item.get("arguments")
                if isinstance(call_id, str) and call_id and isinstance(name, str) and name:
                    parts.append(
                        Part.tool_call(
                            tool_call_id=call_id,
                            name=name,
                            arguments=_parse_tool_call_arguments(arguments),
                        )
                    )

        if not parts:
            parts.append(Part.from_text(""))
        usage = _usage_from_openai_responses(obj)
        return GenerateResponse(
            id=str(resp_id),
            provider=provider,
            model=model,
            status="completed",
            output=[Message(role="assistant", content=parts)],
            usage=usage,
        )

    def _chat_stream(self, request: GenerateRequest, *, model_id: str) -> Iterator[GenerateEvent]:
        url = f"{self.base_url}/chat/completions"
        body = self._chat_body(request, model_id=model_id)
        body["stream"] = True
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
                choices = obj.get("choices")
                if not isinstance(choices, list) or not choices:
                    continue
                delta = choices[0].get("delta")
                if not isinstance(delta, dict):
                    continue
                text = delta.get("content")
                if isinstance(text, str) and text:
                    yield GenerateEvent(type="output.text.delta", data={"delta": text})
            yield GenerateEvent(type="done", data={})

        return _iter()

    def _parse_chat_response(
        self,
        obj: dict[str, Any],
        *,
        provider: str,
        model: str,
        request: GenerateRequest | None = None,
    ) -> GenerateResponse:
        resp_id = obj.get("id") or f"sdk_{uuid4().hex}"
        choices = obj.get("choices")
        if not isinstance(choices, list) or not choices:
            raise provider_error("openai chat response missing choices")
        msg = choices[0].get("message")
        if not isinstance(msg, dict):
            raise provider_error("openai chat response missing message")

        parts: list[Part] = []
        content_text = msg.get("content")
        if isinstance(content_text, str) and content_text:
            parts.append(Part.from_text(content_text))

        audio = msg.get("audio")
        if isinstance(audio, dict):
            data_b64 = audio.get("data")
            if isinstance(data_b64, str) and data_b64:
                fmt = None
                if request and request.output.audio and request.output.audio.format:
                    fmt = request.output.audio.format
                mime = _audio_mime_from_format(fmt or "wav")
                parts.append(Part(type="audio", mime_type=mime, source=PartSourceBytes(data=data_b64, encoding="base64")))
            transcript = audio.get("transcript")
            if (
                isinstance(transcript, str)
                and transcript
                and not (isinstance(content_text, str) and content_text)
            ):
                parts.append(Part.from_text(transcript))

        tool_calls = msg.get("tool_calls")
        if isinstance(tool_calls, list):
            for call in tool_calls:
                if not isinstance(call, dict):
                    continue
                tool_call_id = call.get("id")
                if not isinstance(tool_call_id, str) or not tool_call_id:
                    continue
                fn = call.get("function")
                if not isinstance(fn, dict):
                    continue
                name = fn.get("name")
                args = fn.get("arguments")
                if not isinstance(name, str) or not name:
                    continue
                parts.append(
                    Part.tool_call(
                        tool_call_id=tool_call_id,
                        name=name,
                        arguments=_parse_tool_call_arguments(args),
                    )
                )

        if not parts:
            parts.append(Part.from_text(""))

        usage = _usage_from_openai(obj)
        return GenerateResponse(
            id=str(resp_id),
            provider=provider,
            model=model,
            status="completed",
            output=[Message(role="assistant", content=parts)],
            usage=usage,
        )

    def _images(self, request: GenerateRequest, *, model_id: str) -> GenerateResponse:
        if self.provider_name == "openai" and not (
            model_id.startswith("dall-e-")
            or model_id.startswith("gpt-image-")
            or model_id.startswith("chatgpt-image")
        ):
            raise not_supported_error(f'image generation requires model like "{self.provider_name}:gpt-image-1"')

        texts: list[str] = []
        images: list[Part] = []
        for m in request.input:
            for p in m.content:
                if p.type == "text":
                    t = p.require_text().strip()
                    if t:
                        texts.append(t)
                    continue
                if p.type == "image":
                    images.append(p)
                    continue
                raise invalid_request_error("image generation only supports text (+ optional image)")
        if len(texts) != 1:
            raise invalid_request_error("image generation requires exactly one text part")
        if len(images) > 1:
            raise invalid_request_error("image generation supports at most one image input")

        prompt = texts[0]
        image_part = images[0] if images else None

        response_format: str | None = None
        img = request.output.image
        if img and img.n is not None:
            n = img.n
        else:
            n = None
        if img and img.size is not None:
            size = img.size
        else:
            size = None
        if model_id.startswith("dall-e-"):
            response_format = "url"
        if img and img.format:
            fmt = img.format.strip().lower()
            if fmt in {"url"}:
                response_format = "url"
            elif fmt in {"b64_json", "base64", "bytes"}:
                response_format = "b64_json"

        if image_part is None:
            body: dict[str, Any] = {"model": model_id, "prompt": prompt}
            if n is not None:
                body["n"] = n
            if size is not None:
                body["size"] = size
            if response_format:
                body["response_format"] = response_format
            self._apply_provider_options(body, request)
            url = f"{self.base_url}/images/generations"
            obj = request_json(
                method="POST",
                url=url,
                headers=self._headers(request),
                json_body=body,
                timeout_ms=request.params.timeout_ms,
                proxy_url=self.proxy_url,
            )
        else:
            src = image_part.require_source()
            tmp_path: str | None = None
            if isinstance(src, PartSourceUrl):
                tmp_path = download_to_tempfile(
                    url=src.url,
                    timeout_ms=request.params.timeout_ms,
                    max_bytes=_INLINE_BYTES_LIMIT,
                    proxy_url=self.proxy_url,
                )
                file_path = tmp_path
            elif isinstance(src, PartSourcePath):
                file_path = src.path
            elif isinstance(src, PartSourceBytes) and src.encoding == "base64":
                try:
                    data = base64.b64decode(src.data)
                except Exception:
                    raise invalid_request_error("image base64 data is not valid base64")
                with tempfile.NamedTemporaryFile(prefix="genaisdk-", suffix=".bin", delete=False) as f:
                    f.write(data)
                    tmp_path = f.name
                file_path = tmp_path
            elif isinstance(src, PartSourceRef):
                raise not_supported_error("image edits do not support ref input; use url/bytes/path")
            else:
                assert isinstance(src, PartSourceBytes)
                if not isinstance(src.data, bytes):
                    raise invalid_request_error("image bytes data must be bytes")
                with tempfile.NamedTemporaryFile(prefix="genaisdk-", suffix=".bin", delete=False) as f:
                    f.write(src.data)
                    tmp_path = f.name
                file_path = tmp_path

            try:
                fields: dict[str, str] = {"model": model_id, "prompt": prompt}
                if n is not None:
                    fields["n"] = str(n)
                if size is not None:
                    fields["size"] = str(size)
                if response_format:
                    fields["response_format"] = response_format
                self._apply_provider_options_form_fields(fields, request)
                body = multipart_form_data(
                    fields=fields,
                    file_field="image",
                    file_path=file_path,
                    filename=os.path.basename(file_path),
                    file_mime_type=image_part.mime_type
                    or detect_mime_type(file_path)
                    or "application/octet-stream",
                )
                obj = request_streaming_body_json(
                    method="POST",
                    url=f"{self.base_url}/images/edits",
                    headers=self._headers(request),
                    body=body,
                    timeout_ms=request.params.timeout_ms,
                    proxy_url=self.proxy_url,
                )
            finally:
                if tmp_path is not None:
                    try:
                        os.unlink(tmp_path)
                    except OSError:
                        pass
        resp_id = obj.get("created") or f"sdk_{uuid4().hex}"
        data = obj.get("data")
        items: list[object] | None = None
        if isinstance(data, list):
            items = data
        elif isinstance(data, dict):
            images = data.get("images")
            if isinstance(images, list):
                items = images
            else:
                inner = data.get("data")
                if isinstance(inner, list):
                    items = inner
        if not items:
            raise provider_error("openai images response missing data")
        parts = []
        for item in items:
            if not isinstance(item, dict):
                continue
            u = item.get("url")
            if isinstance(u, str) and u:
                parts.append(Part(type="image", source=PartSourceUrl(url=u)))
                continue
            b64 = item.get("b64_json")
            if isinstance(b64, str) and b64:
                parts.append(Part(type="image", mime_type="image/png", source=PartSourceBytes(data=b64, encoding="base64")))
        if not parts:
            raise provider_error("openai images response missing urls")
        return GenerateResponse(
            id=str(resp_id),
            provider=self.provider_name,
            model=f"{self.provider_name}:{model_id}",
            status="completed",
            output=[Message(role="assistant", content=parts)],
            usage=None,
        )

    def _tts(self, request: GenerateRequest, *, model_id: str) -> GenerateResponse:
        if self.provider_name == "openai" and not (model_id.startswith("tts-") or "-tts" in model_id):
            raise invalid_request_error(f'TTS requires model like "{self.provider_name}:tts-1"')
        text = self._single_text_prompt(request)
        audio = request.output.audio
        if audio is None or not audio.voice:
            raise invalid_request_error("output.audio.voice required for TTS")
        fmt = audio.format or "mp3"
        body: dict[str, Any] = {
            "model": model_id,
            "voice": audio.voice,
            "input": text,
            "response_format": fmt,
        }
        self._apply_provider_options(body, request)
        url = f"{self.base_url}/audio/speech"
        data = request_bytes(
            method="POST",
            url=url,
            headers={**self._headers(request), "Content-Type": "application/json"},
            body=json.dumps(body, separators=(",", ":")).encode("utf-8"),
            timeout_ms=request.params.timeout_ms,
            proxy_url=self.proxy_url,
        )
        part = Part(
            type="audio",
            mime_type=_audio_mime_from_format(fmt),
            source=PartSourceBytes(data=bytes_to_base64(data), encoding="base64"),
        )
        return GenerateResponse(
            id=f"sdk_{uuid4().hex}",
            provider=self.provider_name,
            model=f"{self.provider_name}:{model_id}",
            status="completed",
            output=[Message(role="assistant", content=[part])],
            usage=None,
        )

    def _transcribe(self, request: GenerateRequest, *, model_id: str) -> GenerateResponse:
        audio_part = self._single_audio_part(request)
        prompt = self._transcription_prompt(request, audio_part=audio_part)
        src = audio_part.require_source()
        tmp_path: str | None = None
        if isinstance(src, PartSourceUrl):
            tmp_path = _download_to_temp(
                src.url,
                timeout_ms=request.params.timeout_ms,
                max_bytes=None,
                proxy_url=self.proxy_url,
            )
            file_path = tmp_path
        elif isinstance(src, PartSourcePath):
            file_path = src.path
        elif isinstance(src, PartSourceBytes) and src.encoding == "base64":
            try:
                data = base64.b64decode(src.data)
            except Exception:
                raise invalid_request_error("audio base64 data is not valid base64")
            with tempfile.NamedTemporaryFile(prefix="genaisdk-", suffix=".bin", delete=False) as f:
                f.write(data)
                tmp_path = f.name
            file_path = tmp_path
        elif isinstance(src, PartSourceRef):
            raise not_supported_error("openai transcription does not support ref input")
        else:
            assert isinstance(src, PartSourceBytes)
            if not isinstance(src.data, bytes):
                raise invalid_request_error("audio bytes data must be bytes")
            with tempfile.NamedTemporaryFile(prefix="genaisdk-", suffix=".bin", delete=False) as f:
                f.write(src.data)
                tmp_path = f.name
            file_path = tmp_path

        try:
            fields = {"model": model_id}
            if request.params.temperature is not None:
                fields["temperature"] = str(request.params.temperature)
            lang = audio_part.meta.get("language")
            if isinstance(lang, str) and lang.strip():
                fields["language"] = lang.strip()
            if prompt:
                fields["prompt"] = prompt
            self._apply_provider_options_form_fields(fields, request)
            if "diarize" in model_id and "chunking_strategy" not in fields:
                fields["chunking_strategy"] = "auto"
            body = multipart_form_data(
                fields=fields,
                file_field="file",
                file_path=file_path,
                filename=os.path.basename(file_path),
                file_mime_type=audio_part.mime_type or detect_mime_type(file_path) or "application/octet-stream",
            )
            url = f"{self.base_url}/audio/transcriptions"
            obj = request_streaming_body_json(
                method="POST",
                url=url,
                headers=self._headers(request),
                body=body,
                timeout_ms=request.params.timeout_ms,
                proxy_url=self.proxy_url,
            )
        finally:
            if tmp_path is not None:
                try:
                    os.unlink(tmp_path)
                except OSError:
                    pass

        text = obj.get("text")
        if not isinstance(text, str):
            raise provider_error("openai transcription missing text")
        return GenerateResponse(
            id=f"sdk_{uuid4().hex}",
            provider=self.provider_name,
            model=f"{self.provider_name}:{model_id}",
            status="completed",
            output=[Message(role="assistant", content=[Part.from_text(text)])],
            usage=None,
        )

    def _embed(self, request: GenerateRequest, *, model_id: str) -> GenerateResponse:
        if self.provider_name == "openai" and not model_id.startswith("text-embedding-"):
            raise not_supported_error(f'embedding requires model like "{self.provider_name}:text-embedding-3-small"')
        texts = _gather_text_inputs(request)
        url = f"{self.base_url}/embeddings"
        body: dict[str, Any] = {"model": model_id, "input": texts}
        emb = request.output.embedding
        if emb and emb.dimensions is not None:
            if self.provider_name == "openai" and not model_id.startswith("text-embedding-3-"):
                raise invalid_request_error("embedding.dimensions is only supported for OpenAI text-embedding-3 models")
            body["dimensions"] = emb.dimensions
        self._apply_provider_options(body, request)
        obj = request_json(
            method="POST",
            url=url,
            headers=self._headers(request),
            json_body=body,
            timeout_ms=request.params.timeout_ms,
            proxy_url=self.proxy_url,
        )
        data = obj.get("data")
        if not isinstance(data, list) or len(data) != len(texts):
            raise provider_error("openai embeddings response missing data")
        parts: list[Part] = []
        for item in data:
            if not isinstance(item, dict):
                raise provider_error("openai embeddings item is not object")
            emb = item.get("embedding")
            if not isinstance(emb, list) or not all(isinstance(x, (int, float)) for x in emb):
                raise provider_error("openai embeddings item missing embedding")
            parts.append(Part(type="embedding", embedding=[float(x) for x in emb]))

        usage = None
        u = obj.get("usage")
        if isinstance(u, dict):
            usage = Usage(
                input_tokens=u.get("prompt_tokens"),
                total_tokens=u.get("total_tokens"),
            )

        return GenerateResponse(
            id=f"sdk_{uuid4().hex}",
            provider=self.provider_name,
            model=f"{self.provider_name}:{model_id}",
            status="completed",
            output=[Message(role="assistant", content=parts)],
            usage=usage,
        )

    def _video(self, request: GenerateRequest, *, model_id: str) -> GenerateResponse:
        if self.provider_name == "openai" and not model_id.startswith("sora-"):
            raise not_supported_error(f'video generation requires model like "{self.provider_name}:sora-2"')

        is_tuzi = self.provider_name.startswith("tuzi")

        def _tuzi_prompt_and_image(req: GenerateRequest) -> tuple[str, Part | None]:
            texts: list[str] = []
            images: list[Part] = []
            for msg in req.input:
                for part in msg.content:
                    if part.type == "text":
                        t = part.require_text().strip()
                        if t:
                            texts.append(t)
                        continue
                    if part.type == "image":
                        images.append(part)
                        continue
                    raise invalid_request_error("video generation only supports text (+ optional image)")
            if len(texts) != 1:
                raise invalid_request_error("video generation requires exactly one text part")
            if len(images) > 1:
                raise invalid_request_error("video generation supports at most one image input")
            return texts[0], images[0] if images else None

        if is_tuzi:
            prompt, image_part = _tuzi_prompt_and_image(request)
        else:
            prompt = self._single_text_prompt(request)

        video = request.output.video
        if is_tuzi:
            is_sora = model_id.lower().startswith("sora-")
            fields: dict[str, str] = {"model": model_id, "prompt": prompt}
            if video and video.duration_sec is not None:
                fields["seconds"] = str(_closest_video_seconds(video.duration_sec, is_tuzi=not is_sora))
            if video and video.aspect_ratio:
                size = _video_size_from_aspect_ratio(video.aspect_ratio)
                if size:
                    fields["size"] = size
            self._apply_provider_options_form_fields(fields, request)
            tmp_path: str | None = None
            try:
                if image_part is None:
                    body = multipart_form_data_fields(fields=fields)
                else:
                    src = image_part.require_source()
                    if isinstance(src, PartSourceUrl):
                        fields["first_frame_image"] = src.url
                        fields["input_reference"] = src.url
                        body = multipart_form_data_fields(fields=fields)
                    elif isinstance(src, PartSourceRef):
                        raise not_supported_error("tuzi video generation does not support ref image input")
                    else:
                        if isinstance(src, PartSourcePath):
                            file_path = src.path
                        elif isinstance(src, PartSourceBytes) and src.encoding == "base64":
                            try:
                                data = base64.b64decode(src.data)
                            except Exception:
                                raise invalid_request_error("image base64 data is not valid base64")
                            with tempfile.NamedTemporaryFile(prefix="genaisdk-", suffix=".bin", delete=False) as f:
                                f.write(data)
                                tmp_path = f.name
                            file_path = tmp_path
                        else:
                            assert isinstance(src, PartSourceBytes)
                            if not isinstance(src.data, bytes):
                                raise invalid_request_error("image bytes data must be bytes")
                            with tempfile.NamedTemporaryFile(prefix="genaisdk-", suffix=".bin", delete=False) as f:
                                f.write(src.data)
                                tmp_path = f.name
                            file_path = tmp_path

                        body = multipart_form_data(
                            fields=fields,
                            file_field="input_reference",
                            file_path=file_path,
                            filename=os.path.basename(file_path),
                            file_mime_type=image_part.mime_type
                            or detect_mime_type(file_path)
                            or "application/octet-stream",
                        )
                obj = request_streaming_body_json(
                    method="POST",
                    url=f"{self.base_url}/videos",
                    headers=self._headers(request),
                    body=body,
                    timeout_ms=request.params.timeout_ms,
                    proxy_url=self.proxy_url,
                )
            finally:
                if tmp_path is not None:
                    try:
                        os.unlink(tmp_path)
                    except OSError:
                        pass
        else:
            body: dict[str, Any] = {"model": model_id, "prompt": prompt}
            if video and video.duration_sec is not None:
                body["seconds"] = _closest_video_seconds(video.duration_sec, is_tuzi=False)
            if video and video.aspect_ratio:
                size = _video_size_from_aspect_ratio(video.aspect_ratio)
                if size:
                    body["size"] = size
            self._apply_provider_options(body, request)
            obj = request_json(
                method="POST",
                url=f"{self.base_url}/videos",
                headers=self._headers(request),
                json_body=body,
                timeout_ms=request.params.timeout_ms,
                proxy_url=self.proxy_url,
            )
        video_id = obj.get("id")
        if not isinstance(video_id, str) or not video_id:
            raise provider_error("openai video response missing id")

        if not request.wait:
            return GenerateResponse(
                id=f"sdk_{uuid4().hex}",
                provider=self.provider_name,
                model=f"{self.provider_name}:{model_id}",
                status="running",
                job=JobInfo(job_id=video_id, poll_after_ms=1_000),
            )

        job = self._wait_video_job(video_id, timeout_ms=request.params.timeout_ms)
        status = job.get("status")
        if status != "completed":
            if status == "failed":
                err = job.get("error")
                raise provider_error(f"openai video generation failed: {err}")
            return GenerateResponse(
                id=f"sdk_{uuid4().hex}",
                provider=self.provider_name,
                model=f"{self.provider_name}:{model_id}",
                status="running",
                job=JobInfo(job_id=video_id, poll_after_ms=1_000),
            )

        data = request_bytes(
            method="GET",
            url=f"{self.base_url}/videos/{video_id}/content",
            headers=self._headers(request),
            timeout_ms=request.params.timeout_ms,
            proxy_url=self.proxy_url,
        )
        part = Part(type="video", mime_type="video/mp4", source=PartSourceBytes(data=bytes_to_base64(data), encoding="base64"))
        return GenerateResponse(
            id=f"sdk_{uuid4().hex}",
            provider=self.provider_name,
            model=f"{self.provider_name}:{model_id}",
            status="completed",
            output=[Message(role="assistant", content=[part])],
            usage=None,
        )

    def _wait_video_job(self, video_id: str, *, timeout_ms: int | None) -> dict[str, Any]:
        budget_ms = 120_000 if timeout_ms is None else timeout_ms
        deadline = time.time() + max(1, budget_ms) / 1000.0
        while True:
            remaining_ms = int((deadline - time.time()) * 1000)
            if remaining_ms <= 0:
                break
            obj = request_json(
                method="GET",
                url=f"{self.base_url}/videos/{video_id}",
                headers=self._headers(),
                timeout_ms=min(30_000, remaining_ms),
                proxy_url=self.proxy_url,
            )
            status = obj.get("status")
            if status in {"completed", "failed"}:
                return obj
            time.sleep(min(1.0, max(0.0, deadline - time.time())))
        return {"id": video_id, "status": "in_progress"}

    def _single_text_prompt(self, request: GenerateRequest) -> str:
        texts = _gather_text_inputs(request)
        if len(texts) != 1:
            raise invalid_request_error("this operation requires exactly one text part")
        return texts[0]

    def _single_audio_part(self, request: GenerateRequest) -> Part:
        parts: list[Part] = []
        for m in request.input:
            for p in m.content:
                if p.type == "audio":
                    parts.append(p)
                elif p.type != "text":
                    raise invalid_request_error("transcription only supports audio (+ optional text)")
        if len(parts) != 1:
            raise invalid_request_error("transcription requires exactly one audio part")
        return parts[0]

    def _transcription_prompt(self, request: GenerateRequest, *, audio_part: Part) -> str | None:
        v = audio_part.meta.get("transcription_prompt")
        if isinstance(v, str) and v.strip():
            return v.strip()
        chunks: list[str] = []
        for m in request.input:
            for p in m.content:
                if p.type != "text":
                    continue
                if p.meta.get("transcription_prompt") is not True:
                    continue
                t = p.require_text().strip()
                if t:
                    chunks.append(t)
        if not chunks:
            return None
        return "\n\n".join(chunks)


def _closest_video_seconds(duration_sec: int, *, is_tuzi: bool) -> int:
    if is_tuzi:
        if duration_sec <= 10:
            return 10
        if duration_sec <= 15:
            return 15
        return 25
    if duration_sec <= 4:
        return 4
    if duration_sec <= 8:
        return 8
    return 12


def _video_size_from_aspect_ratio(aspect_ratio: str) -> str | None:
    ar = aspect_ratio.strip()
    if ar == "16:9":
        return "1280x720"
    if ar == "9:16":
        return "720x1280"
    return None
