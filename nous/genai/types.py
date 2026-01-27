from __future__ import annotations

import base64
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

from ._internal.errors import ErrorInfo, invalid_request_error, not_supported_error

Role = Literal["system", "user", "assistant", "tool"]
PartType = Literal[
    "text",
    "image",
    "audio",
    "video",
    "embedding",
    "file",
    "tool_call",
    "tool_result",
]

Modality = Literal["text", "image", "audio", "video", "embedding"]
Status = Literal["completed", "running", "failed", "canceled"]
ReasoningEffort = Literal["none", "minimal", "low", "medium", "high", "xhigh"]
ToolChoiceMode = Literal["none", "auto", "required", "tool"]


@dataclass(frozen=True, slots=True)
class ReasoningSpec:
    effort: ReasoningEffort | None = None


_REASONING_EFFORT_VALUES = {"none", "minimal", "low", "medium", "high", "xhigh"}


def normalize_reasoning_effort(value: object) -> ReasoningEffort:
    if not isinstance(value, str) or not value.strip():
        raise invalid_request_error("reasoning.effort must be a non-empty string")
    effort = value.strip().lower()
    if effort not in _REASONING_EFFORT_VALUES:
        raise invalid_request_error(f"unknown reasoning.effort: {effort}")
    return effort


@dataclass(frozen=True, slots=True)
class PartSourceBytes:
    kind: Literal["bytes"] = "bytes"
    data: bytes | str = b""
    encoding: Literal["base64"] | None = None

    def __post_init__(self) -> None:
        if self.encoding is None:
            if isinstance(self.data, bytearray):
                object.__setattr__(self, "data", bytes(self.data))
                return
            if not isinstance(self.data, bytes):
                raise ValueError("PartSourceBytes.data must be bytes when encoding is None")
            return

        if self.encoding != "base64":
            raise ValueError(f"unknown PartSourceBytes.encoding: {self.encoding}")
        if not isinstance(self.data, str):
            raise ValueError("PartSourceBytes.data must be str when encoding is 'base64'")


@dataclass(frozen=True, slots=True)
class PartSourcePath:
    kind: Literal["path"] = "path"
    path: str = ""


@dataclass(frozen=True, slots=True)
class PartSourceUrl:
    kind: Literal["url"] = "url"
    url: str = ""


@dataclass(frozen=True, slots=True)
class PartSourceRef:
    kind: Literal["ref"] = "ref"
    provider: str = ""
    id: str = ""


PartSource = PartSourceBytes | PartSourcePath | PartSourceUrl | PartSourceRef


@dataclass(frozen=True, slots=True)
class Part:
    type: PartType
    mime_type: str | None = None
    source: PartSource | None = None
    text: str | None = None
    embedding: list[float] | None = None
    meta: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not isinstance(self.meta, dict):
            raise ValueError("Part.meta must be an object")
        if self.mime_type is not None and not isinstance(self.mime_type, str):
            raise ValueError("Part.mime_type must be a string")

        if self.type == "text":
            if not isinstance(self.text, str):
                raise ValueError("text Part requires text")
            if self.source is not None:
                raise ValueError("text Part cannot have source")
            if self.embedding is not None:
                raise ValueError("text Part cannot have embedding")
            return

        if self.type == "embedding":
            if not isinstance(self.embedding, list) or not all(isinstance(x, (int, float)) for x in self.embedding):
                raise ValueError("embedding Part requires embedding: list[number]")
            if self.source is not None:
                raise ValueError("embedding Part cannot have source")
            if self.text is not None:
                raise ValueError("embedding Part cannot have text")
            return

        if self.type in {"image", "audio", "video", "file"}:
            if self.source is None:
                raise ValueError(f"{self.type} Part requires source")
            if not isinstance(
                self.source,
                (PartSourceBytes, PartSourcePath, PartSourceUrl, PartSourceRef),
            ):
                raise ValueError("Part.source must be a PartSource object")
            if self.text is not None:
                raise ValueError(f"{self.type} Part cannot have text")
            if self.embedding is not None:
                raise ValueError(f"{self.type} Part cannot have embedding")
            if self.mime_type and self.type in {"image", "audio", "video"}:
                prefix = f"{self.type}/"
                if not self.mime_type.startswith(prefix):
                    raise ValueError(f"{self.type} Part mime_type must start with {prefix!r}")
            return

        if self.type in {"tool_call", "tool_result"}:
            if self.source is not None:
                raise ValueError(f"{self.type} Part cannot have source")
            if self.text is not None:
                raise ValueError(f"{self.type} Part cannot have text")
            if self.embedding is not None:
                raise ValueError(f"{self.type} Part cannot have embedding")
            return

        raise ValueError(f"unknown Part.type: {self.type}")

    @staticmethod
    def from_text(text: str) -> "Part":
        return Part(type="text", text=text)

    @staticmethod
    def tool_call(*, name: str, arguments: Any, tool_call_id: str | None = None) -> "Part":
        meta: dict[str, Any] = {"name": name, "arguments": arguments}
        if tool_call_id is not None:
            meta["tool_call_id"] = tool_call_id
        return Part(type="tool_call", meta=meta)

    @staticmethod
    def tool_result(
        *,
        name: str,
        result: Any,
        tool_call_id: str | None = None,
        is_error: bool | None = None,
    ) -> "Part":
        meta: dict[str, Any] = {"name": name, "result": result}
        if tool_call_id is not None:
            meta["tool_call_id"] = tool_call_id
        if is_error is not None:
            meta["is_error"] = bool(is_error)
        return Part(type="tool_result", meta=meta)

    @staticmethod
    def embedding_part(vector: list[float]) -> "Part":
        return Part(type="embedding", embedding=vector)

    def require_text(self) -> str:
        if self.type != "text" or self.text is None:
            raise invalid_request_error("Part is not text")
        return self.text

    def require_source(self) -> PartSource:
        if self.source is None:
            raise invalid_request_error("Part has no source")
        return self.source


@dataclass(frozen=True, slots=True)
class Message:
    role: Role
    content: list[Part]


@dataclass(frozen=True, slots=True)
class OutputTextSpec:
    format: Literal["text", "json"] = "text"
    json_schema: Any | None = None
    max_output_tokens: int | None = None


@dataclass(frozen=True, slots=True)
class OutputImageSpec:
    n: int | None = None
    size: str | None = None
    format: str | None = None


@dataclass(frozen=True, slots=True)
class OutputAudioSpec:
    voice: str | None = None
    format: str | None = None
    language: str | None = None


@dataclass(frozen=True, slots=True)
class OutputVideoSpec:
    duration_sec: int | None = None
    aspect_ratio: str | None = None
    fps: int | None = None
    format: str | None = None


@dataclass(frozen=True, slots=True)
class OutputEmbeddingSpec:
    dimensions: int | None = None


@dataclass(frozen=True, slots=True)
class Tool:
    """
    Minimal function tool declaration (provider-agnostic).

    - `parameters` is a JSON Schema object describing the function arguments.
    - Providers may support only a subset of JSON Schema/OpenAPI Schema.
    """

    name: str
    description: str | None = None
    parameters: dict[str, Any] | None = None
    strict: bool | None = None


@dataclass(frozen=True, slots=True)
class ToolChoice:
    mode: ToolChoiceMode = "auto"
    name: str | None = None

    def normalized(self) -> "ToolChoice":
        mode = self.mode.strip().lower()
        if mode not in {"none", "auto", "required", "tool"}:
            raise invalid_request_error(f"unknown tool_choice.mode: {self.mode}")
        name = self.name.strip() if isinstance(self.name, str) else None
        if mode == "tool" and not name:
            raise invalid_request_error("tool_choice.name required when mode='tool'")
        if mode != "tool" and name is not None:
            raise invalid_request_error("tool_choice.name only allowed when mode='tool'")
        return ToolChoice(mode=mode, name=name)


@dataclass(frozen=True, slots=True)
class OutputSpec:
    modalities: list[Modality]
    text: OutputTextSpec | None = None
    image: OutputImageSpec | None = None
    audio: OutputAudioSpec | None = None
    video: OutputVideoSpec | None = None
    embedding: OutputEmbeddingSpec | None = None


@dataclass(frozen=True, slots=True)
class GenerateParams:
    temperature: float | None = None
    top_p: float | None = None
    seed: int | None = None
    max_output_tokens: int | None = None
    stop: list[str] | None = None
    timeout_ms: int | None = None
    idempotency_key: str | None = None
    reasoning: ReasoningSpec | None = None


@dataclass(frozen=True, slots=True)
class GenerateRequest:
    model: str = field(
        metadata={
            "description": 'Model string in the form "{provider}:{model_id}" (e.g. "openai:gpt-4o-mini")',
            "pattern": r"^[^\s:]+:[^\s]+$",
            "examples": ["openai:gpt-4o-mini"],
        }
    )
    input: list[Message]
    output: OutputSpec
    params: GenerateParams = field(default_factory=GenerateParams)
    wait: bool = True
    tools: list[Tool] | None = None
    tool_choice: ToolChoice | None = None
    provider_options: dict[str, Any] = field(default_factory=dict)

    def provider(self) -> str:
        if ":" not in self.model:
            raise invalid_request_error('model must be "{provider}:{model_id}"')
        return self.model.split(":", 1)[0]

    def model_id(self) -> str:
        if ":" not in self.model:
            raise invalid_request_error('model must be "{provider}:{model_id}"')
        return self.model.split(":", 1)[1]


@dataclass(frozen=True, slots=True)
class Usage:
    input_tokens: int | None = None
    output_tokens: int | None = None
    total_tokens: int | None = None
    seconds: float | None = None
    image_count: int | None = None
    video_seconds: float | None = None
    cost_estimate: float | None = None


@dataclass(frozen=True, slots=True)
class JobInfo:
    job_id: str
    poll_after_ms: int = 1_000
    expires_at: str | None = None


@dataclass(frozen=True, slots=True)
class GenerateResponse:
    id: str
    provider: str
    model: str
    status: Status
    output: list[Message] = field(default_factory=list)
    usage: Usage | None = None
    job: JobInfo | None = None
    error: ErrorInfo | None = None


@dataclass(frozen=True, slots=True)
class GenerateEvent:
    type: str
    data: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class Capability:
    input_modalities: set[Modality]
    output_modalities: set[Modality]
    supports_stream: bool
    supports_job: bool
    supports_tools: bool = False
    supports_json_schema: bool = False


def detect_mime_type(path: str) -> str | None:
    suffix = Path(path).suffix.lower()
    if suffix in {".png"}:
        return "image/png"
    if suffix in {".jpg", ".jpeg"}:
        return "image/jpeg"
    if suffix in {".webp"}:
        return "image/webp"
    if suffix in {".wav"}:
        return "audio/wav"
    if suffix in {".mp3"}:
        return "audio/mpeg"
    if suffix in {".m4a"}:
        return "audio/mp4"
    if suffix in {".mp4"}:
        return "video/mp4"
    if suffix in {".mov"}:
        return "video/quicktime"
    return None


def file_to_bytes(path: str, max_bytes: int) -> bytes:
    st = os.stat(path)
    if st.st_size > max_bytes:
        raise not_supported_error(
            f"file too large for inline bytes ({st.st_size} > {max_bytes}); use url/ref instead"
        )
    with open(path, "rb") as f:
        return f.read()


def bytes_to_base64(data: bytes) -> str:
    return base64.b64encode(data).decode("ascii")


def sniff_image_mime_type(data: bytes) -> str | None:
    if len(data) >= 8 and data.startswith(b"\x89PNG\r\n\x1a\n"):
        return "image/png"
    if len(data) >= 3 and data.startswith(b"\xff\xd8\xff"):
        return "image/jpeg"
    if len(data) >= 12 and data.startswith(b"RIFF") and data[8:12] == b"WEBP":
        return "image/webp"
    if len(data) >= 6 and data[:6] in {b"GIF87a", b"GIF89a"}:
        return "image/gif"
    return None
