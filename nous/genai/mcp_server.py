from __future__ import annotations

import argparse
import base64
import hashlib
import hmac
import json
import os
import time
from hmac import compare_digest
from collections import OrderedDict
from contextvars import ContextVar
from dataclasses import asdict, dataclass, replace
from typing import Any, TypedDict
from urllib.parse import parse_qs
from uuid import uuid4

from .client import Client
from .types import (
    GenerateRequest,
    GenerateResponse,
    OutputImageSpec,
    sniff_image_mime_type,
)

_MCP_GENERATE_REQUEST_SCHEMA: dict = {
    "type": "object",
    "title": "GenerateRequest",
    "description": 'GenAISDK request object. `model` must be "{provider}:{model_id}" (e.g. "openai:gpt-4o-mini").',
    "required": ["model", "input", "output"],
    "properties": {
        "model": {
            "type": "string",
            "pattern": r"^[^\s:]+:[^\s]+$",
            "description": 'Model string in the form "{provider}:{model_id}".',
            "examples": ["openai:gpt-4o-mini"],
        },
        "input": {
            "description": 'Chat messages (or a shorthand prompt string). Preferred: [{"role":"user","content":[{"type":"text","text":"..."}]}]. Shorthand also accepted: "..." or [{"role":"user","content":"..."}].',
            "examples": [
                [
                    {
                        "role": "user",
                        "content": [{"type": "text", "text": "Generate an image of a cute cat"}],
                    }
                ]
            ],
            "anyOf": [
                {"type": "string"},
                {
                    "type": "object",
                    "required": ["role", "content"],
                    "properties": {
                        "role": {"type": "string", "enum": ["system", "user", "assistant", "tool"]},
                        "content": {
                            "description": 'List of Part objects (preferred) or a shorthand string. Preferred: [{"type":"text","text":"..."}].',
                            "anyOf": [
                                {"type": "string"},
                                {"type": "object"},
                                {"type": "array", "items": {"type": "object"}},
                            ],
                        },
                    },
                },
                {
                    "type": "array",
                    "minItems": 1,
                    "items": {
                        "type": "object",
                        "required": ["role", "content"],
                        "properties": {
                            "role": {"type": "string", "enum": ["system", "user", "assistant", "tool"]},
                            "content": {
                                "description": 'List of Part objects (preferred) or a shorthand string. Preferred: [{"type":"text","text":"..."}].',
                                "anyOf": [
                                    {"type": "string"},
                                    {"type": "object"},
                                    {
                                        "type": "array",
                                        "minItems": 1,
                                        "examples": [[{"type": "text", "text": "Hello"}]],
                                        "items": {
                                            "type": "object",
                                            "required": ["type"],
                                            "properties": {
                                                "type": {"type": "string"},
                                                "mime_type": {"type": "string"},
                                                "source": {
                                                    "anyOf": [
                                                        {
                                                            "type": "object",
                                                            "required": ["kind", "encoding", "data"],
                                                            "properties": {
                                                                "kind": {"const": "bytes"},
                                                                "encoding": {"const": "base64"},
                                                                "data": {"type": "string"},
                                                            },
                                                        },
                                                        {
                                                            "type": "object",
                                                            "required": ["kind", "url"],
                                                            "properties": {
                                                                "kind": {"const": "url"},
                                                                "url": {"type": "string"},
                                                            },
                                                        },
                                                        {
                                                            "type": "object",
                                                            "required": ["kind", "provider", "id"],
                                                            "properties": {
                                                                "kind": {"const": "ref"},
                                                                "provider": {"type": "string"},
                                                                "id": {"type": "string"},
                                                            },
                                                        },
                                                        {"type": "null"},
                                                    ]
                                                },
                                                "text": {"type": "string"},
                                                "embedding": {"type": "array", "items": {"type": "number"}},
                                                "meta": {"type": "object"},
                                            },
                                        },
                                    },
                                ],
                            },
                        },
                    },
                },
            ],
        },
        "output": {
            "type": "object",
            "required": ["modalities"],
            "properties": {
                "modalities": {"type": "array", "items": {"type": "string"}},
                "image": {
                    "type": "object",
                    "properties": {
                        "format": {
                            "type": "string",
                            "description": 'For image-only output, omit to default to "url" in MCP.',
                        }
                    },
                },
            },
        },
        "params": {"type": "object"},
        "wait": {"type": "boolean"},
        "tools": {"anyOf": [{"type": "array", "items": {"type": "object"}}, {"type": "null"}]},
        "tool_choice": {"anyOf": [{"type": "object"}, {"type": "null"}]},
        "provider_options": {"type": "object"},
    },
}

class ProvidersInfo(TypedDict):
    supported: list[str]
    configured: list[str]


class ModelInfo(TypedDict):
    model: str
    modes: list[str]
    input_modalities: list[str]
    output_modalities: list[str]


class AvailableModelsInfo(TypedDict):
    models: list[ModelInfo]


class McpGenerateResponseBase(TypedDict):
    id: str
    provider: str
    model: str
    status: str
    output: list[dict[str, Any]]


class McpGenerateResponse(McpGenerateResponseBase, total=False):
    usage: dict[str, Any] | None
    job: dict[str, Any] | None
    error: dict[str, Any] | None


_REQUEST_TOKEN: ContextVar[str | None] = ContextVar("nous_genai_mcp_request_token", default=None)


@dataclass(frozen=True, slots=True)
class McpTokenScope:
    providers_all: frozenset[str]
    models: dict[str, frozenset[str]]
    providers: frozenset[str]


_DENY_ALL_SCOPE = McpTokenScope(providers_all=frozenset(), models={}, providers=frozenset())


def _parse_mcp_token_scopes(raw: str) -> dict[str, McpTokenScope]:
    raw = raw.strip()
    if not raw:
        return {}

    # Accept a JSON dict: {"token": ["openai", "openai:gpt-4o-mini"], ...}
    if raw[0] == "{":
        try:
            parsed = json.loads(raw)
        except json.JSONDecodeError:
            parsed = None
        if isinstance(parsed, dict):
            items: list[str] = []
            for token, allow in parsed.items():
                if not isinstance(token, str) or not token.strip():
                    raise ValueError("invalid NOUS_GENAI_MCP_TOKEN_RULES: token must be a non-empty string")
                if allow is None:
                    items.append(f"{token.strip()}: []")
                    continue
                if not isinstance(allow, list) or not all(isinstance(x, str) for x in allow):
                    raise ValueError(
                        "invalid NOUS_GENAI_MCP_TOKEN_RULES: each token value must be a list of strings"
                    )
                joined = " ".join(a.strip() for a in allow if a.strip())
                items.append(f"{token.strip()}: [{joined}]")
            raw = ";".join(items)

    # Bracket syntax: token: [provider provider:model_id ...]; token2: [...]
    text = raw.replace("\\n", "\n")
    entries: list[str] = []
    for line in text.splitlines():
        for part in line.split(";"):
            stripped = part.strip()
            if stripped:
                entries.append(stripped)

    scopes: dict[str, McpTokenScope] = {}
    for entry in entries:
        if entry.startswith("#"):
            continue
        if ":" not in entry:
            raise ValueError(f"invalid NOUS_GENAI_MCP_TOKEN_RULES entry (missing ':'): {entry}")
        token, spec = entry.split(":", 1)
        token = token.strip()
        spec = spec.strip()
        if not token:
            raise ValueError(f"invalid NOUS_GENAI_MCP_TOKEN_RULES entry (empty token): {entry}")
        if token in scopes:
            raise ValueError(f"invalid NOUS_GENAI_MCP_TOKEN_RULES (duplicate token): {token}")
        if not (spec.startswith("[") and spec.endswith("]")):
            raise ValueError(f"invalid NOUS_GENAI_MCP_TOKEN_RULES entry (expected '[...]'): {entry}")
        inner = spec[1:-1].strip()
        providers_all: set[str] = set()
        models: dict[str, set[str]] = {}
        for item in inner.replace(",", " ").split():
            if not item:
                continue
            if ":" in item:
                provider, model_id = item.split(":", 1)
                provider = provider.strip().lower()
                model_id = model_id.strip()
                if not provider or not model_id:
                    raise ValueError(f"invalid NOUS_GENAI_MCP_TOKEN_RULES entry item: {item}")
                if model_id == "*":
                    providers_all.add(provider)
                    continue
                models.setdefault(provider, set()).add(model_id)
                continue
            providers_all.add(item.strip().lower())

        providers_all_fs = frozenset(providers_all)
        models_fs = {p: frozenset(ids) for p, ids in models.items() if ids}
        providers_fs = providers_all_fs | frozenset(models_fs.keys())
        scopes[token] = McpTokenScope(providers_all=providers_all_fs, models=models_fs, providers=providers_fs)

    return scopes


def _env_int(name: str, default: int) -> int:
    raw = os.environ.get(name)
    if raw is None:
        return default
    try:
        value = int(raw)
    except ValueError:
        return default
    return value


def _get_host_port() -> tuple[str, int]:
    host = os.environ.get("NOUS_GENAI_MCP_HOST", "").strip() or "127.0.0.1"
    port = _env_int("NOUS_GENAI_MCP_PORT", 6001)
    if port < 1:
        port = 1
    if port > 65535:
        port = 65535
    return host, port


def build_server(
    *,
    proxy_url: str | None = None,
    host: str | None = None,
    port: int | None = None,
    model_keywords: list[str] | None = None,
    bearer_token: str | None = None,
    token_scopes: dict[str, McpTokenScope] | None = None,
):
    """
    Build a FastMCP server that exposes:
    - generate: GenAISDK Client.generate wrapper (MCP-friendly defaults)
    - list_providers: discover providers configured on this server
    - list_available_models: list available models for a provider (fully-qualified)
    - list_all_available_models: list available models across all providers (fully-qualified)

    Notes for LLM tool callers:
    - Model must be "{provider}:{model_id}".
    - Call `list_providers` first if you don't know which providers are usable.
    """
    try:
        from mcp.server.fastmcp import FastMCP
    except ModuleNotFoundError as e:  # pragma: no cover
        raise SystemExit("missing dependency: install `mcp` to run the MCP server (e.g. `uv sync`)") from e

    try:
        from pydantic import Field, WithJsonSchema
    except ModuleNotFoundError as e:  # pragma: no cover
        raise SystemExit("missing dependency: install `mcp` to run the MCP server (e.g. `uv sync`)") from e
    from typing import Annotated
    from starlette.requests import Request
    from starlette.responses import Response

    os.environ["NOUS_GENAI_TRANSPORT"] = "mcp"
    if host is None or port is None:
        host, port = _get_host_port()
    server = FastMCP(name="GenAISDK", host=host, port=port)

    keywords: list[str] = []
    for raw in model_keywords or []:
        if not isinstance(raw, str):
            continue
        for part in raw.split(","):
            keyword = part.strip().lower()
            if keyword:
                keywords.append(keyword)

    def _model_allowed(model: str) -> bool:
        if not keywords:
            return True
        candidate = model.strip().lower()
        return any(k in candidate for k in keywords)

    max_artifacts = _env_int("NOUS_GENAI_MAX_ARTIFACTS", 64)
    if max_artifacts < 1:
        max_artifacts = 1
    max_artifact_bytes = _env_int("NOUS_GENAI_MAX_ARTIFACT_BYTES", 64 * 1024 * 1024)
    if max_artifact_bytes < 0:
        max_artifact_bytes = 0
    artifact_url_ttl_seconds = _env_int("NOUS_GENAI_ARTIFACT_URL_TTL_SECONDS", 600)
    if artifact_url_ttl_seconds < 1:
        artifact_url_ttl_seconds = 1

    def _public_base_url() -> str:
        base = os.environ.get("NOUS_GENAI_MCP_PUBLIC_BASE_URL", "").strip()
        if base:
            return base.rstrip("/")
        if host in {"0.0.0.0", "::"}:
            return f"http://127.0.0.1:{port}"
        return f"http://{host}:{port}"

    base_url = _public_base_url()
    signing_token = bearer_token.strip() if isinstance(bearer_token, str) and bearer_token.strip() else None
    token_scopes = token_scopes or {}

    def _b64url(data: bytes) -> str:
        return base64.urlsafe_b64encode(data).decode("ascii").rstrip("=")

    def _signing_key() -> str | None:
        current = _REQUEST_TOKEN.get()
        if isinstance(current, str) and current:
            return current
        return signing_token

    def _artifact_sig(key: str, artifact_id: str, exp: int) -> str:
        msg = f"{artifact_id}.{exp}".encode("utf-8", errors="strict")
        digest = hmac.new(key.encode("utf-8", errors="strict"), msg, hashlib.sha256).digest()
        return _b64url(digest)

    def _artifact_url(artifact_id: str) -> str:
        url = f"{base_url}/artifact/{artifact_id}"
        key = _signing_key()
        if key is None:
            return url
        exp = int(time.time()) + artifact_url_ttl_seconds
        sig = _artifact_sig(key, artifact_id, exp)
        return f"{url}?exp={exp}&sig={sig}"

    @dataclass(slots=True)
    class _ArtifactItem:
        data: bytes
        mime_type: str | None
        owner_token: str | None

    artifacts: OrderedDict[str, _ArtifactItem] = OrderedDict()
    artifacts_total_bytes = 0

    def _evict_one() -> None:
        nonlocal artifacts_total_bytes
        _, item = artifacts.popitem(last=False)
        artifacts_total_bytes -= len(item.data)

    def _enforce_artifact_limits() -> None:
        while len(artifacts) > max_artifacts:
            _evict_one()
        if max_artifact_bytes <= 0:
            while artifacts:
                _evict_one()
            return
        while artifacts and artifacts_total_bytes > max_artifact_bytes:
            _evict_one()

    def _artifact_owner() -> str | None:
        token = _REQUEST_TOKEN.get()
        if isinstance(token, str) and token:
            return token
        return signing_token

    def _store_artifact(data: bytes, mime_type: str | None) -> str | None:
        nonlocal artifacts_total_bytes
        if max_artifact_bytes <= 0:
            return None
        if len(data) > max_artifact_bytes:
            return None
        artifact_id = uuid4().hex
        artifacts[artifact_id] = _ArtifactItem(data=data, mime_type=mime_type, owner_token=_artifact_owner())
        artifacts_total_bytes += len(data)
        artifacts.move_to_end(artifact_id)
        _enforce_artifact_limits()
        return artifact_id if artifact_id in artifacts else None

    class _McpArtifactStore:
        def put(self, data: bytes, mime_type: str | None) -> str | None:
            return _store_artifact(data, mime_type)

        def url(self, artifact_id: str) -> str:
            return _artifact_url(artifact_id)

    client = Client(
        proxy_url=proxy_url,
        artifact_store=_McpArtifactStore(),
    )

    @server.custom_route("/artifact/{artifact_id}", methods=["GET", "HEAD"], include_in_schema=False)
    async def artifact_route(request: Request) -> Response:
        artifact_id = request.path_params.get("artifact_id")
        if not isinstance(artifact_id, str) or not artifact_id:
            return Response(status_code=404)
        item = artifacts.get(artifact_id)
        if item is None:
            return Response(status_code=404)
        owner = item.owner_token
        if owner is not None:
            token = _REQUEST_TOKEN.get()
            if token is None or not compare_digest(token, owner):
                return Response(status_code=404)
        artifacts.move_to_end(artifact_id)
        if item.mime_type is None:
            guessed = sniff_image_mime_type(item.data)
            if guessed is not None:
                item.mime_type = guessed
        headers = {"Content-Length": str(len(item.data))}
        if request.method.upper() == "HEAD":
            return Response(
                content=b"",
                media_type=item.mime_type or "application/octet-stream",
                headers=headers,
            )
        return Response(
            content=item.data,
            media_type=item.mime_type or "application/octet-stream",
            headers=headers,
        )

    @server.resource("genaisdk://artifact/{artifact_id}", mime_type="application/json")
    def read_artifact(artifact_id: str) -> dict[str, Any]:
        item = artifacts.get(artifact_id)
        if item is None:
            raise ValueError("artifact not found")
        owner = item.owner_token
        if owner is not None:
            token = _REQUEST_TOKEN.get()
            if token is None or not compare_digest(token, owner):
                raise ValueError("artifact not found")
        artifacts.move_to_end(artifact_id)
        if item.mime_type is None:
            item.mime_type = sniff_image_mime_type(item.data)
        return {
            "id": artifact_id,
            "mime_type": item.mime_type,
            "bytes": len(item.data),
            "url": _artifact_url(artifact_id),
        }

    def _scope() -> McpTokenScope | None:
        if not token_scopes:
            return None
        token = _REQUEST_TOKEN.get()
        if not token:
            return None
        return token_scopes.get(token, _DENY_ALL_SCOPE)

    def _provider_allowed(provider: str) -> bool:
        scope = _scope()
        if scope is None:
            return True
        return provider in scope.providers

    def _model_allowed_by_scope(model: str) -> bool:
        scope = _scope()
        if scope is None:
            return True
        if ":" not in model:
            return False
        provider, model_id = model.split(":", 1)
        provider = provider.strip().lower()
        model_id = model_id.strip()
        if not provider or not model_id:
            return False
        if provider in scope.providers_all:
            return True
        allowed = scope.models.get(provider)
        if allowed is None:
            return False
        return model_id in allowed

    def list_providers() -> ProvidersInfo:
        """
        List providers supported by this MCP server.

        Returns:
        - supported: providers known by the SDK catalog
        - configured: providers that have credentials configured and can be used
        """
        from .reference import get_model_catalog
        from ._internal.errors import GenAIError

        supported = sorted(get_model_catalog().keys())
        if token_scopes:
            supported = [p for p in supported if _provider_allowed(p)]
        configured: list[str] = []
        for p in supported:
            try:
                client._adapter(p)  # type: ignore[attr-defined]
            except GenAIError:
                continue
            configured.append(p)
        return {"supported": supported, "configured": configured}

    def list_available_models(provider: str, *, timeout_ms: int | None = None) -> AvailableModelsInfo:
        """
        List available models (sdk catalog ∩ remotely available) for a provider.

        Always returns fully-qualified model strings like "openai:gpt-4o-mini".

        Returns:
        - models: list[object], each includes:
          - model: "{provider}:{model_id}"
          - modes: ["sync","stream","job","async"] (varies by provider/model)
          - input_modalities: ["text","image",...]
          - output_modalities: ["text","image",...]
        """
        from .client import _normalize_provider
        from .reference import get_sdk_supported_models_for_provider

        p = _normalize_provider(provider)
        if not _provider_allowed(p):
            raise ValueError(f"provider not allowed: {p}")
        ids = client.list_available_models(p, timeout_ms=timeout_ms)
        rows = get_sdk_supported_models_for_provider(p)
        by_model_id = {r["model_id"]: r for r in rows}

        models: list[ModelInfo] = []
        for model_id in ids:
            row = by_model_id.get(model_id)
            if row is None:
                continue
            model = row["model"]
            if not _model_allowed(model) or not _model_allowed_by_scope(model):
                continue
            models.append(
                {
                    "model": model,
                    "modes": row["modes"],
                    "input_modalities": row["input_modalities"],
                    "output_modalities": row["output_modalities"],
                }
            )
        return {"models": models}

    def list_all_available_models(*, timeout_ms: int | None = None) -> AvailableModelsInfo:
        """
        List available models (sdk catalog ∩ remotely available) across all providers.

        Always returns fully-qualified model strings like "openai:gpt-4o-mini".

        Returns:
        - models: list[object], each includes:
          - model: "{provider}:{model_id}"
          - modes: ["sync","stream","job","async"] (varies by provider/model)
          - input_modalities: ["text","image",...]
          - output_modalities: ["text","image",...]
        """
        scope = _scope()
        if scope is None:
            from .reference import get_sdk_supported_models

            rows = get_sdk_supported_models()
            by_model = {r["model"]: r for r in rows}

            models: list[ModelInfo] = []
            for model in client.list_all_available_models(timeout_ms=timeout_ms):
                if not _model_allowed(model) or not _model_allowed_by_scope(model):
                    continue
                row = by_model.get(model)
                if row is None:
                    continue
                models.append(
                    {
                        "model": model,
                        "modes": row["modes"],
                        "input_modalities": row["input_modalities"],
                        "output_modalities": row["output_modalities"],
                    }
                )
            return {"models": models}

        models: list[ModelInfo] = []
        for provider in sorted(scope.providers):
            models.extend(list_available_models(provider, timeout_ms=timeout_ms)["models"])
        models.sort(key=lambda row: row["model"])
        return {"models": models}

    def generate(request: dict[str, Any], *, stream: bool = False) -> McpGenerateResponse:
        """
        MCP-friendly wrapper of `Client.generate`.

        Behavior:
        - Enforces non-stream tool behavior (MCP tool call returns one result).
        - For image-only output, defaults to URL output when format is unspecified.
        - Large base64 binary outputs are stored on this server and returned as a URL.

        Notes:
        - `request.model` must be "{provider}:{model_id}" (e.g. "openai:gpt-4o-mini").
        - `request.input` preferred format: [{"role":"user","content":[{"type":"text","text":"..."}]}].
          Shorthand accepted: `request.input="..."` or [{"role":"user","content":"..."}] (auto-coerced to a text Part).
        - Call `list_providers` / `list_available_models` / `list_all_available_models` first when in doubt.

        Returns:
        - id/provider/model/status/output (+ optional usage/job/error)
        """
        if stream:
            raise ValueError("stream=true is not supported for MCP tool calls; use stream=false")

        from pydantic import TypeAdapter, ValidationError

        if not isinstance(request, dict):
            raise ValueError("request must be an object")

        req_dict: dict[str, Any] = dict(request)
        msgs = req_dict.get("input")
        if isinstance(msgs, str):
            msgs = [{"role": "user", "content": msgs}]
        elif isinstance(msgs, dict):
            msgs = [msgs]
        if isinstance(msgs, list):
            coerced_msgs: list[Any] = []
            for msg in msgs:
                if isinstance(msg, str):
                    coerced_msgs.append({"role": "user", "content": msg})
                    continue
                if not isinstance(msg, dict):
                    coerced_msgs.append(msg)
                    continue
                m = dict(msg)
                if not isinstance(m.get("role"), str):
                    m["role"] = "user"
                content = m.get("content")
                if isinstance(content, str):
                    m["content"] = [{"type": "text", "text": content}]
                elif isinstance(content, dict):
                    part = dict(content)
                    if "type" not in part and isinstance(part.get("text"), str):
                        part["type"] = "text"
                    m["content"] = [part]
                elif isinstance(content, list):
                    parts: list[Any] = []
                    for part in content:
                        if isinstance(part, str):
                            parts.append({"type": "text", "text": part})
                        elif isinstance(part, dict) and "type" not in part and isinstance(part.get("text"), str):
                            p = dict(part)
                            p["type"] = "text"
                            parts.append(p)
                        else:
                            parts.append(part)
                    m["content"] = parts
                coerced_msgs.append(m)
            req_dict["input"] = coerced_msgs

        out_spec = req_dict.get("output")
        if isinstance(out_spec, str):
            req_dict["output"] = {"modalities": [out_spec]}
        elif isinstance(out_spec, dict) and isinstance(out_spec.get("modalities"), str):
            o = dict(out_spec)
            o["modalities"] = [out_spec["modalities"]]
            req_dict["output"] = o

        try:
            req = TypeAdapter(GenerateRequest).validate_python(req_dict)
        except ValidationError as e:
            raise ValueError(str(e)) from e

        if not _model_allowed(req.model):
            raise ValueError(f"model not allowed by server filter: {req.model}")
        if not _model_allowed_by_scope(req.model):
            raise ValueError(f"model not allowed: {req.model}")

        if set(req.output.modalities) == {"image"}:
            img = req.output.image or OutputImageSpec()
            if not img.format:
                img = replace(img, format="url")
                req = replace(req, output=replace(req.output, image=img))

        resp = client.generate(req, stream=False)
        if not isinstance(resp, GenerateResponse):
            raise ValueError("provider returned stream response; expected non-stream response")
        return asdict(resp)

    generate.__annotations__["request"] = Annotated[dict[str, Any], WithJsonSchema(_MCP_GENERATE_REQUEST_SCHEMA)]
    list_available_models.__annotations__["provider"] = Annotated[
        str,
        Field(
            description='Provider name (e.g. "openai"). Call `list_providers` first if unknown.',
            examples=["openai"],
        ),
    ]
    list_available_models.__annotations__["timeout_ms"] = Annotated[
        int | None,
        Field(
            default=None,
            description="Remote request timeout in milliseconds.",
        ),
    ]
    list_all_available_models.__annotations__["timeout_ms"] = Annotated[
        int | None,
        Field(
            default=None,
            description="Remote request timeout in milliseconds.",
        ),
    ]

    server.tool(structured_output=True)(generate)
    server.tool(structured_output=True)(list_providers)
    server.tool(structured_output=True)(list_available_models)
    server.tool(structured_output=True)(list_all_available_models)
    return server


def build_http_app(server: Any) -> Any:
    from starlette.routing import Mount, Route

    app = server.streamable_http_app()
    sse = server.sse_app()
    sse_path = server.settings.sse_path
    message_path = server.settings.message_path.rstrip("/")

    for route in sse.router.routes:
        if isinstance(route, Route) and route.path == sse_path:
            app.router.routes.append(route)
            continue
        if isinstance(route, Mount) and route.path.rstrip("/") == message_path:
            app.router.routes.append(route)
            continue
    return app


def main(argv: list[str] | None = None) -> None:
    from ._internal.config import load_env_files

    load_env_files()

    parser = argparse.ArgumentParser(
        prog="genai-mcp-server",
        description="nous-genai-sdk MCP server (Streamable HTTP: /mcp, SSE: /sse)",
    )
    parser.add_argument(
        "--proxy",
        dest="proxy_url",
        help="HTTP proxy URL for provider requests (e.g. http://127.0.0.1:7890)",
    )
    parser.add_argument(
        "--bearer-token",
        dest="bearer_token",
        help="Require HTTP Authorization: Bearer <token> for all endpoints (or set NOUS_GENAI_MCP_BEARER_TOKEN).",
    )
    parser.add_argument(
        "--model-keyword",
        dest="model_keywords",
        action="append",
        help='Only expose models whose "{provider}:{model_id}" contains this substring (case-insensitive). Repeatable; comma-separated also accepted.',
    )
    args = parser.parse_args(argv)
    bearer = (args.bearer_token or os.environ.get("NOUS_GENAI_MCP_BEARER_TOKEN") or "").strip()
    token_rules = (os.environ.get("NOUS_GENAI_MCP_TOKEN_RULES") or "").strip()
    if token_rules and bearer:
        raise SystemExit("set either NOUS_GENAI_MCP_BEARER_TOKEN/--bearer-token or NOUS_GENAI_MCP_TOKEN_RULES, not both")
    token_scopes: dict[str, McpTokenScope] = {}
    if token_rules:
        try:
            token_scopes = _parse_mcp_token_scopes(token_rules)
        except ValueError as e:
            raise SystemExit(str(e)) from e
        if not token_scopes:
            raise SystemExit("invalid NOUS_GENAI_MCP_TOKEN_RULES: no tokens configured")

    server_host, server_port = _get_host_port()
    server = build_server(
        proxy_url=args.proxy_url,
        host=server_host,
        port=server_port,
        model_keywords=args.model_keywords,
        bearer_token=bearer,
        token_scopes=token_scopes,
    )
    app = build_http_app(server)
    if token_scopes:
        app.add_middleware(_BearerAuthMiddleware, tokens=token_scopes)
    elif bearer:
        app.add_middleware(_BearerAuthMiddleware, token=bearer)

    try:
        import uvicorn
    except ModuleNotFoundError as e:  # pragma: no cover
        raise SystemExit("missing dependency: install `uvicorn` to run the MCP server (e.g. `uv sync`)") from e

    uvicorn.run(app, host=server_host, port=server_port, log_level=server.settings.log_level.lower())


class _BearerAuthMiddleware:
    def __init__(
        self,
        app: Any,
        *,
        token: str | None = None,
        tokens: dict[str, McpTokenScope] | None = None,
    ) -> None:
        self.app = app
        token = token.strip() if isinstance(token, str) else ""
        tokens = tokens or {}
        if token and tokens:
            raise ValueError("pass either token=... or tokens=..., not both")
        if not token and not tokens:
            raise ValueError("missing auth config: pass token=... or tokens=...")
        self._tokens = [token] if token else list(tokens.keys())

    def _match_bearer(self, scope: Any) -> str | None:
        raw = None
        for k, v in scope.get("headers") or []:
            if k.lower() == b"authorization":
                raw = v
                break
        if not raw:
            return None

        try:
            header = raw.decode("utf-8", errors="replace").strip()
        except Exception:
            return None

        if not header.lower().startswith("bearer "):
            return None

        token = header[7:].strip()
        for candidate in self._tokens:
            if compare_digest(token, candidate):
                return candidate
        return None

    def _match_artifact_sig(self, scope: Any) -> str | None:
        if not self._tokens:
            return None

        path = scope.get("path")
        if not isinstance(path, str) or not path.startswith("/artifact/"):
            return None

        method = (scope.get("method") or "").upper()
        if method not in {"GET", "HEAD"}:
            return None

        artifact_id = path[len("/artifact/") :].strip("/")
        if not artifact_id:
            return None

        qs = scope.get("query_string") or b""
        try:
            query = parse_qs(qs.decode("utf-8", errors="replace"), keep_blank_values=True)
        except Exception:
            return None

        exp_raw = (query.get("exp") or [None])[0]
        sig_raw = (query.get("sig") or [None])[0]
        if not isinstance(exp_raw, str) or not exp_raw.strip() or not isinstance(sig_raw, str) or not sig_raw.strip():
            return None

        try:
            exp = int(exp_raw.strip())
        except ValueError:
            return None

        if int(time.time()) > exp:
            return None

        msg = f"{artifact_id}.{exp}".encode("utf-8", errors="strict")
        sig = sig_raw.strip()
        for token in self._tokens:
            digest = hmac.new(token.encode("utf-8", errors="strict"), msg, hashlib.sha256).digest()
            expected = base64.urlsafe_b64encode(digest).decode("ascii").rstrip("=")
            if compare_digest(sig, expected):
                return token
        return None

    async def __call__(self, scope: Any, receive: Any, send: Any) -> None:
        if scope.get("type") != "http":
            await self.app(scope, receive, send)
            return

        token = self._match_bearer(scope) or self._match_artifact_sig(scope)
        if token is None:
            await _send_unauthorized(send)
            return

        ctx = _REQUEST_TOKEN.set(token)
        try:
            await self.app(scope, receive, send)
        finally:
            _REQUEST_TOKEN.reset(ctx)


async def _send_unauthorized(send: Any) -> None:
    body = b'{"error":"invalid_token","error_description":"Authentication required"}'
    await send(
        {
            "type": "http.response.start",
            "status": 401,
            "headers": [
                (b"content-type", b"application/json"),
                (b"content-length", str(len(body)).encode()),
                (b"www-authenticate", b'Bearer error="invalid_token", error_description="Authentication required"'),
            ],
        }
    )
    await send({"type": "http.response.body", "body": body})


if __name__ == "__main__":  # pragma: no cover
    main()
