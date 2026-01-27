from __future__ import annotations

import asyncio
import base64
import os
import urllib.parse
from dataclasses import replace
from typing import Iterator, Protocol

from ._internal.config import get_default_timeout_ms, get_provider_keys, load_env_files
from ._internal.errors import GenAIError, invalid_request_error, not_supported_error
from ._internal.http import download_to_file as _download_to_file
from ._internal.http import download_to_tempfile
from ._internal.json_schema import normalize_json_schema, reject_gemini_response_schema_dict
from .types import (
    Capability,
    GenerateEvent,
    GenerateRequest,
    GenerateResponse,
    Message,
    Part,
    PartSourceBytes,
    PartSourceUrl,
    sniff_image_mime_type,
)
from .providers import (
    AliyunAdapter,
    AnthropicAdapter,
    GeminiAdapter,
    OpenAIAdapter,
    TuziAdapter,
    VolcengineAdapter,
)


class _ArtifactStore(Protocol):
    def put(self, data: bytes, mime_type: str | None) -> str | None: ...
    def url(self, artifact_id: str) -> str: ...


class Client:
    def __init__(
        self,
        *,
        proxy_url: str | None = None,
        artifact_store: _ArtifactStore | None = None,
    ) -> None:
        load_env_files()
        self._transport = os.environ.get("NOUS_GENAI_TRANSPORT", "").strip().lower()
        self._proxy_url = proxy_url.strip() if isinstance(proxy_url, str) and proxy_url.strip() else None
        self._artifact_store = artifact_store
        keys = get_provider_keys()
        self._openai = (
            OpenAIAdapter(api_key=keys.openai_api_key, proxy_url=self._proxy_url)
            if keys.openai_api_key
            else None
        )
        self._gemini = (
            GeminiAdapter(api_key=keys.google_api_key, proxy_url=self._proxy_url) if keys.google_api_key else None
        )
        self._anthropic = (
            AnthropicAdapter(api_key=keys.anthropic_api_key, proxy_url=self._proxy_url)
            if keys.anthropic_api_key
            else None
        )
        self._aliyun = None
        if keys.aliyun_api_key:
            self._aliyun = AliyunAdapter(
                openai=OpenAIAdapter(
                    api_key=keys.aliyun_api_key,
                    base_url=os.environ.get(
                        "ALIYUN_OAI_BASE_URL", "https://dashscope.aliyuncs.com/compatible-mode/v1"
                    ).rstrip("/"),
                    provider_name="aliyun",
                    chat_api="chat_completions",
                    proxy_url=self._proxy_url,
                )
            )
        self._volcengine = None
        if keys.volcengine_api_key:
            self._volcengine = VolcengineAdapter(
                openai=OpenAIAdapter(
                    api_key=keys.volcengine_api_key,
                    base_url=os.environ.get(
                        "VOLCENGINE_OAI_BASE_URL", "https://ark.cn-beijing.volces.com/api/v3"
                    ).rstrip("/"),
                    provider_name="volcengine",
                    chat_api="chat_completions",
                    proxy_url=self._proxy_url,
                )
            )
        base_host = os.environ.get("TUZI_BASE_URL", "https://api.tu-zi.com").rstrip("/")
        self._tuzi_web = None
        if keys.tuzi_web_api_key:
            self._tuzi_web = TuziAdapter(
                openai=OpenAIAdapter(
                    api_key=keys.tuzi_web_api_key,
                    base_url=os.environ.get("TUZI_OAI_BASE_URL", f"{base_host}/v1").rstrip("/"),
                    provider_name="tuzi-web",
                    chat_api="chat_completions",
                    proxy_url=self._proxy_url,
                ),
                gemini=GeminiAdapter(
                    api_key=keys.tuzi_web_api_key,
                    base_url=os.environ.get("TUZI_GOOGLE_BASE_URL", base_host).rstrip("/"),
                    provider_name="tuzi-web",
                    auth_mode="bearer",
                    supports_file_upload=False,
                    proxy_url=self._proxy_url,
                ),
                anthropic=AnthropicAdapter(
                    api_key=keys.tuzi_web_api_key,
                    base_url=os.environ.get("TUZI_ANTHROPIC_BASE_URL", base_host).rstrip("/"),
                    provider_name="tuzi-web",
                    auth_mode="bearer",
                    proxy_url=self._proxy_url,
                ),
                proxy_url=self._proxy_url,
            )
        self._tuzi_openai = None
        if keys.tuzi_openai_api_key:
            self._tuzi_openai = OpenAIAdapter(
                api_key=keys.tuzi_openai_api_key,
                base_url=os.environ.get("TUZI_OAI_BASE_URL", f"{base_host}/v1").rstrip("/"),
                provider_name="tuzi-openai",
                chat_api="chat_completions",
                proxy_url=self._proxy_url,
            )
        self._tuzi_google = None
        if keys.tuzi_google_api_key:
            self._tuzi_google = GeminiAdapter(
                api_key=keys.tuzi_google_api_key,
                base_url=os.environ.get("TUZI_GOOGLE_BASE_URL", base_host).rstrip("/"),
                provider_name="tuzi-google",
                auth_mode="bearer",
                supports_file_upload=False,
                proxy_url=self._proxy_url,
            )
        self._tuzi_anthropic = None
        if keys.tuzi_anthropic_api_key:
            self._tuzi_anthropic = AnthropicAdapter(
                api_key=keys.tuzi_anthropic_api_key,
                base_url=os.environ.get("TUZI_ANTHROPIC_BASE_URL", base_host).rstrip("/"),
                provider_name="tuzi-anthropic",
                auth_mode="bearer",
                proxy_url=self._proxy_url,
            )
        self._default_timeout_ms = get_default_timeout_ms()

    def capabilities(self, model: str) -> Capability:
        provider, model_id = _split_model(model)
        adapter = self._adapter(provider)
        return adapter.capabilities(model_id)

    def list_provider_models(self, provider: str, *, timeout_ms: int | None = None) -> list[str]:
        provider = _normalize_provider(provider)
        try:
            adapter = self._adapter(provider)
        except GenAIError:
            return []
        fn = getattr(adapter, "list_models", None)
        if not callable(fn):
            return []
        try:
            models = fn(timeout_ms=timeout_ms)
        except GenAIError:
            return []
        return [m for m in models if isinstance(m, str) and m]

    def list_available_models(self, provider: str, *, timeout_ms: int | None = None) -> list[str]:
        """
        List models that are both:
        - included in the SDK curated catalog, and
        - remotely available for the current credentials.
        """
        from .reference import get_model_catalog

        p = _normalize_provider(provider)
        supported = {m for m in get_model_catalog().get(p, []) if isinstance(m, str) and m}
        if not supported:
            return []
        remote = set(self.list_provider_models(p, timeout_ms=timeout_ms))
        if not remote:
            return []
        return sorted(supported & remote)

    def list_all_available_models(self, *, timeout_ms: int | None = None) -> list[str]:
        """
        List available models across all SDK-supported providers.

        Returns fully-qualified model strings like "openai:gpt-4o-mini".
        """
        from .reference import get_supported_providers

        out: list[str] = []
        for provider in sorted(get_supported_providers()):
            p = _normalize_provider(provider)
            for model_id in self.list_available_models(p, timeout_ms=timeout_ms):
                out.append(f"{p}:{model_id}")
        return out

    def list_unsupported_models(self, provider: str, *, timeout_ms: int | None = None) -> list[str]:
        """
        List remotely available models that are not in the SDK curated catalog.
        """
        from .reference import get_model_catalog

        p = _normalize_provider(provider)
        supported = {m for m in get_model_catalog().get(p, []) if isinstance(m, str) and m}
        remote = set(self.list_provider_models(p, timeout_ms=timeout_ms))
        if not remote:
            return []
        return sorted(remote - supported)

    def list_stale_models(self, provider: str, *, timeout_ms: int | None = None) -> list[str]:
        """
        List models that are in the SDK curated catalog, but not remotely available for the current credentials.
        """
        from .reference import get_model_catalog

        p = _normalize_provider(provider)
        supported = {m for m in get_model_catalog().get(p, []) if isinstance(m, str) and m}
        if not supported:
            return []
        remote = set(self.list_provider_models(p, timeout_ms=timeout_ms))
        if not remote:
            return []
        return sorted(supported - remote)

    def generate(
        self,
        request: GenerateRequest,
        *,
        stream: bool = False,
    ) -> GenerateResponse | Iterator[GenerateEvent]:
        if _is_mcp_transport_marker(self._transport):
            _validate_mcp_wire_request(request)
        provider = _normalize_provider(request.provider())
        adapter = self._adapter(provider)
        if request.params.timeout_ms is None:
            request = replace(request, params=replace(request.params, timeout_ms=self._default_timeout_ms))
        request = _normalize_output_text_json_schema(request)
        cap = adapter.capabilities(request.model_id())
        in_modalities: set[str] = set()
        for msg in request.input:
            for part in msg.content:
                if part.type in {"text", "image", "audio", "video", "embedding"}:
                    in_modalities.add(part.type)
                elif part.type == "file":
                    raise not_supported_error("file parts are not supported in request.input; use image/audio/video parts")
        if not in_modalities.issubset(cap.input_modalities):
            raise not_supported_error(
                f"requested input modalities not supported: {sorted(in_modalities)} (supported: {sorted(cap.input_modalities)})"
            )
        out_modalities = set(request.output.modalities)
        if not out_modalities:
            raise invalid_request_error("output.modalities must not be empty")
        if not out_modalities.issubset(cap.output_modalities):
            raise not_supported_error(
                f"requested output modalities not supported: {sorted(out_modalities)} (supported: {sorted(cap.output_modalities)})"
            )
        if stream and not cap.supports_stream:
            raise not_supported_error("streaming is not supported for this model")
        out = adapter.generate(request, stream=stream)
        if isinstance(out, GenerateResponse):
            out = self._externalize_large_base64_parts(out)
            return self._externalize_protected_url_parts(out, adapter=adapter, timeout_ms=request.params.timeout_ms)
        return out

    def _externalize_large_base64_parts(self, resp: GenerateResponse) -> GenerateResponse:
        store = self._artifact_store
        if store is None:
            return resp

        max_inline_b64_chars = _env_int("NOUS_GENAI_MAX_INLINE_BASE64_CHARS", 4096)
        if max_inline_b64_chars < 0:
            max_inline_b64_chars = 0
        if max_inline_b64_chars == 0:
            threshold = 1
        else:
            threshold = max_inline_b64_chars

        max_artifact_bytes = _env_int("NOUS_GENAI_MAX_ARTIFACT_BYTES", 64 * 1024 * 1024)
        if max_artifact_bytes < 0:
            max_artifact_bytes = 0
        if max_artifact_bytes == 0:
            return resp

        out_msgs: list[Message] = []
        changed = False
        for msg in resp.output:
            out_parts: list[Part] = []
            for part in msg.content:
                src = part.source
                if not isinstance(src, PartSourceBytes) or src.encoding != "base64":
                    out_parts.append(part)
                    continue
                data_b64 = src.data
                if not isinstance(data_b64, str) or len(data_b64) < threshold:
                    out_parts.append(part)
                    continue

                estimated_bytes = (len(data_b64) * 3) // 4
                if estimated_bytes > max_artifact_bytes:
                    out_parts.append(part)
                    continue

                try:
                    data = base64.b64decode(data_b64)
                except Exception:
                    out_parts.append(part)
                    continue

                mime_type = part.mime_type.strip() if isinstance(part.mime_type, str) and part.mime_type.strip() else None
                if mime_type is None and part.type == "image":
                    mime_type = sniff_image_mime_type(data)

                artifact_id = store.put(data, mime_type)
                if artifact_id is None:
                    out_parts.append(part)
                    continue

                out_parts.append(
                    Part(
                        type=part.type,
                        mime_type=mime_type,
                        source=PartSourceUrl(url=store.url(artifact_id)),
                        text=part.text,
                        embedding=part.embedding,
                        meta=part.meta,
                    )
                )
                changed = True
            out_msgs.append(Message(role=msg.role, content=out_parts))

        if not changed:
            return resp
        return replace(resp, output=out_msgs)

    def _externalize_protected_url_parts(
        self,
        resp: GenerateResponse,
        *,
        adapter: object,
        timeout_ms: int | None,
    ) -> GenerateResponse:
        store = self._artifact_store
        if store is None:
            return resp

        max_artifact_bytes = _env_int("NOUS_GENAI_MAX_ARTIFACT_BYTES", 64 * 1024 * 1024)
        if max_artifact_bytes < 0:
            max_artifact_bytes = 0
        if max_artifact_bytes == 0:
            return resp

        header_fn = getattr(adapter, "_download_headers", None)
        if not callable(header_fn):
            return resp
        base_url = getattr(adapter, "base_url", None)
        if not isinstance(base_url, str) or not base_url.strip():
            return resp

        try:
            raw_headers = header_fn()
        except Exception:
            return resp
        if not isinstance(raw_headers, dict) or not raw_headers:
            return resp
        headers: dict[str, str] = {}
        for k, v in raw_headers.items():
            if isinstance(k, str) and k and isinstance(v, str) and v:
                headers[k] = v
        if not headers:
            return resp

        host = urllib.parse.urlparse(base_url).hostname
        if not isinstance(host, str) or not host:
            return resp
        host_l = host.lower()

        out_msgs: list[Message] = []
        changed = False
        for msg in resp.output:
            out_parts: list[Part] = []
            for part in msg.content:
                src = part.source
                if not isinstance(src, PartSourceUrl):
                    out_parts.append(part)
                    continue
                url = src.url
                if not isinstance(url, str) or not url:
                    out_parts.append(part)
                    continue
                url_host = urllib.parse.urlparse(url).hostname
                if not isinstance(url_host, str) or not url_host or url_host.lower() != host_l:
                    out_parts.append(part)
                    continue

                tmp_path: str | None = None
                try:
                    tmp_path = download_to_tempfile(
                        url=url,
                        timeout_ms=timeout_ms,
                        max_bytes=max_artifact_bytes,
                        headers=headers,
                        proxy_url=self._proxy_url,
                    )
                    with open(tmp_path, "rb") as f:
                        data = f.read()
                except Exception:
                    out_parts.append(part)
                    continue
                finally:
                    if tmp_path is not None:
                        try:
                            os.unlink(tmp_path)
                        except OSError:
                            pass

                mime_type = part.mime_type.strip() if isinstance(part.mime_type, str) and part.mime_type.strip() else None
                if mime_type is None and part.type == "image":
                    mime_type = sniff_image_mime_type(data)

                artifact_id = store.put(data, mime_type)
                if artifact_id is None:
                    out_parts.append(part)
                    continue

                out_parts.append(
                    Part(
                        type=part.type,
                        mime_type=mime_type,
                        source=PartSourceUrl(url=store.url(artifact_id)),
                        text=part.text,
                        embedding=part.embedding,
                        meta=part.meta,
                    )
                )
                changed = True

            out_msgs.append(Message(role=msg.role, content=out_parts))

        if not changed:
            return resp
        return replace(resp, output=out_msgs)

    def download_to_file(
        self,
        *,
        url: str,
        output_path: str,
        provider: str | None = None,
        timeout_ms: int | None = None,
        max_bytes: int | None = None,
    ) -> None:
        headers: dict[str, str] | None = None
        if provider is not None:
            try:
                adapter = self._adapter(_normalize_provider(provider))
            except GenAIError:
                adapter = None
            if adapter is not None:
                header_fn = getattr(adapter, "_download_headers", None)
                base_url = getattr(adapter, "base_url", None)
                if callable(header_fn) and isinstance(base_url, str) and base_url.strip():
                    base_host = urllib.parse.urlparse(base_url).hostname
                    url_host = urllib.parse.urlparse(url).hostname
                    if (
                        isinstance(base_host, str)
                        and base_host
                        and isinstance(url_host, str)
                        and url_host
                        and base_host.lower() == url_host.lower()
                    ):
                        try:
                            raw = header_fn()
                        except Exception:
                            raw = None
                        if isinstance(raw, dict) and raw:
                            sanitized: dict[str, str] = {}
                            for k, v in raw.items():
                                if isinstance(k, str) and k and isinstance(v, str) and v:
                                    sanitized[k] = v
                            if sanitized:
                                headers = sanitized

        _download_to_file(
            url=url,
            output_path=output_path,
            timeout_ms=timeout_ms,
            max_bytes=max_bytes,
            headers=headers,
            proxy_url=self._proxy_url,
        )

    def generate_stream(self, request: GenerateRequest) -> Iterator[GenerateEvent]:
        out = self.generate(request, stream=True)
        if not isinstance(out, Iterator):
            raise not_supported_error("provider returned non-stream response")
        return out

    async def generate_async(self, request: GenerateRequest) -> GenerateResponse:
        """
        Async non-streaming wrapper for `generate()`.

        Implementation: run sync HTTP calls in a worker thread via `asyncio.to_thread`.
        """
        out = await asyncio.to_thread(self.generate, request, stream=False)
        if isinstance(out, GenerateResponse):
            return out
        raise not_supported_error("provider returned stream response")

    def _adapter(self, provider: str):
        provider = _normalize_provider(provider)
        if provider == "openai":
            if self._openai is None:
                raise invalid_request_error("NOUS_GENAI_OPENAI_API_KEY/OPENAI_API_KEY not configured")
            return self._openai
        if provider == "google":
            if self._gemini is None:
                raise invalid_request_error("NOUS_GENAI_GOOGLE_API_KEY/GOOGLE_API_KEY not configured")
            return self._gemini
        if provider == "anthropic":
            if self._anthropic is None:
                raise invalid_request_error("NOUS_GENAI_ANTHROPIC_API_KEY/ANTHROPIC_API_KEY not configured")
            return self._anthropic
        if provider == "aliyun":
            if self._aliyun is None:
                raise invalid_request_error("NOUS_GENAI_ALIYUN_API_KEY/ALIYUN_API_KEY not configured")
            return self._aliyun
        if provider == "volcengine":
            if self._volcengine is None:
                raise invalid_request_error("NOUS_GENAI_VOLCENGINE_API_KEY/VOLCENGINE_API_KEY not configured")
            return self._volcengine
        if provider == "tuzi-web":
            if self._tuzi_web is None:
                raise invalid_request_error("NOUS_GENAI_TUZI_WEB_API_KEY/TUZI_WEB_API_KEY not configured")
            return self._tuzi_web
        if provider == "tuzi-openai":
            if self._tuzi_openai is None:
                raise invalid_request_error("NOUS_GENAI_TUZI_OPENAI_API_KEY/TUZI_OPENAI_API_KEY not configured")
            return self._tuzi_openai
        if provider == "tuzi-google":
            if self._tuzi_google is None:
                raise invalid_request_error("NOUS_GENAI_TUZI_GOOGLE_API_KEY/TUZI_GOOGLE_API_KEY not configured")
            return self._tuzi_google
        if provider == "tuzi-anthropic":
            if self._tuzi_anthropic is None:
                raise invalid_request_error("NOUS_GENAI_TUZI_ANTHROPIC_API_KEY/TUZI_ANTHROPIC_API_KEY not configured")
            return self._tuzi_anthropic
        raise invalid_request_error(f"unknown provider: {provider}")


def _normalize_provider(provider: str) -> str:
    p = provider.strip().lower()
    if p == "gemini":
        return "google"
    if p in {"volc", "ark"}:
        return "volcengine"
    return p


def _split_model(model: str) -> tuple[str, str]:
    if ":" not in model:
        raise invalid_request_error('model must be "{provider}:{model_id}"')
    provider, model_id = model.split(":", 1)
    return provider, model_id


def _env_int(name: str, default: int) -> int:
    raw = os.environ.get(name)
    if raw is None:
        return default
    try:
        value = int(raw)
    except ValueError:
        return default
    return value


def _is_mcp_transport_marker(value: str) -> bool:
    return value in {"mcp", "sse", "streamable", "streamable-http", "streamable_http"}


def _validate_mcp_wire_request(request: GenerateRequest) -> None:
    for msg in request.input:
        for part in msg.content:
            source = part.source
            if source is None:
                continue
            kind = getattr(source, "kind", None)
            if kind == "path":
                raise invalid_request_error(
                    "MCP transport does not support local file sources; use bytes(encoding=base64)/url/ref"
                )
            if kind == "bytes":
                enc = getattr(source, "encoding", None)
                data = getattr(source, "data", None)
                if enc == "base64" and isinstance(data, str):
                    continue
                raise invalid_request_error(
                    "MCP transport does not support raw bytes; use bytes(encoding=base64)/url/ref"
                )


def _normalize_output_text_json_schema(request: GenerateRequest) -> GenerateRequest:
    spec = request.output.text
    if spec is None:
        return request
    schema = spec.json_schema
    if schema is None:
        return request
    if isinstance(schema, dict):
        reject_gemini_response_schema_dict(schema)
        return request
    coerced = normalize_json_schema(schema)
    return replace(request, output=replace(request.output, text=replace(spec, json_schema=coerced)))
