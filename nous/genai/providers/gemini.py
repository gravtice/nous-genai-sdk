from __future__ import annotations

import os
import re
import tempfile
import time
import urllib.parse
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any, Iterator, Literal
from uuid import uuid4

from .._internal.capability_rules import (
    gemini_image_input_modalities,
    gemini_model_kind,
    gemini_output_modalities,
)
from .._internal.errors import (
    invalid_request_error,
    not_supported_error,
    provider_error,
    timeout_error,
)
from .._internal.http import (
    download_to_tempfile,
    multipart_form_data_json_and_file,
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
    PartType,
    PartSourceBytes,
    PartSourcePath,
    PartSourceRef,
    PartSourceUrl,
    Usage,
    bytes_to_base64,
    detect_mime_type,
    file_to_bytes,
    normalize_reasoning_effort,
)

_GEMINI_DEFAULT_BASE_URL = "https://generativelanguage.googleapis.com"

_ASYNCDATA_BASE_URL = "https://asyncdata.net"

_TUZI_TASK_ID_RE = re.compile(r"Task ID:\s*`([^`]+)`")
_MP4_URL_RE = re.compile(r"https?://[^\s)]+\.mp4")
_MD_IMAGE_URL_RE = re.compile(r"!\[[^\]]*]\((https?://[^\s)]+)\)")

_GEMINI_SCHEMA_TYPES = frozenset(
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

_JSON_SCHEMA_TO_GEMINI_TYPE: dict[str, str] = {
    "string": "STRING",
    "number": "NUMBER",
    "integer": "INTEGER",
    "boolean": "BOOLEAN",
    "array": "ARRAY",
    "object": "OBJECT",
}


def _looks_like_gemini_schema(schema: dict[str, Any]) -> bool:
    t = schema.get("type")
    return isinstance(t, str) and t in _GEMINI_SCHEMA_TYPES


def _resolve_json_schema_ref(ref: str, defs: dict[str, Any]) -> dict[str, Any]:
    if not ref.startswith("#/$defs/"):
        raise invalid_request_error(
            "Gemini responseSchema only supports local $defs $ref"
        )
    name = ref[len("#/$defs/") :]
    if not name or "/" in name:
        raise invalid_request_error(
            "Gemini responseSchema only supports simple $defs refs"
        )
    resolved = defs.get(name)
    if not isinstance(resolved, dict):
        raise invalid_request_error(f"unresolved $ref: {ref}")
    return resolved


def _is_json_schema_null(schema: Any) -> bool:
    if not isinstance(schema, dict):
        return False
    return schema.get("type") == "null"


def _convert_nullable_union(
    *,
    tag: str,
    options: Any,
    defs: dict[str, Any],
    ref_stack: tuple[str, ...],
) -> dict[str, Any]:
    if not isinstance(options, list) or not options:
        raise invalid_request_error(
            f"Gemini responseSchema {tag} must be a non-empty array"
        )
    non_null: list[dict[str, Any]] = []
    null_count = 0
    for item in options:
        if _is_json_schema_null(item):
            null_count += 1
            continue
        if not isinstance(item, dict):
            raise invalid_request_error(
                f"Gemini responseSchema {tag} items must be objects"
            )
        non_null.append(item)
    if null_count != 1 or len(non_null) != 1:
        raise invalid_request_error(
            f"Gemini responseSchema only supports nullable unions ({tag} with exactly one null and one schema)"
        )
    out = _json_schema_to_gemini_schema(non_null[0], defs=defs, ref_stack=ref_stack)
    out["nullable"] = True
    return out


def _json_schema_to_gemini_schema(
    schema: Any,
    *,
    defs: dict[str, Any],
    ref_stack: tuple[str, ...],
) -> dict[str, Any]:
    if not isinstance(schema, dict):
        raise invalid_request_error(
            "output.text.json_schema must be an object for Gemini"
        )

    local_defs = schema.get("$defs")
    if isinstance(local_defs, dict) and local_defs:
        merged = dict(defs)
        merged.update(local_defs)
        defs = merged

    ref = schema.get("$ref")
    if ref is not None:
        if not isinstance(ref, str) or not ref:
            raise invalid_request_error("$ref must be a non-empty string")
        if ref in ref_stack:
            raise invalid_request_error(
                "Gemini responseSchema does not support recursive $ref"
            )
        resolved = _resolve_json_schema_ref(ref, defs)
        out = _json_schema_to_gemini_schema(
            resolved, defs=defs, ref_stack=ref_stack + (ref,)
        )
        desc = schema.get("description")
        if isinstance(desc, str) and desc.strip() and "description" not in out:
            out["description"] = desc.strip()
        return out

    all_of = schema.get("allOf")
    if all_of is not None:
        if isinstance(all_of, list) and len(all_of) == 1:
            out = _json_schema_to_gemini_schema(
                all_of[0], defs=defs, ref_stack=ref_stack
            )
            desc = schema.get("description")
            if isinstance(desc, str) and desc.strip() and "description" not in out:
                out["description"] = desc.strip()
            return out
        raise invalid_request_error(
            "Gemini responseSchema does not support allOf (except a single item)"
        )

    any_of = schema.get("anyOf")
    if any_of is not None:
        out = _convert_nullable_union(
            tag="anyOf", options=any_of, defs=defs, ref_stack=ref_stack
        )
        desc = schema.get("description")
        if isinstance(desc, str) and desc.strip() and "description" not in out:
            out["description"] = desc.strip()
        return out

    one_of = schema.get("oneOf")
    if one_of is not None:
        out = _convert_nullable_union(
            tag="oneOf", options=one_of, defs=defs, ref_stack=ref_stack
        )
        desc = schema.get("description")
        if isinstance(desc, str) and desc.strip() and "description" not in out:
            out["description"] = desc.strip()
        return out

    nullable = False
    t = schema.get("type")
    if isinstance(t, list):
        types = [x for x in t if isinstance(x, str) and x]
        if "null" in types:
            types = [x for x in types if x != "null"]
            if len(types) != 1:
                raise invalid_request_error(
                    "Gemini responseSchema only supports nullable union with one non-null type"
                )
            t = types[0]
            nullable = True
        elif len(types) == 1:
            t = types[0]
        else:
            raise invalid_request_error(
                "Gemini responseSchema does not support union types"
            )

    if not isinstance(t, str) or not t:
        if isinstance(schema.get("properties"), dict):
            t = "object"
        else:
            raise invalid_request_error("output.text.json_schema missing type")

    t_norm = t.strip().lower()
    gemini_type = _JSON_SCHEMA_TO_GEMINI_TYPE.get(t_norm)
    if gemini_type is None:
        raise invalid_request_error(f"Gemini responseSchema unsupported type: {t}")

    out = {"type": gemini_type}
    if nullable:
        out["nullable"] = True

    desc = schema.get("description")
    if isinstance(desc, str) and desc.strip():
        out["description"] = desc.strip()

    const = schema.get("const")
    if const is not None:
        out["enum"] = [const]
    enum = schema.get("enum")
    if isinstance(enum, list) and enum:
        out["enum"] = enum

    if gemini_type == "OBJECT":
        props = schema.get("properties")
        if isinstance(props, dict):
            out_props: dict[str, Any] = {}
            for k, v in props.items():
                if not isinstance(k, str) or not k:
                    continue
                out_props[k] = _json_schema_to_gemini_schema(
                    v, defs=defs, ref_stack=ref_stack
                )
            if out_props:
                out["properties"] = out_props
        required = schema.get("required")
        if isinstance(required, list):
            req = [x for x in required if isinstance(x, str) and x]
            if req:
                out["required"] = req

        addl = schema.get("additionalProperties")
        if isinstance(addl, dict):
            out["additionalProperties"] = _json_schema_to_gemini_schema(
                addl, defs=defs, ref_stack=ref_stack
            )
        elif isinstance(addl, bool):
            out["additionalProperties"] = addl

    if gemini_type == "ARRAY":
        items = schema.get("items")
        if isinstance(items, dict):
            out["items"] = _json_schema_to_gemini_schema(
                items, defs=defs, ref_stack=ref_stack
            )
        elif items is not None:
            raise invalid_request_error(
                "Gemini responseSchema array items must be an object"
            )

    return out


def _to_gemini_response_schema(schema: Any) -> dict[str, Any]:
    if not isinstance(schema, dict):
        raise invalid_request_error("output.text.json_schema must be an object")
    if _looks_like_gemini_schema(schema):
        raise invalid_request_error(
            "output.text.json_schema must be JSON Schema (not Gemini responseSchema); "
            "pass a Python type/model or use provider_options.google.generationConfig.responseSchema"
        )
    defs = schema.get("$defs")
    defs_map = defs if isinstance(defs, dict) else {}
    return _json_schema_to_gemini_schema(schema, defs=defs_map, ref_stack=())


@dataclass(frozen=True, slots=True)
class GeminiAdapter:
    api_key: str
    base_url: str = _GEMINI_DEFAULT_BASE_URL
    provider_name: str = "google"
    auth_mode: Literal["query_key", "bearer"] = "query_key"
    supports_file_upload: bool = True
    proxy_url: str | None = None

    def _auth_headers(self) -> dict[str, str]:
        if self.auth_mode == "bearer":
            return {"Authorization": f"Bearer {self.api_key}"}
        return {}

    def _with_key(self, url: str) -> str:
        if self.auth_mode != "query_key":
            return url
        sep = "&" if "?" in url else "?"
        return f"{url}{sep}key={self.api_key}"

    def _v1beta_url(self, path: str) -> str:
        base = self.base_url.rstrip("/")
        return self._with_key(f"{base}/v1beta/{path.lstrip('/')}")

    def _upload_url(self, path: str) -> str:
        base = self.base_url.rstrip("/")
        return self._with_key(f"{base}/upload/v1beta/{path.lstrip('/')}")

    def _download_headers(self) -> dict[str, str] | None:
        if self.auth_mode == "bearer":
            return {"Authorization": f"Bearer {self.api_key}"}
        return {"x-goog-api-key": self.api_key}

    def capabilities(self, model_id: str) -> Capability:
        kind = gemini_model_kind(model_id)
        out_mods = gemini_output_modalities(kind)

        if kind == "video":
            return Capability(
                input_modalities={"text"},
                output_modalities=out_mods,
                supports_stream=False,
                supports_job=True,
                supports_tools=False,
                supports_json_schema=False,
            )
        if kind == "embedding":
            return Capability(
                input_modalities={"text"},
                output_modalities=out_mods,
                supports_stream=False,
                supports_job=False,
                supports_tools=False,
                supports_json_schema=False,
            )
        if kind == "tts":
            return Capability(
                input_modalities={"text"},
                output_modalities=out_mods,
                supports_stream=False,
                supports_job=False,
                supports_tools=False,
                supports_json_schema=False,
            )
        if kind == "native_audio":
            return Capability(
                input_modalities={"text", "audio", "video"},
                output_modalities=out_mods,
                supports_stream=True,
                supports_job=False,
                supports_tools=True,
                supports_json_schema=True,
            )
        if kind == "image":
            return Capability(
                input_modalities=gemini_image_input_modalities(model_id),
                output_modalities=out_mods,
                supports_stream=False,
                supports_job=False,
                supports_tools=False,
                supports_json_schema=False,
            )
        return Capability(
            input_modalities={"text", "image", "audio", "video"},
            output_modalities=out_mods,
            supports_stream=True,
            supports_job=False,
            supports_tools=True,
            supports_json_schema=True,
        )

    def list_models(self, *, timeout_ms: int | None = None) -> list[str]:
        """
        Fetch remote model ids via Gemini Developer API GET /v1beta/models.

        Returns model ids without the leading "models/" prefix.
        """
        out: list[str] = []
        page_token: str | None = None
        for _ in range(20):
            url = self._v1beta_url("models")
            if page_token:
                token = urllib.parse.quote(page_token, safe="")
                sep = "&" if "?" in url else "?"
                url = f"{url}{sep}pageToken={token}"
            obj = request_json(
                method="GET",
                url=url,
                headers=self._auth_headers(),
                timeout_ms=timeout_ms,
                proxy_url=self.proxy_url,
            )
            models = obj.get("models")
            if isinstance(models, list):
                for m in models:
                    if not isinstance(m, dict):
                        continue
                    name = m.get("name")
                    if not isinstance(name, str) or not name:
                        continue
                    out.append(
                        name[len("models/") :] if name.startswith("models/") else name
                    )
            next_token = obj.get("nextPageToken")
            if not isinstance(next_token, str) or not next_token:
                break
            page_token = next_token
        return sorted(set(out))

    def generate(
        self, request: GenerateRequest, *, stream: bool
    ) -> GenerateResponse | Iterator[GenerateEvent]:
        model_id = request.model_id()
        model_name = (
            model_id if model_id.startswith("models/") else f"models/{model_id}"
        )
        modalities = set(request.output.modalities)
        if "embedding" in modalities:
            if modalities != {"embedding"}:
                raise not_supported_error(
                    "embedding cannot be combined with other output modalities"
                )
            if stream:
                raise not_supported_error("embedding does not support streaming")
            return self._embed(request, model_name=model_name)

        if "video" in modalities:
            if modalities != {"video"}:
                raise not_supported_error(
                    "video cannot be combined with other output modalities"
                )
            if stream:
                raise not_supported_error("video does not support streaming")
            return self._video(request, model_name=model_name)
        if stream and (modalities & {"image", "audio"}):
            raise not_supported_error(
                "streaming image/audio output is not supported in this SDK yet"
            )

        if stream:
            return self._generate_stream(request, model_name=model_name)
        return self._generate(request, model_name=model_name)

    def _generate(
        self, request: GenerateRequest, *, model_name: str
    ) -> GenerateResponse:
        url = self._v1beta_url(f"{model_name}:generateContent")
        body = self._generate_body(request, model_name=model_name)
        obj = request_json(
            method="POST",
            url=url,
            headers=self._auth_headers(),
            json_body=body,
            timeout_ms=request.params.timeout_ms,
            proxy_url=self.proxy_url,
        )
        return self._parse_generate(obj, model=model_name)

    def _video(self, request: GenerateRequest, *, model_name: str) -> GenerateResponse:
        if self.provider_name.startswith("tuzi") and model_name.startswith(
            "models/veo2"
        ):
            return self._tuzi_veo2_video(request, model_name=model_name)

        if not model_name.startswith("models/veo-"):
            raise not_supported_error(
                'video generation requires model like "google:veo-3.1-generate-preview"'
            )

        texts = _gather_text_inputs(request)
        if len(texts) != 1:
            raise invalid_request_error(
                "video generation requires exactly one text part"
            )
        prompt = texts[0]

        body: dict[str, Any] = {"instances": [{"prompt": prompt}]}
        params: dict[str, Any] = {}
        video = request.output.video
        if video and video.duration_sec is not None:
            duration = int(video.duration_sec)
            if duration < 5 or duration > 8:
                raise invalid_request_error(
                    "google veo duration_sec must be between 5 and 8 seconds"
                )
            params["durationSeconds"] = duration
        if video and video.aspect_ratio:
            params["aspectRatio"] = video.aspect_ratio

        opts = (
            request.provider_options.get(self.provider_name)
            or request.provider_options.get("google")
            or request.provider_options.get("gemini")
        )
        if isinstance(opts, dict):
            opt_params = opts.get("parameters")
            if opt_params is not None:
                if not isinstance(opt_params, dict):
                    raise invalid_request_error(
                        "provider_options.google.parameters must be an object"
                    )
                for k, v in opt_params.items():
                    if k in params:
                        raise invalid_request_error(
                            f"provider_options cannot override parameters.{k}"
                        )
                    params[k] = v
            for k, v in opts.items():
                if k == "parameters":
                    continue
                if k in body:
                    raise invalid_request_error(
                        f"provider_options cannot override body.{k}"
                    )
                body[k] = v

        if params:
            body["parameters"] = params

        budget_ms = (
            120_000 if request.params.timeout_ms is None else request.params.timeout_ms
        )
        deadline = time.time() + max(1, budget_ms) / 1000.0
        url = self._v1beta_url(f"{model_name}:predictLongRunning")
        obj = request_json(
            method="POST",
            url=url,
            headers=self._auth_headers(),
            json_body=body,
            timeout_ms=min(30_000, max(1, budget_ms)),
            proxy_url=self.proxy_url,
        )
        name = obj.get("name")
        if not isinstance(name, str) or not name:
            raise provider_error("gemini veo response missing operation name")

        if not request.wait:
            return GenerateResponse(
                id=f"sdk_{uuid4().hex}",
                provider="google",
                model=f"google:{model_name}",
                status="running",
                job=JobInfo(job_id=name, poll_after_ms=1_000),
            )

        if obj.get("done") is not True:
            obj = self._wait_operation_done(name=name, deadline=deadline)
        if obj.get("done") is not True:
            return GenerateResponse(
                id=f"sdk_{uuid4().hex}",
                provider="google",
                model=f"google:{model_name}",
                status="running",
                job=JobInfo(job_id=name, poll_after_ms=1_000),
            )

        err = obj.get("error")
        if isinstance(err, dict):
            msg = err.get("message") or str(err)
            raise provider_error(f"gemini veo operation failed: {msg}")

        video_uri = _extract_veo_video_uri(obj.get("response"))
        if not video_uri:
            raise provider_error("gemini veo operation response missing video uri")
        scheme = urllib.parse.urlparse(video_uri).scheme.lower()
        source = (
            PartSourceUrl(url=video_uri)
            if scheme in {"http", "https"}
            else PartSourceRef(provider=self.provider_name, id=video_uri)
        )
        part = Part(type="video", mime_type="video/mp4", source=source)
        return GenerateResponse(
            id=f"sdk_{uuid4().hex}",
            provider=self.provider_name,
            model=f"{self.provider_name}:{model_name}",
            status="completed",
            output=[Message(role="assistant", content=[part])],
            usage=None,
        )

    def _tuzi_veo2_video(
        self, request: GenerateRequest, *, model_name: str
    ) -> GenerateResponse:
        texts = _gather_text_inputs(request)
        if len(texts) != 1:
            raise invalid_request_error(
                "video generation requires exactly one text part"
            )
        prompt = texts[0]

        params: dict[str, Any] = {}
        video = request.output.video
        if video and video.duration_sec is not None:
            params["durationSeconds"] = int(video.duration_sec)
        if video and video.aspect_ratio:
            params["aspectRatio"] = video.aspect_ratio

        body: dict[str, Any] = {
            "contents": [{"role": "user", "parts": [{"text": prompt}]}]
        }
        if params:
            body["parameters"] = params

        budget_ms = (
            120_000 if request.params.timeout_ms is None else request.params.timeout_ms
        )
        obj = request_json(
            method="POST",
            url=self._v1beta_url(f"{model_name}:predictLongRunning"),
            headers=self._auth_headers(),
            json_body=body,
            timeout_ms=max(1, budget_ms),
            proxy_url=self.proxy_url,
        )

        text = _first_candidate_text(obj)
        mp4_url = _extract_first_mp4_url(text) if text else None
        task_id = _extract_tuzi_task_id(text) if text else None

        if mp4_url:
            part = Part(
                type="video", mime_type="video/mp4", source=PartSourceUrl(url=mp4_url)
            )
            return GenerateResponse(
                id=f"sdk_{uuid4().hex}",
                provider=self.provider_name,
                model=f"{self.provider_name}:{model_name}",
                status="completed",
                output=[Message(role="assistant", content=[part])],
                usage=_usage_from_gemini(obj.get("usageMetadata")),
            )

        if task_id and request.wait:
            deadline = time.time() + max(1, budget_ms) / 1000.0
            mp4_url = _poll_tuzi_video_mp4(
                task_id=task_id, deadline=deadline, proxy_url=self.proxy_url
            )
            if mp4_url:
                part = Part(
                    type="video",
                    mime_type="video/mp4",
                    source=PartSourceUrl(url=mp4_url),
                )
                return GenerateResponse(
                    id=f"sdk_{uuid4().hex}",
                    provider=self.provider_name,
                    model=f"{self.provider_name}:{model_name}",
                    status="completed",
                    output=[Message(role="assistant", content=[part])],
                    usage=None,
                )

        job = JobInfo(job_id=task_id, poll_after_ms=2_000) if task_id else None
        return GenerateResponse(
            id=f"sdk_{uuid4().hex}",
            provider=self.provider_name,
            model=f"{self.provider_name}:{model_name}",
            status="running",
            job=job,
        )

    def _generate_stream(
        self, request: GenerateRequest, *, model_name: str
    ) -> Iterator[GenerateEvent]:
        url = self._v1beta_url(f"{model_name}:streamGenerateContent?alt=sse")
        body = self._generate_body(request, model_name=model_name)
        events = request_stream_json_sse(
            method="POST",
            url=url,
            headers=self._auth_headers(),
            json_body=body,
            timeout_ms=request.params.timeout_ms,
            proxy_url=self.proxy_url,
        )

        def _iter() -> Iterator[GenerateEvent]:
            for obj in events:
                delta = self._extract_text_delta(obj)
                if delta:
                    yield GenerateEvent(type="output.text.delta", data={"delta": delta})
            yield GenerateEvent(type="done", data={})

        return _iter()

    def _embed(self, request: GenerateRequest, *, model_name: str) -> GenerateResponse:
        texts = _gather_text_inputs(request)
        emb = request.output.embedding
        dims = emb.dimensions if emb and emb.dimensions is not None else None
        if model_name == "models/embedding-gecko-001":
            if dims is not None:
                raise invalid_request_error(
                    "models/embedding-gecko-001 does not support embedding.dimensions"
                )
            if len(texts) != 1:
                raise invalid_request_error(
                    "models/embedding-gecko-001 only supports single text per request"
                )
            url = self._v1beta_url(f"{model_name}:embedText")
            obj = request_json(
                method="POST",
                url=url,
                headers=self._auth_headers(),
                json_body={"model": model_name, "text": texts[0]},
                timeout_ms=request.params.timeout_ms,
                proxy_url=self.proxy_url,
            )
            embedding = obj.get("embedding")
            if not isinstance(embedding, dict):
                raise provider_error("gemini embedText response missing embedding")
            values = embedding.get("value")
            if not isinstance(values, list) or not all(
                isinstance(x, (int, float)) for x in values
            ):
                raise provider_error(
                    "gemini embedText response missing embedding.value"
                )
            parts = [Part(type="embedding", embedding=[float(x) for x in values])]
        else:
            url = self._v1beta_url(f"{model_name}:batchEmbedContents")
            reqs: list[dict[str, Any]] = [
                {"model": model_name, "content": {"parts": [{"text": t}]}}
                for t in texts
            ]
            if dims is not None:
                for r in reqs:
                    r["outputDimensionality"] = dims
            obj = request_json(
                method="POST",
                url=url,
                headers=self._auth_headers(),
                json_body={"requests": reqs},
                timeout_ms=request.params.timeout_ms,
                proxy_url=self.proxy_url,
            )
            embeddings = obj.get("embeddings")
            if not isinstance(embeddings, list) or len(embeddings) != len(texts):
                raise provider_error(
                    "gemini batchEmbedContents response missing embeddings"
                )
            parts = []
            for emb in embeddings:
                if not isinstance(emb, dict):
                    raise provider_error("gemini embedding item is not object")
                values = emb.get("values")
                if not isinstance(values, list) or not all(
                    isinstance(x, (int, float)) for x in values
                ):
                    raise provider_error("gemini embedding item missing values")
                parts.append(
                    Part(type="embedding", embedding=[float(x) for x in values])
                )
        return GenerateResponse(
            id=f"sdk_{uuid4().hex}",
            provider=self.provider_name,
            model=f"{self.provider_name}:{model_name}",
            status="completed",
            output=[Message(role="assistant", content=parts)],
            usage=None,
        )

    def _generate_body(
        self, request: GenerateRequest, *, model_name: str
    ) -> dict[str, Any]:
        system_text = _extract_system_text(request)
        contents = [
            _message_to_content(self, m, timeout_ms=request.params.timeout_ms)
            for m in request.input
            if m.role != "system"
        ]
        if not contents:
            raise invalid_request_error(
                "request.input must contain at least one non-system message"
            )

        gen_cfg: dict[str, Any] = {}
        params = request.params
        if params.temperature is not None:
            gen_cfg["temperature"] = params.temperature
        if params.top_p is not None:
            gen_cfg["topP"] = params.top_p
        if params.seed is not None:
            gen_cfg["seed"] = params.seed
        text_spec = request.output.text
        max_out = (
            text_spec.max_output_tokens
            if text_spec and text_spec.max_output_tokens is not None
            else params.max_output_tokens
        )
        if max_out is not None:
            gen_cfg["maxOutputTokens"] = max_out
        if params.stop is not None:
            gen_cfg["stopSequences"] = params.stop

        if params.reasoning is not None:
            thinking_cfg = _thinking_config(params.reasoning, model_name=model_name)
            if thinking_cfg is not None:
                gen_cfg["thinkingConfig"] = thinking_cfg

        if text_spec and (
            text_spec.format != "text" or text_spec.json_schema is not None
        ):
            if set(request.output.modalities) != {"text"}:
                raise invalid_request_error("json output requires text-only modality")
            gen_cfg["responseMimeType"] = "application/json"
            if text_spec.json_schema is not None:
                gen_cfg["responseSchema"] = _to_gemini_response_schema(
                    text_spec.json_schema
                )

        response_modalities = _gemini_response_modalities(request.output.modalities)
        if response_modalities == ["IMAGE"] and model_name.endswith("image-generation"):
            response_modalities = ["TEXT", "IMAGE"]
        if response_modalities:
            gen_cfg["responseModalities"] = response_modalities

        if request.output.image and request.output.image.n is not None:
            gen_cfg["candidateCount"] = request.output.image.n

        if "audio" in request.output.modalities:
            audio = request.output.audio
            if audio is None or not audio.voice:
                raise invalid_request_error(
                    "output.audio.voice required for Gemini audio output"
                )
            speech_cfg: dict[str, Any] = {
                "voiceConfig": {"prebuiltVoiceConfig": {"voiceName": audio.voice}},
            }
            if audio.language:
                speech_cfg["languageCode"] = audio.language
            gen_cfg["speechConfig"] = speech_cfg

        if "image" in request.output.modalities and request.output.image:
            img_cfg = {}
            if request.output.image.size:
                img_cfg["imageSize"] = request.output.image.size
            if img_cfg:
                gen_cfg["imageConfig"] = img_cfg

        opts = (
            request.provider_options.get(self.provider_name)
            or request.provider_options.get("google")
            or request.provider_options.get("gemini")
        )
        if isinstance(opts, dict):
            opt_gen_cfg = opts.get("generationConfig")
            if opt_gen_cfg is not None:
                if not isinstance(opt_gen_cfg, dict):
                    raise invalid_request_error(
                        "provider_options.google.generationConfig must be an object"
                    )
                for k, v in opt_gen_cfg.items():
                    if k in gen_cfg:
                        raise invalid_request_error(
                            f"provider_options cannot override generationConfig.{k}"
                        )
                    gen_cfg[k] = v

        body: dict[str, Any] = {"contents": contents}
        if system_text:
            body["systemInstruction"] = {
                "role": "user",
                "parts": [{"text": system_text}],
            }
        if gen_cfg:
            body["generationConfig"] = gen_cfg

        if request.tools:
            decls: list[dict[str, Any]] = []
            for t in request.tools:
                name = t.name.strip()
                if not name:
                    raise invalid_request_error("tool.name must be non-empty")
                decl: dict[str, Any] = {"name": name}
                if isinstance(t.description, str) and t.description.strip():
                    decl["description"] = t.description.strip()
                decl["parameters"] = (
                    t.parameters if t.parameters is not None else {"type": "object"}
                )
                decls.append(decl)
            body["tools"] = [{"functionDeclarations": decls}]

        if request.tool_choice is not None:
            choice = request.tool_choice.normalized()
            if choice.mode in {"required", "tool"} and not request.tools:
                raise invalid_request_error("tool_choice requires request.tools")
            if choice.mode == "none":
                body["toolConfig"] = {"functionCallingConfig": {"mode": "NONE"}}
            elif choice.mode == "required":
                body["toolConfig"] = {"functionCallingConfig": {"mode": "ANY"}}
            elif choice.mode == "tool":
                body["toolConfig"] = {
                    "functionCallingConfig": {
                        "mode": "ANY",
                        "allowedFunctionNames": [choice.name],
                    }
                }

        if isinstance(opts, dict):
            for k, v in opts.items():
                if k == "generationConfig":
                    continue
                if k in body:
                    raise invalid_request_error(
                        f"provider_options cannot override body.{k}"
                    )
                body[k] = v
        return body

    def _parse_generate(self, obj: dict[str, Any], *, model: str) -> GenerateResponse:
        candidates = obj.get("candidates")
        if not isinstance(candidates, list) or not candidates:
            raise provider_error("gemini response missing candidates")
        cand0 = candidates[0]
        if not isinstance(cand0, dict):
            raise provider_error("gemini candidate is not object")
        content = cand0.get("content")
        if not isinstance(content, dict):
            raise provider_error("gemini candidate missing content")
        parts = content.get("parts")
        if not isinstance(parts, list):
            raise provider_error("gemini candidate content missing parts")
        out_parts: list[Part] = []
        for p in parts:
            if not isinstance(p, dict):
                continue
            out_parts.extend(_gemini_part_to_parts(p, provider_name=self.provider_name))
        usage = _usage_from_gemini(obj.get("usageMetadata"))
        if not out_parts:
            out_parts.append(Part.from_text(""))
        return GenerateResponse(
            id=f"sdk_{uuid4().hex}",
            provider=self.provider_name,
            model=f"{self.provider_name}:{model}",
            status="completed",
            output=[Message(role="assistant", content=out_parts)],
            usage=usage,
        )

    def _extract_text_delta(self, obj: dict[str, Any]) -> str | None:
        candidates = obj.get("candidates")
        if not isinstance(candidates, list) or not candidates:
            return None
        cand0 = candidates[0]
        if not isinstance(cand0, dict):
            return None
        content = cand0.get("content")
        if not isinstance(content, dict):
            return None
        parts = content.get("parts")
        if not isinstance(parts, list) or not parts:
            return None
        p0 = parts[0]
        if not isinstance(p0, dict):
            return None
        text = p0.get("text")
        if isinstance(text, str) and text:
            return text
        return None

    def _upload_to_file_uri(
        self, file_path: str, *, mime_type: str, timeout_ms: int | None
    ) -> str:
        if not self.supports_file_upload:
            raise not_supported_error(
                "Gemini file upload is not supported for this provider"
            )
        meta = {"file": {"displayName": os.path.basename(file_path)}}
        body = multipart_form_data_json_and_file(
            metadata_field="metadata",
            metadata=meta,
            file_field="file",
            file_path=file_path,
            filename=os.path.basename(file_path),
            file_mime_type=mime_type,
        )
        url = self._upload_url("files")
        headers = {"X-Goog-Upload-Protocol": "multipart"}
        headers.update(self._auth_headers())
        obj = request_streaming_body_json(
            method="POST",
            url=url,
            headers=headers,
            body=body,
            timeout_ms=timeout_ms,
            proxy_url=self.proxy_url,
        )
        file_obj = obj.get("file")
        if not isinstance(file_obj, dict):
            raise provider_error("gemini upload response missing file")
        name = file_obj.get("name")
        uri = file_obj.get("uri")
        if not isinstance(name, str) or not name:
            raise provider_error("gemini upload response missing file.name")
        if not isinstance(uri, str) or not uri:
            raise provider_error("gemini upload response missing file.uri")
        state = file_obj.get("state")
        if state == "ACTIVE":
            return uri
        return self._wait_file_active(name=name, uri=uri, timeout_ms=timeout_ms)

    def _wait_file_active(self, *, name: str, uri: str, timeout_ms: int | None) -> str:
        if not self.supports_file_upload:
            raise not_supported_error(
                "Gemini file upload is not supported for this provider"
            )
        url = self._v1beta_url(name)
        budget_ms = 120_000 if timeout_ms is None else timeout_ms
        deadline = time.time() + max(1, budget_ms) / 1000.0
        while True:
            remaining_ms = int((deadline - time.time()) * 1000)
            if remaining_ms <= 0:
                break
            obj = request_json(
                method="GET",
                url=url,
                headers=self._auth_headers(),
                json_body=None,
                timeout_ms=min(30_000, remaining_ms),
                proxy_url=self.proxy_url,
            )
            if not isinstance(obj, dict):
                raise provider_error("gemini get file response is not object")
            state = obj.get("state")
            if state == "ACTIVE":
                return uri
            if state == "FAILED":
                err = obj.get("error")
                raise provider_error(f"gemini file processing failed: {err}")
            time.sleep(min(1.0, max(0.0, deadline - time.time())))
        raise timeout_error("gemini file processing timeout")

    def _wait_operation_done(self, *, name: str, deadline: float) -> dict[str, Any]:
        url = self._v1beta_url(name)
        while True:
            remaining_ms = int((deadline - time.time()) * 1000)
            if remaining_ms <= 0:
                break
            obj = request_json(
                method="GET",
                url=url,
                headers=self._auth_headers(),
                json_body=None,
                timeout_ms=min(30_000, remaining_ms),
                proxy_url=self.proxy_url,
            )
            if not isinstance(obj, dict):
                raise provider_error("gemini operation get response is not object")
            if obj.get("done") is True:
                return obj
            time.sleep(min(1.0, max(0.0, deadline - time.time())))
        return {"name": name, "done": False}


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


def _extract_system_text(request: GenerateRequest) -> str | None:
    chunks: list[str] = []
    for m in request.input:
        if m.role != "system":
            continue
        for p in m.content:
            if p.type != "text":
                raise invalid_request_error(
                    "system message only supports text for Gemini"
                )
            chunks.append(p.require_text())
    joined = "\n\n".join([c for c in chunks if c.strip()])
    return joined or None


def _gemini_supports_thinking_level(model_name: str) -> bool:
    mid = (
        model_name[len("models/") :] if model_name.startswith("models/") else model_name
    )
    mid = mid.strip().lower()
    if not mid.startswith("gemini-"):
        return False
    rest = mid[len("gemini-") :]
    digits = []
    for ch in rest:
        if ch.isdigit():
            digits.append(ch)
        else:
            break
    if not digits:
        return False
    try:
        major = int("".join(digits))
    except ValueError:
        return False
    return major >= 3


def _gemini_supports_thinking_budget(model_name: str) -> bool:
    mid = (
        model_name[len("models/") :] if model_name.startswith("models/") else model_name
    )
    mid = mid.strip().lower()
    return (
        "gemini-2.5-" in mid or "robotics-er" in mid or "flash-live-native-audio" in mid
    )


def _thinking_config(reasoning, *, model_name: str) -> dict[str, Any] | None:
    cfg: dict[str, Any] = {}
    if reasoning.effort is not None:
        if _gemini_supports_thinking_level(model_name):
            cfg["thinkingLevel"] = _map_effort_to_thinking_level(
                reasoning.effort, model_name=model_name
            )
        elif _gemini_supports_thinking_budget(model_name):
            cfg["thinkingBudget"] = _map_effort_to_thinking_budget(
                reasoning.effort, model_name=model_name
            )
    return cfg or None


def _map_effort_to_thinking_level(effort: object, *, model_name: str) -> str:
    """
    Gemini 3 thinkingLevel mapping (per Google docs):
    - Gemini 3 Pro: low/high only
    - Gemini 3 Flash: minimal/low/medium/high
    """
    eff = normalize_reasoning_effort(effort)
    mid = (
        model_name[len("models/") :] if model_name.startswith("models/") else model_name
    )
    mid = mid.strip().lower()
    is_flash = "gemini-3-flash" in mid
    if eff in {"none", "minimal"}:
        return "minimal" if is_flash else "low"
    if eff == "low":
        return "low"
    if eff == "medium":
        return "medium" if is_flash else "high"
    return "high"


def _map_effort_to_thinking_budget(effort: object, *, model_name: str) -> int:
    """
    Gemini 2.5+ thinkingBudget mapping (参考 Google docs + LiteLLM 默认值)。

    - `none` 尽量关闭（2.5 Pro 不能关闭，取最小预算）
    - `minimal/low/medium/high/xhigh` 映射到逐步增加的 budget
    """
    eff = normalize_reasoning_effort(effort)
    mid = (
        model_name[len("models/") :] if model_name.startswith("models/") else model_name
    )
    mid = mid.strip().lower()

    is_25_flash_lite = "gemini-2.5-flash-lite" in mid
    is_25_pro = "gemini-2.5-pro" in mid
    is_25_flash = "gemini-2.5-flash" in mid

    if eff == "none":
        return 128 if is_25_pro else 0
    if eff == "minimal":
        if is_25_flash_lite:
            return 512
        if is_25_pro:
            return 128
        if is_25_flash:
            return 1
        return 128
    if eff == "low":
        return 1024
    if eff == "medium":
        return 2048
    if eff == "high":
        return 4096
    return 8192


def _gemini_response_modalities(modalities: Sequence[str]) -> list[str]:
    out: list[str] = []
    for m in modalities:
        m = m.lower()
        if m == "text":
            out.append("TEXT")
        elif m == "image":
            out.append("IMAGE")
        elif m == "audio":
            out.append("AUDIO")
        elif m == "video":
            raise not_supported_error("Gemini does not support video response modality")
        elif m == "embedding":
            raise not_supported_error(
                "embedding should use embedContent/batchEmbedContents"
            )
        else:
            raise invalid_request_error(f"unknown modality: {m}")
    return out


def _message_to_content(
    adapter: GeminiAdapter, message: Message, *, timeout_ms: int | None
) -> dict[str, Any]:
    if message.role == "user" and any(p.type == "tool_call" for p in message.content):
        raise invalid_request_error(
            "tool_call parts are only allowed in assistant messages"
        )
    if message.role == "assistant" and any(
        p.type == "tool_result" for p in message.content
    ):
        raise invalid_request_error("tool_result parts must be sent as role='tool'")
    if message.role == "tool" and any(p.type != "tool_result" for p in message.content):
        raise invalid_request_error("tool messages may only contain tool_result parts")
    role = (
        "user"
        if message.role in {"user", "tool"}
        else "model"
        if message.role == "assistant"
        else message.role
    )
    if role not in {"user", "model"}:
        raise not_supported_error(f"Gemini does not support role: {message.role}")
    parts = [
        _part_to_gemini_part(adapter, p, timeout_ms=timeout_ms) for p in message.content
    ]
    return {"role": role, "parts": parts}


def _require_tool_call_meta(part: Part) -> tuple[str, Any]:
    name = part.meta.get("name")
    if not isinstance(name, str) or not name.strip():
        raise invalid_request_error("tool_call.meta.name must be a non-empty string")
    arguments = part.meta.get("arguments")
    if not isinstance(arguments, dict):
        raise invalid_request_error("Gemini tool_call.meta.arguments must be an object")
    return (name.strip(), arguments)


def _require_tool_result_meta(part: Part) -> tuple[str, Any]:
    name = part.meta.get("name")
    if not isinstance(name, str) or not name.strip():
        raise invalid_request_error("tool_result.meta.name must be a non-empty string")
    return (name.strip(), part.meta.get("result"))


def _part_to_gemini_part(
    adapter: GeminiAdapter, part: Part, *, timeout_ms: int | None
) -> dict[str, Any]:
    if part.type == "text":
        return {"text": part.require_text()}
    if part.type == "tool_call":
        name, arguments = _require_tool_call_meta(part)
        return {"functionCall": {"name": name, "args": arguments}}
    if part.type == "tool_result":
        name, result = _require_tool_result_meta(part)
        response = result if isinstance(result, dict) else {"result": result}
        return {"functionResponse": {"name": name, "response": response}}
    if part.type in {"image", "audio", "video"}:
        source = part.require_source()
        mime_type = part.mime_type
        if mime_type is None and isinstance(source, PartSourcePath):
            mime_type = detect_mime_type(source.path)
        if not mime_type:
            raise invalid_request_error(
                f"{part.type} requires mime_type (or path extension)"
            )

        if isinstance(source, PartSourceRef):
            return {"fileData": {"mimeType": mime_type, "fileUri": source.id}}

        if isinstance(source, PartSourceUrl):
            tmp = download_to_tempfile(
                url=source.url,
                timeout_ms=timeout_ms,
                max_bytes=None,
                proxy_url=adapter.proxy_url,
            )
            try:
                return _upload_or_inline(
                    adapter,
                    part,
                    file_path=tmp,
                    mime_type=mime_type,
                    timeout_ms=timeout_ms,
                )
            finally:
                try:
                    os.unlink(tmp)
                except OSError:
                    pass

        if isinstance(source, PartSourcePath):
            return _upload_or_inline(
                adapter,
                part,
                file_path=source.path,
                mime_type=mime_type,
                timeout_ms=timeout_ms,
            )

        if isinstance(source, PartSourceBytes) and source.encoding == "base64":
            b64 = source.data
            if not isinstance(b64, str) or not b64:
                raise invalid_request_error(
                    f"{part.type} base64 data must be non-empty"
                )
            return {"inlineData": {"mimeType": mime_type, "data": b64}}

        assert isinstance(source, PartSourceBytes)
        data = source.data
        if not isinstance(data, bytes):
            raise invalid_request_error(f"{part.type} bytes data must be bytes")
        if len(data) <= 20 * 1024 * 1024 and part.type == "image":
            return {
                "inlineData": {"mimeType": mime_type, "data": bytes_to_base64(data)}
            }
        with tempfile.NamedTemporaryFile(
            prefix="genaisdk-", suffix=".bin", delete=False
        ) as f:
            f.write(data)
            tmp = f.name
        try:
            return _upload_or_inline(
                adapter, part, file_path=tmp, mime_type=mime_type, timeout_ms=timeout_ms
            )
        finally:
            try:
                os.unlink(tmp)
            except OSError:
                pass

    if part.type == "embedding":
        raise not_supported_error(
            "embedding is not a valid Gemini generateContent input part"
        )
    raise not_supported_error(f"unsupported part type: {part.type}")


def _upload_or_inline(
    adapter: GeminiAdapter,
    part: Part,
    *,
    file_path: str,
    mime_type: str,
    timeout_ms: int | None,
) -> dict[str, Any]:
    max_inline = 20 * 1024 * 1024
    if part.type == "image":
        st = os.stat(file_path)
        if st.st_size <= max_inline:
            data = file_to_bytes(file_path, max_inline)
            return {
                "inlineData": {"mimeType": mime_type, "data": bytes_to_base64(data)}
            }
    if part.type == "audio":
        st = os.stat(file_path)
        if st.st_size <= max_inline:
            data = file_to_bytes(file_path, max_inline)
            return {
                "inlineData": {"mimeType": mime_type, "data": bytes_to_base64(data)}
            }
    if part.type == "video":
        st = os.stat(file_path)
        if not adapter.supports_file_upload:
            if st.st_size <= max_inline:
                data = file_to_bytes(file_path, max_inline)
                return {
                    "inlineData": {"mimeType": mime_type, "data": bytes_to_base64(data)}
                }
            raise not_supported_error(
                f"file too large for inline bytes ({st.st_size} > {max_inline}); use url/ref instead"
            )

    file_uri = adapter._upload_to_file_uri(
        file_path, mime_type=mime_type, timeout_ms=timeout_ms
    )
    return {"fileData": {"mimeType": mime_type, "fileUri": file_uri}}


def _gemini_part_to_parts(part: dict[str, Any], *, provider_name: str) -> list[Part]:
    if "text" in part and isinstance(part["text"], str):
        text = part["text"]
        if provider_name.startswith("tuzi"):
            urls = [
                m.group(1).strip()
                for m in _MD_IMAGE_URL_RE.finditer(text)
                if m.group(1).strip()
            ]
            if urls:
                out: list[Part] = []
                remaining = _MD_IMAGE_URL_RE.sub("", text).strip()
                if remaining:
                    out.append(Part.from_text(remaining))
                out.extend(
                    [Part(type="image", source=PartSourceUrl(url=u)) for u in urls]
                )
                return out
        return [Part.from_text(text)]
    fc_obj = part.get("functionCall") or part.get("function_call")
    if isinstance(fc_obj, dict):
        fc = fc_obj
        name = fc.get("name")
        args = fc.get("args")
        if isinstance(name, str) and name and isinstance(args, dict):
            return [Part.tool_call(name=name, arguments=args)]
        return []
    if "inlineData" in part and isinstance(part["inlineData"], dict):
        blob = part["inlineData"]
        mime = blob.get("mimeType")
        data_b64 = blob.get("data")
        if not isinstance(mime, str) or not isinstance(data_b64, str):
            return []
        if mime.startswith("image/"):
            return [
                Part(
                    type="image",
                    mime_type=mime,
                    source=PartSourceBytes(data=data_b64, encoding="base64"),
                )
            ]
        if mime.startswith("audio/"):
            return [
                Part(
                    type="audio",
                    mime_type=mime,
                    source=PartSourceBytes(data=data_b64, encoding="base64"),
                )
            ]
        if mime.startswith("video/"):
            return [
                Part(
                    type="video",
                    mime_type=mime,
                    source=PartSourceBytes(data=data_b64, encoding="base64"),
                )
            ]
        return [
            Part(
                type="file",
                mime_type=mime,
                source=PartSourceBytes(data=data_b64, encoding="base64"),
            )
        ]
    if "fileData" in part and isinstance(part["fileData"], dict):
        fd = part["fileData"]
        uri = fd.get("fileUri")
        mime = fd.get("mimeType")
        if not isinstance(uri, str) or not uri:
            return []
        mime_s = mime if isinstance(mime, str) else None
        kind: PartType = "file"
        if mime_s and mime_s.startswith("image/"):
            kind = "image"
        elif mime_s and mime_s.startswith("audio/"):
            kind = "audio"
        elif mime_s and mime_s.startswith("video/"):
            kind = "video"
        return [
            Part(
                type=kind,
                mime_type=mime_s,
                source=PartSourceRef(provider=provider_name, id=uri),
            )
        ]
    return []


def _usage_from_gemini(usage: Any) -> Usage | None:
    if not isinstance(usage, dict):
        return None
    return Usage(
        input_tokens=usage.get("promptTokenCount"),
        output_tokens=usage.get("candidatesTokenCount"),
        total_tokens=usage.get("totalTokenCount"),
    )


def _extract_veo_video_uri(response: Any) -> str | None:
    if not isinstance(response, dict):
        return None
    gvr = response.get("generateVideoResponse")
    if isinstance(gvr, dict):
        uri = _extract_veo_video_uri_from_samples(gvr.get("generatedSamples"))
        if uri:
            return uri
    uri = _extract_veo_video_uri_from_samples(response.get("generatedSamples"))
    if uri:
        return uri
    videos = response.get("generatedVideos") or response.get("generated_videos")
    if isinstance(videos, list) and videos:
        v0 = videos[0]
        if isinstance(v0, dict):
            v = v0.get("video")
            if isinstance(v, dict):
                uri = v.get("uri") or v.get("downloadUri") or v.get("fileUri")
                if isinstance(uri, str) and uri:
                    return uri
    return None


def _extract_veo_video_uri_from_samples(samples: Any) -> str | None:
    if not isinstance(samples, list) or not samples:
        return None
    s0 = samples[0]
    if not isinstance(s0, dict):
        return None
    v = s0.get("video")
    if isinstance(v, dict):
        uri = v.get("uri") or v.get("downloadUri") or v.get("fileUri")
        if isinstance(uri, str) and uri:
            return uri
    uri = s0.get("uri")
    if isinstance(uri, str) and uri:
        return uri
    return None


def _first_candidate_text(obj: dict[str, Any]) -> str | None:
    candidates = obj.get("candidates")
    if not isinstance(candidates, list) or not candidates:
        return None
    cand0 = candidates[0]
    if not isinstance(cand0, dict):
        return None
    content = cand0.get("content")
    if not isinstance(content, dict):
        return None
    parts = content.get("parts")
    if not isinstance(parts, list):
        return None
    for p in parts:
        if not isinstance(p, dict):
            continue
        text = p.get("text")
        if isinstance(text, str) and text:
            return text
    return None


def _extract_tuzi_task_id(text: str) -> str | None:
    m = _TUZI_TASK_ID_RE.search(text)
    if m is None:
        return None
    tid = m.group(1).strip()
    return tid or None


def _extract_first_mp4_url(text: str) -> str | None:
    m = _MP4_URL_RE.search(text)
    if m is None:
        return None
    url = m.group(0).strip()
    return url or None


def _poll_tuzi_video_mp4(
    *, task_id: str, deadline: float, proxy_url: str | None
) -> str | None:
    poll_url = f"{_ASYNCDATA_BASE_URL}/source/{task_id}"
    while True:
        remaining_ms = int((deadline - time.time()) * 1000)
        if remaining_ms <= 0:
            return None
        obj = request_json(
            method="GET",
            url=poll_url,
            headers=None,
            json_body=None,
            timeout_ms=min(30_000, remaining_ms),
            proxy_url=proxy_url,
        )
        content = obj.get("content")
        if isinstance(content, str) and content:
            mp4_url = _extract_first_mp4_url(content)
            if mp4_url:
                return mp4_url
        time.sleep(min(2.0, max(0.0, deadline - time.time())))
