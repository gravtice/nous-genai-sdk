from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any, Iterator
from uuid import uuid4

from .._internal.errors import invalid_request_error, not_supported_error, provider_error
from .._internal.http import request_json
from ..types import Capability, GenerateEvent, GenerateRequest, GenerateResponse
from ..types import JobInfo, Message, Part, PartSourceUrl
from .openai import OpenAIAdapter


@dataclass(frozen=True, slots=True)
class VolcengineAdapter:
    """
    Volcengine Ark (Doubao).

    Supported in this SDK:
    - chat (text/image -> text), stream supported
    - embeddings (text -> embedding) for text embedding models
    - image generation (text -> image) for Seedream models
    - video generation (text -> video) for Seedance models via content generation tasks
    """

    openai: OpenAIAdapter

    def capabilities(self, model_id: str) -> Capability:
        if _is_text_embedding_model(model_id):
            return Capability(
                input_modalities={"text"},
                output_modalities={"embedding"},
                supports_stream=False,
                supports_job=False,
                supports_tools=False,
                supports_json_schema=False,
            )
        if _is_seedream_model(model_id):
            return Capability(
                input_modalities={"text"},
                output_modalities={"image"},
                supports_stream=False,
                supports_job=False,
                supports_tools=False,
                supports_json_schema=False,
            )
        if _is_seedance_video_model(model_id):
            return Capability(
                input_modalities={"text"},
                output_modalities={"video"},
                supports_stream=False,
                supports_job=True,
                supports_tools=False,
                supports_json_schema=False,
            )
        return Capability(
            input_modalities={"text", "image"},
            output_modalities={"text"},
            supports_stream=True,
            supports_job=False,
            supports_tools=True,
            supports_json_schema=True,
        )

    def list_models(self, *, timeout_ms: int | None = None) -> list[str]:
        return self.openai.list_models(timeout_ms=timeout_ms)

    def generate(self, request: GenerateRequest, *, stream: bool) -> GenerateResponse | Iterator[GenerateEvent]:
        modalities = set(request.output.modalities)
        model_id = request.model_id()

        if modalities == {"embedding"}:
            if stream:
                raise not_supported_error("Volcengine embeddings do not support streaming")
            if not _is_text_embedding_model(model_id):
                raise not_supported_error("Volcengine embedding requires a text embedding model")
            return self.openai.generate(request, stream=False)

        if modalities == {"image"}:
            if stream:
                raise not_supported_error("Volcengine image generation does not support streaming")
            if not _is_seedream_model(model_id):
                raise not_supported_error("Volcengine image generation requires a Seedream model")
            return self.openai.generate(request, stream=False)

        if modalities == {"video"}:
            if stream:
                raise not_supported_error("Volcengine video generation does not support streaming")
            return self._video(request, model_id=model_id)

        if modalities != {"text"}:
            raise not_supported_error(
                "Volcengine only supports text chat, embeddings, Seedream images, and Seedance video in this SDK"
            )
        if _is_text_embedding_model(model_id):
            raise not_supported_error("Volcengine embedding models must be called with output.modalities=['embedding']")
        if _is_seedream_model(model_id):
            raise not_supported_error("Volcengine Seedream models must be called with output.modalities=['image']")
        if _is_seedance_video_model(model_id):
            raise not_supported_error("Volcengine Seedance models must be called with output.modalities=['video']")
        if _has_audio_input(request):
            raise not_supported_error("Volcengine chat input does not support audio in this SDK")
        return self.openai.generate(request, stream=stream)

    def _video(self, request: GenerateRequest, *, model_id: str) -> GenerateResponse:
        if not _is_seedance_video_model(model_id):
            raise not_supported_error(
                'Volcengine video generation requires model like "volcengine:doubao-seedance-1-0-lite-t2v-250428"'
            )

        prompt = _single_text_prompt(request)
        body: dict[str, Any] = {"model": model_id, "content": [{"type": "text", "text": prompt}]}

        opts = request.provider_options.get("volcengine")
        if isinstance(opts, dict):
            if "model" in opts and opts["model"] != model_id:
                raise invalid_request_error("provider_options cannot override model")
            body.update({k: v for k, v in opts.items() if k != "model"})

        budget_ms = 120_000 if request.params.timeout_ms is None else request.params.timeout_ms
        deadline = time.time() + max(1, budget_ms) / 1000.0
        obj = request_json(
            method="POST",
            url=f"{self.openai.base_url}/contents/generations/tasks",
            headers=_headers(self.openai.api_key, request=request),
            json_body=body,
            timeout_ms=min(30_000, max(1, budget_ms)),
            proxy_url=self.openai.proxy_url,
        )
        task_id = obj.get("id")
        if not isinstance(task_id, str) or not task_id:
            raise provider_error("volcengine video response missing task id")

        if not request.wait:
            return GenerateResponse(
                id=f"sdk_{uuid4().hex}",
                provider="volcengine",
                model=f"volcengine:{model_id}",
                status="running",
                job=JobInfo(job_id=task_id, poll_after_ms=1_000),
            )

        final = _wait_task_done(
            base_url=self.openai.base_url,
            api_key=self.openai.api_key,
            task_id=task_id,
            deadline=deadline,
            proxy_url=self.openai.proxy_url,
        )
        status = final.get("status")
        if status != "succeeded":
            if status == "failed":
                raise provider_error(f"volcengine video generation failed: {final}")
            return GenerateResponse(
                id=f"sdk_{uuid4().hex}",
                provider="volcengine",
                model=f"volcengine:{model_id}",
                status="running",
                job=JobInfo(job_id=task_id, poll_after_ms=1_000),
            )

        content = final.get("content")
        video_url = content.get("video_url") if isinstance(content, dict) else None
        if not isinstance(video_url, str) or not video_url:
            raise provider_error("volcengine video task missing video_url")
        part = Part(type="video", mime_type="video/mp4", source=PartSourceUrl(url=video_url))
        return GenerateResponse(
            id=f"sdk_{uuid4().hex}",
            provider="volcengine",
            model=f"volcengine:{model_id}",
            status="completed",
            output=[Message(role="assistant", content=[part])],
            usage=None,
        )


def _is_text_embedding_model(model_id: str) -> bool:
    model_id = model_id.lower()
    return "embedding" in model_id and "text" in model_id


def _is_seedream_model(model_id: str) -> bool:
    return "seedream" in model_id.lower()


def _is_seedance_video_model(model_id: str) -> bool:
    mid = model_id.lower()
    return "seedance" in mid and ("t2v" in mid or "i2v" in mid)


def _single_text_prompt(request: GenerateRequest) -> str:
    texts: list[str] = []
    for m in request.input:
        for p in m.content:
            if p.type != "text":
                raise invalid_request_error("this operation requires exactly one text part")
            t = p.require_text().strip()
            if t:
                texts.append(t)
    if len(texts) != 1:
        raise invalid_request_error("this operation requires exactly one text part")
    return texts[0]


def _headers(api_key: str, *, request: GenerateRequest | None = None) -> dict[str, str]:
    headers = {"Authorization": f"Bearer {api_key}"}
    if request and request.params.idempotency_key:
        headers["Idempotency-Key"] = request.params.idempotency_key
    return headers


def _wait_task_done(
    *, base_url: str, api_key: str, task_id: str, deadline: float, proxy_url: str | None
) -> dict[str, Any]:
    url = f"{base_url}/contents/generations/tasks/{task_id}"
    while True:
        remaining_ms = int((deadline - time.time()) * 1000)
        if remaining_ms <= 0:
            break
        obj = request_json(
            method="GET",
            url=url,
            headers=_headers(api_key),
            timeout_ms=min(30_000, remaining_ms),
            proxy_url=proxy_url,
        )
        st = obj.get("status")
        if st in {"succeeded", "failed", "cancelled"}:
            return obj
        time.sleep(1.0)
    return {"id": task_id, "status": "running"}


def _has_audio_input(request: GenerateRequest) -> bool:
    for m in request.input:
        for p in m.content:
            if p.type == "audio":
                return True
    return False
