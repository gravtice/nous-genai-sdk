from __future__ import annotations

import json
import re
import time
from dataclasses import dataclass, replace
from typing import Iterator
from uuid import uuid4

from .._internal.errors import GenAIError, invalid_request_error, not_supported_error, provider_error
from .._internal.http import request_json
from ..types import Capability, GenerateEvent, GenerateRequest, GenerateResponse, JobInfo, Message, Part, PartSourceUrl
from .anthropic import AnthropicAdapter
from .gemini import GeminiAdapter
from .openai import OpenAIAdapter

_ASYNCDATA_BASE_URL = "https://asyncdata.net"
_ASYNCDATA_PRO_BASE_URL = "https://pro.asyncdata.net"

_DEEPSEARCH_MODELS = frozenset({
    "gemini-2.5-flash-deepsearch",
    "gemini-2.5-flash-deepsearch-async",
    "gemini-2.5-pro-deepsearch",
    "gemini-2.5-pro-deepsearch-async",
    "gemini-3-pro-deepsearch",
    "gemini-3-pro-deepsearch-async",
})


def _is_deepsearch_model(model_id: str) -> bool:
    return model_id.lower().strip() in _DEEPSEARCH_MODELS


_MP4_URL_RE = re.compile(r"https?://[^\s\"'<>]+?\.mp4(?:\?[^\s\"'<>]*)?", re.IGNORECASE)
_AUDIO_URL_RE = re.compile(
    r"https?://[^\s\"'<>]+?\.(?:mp3|wav|m4a|aac|flac|ogg|opus)(?:\?[^\s\"'<>]*)?",
    re.IGNORECASE,
)

_SUNO_WORKFLOW_MODELS = frozenset(
    {
        "suno-all-stems",
        "suno-continue",
        "suno-continue-uploaded",
        "suno-infill",
        "suno-infill-uploaded",
        "suno-midi",
        "suno-overpainting",
        "suno-remix",
        "suno-remix-uploaded",
        "suno-rewrite",
        "suno-tags",
        "suno-vocal-stems",
        "suno_act_midi",
        "suno_act_mp4",
        "suno_act_stems",
        "suno_act_tags",
        "suno_act_timing",
        "suno_act_wav",
        "suno_concat",
        "suno_persona_create",
        "suno_uploads",
    }
)


def _extract_first_url(pattern: re.Pattern[str], text: str) -> str | None:
    m = pattern.search(text)
    if m is None:
        return None
    return m.group(0)


def _closest_kling_duration(duration_sec: int | None) -> str:
    if duration_sec is None:
        return "5"
    try:
        sec = int(duration_sec)
    except Exception:
        return "5"
    return "5" if sec <= 5 else "10"


def _sora_api_model_and_prompt_suffix(model_id: str) -> tuple[str, str | None]:
    mid = model_id.strip()
    mid_l = mid.lower()
    if mid_l in {"sora-2", "sora-2-pro", "sora-2-character", "sora-2-pro-character"}:
        return mid, None
    if mid_l.startswith("sora-") and ":" in mid_l:
        parts = mid_l.split("-")
        ratio = parts[1] if len(parts) > 1 else ""
        res = parts[2] if len(parts) > 2 else ""
        dur = parts[3] if len(parts) > 3 else ""
        api_model = "sora-2-pro" if "720p" in res else "sora-2"
        suffix_parts: list[str] = []
        if ratio:
            suffix_parts.append(ratio)
        if res:
            suffix_parts.append(res)
        if dur:
            suffix_parts.append(dur if dur.endswith("s") else f"{dur}s")
        return api_model, " ".join(suffix_parts) if suffix_parts else None
    return mid, None


@dataclass(frozen=True, slots=True)
class TuziAdapter:
    """
    Tuzi exposes multiple protocols (OpenAI-compatible, Gemini v1beta, Anthropic /v1/messages)
    under a single API key. Route by model_id.

    For deepsearch models, uses the asyncdata.net async API.
    """

    openai: OpenAIAdapter | None
    gemini: GeminiAdapter | None
    anthropic: AnthropicAdapter | None
    proxy_url: str | None = None

    def capabilities(self, model_id: str) -> Capability:
        mid_l = model_id.lower().strip()
        if mid_l in {"kling_image", "seededit"}:
            return Capability(
                input_modalities={"text", "image"},
                output_modalities={"image"},
                supports_stream=False,
                supports_job=(mid_l == "kling_image"),
                supports_tools=False,
                supports_json_schema=False,
            )
        if mid_l in _SUNO_WORKFLOW_MODELS:
            return Capability(
                input_modalities={"text"},
                output_modalities={"audio"},
                supports_stream=False,
                supports_job=True,
                supports_tools=False,
                supports_json_schema=False,
            )
        if mid_l == "suno_lyrics":
            return Capability(
                input_modalities={"text"},
                output_modalities={"text"},
                supports_stream=False,
                supports_job=True,
                supports_tools=False,
                supports_json_schema=False,
            )
        if mid_l == "suno_music" or (mid_l.startswith("chirp-") and mid_l != "chirp-v3"):
            return Capability(
                input_modalities={"text"},
                output_modalities={"audio"},
                supports_stream=False,
                supports_job=True,
                supports_tools=False,
                supports_json_schema=False,
            )
        if _is_deepsearch_model(model_id):
            return Capability(
                input_modalities={"text"},
                output_modalities={"text"},
                supports_stream=False,
                supports_job=True,
                supports_tools=False,
                supports_json_schema=False,
            )
        return self._route(model_id).capabilities(model_id)

    def generate(self, request: GenerateRequest, *, stream: bool) -> GenerateResponse | Iterator[GenerateEvent]:
        model_id = request.model_id()
        mid_l = model_id.lower().strip()
        modalities = set(request.output.modalities)

        if modalities == {"video"} and mid_l.startswith("pika-"):
            raise not_supported_error("tuzi pika endpoints are not available on api.tu-zi.com (returns HTML)")

        if modalities == {"video"} and "seedance" in mid_l:
            raise not_supported_error("doubao seedance video is not supported on tuzi-web (upstream returns multipart: NextPart: EOF)")

        if modalities == {"video"} and mid_l.startswith("kling"):
            if stream:
                raise invalid_request_error("kling video generation does not support streaming")
            return self._kling_text2video(request, model_id=model_id)

        if modalities == {"video"} and mid_l.startswith("sora-"):
            if stream:
                raise invalid_request_error("sora video generation does not support streaming")
            return self._async_chat_video(request, model_id=model_id)

        if modalities == {"video"} and mid_l.startswith("runway-"):
            raise not_supported_error("tuzi runway endpoints are not available on api.tu-zi.com (returns HTML)")

        if modalities == {"image"} and mid_l in {"kling_image", "seededit"}:
            if stream:
                raise invalid_request_error(f"{mid_l} image generation does not support streaming")
            if mid_l == "kling_image" and self._has_image_input(request):
                return self._route(model_id).generate(request, stream=False)
            if mid_l == "kling_image":
                return self._kling_text2image(request, model_id=model_id)
            return self._seededit(request, model_id=model_id)

        if modalities == {"text"} and mid_l == "suno_lyrics":
            if stream:
                raise invalid_request_error("suno lyrics generation does not support streaming")
            return self._suno_lyrics(request)

        if modalities == {"audio"} and mid_l in _SUNO_WORKFLOW_MODELS:
            if stream:
                raise invalid_request_error("suno workflow endpoints do not support streaming")
            return self._suno_workflow(request, model_id=model_id)

        if modalities == {"audio"} and (mid_l == "suno_music" or (mid_l.startswith("chirp-") and mid_l != "chirp-v3")):
            if stream:
                raise invalid_request_error("suno music generation does not support streaming")
            return self._suno_music(request, model_id=model_id)

        if _is_deepsearch_model(model_id):
            if stream:
                raise invalid_request_error("deepsearch models do not support streaming; use stream=False")
            return self._deepsearch(request, model_id=model_id)
        return self._route(model_id).generate(request, stream=stream)

    def _has_image_input(self, request: GenerateRequest) -> bool:
        for msg in request.input:
            for part in msg.content:
                if part.type == "image":
                    return True
        return False

    def _base_host(self) -> str:
        if self.gemini is not None and self.gemini.base_url:
            return self.gemini.base_url.rstrip("/")
        if self.anthropic is not None and self.anthropic.base_url:
            return self.anthropic.base_url.rstrip("/")
        if self.openai is not None and self.openai.base_url:
            base = self.openai.base_url.rstrip("/")
            if base.endswith("/v1"):
                return base[:-3]
            return base
        raise invalid_request_error("tuzi base url not configured")

    def _bearer_headers(self) -> dict[str, str]:
        if self.openai is not None and self.openai.api_key:
            return {"Authorization": f"Bearer {self.openai.api_key}"}
        if self.gemini is not None and self.gemini.api_key:
            return {"Authorization": f"Bearer {self.gemini.api_key}"}
        if self.anthropic is not None and self.anthropic.api_key:
            return {"Authorization": f"Bearer {self.anthropic.api_key}"}
        raise invalid_request_error("tuzi api key not configured")

    def _single_text_prompt(self, request: GenerateRequest) -> str:
        texts: list[str] = []
        for msg in request.input:
            for part in msg.content:
                if part.type != "text":
                    continue
                t = part.require_text().strip()
                if t:
                    texts.append(t)
        if len(texts) != 1:
            raise invalid_request_error("this operation requires exactly one text part")
        return texts[0]

    def _text_prompt_or_none(self, request: GenerateRequest) -> str | None:
        chunks: list[str] = []
        for msg in request.input:
            for part in msg.content:
                if part.type != "text":
                    continue
                t = part.require_text().strip()
                if t:
                    chunks.append(t)
        if not chunks:
            return None
        return "\n".join(chunks).strip()

    def _seededit(self, request: GenerateRequest, *, model_id: str) -> GenerateResponse:
        if self.openai is None:
            raise invalid_request_error("tuzi openai adapter not configured")
        if not self._has_image_input(request):
            raise invalid_request_error("seededit requires image input")
        req = replace(request, model="tuzi-web:api-images-seededit")
        resp = self.openai.generate(req, stream=False)
        assert isinstance(resp, GenerateResponse)
        return replace(resp, model=f"tuzi-web:{model_id}")

    def _kling_text2image(self, request: GenerateRequest, *, model_id: str) -> GenerateResponse:
        prompt = self._single_text_prompt(request)
        host = self._base_host()
        body: dict[str, object] = {
            "prompt": prompt,
            "negative_prompt": "",
            "aspect_ratio": "1:1",
            "callback_url": "",
        }
        opts = request.provider_options.get("tuzi-web")
        if isinstance(opts, dict):
            for k, v in opts.items():
                if k in body:
                    raise invalid_request_error(f"provider_options cannot override {k}")
                body[k] = v

        obj = request_json(
            method="POST",
            url=f"{host}/kling/v1/images/text2image",
            headers=self._bearer_headers(),
            json_body=body,
            timeout_ms=max(request.params.timeout_ms or 60_000, 60_000),
            proxy_url=self.proxy_url,
        )
        data = obj.get("data")
        task_id = data.get("task_id") if isinstance(data, dict) else None
        if not isinstance(task_id, str) or not task_id:
            raise provider_error("kling submit missing task_id")

        if not request.wait:
            return GenerateResponse(
                id=f"sdk_{uuid4().hex}",
                provider="tuzi-web",
                model=f"tuzi-web:{model_id}",
                status="running",
                job=JobInfo(job_id=task_id, poll_after_ms=1_000),
            )

        poll_url = f"{host}/kling/v1/images/text2image/{task_id}"
        budget_ms = 120_000 if request.params.timeout_ms is None else request.params.timeout_ms
        deadline = time.time() + max(1, budget_ms) / 1000.0
        while True:
            remaining_ms = int((deadline - time.time()) * 1000)
            if remaining_ms <= 0:
                break
            obj = request_json(
                method="GET",
                url=poll_url,
                headers=self._bearer_headers(),
                json_body=None,
                timeout_ms=min(30_000, remaining_ms),
                proxy_url=self.proxy_url,
            )
            data = obj.get("data")
            if not isinstance(data, dict):
                time.sleep(1.0)
                continue
            status = data.get("task_status")
            if status == "failed":
                raise provider_error(f"kling task failed: {data.get('task_status_msg')}")
            if status == "succeed":
                task_result = data.get("task_result")
                if isinstance(task_result, dict):
                    images = task_result.get("images")
                    if isinstance(images, list) and images:
                        first = images[0]
                        if isinstance(first, str) and first:
                            part = Part(type="image", source=PartSourceUrl(url=first))
                            return GenerateResponse(
                                id=f"sdk_{uuid4().hex}",
                                provider="tuzi-web",
                                model=f"tuzi-web:{model_id}",
                                status="completed",
                                output=[Message(role="assistant", content=[part])],
                            )
                        if isinstance(first, dict):
                            u = first.get("url")
                            if isinstance(u, str) and u:
                                part = Part(type="image", source=PartSourceUrl(url=u))
                                return GenerateResponse(
                                    id=f"sdk_{uuid4().hex}",
                                    provider="tuzi-web",
                                    model=f"tuzi-web:{model_id}",
                                    status="completed",
                                    output=[Message(role="assistant", content=[part])],
                                )
                raise provider_error("kling task succeeded but missing image url")
            time.sleep(min(1.0, max(0.0, deadline - time.time())))

        return GenerateResponse(
            id=f"sdk_{uuid4().hex}",
            provider="tuzi-web",
            model=f"tuzi-web:{model_id}",
            status="running",
            job=JobInfo(job_id=task_id, poll_after_ms=1_000),
        )

    def _async_chat_video(self, request: GenerateRequest, *, model_id: str) -> GenerateResponse:
        if self.openai is None:
            raise invalid_request_error("NOUS_GENAI_TUZI_OPENAI_API_KEY required for async chat video models")

        api_model, suffix = _sora_api_model_and_prompt_suffix(model_id)
        messages = []
        for msg in request.input:
            role = msg.role if msg.role in {"system", "assistant"} else "user"
            content = "".join(p.require_text() for p in msg.content if p.type == "text").strip()
            if not content:
                continue
            if suffix and role == "user":
                content = f"{content} {suffix}".strip()
                suffix = None
            messages.append({"role": role, "content": content})
        if not messages:
            raise invalid_request_error("video generation requires at least one text message")

        original_url = f"{self.openai.base_url}/chat/completions"
        submit_url = f"{_ASYNCDATA_BASE_URL}/tran/{original_url}"
        obj = request_json(
            method="POST",
            url=submit_url,
            headers=self._bearer_headers(),
            json_body={"model": api_model, "messages": messages},
            timeout_ms=max(request.params.timeout_ms or 120_000, 120_000),
            proxy_url=self.proxy_url,
        )

        task_id = obj.get("id")
        if not isinstance(task_id, str) or not task_id:
            raise provider_error("async video submit missing id")
        source_url = obj.get("source_url")

        if not request.wait:
            return GenerateResponse(
                id=f"sdk_{uuid4().hex}",
                provider="tuzi-web",
                model=f"tuzi-web:{model_id}",
                status="running",
                job=JobInfo(job_id=task_id, poll_after_ms=2_000),
            )

        content = self._poll_asyncdata_content(task_id=task_id, source_url=source_url, timeout_ms=request.params.timeout_ms)
        if content is None:
            return GenerateResponse(
                id=f"sdk_{uuid4().hex}",
                provider="tuzi-web",
                model=f"tuzi-web:{model_id}",
                status="running",
                job=JobInfo(job_id=task_id, poll_after_ms=2_000),
            )

        mp4 = _extract_first_url(_MP4_URL_RE, content)
        if not mp4:
            raise provider_error("async video completed but no mp4 url found")
        part = Part(type="video", mime_type="video/mp4", source=PartSourceUrl(url=mp4))
        return GenerateResponse(
            id=f"sdk_{uuid4().hex}",
            provider="tuzi-web",
            model=f"tuzi-web:{model_id}",
            status="completed",
            output=[Message(role="assistant", content=[part])],
        )

    def _poll_asyncdata_content(self, *, task_id: str, source_url: object, timeout_ms: int | None) -> str | None:
        poll_urls: list[str] = []
        if isinstance(source_url, str) and source_url.strip():
            poll_urls.append(source_url.strip())
        poll_urls.extend([f"{_ASYNCDATA_BASE_URL}/source/{task_id}", f"{_ASYNCDATA_PRO_BASE_URL}/source/{task_id}"])

        budget_ms = 120_000 if timeout_ms is None else timeout_ms
        deadline = time.time() + max(1, budget_ms) / 1000.0
        while True:
            remaining_ms = int((deadline - time.time()) * 1000)
            if remaining_ms <= 0:
                return None
            for url in poll_urls:
                obj = request_json(
                    method="GET",
                    url=url,
                    headers=None,
                    json_body=None,
                    timeout_ms=min(30_000, remaining_ms),
                    proxy_url=self.proxy_url,
                )
                content = obj.get("content")
                if isinstance(content, str) and content:
                    return content
            time.sleep(min(2.0, max(0.0, deadline - time.time())))

    def _kling_text2video(self, request: GenerateRequest, *, model_id: str) -> GenerateResponse:
        prompt = self._single_text_prompt(request)
        host = self._base_host()
        video = request.output.video
        body: dict[str, object] = {
            "prompt": prompt,
            "negative_prompt": "",
            "aspect_ratio": (video.aspect_ratio if video and video.aspect_ratio else "16:9"),
            "duration": _closest_kling_duration(video.duration_sec if video else None),
            "callback_url": "",
        }
        obj = request_json(
            method="POST",
            url=f"{host}/kling/v1/videos/text2video",
            headers=self._bearer_headers(),
            json_body=body,
            timeout_ms=max(request.params.timeout_ms or 60_000, 60_000),
            proxy_url=self.proxy_url,
        )
        data = obj.get("data")
        task_id = data.get("task_id") if isinstance(data, dict) else None
        if not isinstance(task_id, str) or not task_id:
            raise provider_error("kling submit missing task_id")

        if not request.wait:
            return GenerateResponse(
                id=f"sdk_{uuid4().hex}",
                provider="tuzi-web",
                model=f"tuzi-web:{model_id}",
                status="running",
                job=JobInfo(job_id=task_id, poll_after_ms=1_000),
            )

        poll_url = f"{host}/kling/v1/videos/text2video/{task_id}"
        budget_ms = 120_000 if request.params.timeout_ms is None else request.params.timeout_ms
        deadline = time.time() + max(1, budget_ms) / 1000.0
        while True:
            remaining_ms = int((deadline - time.time()) * 1000)
            if remaining_ms <= 0:
                break
            obj = request_json(
                method="GET",
                url=poll_url,
                headers=self._bearer_headers(),
                json_body=None,
                timeout_ms=min(30_000, remaining_ms),
                proxy_url=self.proxy_url,
            )
            data = obj.get("data")
            if not isinstance(data, dict):
                time.sleep(1.0)
                continue
            status = data.get("task_status")
            if status == "failed":
                raise provider_error(f"kling task failed: {data.get('task_status_msg')}")
            if status == "succeed":
                task_result = data.get("task_result")
                if isinstance(task_result, dict):
                    videos = task_result.get("videos")
                    if isinstance(videos, list) and videos:
                        first = videos[0]
                        if isinstance(first, dict):
                            u = first.get("url")
                            if isinstance(u, str) and u:
                                part = Part(type="video", mime_type="video/mp4", source=PartSourceUrl(url=u))
                                return GenerateResponse(
                                    id=f"sdk_{uuid4().hex}",
                                    provider="tuzi-web",
                                    model=f"tuzi-web:{model_id}",
                                    status="completed",
                                    output=[Message(role="assistant", content=[part])],
                                )
                raise provider_error("kling task succeeded but missing video url")
            time.sleep(min(1.0, max(0.0, deadline - time.time())))

        return GenerateResponse(
            id=f"sdk_{uuid4().hex}",
            provider="tuzi-web",
            model=f"tuzi-web:{model_id}",
            status="running",
            job=JobInfo(job_id=task_id, poll_after_ms=1_000),
        )

    def _suno_lyrics(self, request: GenerateRequest) -> GenerateResponse:
        prompt = self._single_text_prompt(request)
        host = self._base_host()
        obj = request_json(
            method="POST",
            url=f"{host}/suno/submit/lyrics",
            headers=self._bearer_headers(),
            json_body={"prompt": prompt},
            timeout_ms=max(request.params.timeout_ms or 60_000, 60_000),
            proxy_url=self.proxy_url,
        )
        task_id = obj.get("data")
        if not isinstance(task_id, str) or not task_id:
            raise provider_error("suno lyrics submit missing task id")
        return self._suno_wait_fetch_text(
            task_id=task_id,
            model_id="suno_lyrics",
            timeout_ms=request.params.timeout_ms,
            wait=request.wait,
        )

    def _suno_music(self, request: GenerateRequest, *, model_id: str) -> GenerateResponse:
        prompt = self._single_text_prompt(request)
        host = self._base_host()
        mv = model_id if model_id.lower().startswith("chirp-") else "chirp-v3-5"
        body: dict[str, object] = {
            "prompt": prompt,
            "tags": "",
            "mv": mv,
            "title": "suno",
            "infill_start_s": None,
            "infill_end_s": None,
        }
        obj = request_json(
            method="POST",
            url=f"{host}/suno/submit/music",
            headers=self._bearer_headers(),
            json_body=body,
            timeout_ms=max(request.params.timeout_ms or 60_000, 60_000),
            proxy_url=self.proxy_url,
        )
        task_id = obj.get("data")
        if not isinstance(task_id, str) or not task_id:
            raise provider_error("suno music submit missing task id")
        return self._suno_wait_fetch_audio(
            task_id=task_id,
            model_id=model_id,
            timeout_ms=request.params.timeout_ms,
            wait=request.wait,
        )

    def _suno_workflow_endpoint(self, model_id: str) -> str:
        mid_l = model_id.lower().strip()
        if mid_l == "suno_concat":
            return "/suno/submit/concat"
        if mid_l == "suno_uploads":
            return "/suno/submit/upload"
        if mid_l == "suno_persona_create":
            return "/suno/submit/persona-create"
        if mid_l.startswith("suno_act_"):
            suffix = mid_l[len("suno_act_") :]
            if not suffix:
                raise invalid_request_error("invalid suno_act model id")
            return f"/suno/submit/act-{suffix}"
        if mid_l.startswith("suno-"):
            suffix = mid_l[len("suno-") :]
            if not suffix:
                raise invalid_request_error("invalid suno model id")
            return f"/suno/submit/{suffix}"
        raise invalid_request_error(f"unsupported suno workflow model: {model_id}")

    def _suno_workflow(self, request: GenerateRequest, *, model_id: str) -> GenerateResponse:
        host = self._base_host()
        endpoint = self._suno_workflow_endpoint(model_id)

        body: dict[str, object] = {}
        opts = request.provider_options.get("tuzi-web")
        if isinstance(opts, dict):
            body.update(opts)

        prompt = self._text_prompt_or_none(request)
        if prompt and "prompt" not in body:
            body["prompt"] = prompt

        obj = request_json(
            method="POST",
            url=f"{host}{endpoint}",
            headers=self._bearer_headers(),
            json_body=body,
            timeout_ms=max(request.params.timeout_ms or 60_000, 60_000),
            proxy_url=self.proxy_url,
        )
        task_id = obj.get("data")
        if not isinstance(task_id, str) or not task_id:
            raise provider_error("suno submit missing task id")
        return self._suno_wait_fetch_any(
            task_id=task_id,
            model_id=model_id,
            timeout_ms=request.params.timeout_ms,
            wait=request.wait,
        )

    def _suno_fetch(self, *, host: str, task_id: str, timeout_ms: int | None) -> dict[str, object]:
        obj = request_json(
            method="GET",
            url=f"{host}/suno/fetch/{task_id}",
            headers=self._bearer_headers(),
            json_body=None,
            timeout_ms=timeout_ms,
            proxy_url=self.proxy_url,
        )
        data = obj.get("data")
        if not isinstance(data, dict):
            raise provider_error("suno fetch missing data")
        return data

    def _suno_wait_fetch_text(self, *, task_id: str, model_id: str, timeout_ms: int | None, wait: bool) -> GenerateResponse:
        if not wait:
            return GenerateResponse(
                id=f"sdk_{uuid4().hex}",
                provider="tuzi-web",
                model=f"tuzi-web:{model_id}",
                status="running",
                job=JobInfo(job_id=task_id, poll_after_ms=2_000),
            )
        host = self._base_host()
        budget_ms = 60_000 if timeout_ms is None else timeout_ms
        deadline = time.time() + max(1, budget_ms) / 1000.0
        while True:
            remaining_ms = int((deadline - time.time()) * 1000)
            if remaining_ms <= 0:
                break
            data = self._suno_fetch(host=host, task_id=task_id, timeout_ms=min(30_000, remaining_ms))
            status = data.get("status")
            if status == "SUCCESS":
                inner = data.get("data")
                if isinstance(inner, dict):
                    text = inner.get("text")
                    if isinstance(text, str):
                        return GenerateResponse(
                            id=f"sdk_{uuid4().hex}",
                            provider="tuzi-web",
                            model=f"tuzi-web:{model_id}",
                            status="completed",
                            output=[Message(role="assistant", content=[Part.from_text(text)])],
                        )
                raise provider_error("suno lyrics succeeded but missing text")
            if status == "FAIL":
                raise provider_error(f"suno task failed: {data.get('fail_reason')}")
            time.sleep(min(2.0, max(0.0, deadline - time.time())))

        return GenerateResponse(
            id=f"sdk_{uuid4().hex}",
            provider="tuzi-web",
            model=f"tuzi-web:{model_id}",
            status="running",
            job=JobInfo(job_id=task_id, poll_after_ms=2_000),
        )

    def _suno_wait_fetch_audio(self, *, task_id: str, model_id: str, timeout_ms: int | None, wait: bool) -> GenerateResponse:
        if not wait:
            return GenerateResponse(
                id=f"sdk_{uuid4().hex}",
                provider="tuzi-web",
                model=f"tuzi-web:{model_id}",
                status="running",
                job=JobInfo(job_id=task_id, poll_after_ms=2_000),
            )
        host = self._base_host()
        budget_ms = 120_000 if timeout_ms is None else timeout_ms
        deadline = time.time() + max(1, budget_ms) / 1000.0
        while True:
            remaining_ms = int((deadline - time.time()) * 1000)
            if remaining_ms <= 0:
                break
            data = self._suno_fetch(host=host, task_id=task_id, timeout_ms=min(30_000, remaining_ms))
            status = data.get("status")
            if status == "SUCCESS":
                inner = data.get("data")
                urls: list[str] = []
                if isinstance(inner, dict):
                    u = inner.get("audio_url")
                    if isinstance(u, str) and u:
                        urls.append(u)
                    clips = inner.get("clips")
                    if isinstance(clips, list):
                        for clip in clips:
                            if not isinstance(clip, dict):
                                continue
                            u = clip.get("audio_url")
                            if isinstance(u, str) and u:
                                urls.append(u)
                elif isinstance(inner, list):
                    for clip in inner:
                        if not isinstance(clip, dict):
                            continue
                        u = clip.get("audio_url")
                        if isinstance(u, str) and u:
                            urls.append(u)
                if not urls:
                    blob = json.dumps(inner, ensure_ascii=False)
                    u = _extract_first_url(_AUDIO_URL_RE, blob)
                    if u:
                        urls.append(u)
                if urls:
                    part = Part(type="audio", mime_type="audio/mpeg", source=PartSourceUrl(url=urls[0]))
                    return GenerateResponse(
                        id=f"sdk_{uuid4().hex}",
                        provider="tuzi-web",
                        model=f"tuzi-web:{model_id}",
                        status="completed",
                        output=[Message(role="assistant", content=[part])],
                    )
                raise provider_error("suno music succeeded but missing audio url")
            if status == "FAIL":
                raise provider_error(f"suno task failed: {data.get('fail_reason')}")
            time.sleep(min(2.0, max(0.0, deadline - time.time())))

        return GenerateResponse(
            id=f"sdk_{uuid4().hex}",
            provider="tuzi-web",
            model=f"tuzi-web:{model_id}",
            status="running",
            job=JobInfo(job_id=task_id, poll_after_ms=2_000),
        )

    def _suno_wait_fetch_any(self, *, task_id: str, model_id: str, timeout_ms: int | None, wait: bool) -> GenerateResponse:
        if not wait:
            return GenerateResponse(
                id=f"sdk_{uuid4().hex}",
                provider="tuzi-web",
                model=f"tuzi-web:{model_id}",
                status="running",
                job=JobInfo(job_id=task_id, poll_after_ms=2_000),
            )
        host = self._base_host()
        budget_ms = 120_000 if timeout_ms is None else timeout_ms
        deadline = time.time() + max(1, budget_ms) / 1000.0
        while True:
            remaining_ms = int((deadline - time.time()) * 1000)
            if remaining_ms <= 0:
                break
            data = self._suno_fetch(host=host, task_id=task_id, timeout_ms=min(30_000, remaining_ms))
            status = data.get("status")
            if status == "SUCCESS":
                inner = data.get("data")
                parts: list[Part] = []
                blob = json.dumps(inner, ensure_ascii=False)

                audio_urls: list[str] = []
                if isinstance(inner, dict):
                    clips = inner.get("clips")
                    if isinstance(clips, list):
                        for clip in clips:
                            if not isinstance(clip, dict):
                                continue
                            u = clip.get("audio_url")
                            if isinstance(u, str) and u:
                                audio_urls.append(u)

                    text = inner.get("text")
                    if isinstance(text, str) and text:
                        parts.append(Part.from_text(text))

                if not audio_urls:
                    u = _extract_first_url(_AUDIO_URL_RE, blob)
                    if u:
                        audio_urls.append(u)
                for u in audio_urls:
                    parts.append(Part(type="audio", mime_type="audio/mpeg", source=PartSourceUrl(url=u)))

                mp4 = _extract_first_url(_MP4_URL_RE, blob)
                if mp4:
                    parts.append(Part(type="video", mime_type="video/mp4", source=PartSourceUrl(url=mp4)))

                if not parts:
                    parts.append(Part.from_text(blob if blob and blob != "null" else "{}"))

                return GenerateResponse(
                    id=f"sdk_{uuid4().hex}",
                    provider="tuzi-web",
                    model=f"tuzi-web:{model_id}",
                    status="completed",
                    output=[Message(role="assistant", content=parts)],
                )
            if status == "FAIL":
                raise provider_error(f"suno task failed: {data.get('fail_reason')}")
            time.sleep(min(2.0, max(0.0, deadline - time.time())))

        return GenerateResponse(
            id=f"sdk_{uuid4().hex}",
            provider="tuzi-web",
            model=f"tuzi-web:{model_id}",
            status="running",
            job=JobInfo(job_id=task_id, poll_after_ms=2_000),
        )

    def _deepsearch(self, request: GenerateRequest, *, model_id: str) -> GenerateResponse:
        """Handle deepsearch models via asyncdata.net async API."""
        if self.openai is None:
            raise invalid_request_error("NOUS_GENAI_TUZI_OPENAI_API_KEY required for deepsearch models")

        # Build chat completions body
        messages = []
        for msg in request.input:
            role = msg.role
            if role == "system":
                role = "system"
            elif role == "assistant":
                role = "assistant"
            else:
                role = "user"
            content = "".join(p.require_text() for p in msg.content if p.type == "text")
            if content:
                messages.append({"role": role, "content": content})

        if not messages:
            raise invalid_request_error("deepsearch requires at least one message")

        # asyncdata.net requires -async suffix for deepsearch models
        api_model_id = model_id if model_id.endswith("-async") else f"{model_id}-async"
        body = {"model": api_model_id, "messages": messages}
        if request.params.temperature is not None:
            body["temperature"] = request.params.temperature

        # Submit async task
        # Note: URL is NOT encoded per official API docs
        original_url = f"{self.openai.base_url}/chat/completions"
        submit_url = f"{_ASYNCDATA_BASE_URL}/tran/{original_url}"

        # asyncdata.net may take a long time to return task_id; retry on transient errors
        submit_timeout_ms = max(request.params.timeout_ms or 300_000, 300_000)
        last_error: str | None = None
        for attempt in range(3):
            obj = request_json(
                method="POST",
                url=submit_url,
                headers={"Authorization": f"Bearer {self.openai.api_key}"},
                json_body=body,
                timeout_ms=submit_timeout_ms,
                proxy_url=self.proxy_url,
            )
            task_id = obj.get("id")
            if isinstance(task_id, str) and task_id:
                break
            last_error = obj.get("error", "missing task id")
            time.sleep(1.0)  # Brief delay before retry
        else:
            raise provider_error(f"deepsearch submit failed: {last_error}")

        # Non-blocking mode
        if not request.wait:
            return GenerateResponse(
                id=f"sdk_{uuid4().hex}",
                provider="tuzi-web",
                model=f"tuzi-web:{model_id}",
                status="running",
                job=JobInfo(job_id=task_id, poll_after_ms=2_000),
            )

        # Blocking mode: poll until complete
        budget_ms = request.params.timeout_ms or 300_000  # 5 min default for deepsearch
        deadline = time.time() + max(1, budget_ms) / 1000.0
        content = self._poll_deepsearch(task_id=task_id, deadline=deadline)

        if content is None:
            return GenerateResponse(
                id=f"sdk_{uuid4().hex}",
                provider="tuzi-web",
                model=f"tuzi-web:{model_id}",
                status="running",
                job=JobInfo(job_id=task_id, poll_after_ms=2_000),
            )

        return GenerateResponse(
            id=f"sdk_{uuid4().hex}",
            provider="tuzi-web",
            model=f"tuzi-web:{model_id}",
            status="completed",
            output=[Message(role="assistant", content=[Part.from_text(content)])],
        )

    def _poll_deepsearch(self, *, task_id: str, deadline: float) -> str | None:
        """Poll asyncdata.net until task completes or deadline reached."""
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
                proxy_url=self.proxy_url,
            )

            content = obj.get("content")
            if isinstance(content, str) and content:
                return content

            # Still processing, wait before next poll
            time.sleep(min(2.0, max(0.0, deadline - time.time())))

    def list_models(self, *, timeout_ms: int | None = None) -> list[str]:
        """
        Fetch remote model ids by querying each underlying protocol adapter (when configured).
        """
        out: set[str] = set()
        if self.openai is not None:
            try:
                openai_models = self.openai.list_models(timeout_ms=timeout_ms)
            except GenAIError:
                openai_models = []
            if openai_models:
                return openai_models
        if self.gemini is not None:
            try:
                out.update(self.gemini.list_models(timeout_ms=timeout_ms))
            except GenAIError:
                pass
        if self.anthropic is not None:
            try:
                out.update(self.anthropic.list_models(timeout_ms=timeout_ms))
            except GenAIError:
                pass
        return sorted(out)

    def _route(self, model_id: str):
        mid = model_id.strip()
        if not mid:
            raise invalid_request_error("model_id must not be empty")
        mid_l = mid.lower()

        if mid_l.startswith("claude-"):
            if self.anthropic is None:
                raise invalid_request_error(
                    "NOUS_GENAI_TUZI_ANTHROPIC_API_KEY/TUZI_ANTHROPIC_API_KEY "
                    "(or NOUS_GENAI_TUZI_WEB_API_KEY/TUZI_WEB_API_KEY) not configured"
                )
            return self.anthropic

        if mid_l.startswith(("models/", "gemini-", "gemma-", "veo-")):
            if self.gemini is None:
                raise invalid_request_error(
                    "NOUS_GENAI_TUZI_GOOGLE_API_KEY/TUZI_GOOGLE_API_KEY "
                    "(or NOUS_GENAI_TUZI_WEB_API_KEY/TUZI_WEB_API_KEY) not configured"
                )
            return self.gemini
        if mid_l.startswith("veo2"):
            if self.gemini is None:
                raise invalid_request_error(
                    "NOUS_GENAI_TUZI_GOOGLE_API_KEY/TUZI_GOOGLE_API_KEY "
                    "(or NOUS_GENAI_TUZI_WEB_API_KEY/TUZI_WEB_API_KEY) not configured"
                )
            return self.gemini

        if mid_l in {
            "text-embedding-004",
            "embedding-001",
            "embedding-gecko-001",
            "gemini-embedding-001",
            "gemini-embedding-exp-03-07",
        }:
            if self.gemini is None:
                raise invalid_request_error(
                    "NOUS_GENAI_TUZI_GOOGLE_API_KEY/TUZI_GOOGLE_API_KEY "
                    "(or NOUS_GENAI_TUZI_WEB_API_KEY/TUZI_WEB_API_KEY) not configured"
                )
            return self.gemini

        if self.openai is None:
            raise invalid_request_error(
                "NOUS_GENAI_TUZI_OPENAI_API_KEY/TUZI_OPENAI_API_KEY "
                "(or NOUS_GENAI_TUZI_WEB_API_KEY/TUZI_WEB_API_KEY) not configured"
            )
        return self.openai
