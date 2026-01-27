from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any, Iterator
from uuid import uuid4

from .._internal.errors import invalid_request_error, not_supported_error, provider_error
from .._internal.http import request_json
from ..types import Capability, GenerateEvent, GenerateRequest, GenerateResponse
from ..types import JobInfo, Message, Part, PartSourceBytes, PartSourceUrl
from .openai import OpenAIAdapter


@dataclass(frozen=True, slots=True)
class AliyunAdapter:
    """
    Aliyun DashScope (Bailian / Model Studio).

    Supported in this SDK:
    - chat (text/image -> text), stream supported
    - embeddings (text -> embedding)
    - image generation (text -> image) via DashScope AIGC endpoint
    - video generation (text -> video) via DashScope AIGC async task endpoint
    - speech synthesis (text -> audio) via DashScope AIGC endpoint
    - speech recognition (audio -> text) for Qwen-ASR models via OpenAI-compatible chat endpoint
    """

    openai: OpenAIAdapter

    def capabilities(self, model_id: str) -> Capability:
        if _is_embedding_model(model_id):
            return Capability(
                input_modalities={"text"},
                output_modalities={"embedding"},
                supports_stream=False,
                supports_job=False,
                supports_tools=False,
                supports_json_schema=False,
            )
        if _is_speech_synthesis_model(model_id):
            return Capability(
                input_modalities={"text"},
                output_modalities={"audio"},
                supports_stream=False,
                supports_job=False,
                supports_tools=False,
                supports_json_schema=False,
            )
        if _is_asr_model(model_id):
            return Capability(
                input_modalities={"audio"},
                output_modalities={"text"},
                supports_stream=False,
                supports_job=False,
                supports_tools=False,
                supports_json_schema=False,
            )
        if _is_image_generation_model(model_id):
            return Capability(
                input_modalities={"text"},
                output_modalities={"image"},
                supports_stream=False,
                supports_job=False,
                supports_tools=False,
                supports_json_schema=False,
            )
        if _is_video_generation_model(model_id):
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
                raise not_supported_error("Aliyun embeddings do not support streaming")
            return self.openai.generate(request, stream=False)

        if modalities == {"image"}:
            if stream:
                raise not_supported_error("Aliyun image generation does not support streaming")
            return self._image(request, model_id=model_id)

        if modalities == {"video"}:
            if stream:
                raise not_supported_error("Aliyun video generation does not support streaming")
            return self._video(request, model_id=model_id)

        if modalities == {"audio"}:
            if stream:
                raise not_supported_error("Aliyun speech synthesis does not support streaming")
            return self._audio(request, model_id=model_id)

        if modalities != {"text"}:
            raise not_supported_error("Aliyun only supports chat/embeddings/image/video/audio in this SDK")
        if _is_embedding_model(model_id):
            raise not_supported_error("Aliyun embedding models must be called with output.modalities=['embedding']")
        if _is_speech_synthesis_model(model_id):
            raise not_supported_error("Aliyun speech synthesis models must be called with output.modalities=['audio']")
        if _is_image_generation_model(model_id):
            raise not_supported_error("Aliyun image models must be called with output.modalities=['image']")
        if _is_video_generation_model(model_id):
            raise not_supported_error("Aliyun video models must be called with output.modalities=['video']")
        if _has_audio_input(request) and not _is_asr_model(model_id):
            raise not_supported_error("Aliyun chat input only supports audio for ASR models in this SDK")
        return self.openai.generate(request, stream=stream)

    def _image(self, request: GenerateRequest, *, model_id: str) -> GenerateResponse:
        if not _is_image_generation_model(model_id):
            raise not_supported_error('Aliyun image generation requires model like "aliyun:qwen-image-max"')

        prompt = _single_text_prompt(request)
        body: dict[str, Any] = {
            "model": model_id,
            "input": {
                "messages": [
                    {
                        "role": "user",
                        "content": [{"text": prompt}],
                    }
                ]
            },
            "parameters": {
                "watermark": False,
            },
        }

        img = request.output.image
        if img and img.size:
            size = img.size.strip()
            if "x" in size and "*" not in size:
                size = size.replace("x", "*")
            body["parameters"]["size"] = size

        opts = request.provider_options.get("aliyun")
        if isinstance(opts, dict):
            _merge_provider_options(body=body, opts=opts)

        obj = request_json(
            method="POST",
            url="https://dashscope.aliyuncs.com/api/v1/services/aigc/multimodal-generation/generation",
            headers={**_aliyun_headers(self.openai.api_key, request=request)},
            json_body=body,
            timeout_ms=request.params.timeout_ms,
            proxy_url=self.openai.proxy_url,
        )

        image_url = _extract_image_url(obj)
        if not image_url:
            raise provider_error("aliyun image response missing image url")

        part = Part(type="image", source=PartSourceUrl(url=image_url))
        return GenerateResponse(
            id=f"sdk_{uuid4().hex}",
            provider="aliyun",
            model=f"aliyun:{model_id}",
            status="completed",
            output=[Message(role="assistant", content=[part])],
            usage=None,
        )

    def _audio(self, request: GenerateRequest, *, model_id: str) -> GenerateResponse:
        if not _is_speech_synthesis_model(model_id):
            raise not_supported_error('Aliyun speech synthesis requires model like "aliyun:qwen3-tts-flash"')

        prompt = _single_text_prompt(request)
        audio = request.output.audio
        if audio is None or not audio.voice:
            raise invalid_request_error("output.audio.voice required for Aliyun speech synthesis")
        if audio.format and audio.format.strip().lower() not in {"wav", "wave"}:
            raise not_supported_error("Aliyun speech synthesis only supports wav output in this SDK")

        body: dict[str, Any] = {
            "model": model_id,
            "input": {
                "text": prompt,
                "voice": audio.voice,
            },
        }
        if audio.language:
            body["input"]["language_type"] = _map_language_type(audio.language)

        opts = request.provider_options.get("aliyun")
        if isinstance(opts, dict):
            _merge_provider_options(body=body, opts=opts)

        obj = request_json(
            method="POST",
            url="https://dashscope.aliyuncs.com/api/v1/services/aigc/multimodal-generation/generation",
            headers={**_aliyun_headers(self.openai.api_key, request=request)},
            json_body=body,
            timeout_ms=request.params.timeout_ms,
            proxy_url=self.openai.proxy_url,
        )

        source, mime_type = _extract_audio_source(obj)
        part = Part(type="audio", mime_type=mime_type, source=source)
        return GenerateResponse(
            id=f"sdk_{uuid4().hex}",
            provider="aliyun",
            model=f"aliyun:{model_id}",
            status="completed",
            output=[Message(role="assistant", content=[part])],
            usage=None,
        )

    def _video(self, request: GenerateRequest, *, model_id: str) -> GenerateResponse:
        if not _is_video_generation_model(model_id):
            raise not_supported_error('Aliyun video generation requires model like "aliyun:wan2.5-t2v-preview"')

        prompt = _single_text_prompt(request)
        body: dict[str, Any] = {"model": model_id, "input": {"prompt": prompt}}

        video = request.output.video
        if video and video.duration_sec is not None:
            body.setdefault("parameters", {})["duration"] = int(video.duration_sec)
        if video and video.aspect_ratio:
            body.setdefault("parameters", {})["ratio"] = video.aspect_ratio

        opts = request.provider_options.get("aliyun")
        if isinstance(opts, dict):
            _merge_provider_options(body=body, opts=opts)

        budget_ms = 120_000 if request.params.timeout_ms is None else request.params.timeout_ms
        deadline = time.time() + max(1, budget_ms) / 1000.0
        obj = request_json(
            method="POST",
            url="https://dashscope.aliyuncs.com/api/v1/services/aigc/video-generation/video-synthesis",
            headers={**_aliyun_headers(self.openai.api_key, request=request), "X-DashScope-Async": "enable"},
            json_body=body,
            timeout_ms=min(30_000, max(1, budget_ms)),
            proxy_url=self.openai.proxy_url,
        )
        task_id = _extract_task_id(obj)
        if not task_id:
            raise provider_error("aliyun video response missing task_id")

        if not request.wait:
            return GenerateResponse(
                id=f"sdk_{uuid4().hex}",
                provider="aliyun",
                model=f"aliyun:{model_id}",
                status="running",
                job=JobInfo(job_id=task_id, poll_after_ms=1_000),
            )

        final = _wait_task_done(
            task_id=task_id,
            api_key=self.openai.api_key,
            deadline=deadline,
            proxy_url=self.openai.proxy_url,
        )
        status = _extract_task_status(final)
        if status != "SUCCEEDED":
            if status == "FAILED":
                raise provider_error(f"aliyun video generation failed: {final}")
            return GenerateResponse(
                id=f"sdk_{uuid4().hex}",
                provider="aliyun",
                model=f"aliyun:{model_id}",
                status="running",
                job=JobInfo(job_id=task_id, poll_after_ms=1_000),
            )

        video_url = _extract_video_url(final)
        if not video_url:
            raise provider_error("aliyun video task missing video_url")
        part = Part(type="video", mime_type="video/mp4", source=PartSourceUrl(url=video_url))
        return GenerateResponse(
            id=f"sdk_{uuid4().hex}",
            provider="aliyun",
            model=f"aliyun:{model_id}",
            status="completed",
            output=[Message(role="assistant", content=[part])],
            usage=None,
        )


def _has_audio_input(request: GenerateRequest) -> bool:
    for m in request.input:
        for p in m.content:
            if p.type == "audio":
                return True
    return False


def _is_embedding_model(model_id: str) -> bool:
    mid = model_id.lower()
    return "embedding" in mid and "text" in mid


def _is_speech_synthesis_model(model_id: str) -> bool:
    return "tts" in model_id.lower()


def _is_asr_model(model_id: str) -> bool:
    return "asr" in model_id.lower()


def _is_image_generation_model(model_id: str) -> bool:
    mid = model_id.lower()
    return "image" in mid and "embedding" not in mid


def _is_video_generation_model(model_id: str) -> bool:
    mid = model_id.lower()
    return "t2v" in mid or "i2v" in mid


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


def _map_language_type(language: str) -> str:
    lang = language.strip().lower()
    if lang.startswith("zh"):
        return "Chinese"
    if lang.startswith("en"):
        return "English"
    return language


def _aliyun_headers(api_key: str, *, request: GenerateRequest | None = None) -> dict[str, str]:
    headers = {"Authorization": f"Bearer {api_key}"}
    if request and request.params.idempotency_key:
        headers["Idempotency-Key"] = request.params.idempotency_key
    return headers


def _merge_provider_options(*, body: dict[str, Any], opts: dict[str, Any]) -> None:
    if "model" in opts and opts["model"] != body.get("model"):
        raise invalid_request_error("provider_options cannot override model")
    if "input" in opts:
        if not isinstance(opts["input"], dict):
            raise invalid_request_error("provider_options.input must be an object")
        inp = body.setdefault("input", {})
        if not isinstance(inp, dict):
            raise invalid_request_error("internal error: body.input is not an object")
        for k, v in opts["input"].items():
            if k in inp:
                raise invalid_request_error(f"provider_options cannot override input.{k}")
            inp[k] = v
    if "parameters" in opts:
        if not isinstance(opts["parameters"], dict):
            raise invalid_request_error("provider_options.parameters must be an object")
        params = body.setdefault("parameters", {})
        if not isinstance(params, dict):
            raise invalid_request_error("internal error: body.parameters is not an object")
        for k, v in opts["parameters"].items():
            if k in params:
                raise invalid_request_error(f"provider_options cannot override parameters.{k}")
            params[k] = v
    for k, v in opts.items():
        if k in {"model", "input", "parameters"}:
            continue
        if k in body:
            raise invalid_request_error(f"provider_options cannot override body.{k}")
        body[k] = v


def _extract_image_url(obj: dict[str, Any]) -> str | None:
    output = obj.get("output")
    if not isinstance(output, dict):
        return None
    choices = output.get("choices")
    if not isinstance(choices, list) or not choices:
        return None
    first = choices[0]
    if not isinstance(first, dict):
        return None
    msg = first.get("message")
    if not isinstance(msg, dict):
        return None
    content = msg.get("content")
    if not isinstance(content, list) or not content:
        return None
    item = content[0]
    if not isinstance(item, dict):
        return None
    u = item.get("image")
    if isinstance(u, str) and u:
        return u
    return None


def _extract_audio_source(obj: dict[str, Any]) -> tuple[PartSourceBytes | PartSourceUrl, str]:
    output = obj.get("output")
    if not isinstance(output, dict):
        raise provider_error("aliyun audio response missing output")
    audio = output.get("audio")
    if not isinstance(audio, dict):
        raise provider_error("aliyun audio response missing output.audio")

    data = audio.get("data")
    if isinstance(data, str) and data:
        return PartSourceBytes(data=data, encoding="base64"), "audio/wav"

    url = audio.get("url")
    if isinstance(url, str) and url:
        return PartSourceUrl(url=url), "audio/wav"
    raise provider_error("aliyun audio response missing url/data")


def _extract_task_id(obj: dict[str, Any]) -> str | None:
    output = obj.get("output")
    if not isinstance(output, dict):
        return None
    tid = output.get("task_id") or output.get("taskId")
    if isinstance(tid, str) and tid:
        return tid
    return None


def _extract_task_status(obj: dict[str, Any]) -> str | None:
    output = obj.get("output")
    if not isinstance(output, dict):
        return None
    st = output.get("task_status") or output.get("taskStatus")
    if isinstance(st, str) and st:
        return st
    return None


def _extract_video_url(obj: dict[str, Any]) -> str | None:
    output = obj.get("output")
    if not isinstance(output, dict):
        return None
    u = output.get("video_url") or output.get("videoUrl")
    if isinstance(u, str) and u:
        return u
    return None


def _wait_task_done(*, task_id: str, api_key: str, deadline: float, proxy_url: str | None) -> dict[str, Any]:
    url = f"https://dashscope.aliyuncs.com/api/v1/tasks/{task_id}"
    while True:
        remaining_ms = int((deadline - time.time()) * 1000)
        if remaining_ms <= 0:
            break
        obj = request_json(
            method="GET",
            url=url,
            headers=_aliyun_headers(api_key),
            timeout_ms=min(30_000, remaining_ms),
            proxy_url=proxy_url,
        )
        st = _extract_task_status(obj)
        if st in {"SUCCEEDED", "FAILED"}:
            return obj
        time.sleep(1.0)
    return {"output": {"task_id": task_id, "task_status": "RUNNING"}}
