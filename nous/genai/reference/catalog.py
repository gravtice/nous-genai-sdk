from __future__ import annotations

import copy
from typing import Any, Literal

from ..types import Capability

from .model_catalog import MODEL_CATALOG
from .mode_overrides import CAPABILITY_OVERRIDES


def get_model_catalog() -> dict[str, list[str]]:
    return copy.deepcopy(MODEL_CATALOG)


def get_supported_providers() -> list[str]:
    return list(MODEL_CATALOG.keys())


def get_sdk_supported_models() -> list[dict[str, Any]]:
    """
    Return a JSON-friendly table of all curated models and their SDK-level capabilities.
    """
    out: list[dict[str, Any]] = []
    for provider, model_ids in MODEL_CATALOG.items():
        protocol = _default_protocol(provider)
        for model_id in model_ids:
            cap = _capability_for(
                provider=provider, protocol=protocol, model_id=model_id
            )
            cap = _apply_capability_overrides(
                provider=provider, model_id=model_id, cap=cap
            )
            category = _category_for_capability(cap)
            modes = _modes_for(cap)
            notes: list[str] = []
            if cap.supports_job:
                notes.append("可能返回 running(job)，需轮询/等待")
            out.append(
                {
                    "provider": provider,
                    "model_id": model_id,
                    "model": f"{provider}:{model_id}",
                    "category": category,
                    "protocol": protocol,
                    "modes": modes,
                    "input_modalities": sorted(cap.input_modalities),
                    "output_modalities": sorted(cap.output_modalities),
                    "supports_job": cap.supports_job,
                    "notes": notes,
                }
            )
    return out


def get_sdk_supported_models_for_provider(provider: str) -> list[dict[str, Any]]:
    """
    Return SDK-curated supported models for a single provider (same rows as `get_sdk_supported_models()`).

    Note: `provider` must match keys used in `MODEL_CATALOG` (e.g. "google", not "gemini").
    """
    p = provider.strip().lower()
    if not p:
        return []
    return [m for m in get_sdk_supported_models() if m.get("provider") == p]


def _default_protocol(provider: str) -> str:
    if provider in {"google", "tuzi-google"}:
        return "gemini"
    if provider in {"anthropic", "tuzi-anthropic"}:
        return "messages"
    if provider == "tuzi-web":
        return "multi"
    return "chat_completions"


def _category_for_capability(cap: Capability) -> str:
    out = set(cap.output_modalities)
    if out == {"embedding"}:
        return "embedding"
    if out == {"image"}:
        return "image"
    if out == {"video"}:
        return "video"
    if out == {"audio"}:
        return "audio"
    if (
        out == {"text"}
        and "audio" in set(cap.input_modalities)
        and not cap.supports_stream
    ):
        return "transcription"
    return "chat"


def _capability_for(*, provider: str, protocol: str, model_id: str) -> Capability:
    if provider in {"openai", "tuzi-openai"}:
        from ..providers import OpenAIAdapter

        return OpenAIAdapter(
            api_key="__demo__", provider_name=provider, chat_api=protocol
        ).capabilities(model_id)

    if provider in {"google", "tuzi-google"}:
        from ..providers import GeminiAdapter

        gemini_auth_mode: Literal["bearer", "query_key"] = (
            "bearer" if provider.startswith("tuzi-") else "query_key"
        )
        return GeminiAdapter(
            api_key="__demo__", provider_name=provider, auth_mode=gemini_auth_mode
        ).capabilities(model_id)

    if provider in {"anthropic", "tuzi-anthropic"}:
        from ..providers import AnthropicAdapter

        anthropic_auth_mode: Literal["bearer", "x-api-key"] = (
            "bearer" if provider.startswith("tuzi-") else "x-api-key"
        )
        return AnthropicAdapter(
            api_key="__demo__", provider_name=provider, auth_mode=anthropic_auth_mode
        ).capabilities(model_id)

    if provider == "volcengine":
        from ..providers import OpenAIAdapter, VolcengineAdapter

        volc = VolcengineAdapter(
            openai=OpenAIAdapter(
                api_key="__demo__",
                provider_name="volcengine",
                chat_api="chat_completions",
            )
        )
        return volc.capabilities(model_id)

    if provider == "aliyun":
        from ..providers import AliyunAdapter, OpenAIAdapter

        aliyun = AliyunAdapter(
            openai=OpenAIAdapter(
                api_key="__demo__", provider_name="aliyun", chat_api="chat_completions"
            )
        )
        return aliyun.capabilities(model_id)

    if provider == "tuzi-web":
        from ..providers import (
            AnthropicAdapter,
            GeminiAdapter,
            OpenAIAdapter,
            TuziAdapter,
        )

        return TuziAdapter(
            openai=OpenAIAdapter(
                api_key="__demo__", provider_name=provider, chat_api="chat_completions"
            ),
            gemini=GeminiAdapter(
                api_key="__demo__",
                base_url="https://api.tu-zi.com",
                provider_name=provider,
                auth_mode="bearer",
                supports_file_upload=False,
            ),
            anthropic=AnthropicAdapter(
                api_key="__demo__", provider_name=provider, auth_mode="bearer"
            ),
        ).capabilities(model_id)

    raise ValueError(f"unknown provider: {provider}")


def _apply_capability_overrides(
    *, provider: str, model_id: str, cap: Capability
) -> Capability:
    overrides = CAPABILITY_OVERRIDES.get(provider)
    if not overrides:
        return cap
    override = overrides.get(model_id)
    if not override:
        return cap
    supports_stream = cap.supports_stream
    if "supports_stream" in override:
        supports_stream = override["supports_stream"]
    supports_job = cap.supports_job
    if "supports_job" in override:
        supports_job = override["supports_job"]
    if supports_stream == cap.supports_stream and supports_job == cap.supports_job:
        return cap
    return Capability(
        input_modalities=set(cap.input_modalities),
        output_modalities=set(cap.output_modalities),
        supports_stream=supports_stream,
        supports_job=supports_job,
    )


def _modes_for(cap: Capability) -> list[str]:
    modes: list[str] = ["sync"]
    if cap.supports_stream:
        modes.append("stream")
    if cap.supports_job:
        modes.append("job")
    modes.append("async")
    return modes
