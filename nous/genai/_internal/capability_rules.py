from __future__ import annotations

"""
Central place for capability rules based on model series (usually shared prefixes).

Rules here only look at model_id strings and are provider-agnostic. Provider adapters
still own protocol details (stream/job semantics, routing, endpoint quirks).
"""

import re
from typing import Final, Literal

ModelKind = Literal["video", "image", "embedding", "tts", "transcribe", "chat"]
GeminiModelKind = Literal["video", "embedding", "tts", "native_audio", "image", "chat"]


def _norm(model_id: str) -> str:
    return model_id.lower().strip()


def _starts_with_any(s: str, prefixes: tuple[str, ...]) -> bool:
    return s.startswith(prefixes)


_TOKEN_SPLIT_RE: Final[re.Pattern[str]] = re.compile(r"[^a-z0-9]+")


def _has_token(s: str, token: str) -> bool:
    # Token boundary: split by any non [a-z0-9] to avoid substring false-positives.
    toks = _TOKEN_SPLIT_RE.split(s.lower())
    return token.lower() in toks


# ---- Model series for /v1-style protocols ----

_SORA_PREFIX: Final[str] = "sora-"

_VEO_PREFIX: Final[str] = "veo"
_PIKA_PREFIX: Final[str] = "pika-"
_RUNWAY_PREFIX: Final[str] = "runway-"
_SEEDANCE_PREFIXES: Final[tuple[str, ...]] = ("seedance", "doubao-seedance")
_KLING_VIDEO_PREFIXES: Final[tuple[str, ...]] = ("kling_video", "kling-video-")

_DALLE_PREFIX: Final[str] = "dall-e-"
_GPT_IMAGE_PREFIX: Final[str] = "gpt-image-"
_CHATGPT_IMAGE_PREFIX: Final[str] = "chatgpt-image"
_SEEDREAM_PREFIXES: Final[tuple[str, ...]] = ("seedream", "doubao-seedream")
_SEEDEDIT_PREFIXES: Final[tuple[str, ...]] = ("seededit", "api-images-seededit")
_FLUX_PREFIXES: Final[tuple[str, ...]] = ("flux-", "flux.")
_SD3_PREFIX: Final[str] = "sd3"

_TEXT_EMBEDDING_PREFIX: Final[str] = "text-embedding-"
_EMBEDDING_PREFIX: Final[str] = "embedding-"
_DOUBAO_EMBEDDING_PREFIX: Final[str] = "doubao-embedding-"

_TTS_PREFIX: Final[str] = "tts-"
_TTS_SUFFIX: Final[str] = "-tts"
_VOICE_SUFFIXES: Final[tuple[str, ...]] = ("-voice", "_voice")
_ADVANCED_VOICE_MODEL: Final[str] = "advanced-voice"
_SUNO_PREFIX: Final[str] = "suno-"
_CHIRP_PREFIX: Final[str] = "chirp-"

_WHISPER_PREFIX: Final[str] = "whisper-"
_DISTIL_WHISPER_PREFIX: Final[str] = "distil-whisper-"
_TRANSCRIBE_MARKER: Final[str] = "-transcribe"
_ASR_PREFIX: Final[str] = "asr"

_IMAGE_MODELS_WITH_IMAGE_INPUT_PREFIXES: Final[tuple[str, ...]] = ("gpt-image-1", "chatgpt-image")

_Z_IMAGE_PREFIX: Final[str] = "z-image"

_CHAT_IMAGE_INPUT_EXACT: Final[tuple[str, ...]] = (
    "chatgpt-4o-latest",
    "codex-mini-latest",
    "computer-use-preview",
    "omni-moderation-latest",
)

_DEEPSEEK_V3_PREFIX: Final[str] = "deepseek-v3"
_KIMI_K2_PREFIX: Final[str] = "kimi-k2"
_KIMI_LATEST_PREFIX: Final[str] = "kimi-latest"
_QWEN_PREFIX: Final[str] = "qwen"
_QVQ_PREFIX: Final[str] = "qvq"

_CHAT_NO_IMAGE_PREFIXES: Final[tuple[str, ...]] = (_DEEPSEEK_V3_PREFIX, _KIMI_K2_PREFIX, _QVQ_PREFIX)

_CHAT_TEXT_ONLY_EXACT: Final[tuple[str, ...]] = (
    "gpt-4",
    "gpt-4-turbo-preview",
    "o1-mini",
    "o1-preview",
    "o3-mini",
)

_CHAT_TEXT_ONLY_MARKERS: Final[tuple[str, ...]] = ("search-preview",)

_CHAT_AUDIO_PREFIXES: Final[tuple[str, ...]] = ("gpt-audio", "gpt-realtime")
_CHAT_AUDIO_SUFFIXES: Final[tuple[str, ...]] = ("-audio-preview", "-realtime-preview")

_CLAUDE_PREFIX: Final[str] = "claude-"


def is_sora_model(model_id: str) -> bool:
    return _norm(model_id).startswith(_SORA_PREFIX)


def is_veo_model(model_id: str) -> bool:
    return _norm(model_id).startswith(_VEO_PREFIX)


def is_pika_model(model_id: str) -> bool:
    return _norm(model_id).startswith(_PIKA_PREFIX)


def is_runway_model(model_id: str) -> bool:
    return _norm(model_id).startswith(_RUNWAY_PREFIX)


def is_seedance_model(model_id: str) -> bool:
    return _starts_with_any(_norm(model_id), _SEEDANCE_PREFIXES)


def is_kling_video_model(model_id: str) -> bool:
    return _starts_with_any(_norm(model_id), _KLING_VIDEO_PREFIXES)


def is_video_model(model_id: str) -> bool:
    return (
        is_sora_model(model_id)
        or is_veo_model(model_id)
        or is_pika_model(model_id)
        or is_runway_model(model_id)
        or is_seedance_model(model_id)
        or is_kling_video_model(model_id)
    )


def is_dall_e_model(model_id: str) -> bool:
    return _norm(model_id).startswith(_DALLE_PREFIX)


def is_gpt_image_model(model_id: str) -> bool:
    return _norm(model_id).startswith(_GPT_IMAGE_PREFIX)


def is_chatgpt_image_model(model_id: str) -> bool:
    return _norm(model_id).startswith(_CHATGPT_IMAGE_PREFIX)


def is_seedream_model(model_id: str) -> bool:
    return _starts_with_any(_norm(model_id), _SEEDREAM_PREFIXES)


def is_seededit_model(model_id: str) -> bool:
    return _starts_with_any(_norm(model_id), _SEEDEDIT_PREFIXES)


def is_flux_model(model_id: str) -> bool:
    return _starts_with_any(_norm(model_id), _FLUX_PREFIXES)


def is_sd3_model(model_id: str) -> bool:
    return _norm(model_id).startswith(_SD3_PREFIX)


def is_z_image_model(model_id: str) -> bool:
    return _norm(model_id).startswith(_Z_IMAGE_PREFIX)


def is_gpt_dash_image_model(model_id: str) -> bool:
    mid_l = _norm(model_id)
    return mid_l.startswith("gpt-") and "-image" in mid_l


def is_image_model(model_id: str) -> bool:
    return (
        is_dall_e_model(model_id)
        or is_gpt_image_model(model_id)
        or is_gpt_dash_image_model(model_id)
        or is_chatgpt_image_model(model_id)
        or is_z_image_model(model_id)
        or is_seedream_model(model_id)
        or is_seededit_model(model_id)
        or is_flux_model(model_id)
        or is_sd3_model(model_id)
    )


def is_text_embedding_model(model_id: str) -> bool:
    return _norm(model_id).startswith(_TEXT_EMBEDDING_PREFIX)


def is_embedding_series_model(model_id: str) -> bool:
    return _norm(model_id).startswith(_EMBEDDING_PREFIX)


def is_doubao_embedding_model(model_id: str) -> bool:
    return _norm(model_id).startswith(_DOUBAO_EMBEDDING_PREFIX)


def is_embedding_model(model_id: str) -> bool:
    return (
        is_text_embedding_model(model_id)
        or is_embedding_series_model(model_id)
        or is_doubao_embedding_model(model_id)
    )


def is_tts_model(model_id: str) -> bool:
    mid_l = _norm(model_id)
    return (
        mid_l.startswith(_TTS_PREFIX)
        or mid_l.startswith(_SUNO_PREFIX)
        or mid_l.startswith(_CHIRP_PREFIX)
        or mid_l.endswith(_TTS_SUFFIX)
        or mid_l.endswith(_VOICE_SUFFIXES)
        or mid_l == _ADVANCED_VOICE_MODEL
    )


def is_whisper_model(model_id: str) -> bool:
    return _norm(model_id).startswith(_WHISPER_PREFIX)


def is_distil_whisper_model(model_id: str) -> bool:
    return _norm(model_id).startswith(_DISTIL_WHISPER_PREFIX)


def is_transcribe_marker_model(model_id: str) -> bool:
    return _TRANSCRIBE_MARKER in _norm(model_id)


def is_asr_model(model_id: str) -> bool:
    return _norm(model_id).startswith(_ASR_PREFIX)


def is_transcribe_model(model_id: str) -> bool:
    return (
        is_whisper_model(model_id)
        or is_distil_whisper_model(model_id)
        or is_transcribe_marker_model(model_id)
        or is_asr_model(model_id)
    )


def infer_model_kind(model_id: str) -> ModelKind:
    if is_video_model(model_id):
        return "video"
    if is_image_model(model_id):
        return "image"
    if is_embedding_model(model_id):
        return "embedding"
    if is_tts_model(model_id):
        return "tts"
    if is_transcribe_model(model_id):
        return "transcribe"
    return "chat"


def output_modalities_for_kind(kind: ModelKind) -> set[str] | None:
    if kind == "video":
        return {"video"}
    if kind == "image":
        return {"image"}
    if kind == "embedding":
        return {"embedding"}
    if kind == "tts":
        return {"audio"}
    if kind == "transcribe":
        return {"text"}
    return None


def transcribe_input_modalities(_: str) -> set[str]:
    return {"audio"}


def is_claude_model(model_id: str) -> bool:
    return _norm(model_id).startswith(_CLAUDE_PREFIX)


def claude_input_modalities(model_id: str) -> set[str]:
    if is_claude_model(model_id):
        return {"text", "image"}
    return {"text"}


def image_input_modalities(model_id: str) -> set[str]:
    mid_l = _norm(model_id)
    if _starts_with_any(mid_l, _IMAGE_MODELS_WITH_IMAGE_INPUT_PREFIXES) or mid_l.startswith(_Z_IMAGE_PREFIX):
        return {"text", "image"}
    return {"text"}


def video_input_modalities(model_id: str) -> set[str]:
    if is_veo_model(model_id):
        return {"text", "image"}
    return {"text"}


def chat_input_modalities(model_id: str) -> set[str]:
    mid_l = _norm(model_id)
    out = {"text"}
    if _chat_supports_image_input(mid_l):
        out.add("image")
    if _chat_supports_audio_input(mid_l):
        out.add("audio")
    return out


def chat_output_modalities(model_id: str) -> set[str]:
    mid_l = _norm(model_id)
    out = {"text"}
    if _chat_supports_audio_io(mid_l):
        out.add("audio")
    return out


def chat_supports_audio_io(model_id: str) -> bool:
    return _chat_supports_audio_io(_norm(model_id))


def _chat_supports_audio_io(mid_l: str) -> bool:
    return _starts_with_any(mid_l, _CHAT_AUDIO_PREFIXES) or mid_l.endswith(_CHAT_AUDIO_SUFFIXES)


def _qwen_supports_image_input(mid_l: str) -> bool:
    return _has_token(mid_l, "image") or _has_token(mid_l, "vl")


def _qwen_supports_audio_input(mid_l: str) -> bool:
    return _has_token(mid_l, "asr")


def chat_supports_audio_input(model_id: str) -> bool:
    return _chat_supports_audio_input(_norm(model_id))


def _chat_supports_audio_input(mid_l: str) -> bool:
    if _chat_supports_audio_io(mid_l):
        return True
    return mid_l.startswith(_QWEN_PREFIX) and _qwen_supports_audio_input(mid_l)


def chat_supports_image_input(model_id: str) -> bool:
    return _chat_supports_image_input(_norm(model_id))


def _chat_supports_image_input(mid_l: str) -> bool:
    if mid_l in _CHAT_TEXT_ONLY_EXACT:
        return False
    if any(marker in mid_l for marker in _CHAT_TEXT_ONLY_MARKERS):
        return False

    if _starts_with_any(mid_l, _CHAT_NO_IMAGE_PREFIXES):
        return False
    if mid_l.startswith(_KIMI_LATEST_PREFIX):
        return True
    if mid_l.startswith(_QWEN_PREFIX):
        return _qwen_supports_image_input(mid_l)

    if mid_l in _CHAT_IMAGE_INPUT_EXACT:
        return True

    return _openai_style_chat_supports_image_input(mid_l)


def _openai_style_chat_supports_image_input(mid_l: str) -> bool:
    if mid_l.startswith("gpt-realtime"):
        return True
    if mid_l.startswith("gpt-4o") and not _chat_supports_audio_io(mid_l):
        return True
    if mid_l.startswith("gpt-4-turbo") and mid_l != "gpt-4-turbo-preview":
        return True
    if mid_l.startswith(("gpt-4.1", "gpt-4.5", "gpt-5")):
        return True
    if mid_l == "o1" or mid_l.startswith("o1-"):
        return True
    if mid_l == "o3" or mid_l.startswith("o3-"):
        return True
    return mid_l.startswith("o4-")


# ---- Gemini model series ----

def _gemini_norm(model_id: str) -> str:
    mid = model_id.lower().strip()
    if mid.startswith("models/"):
        mid = mid[len("models/") :]
    return mid


def _gemini_is_image_model(mid_l: str) -> bool:
    return (
        mid_l.startswith("imagen-")
        or "image-generation" in mid_l
        or mid_l.endswith("-image")
        or mid_l.endswith("-image-preview")
        or "-image-" in mid_l
    )


def gemini_is_video_model(model_id: str) -> bool:
    return _gemini_norm(model_id).startswith("veo")


def gemini_is_embedding_model(model_id: str) -> bool:
    return "embedding" in _gemini_norm(model_id)


def gemini_is_tts_model(model_id: str) -> bool:
    return "tts" in _gemini_norm(model_id)


def gemini_is_native_audio_model(model_id: str) -> bool:
    return "native-audio" in _gemini_norm(model_id)


def gemini_is_image_model(model_id: str) -> bool:
    return _gemini_is_image_model(_gemini_norm(model_id))


def gemini_image_input_modalities(model_id: str) -> set[str]:
    mid_l = _gemini_norm(model_id)
    if mid_l.startswith("imagen-"):
        return {"text"}
    if "image-generation" in mid_l:
        return {"text"}
    return {"text", "image"}


def gemini_model_kind(model_id: str) -> GeminiModelKind:
    mid_l = _gemini_norm(model_id)
    if mid_l.startswith("veo"):
        return "video"
    if "embedding" in mid_l:
        return "embedding"
    if "tts" in mid_l:
        return "tts"
    if "native-audio" in mid_l:
        return "native_audio"
    if _gemini_is_image_model(mid_l):
        return "image"
    return "chat"


def gemini_output_modalities(kind: GeminiModelKind) -> set[str]:
    if kind == "video":
        return {"video"}
    if kind == "embedding":
        return {"embedding"}
    if kind == "tts":
        return {"audio"}
    if kind == "native_audio":
        return {"text", "audio"}
    if kind == "image":
        return {"image"}
    return {"text"}
