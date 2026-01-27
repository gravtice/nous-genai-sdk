from __future__ import annotations

# Curated model catalog used by demos/CLI.
#
# Catalog only lists model ids per provider.
# Capabilities (and display categories) are inferred at runtime from Adapter.capabilities().

from .model_catalog_data.aliyun import MODELS as ALIYUN_MODELS
from .model_catalog_data.anthropic import MODELS as ANTHROPIC_MODELS
from .model_catalog_data.google import MODELS as GOOGLE_MODELS
from .model_catalog_data.openai import MODELS as OPENAI_MODELS
from .model_catalog_data.tuzi_anthropic import MODELS as TUZI_ANTHROPIC_MODELS
from .model_catalog_data.tuzi_google import MODELS as TUZI_GOOGLE_MODELS
from .model_catalog_data.tuzi_openai import MODELS as TUZI_OPENAI_MODELS
from .model_catalog_data.tuzi_web import MODELS as TUZI_WEB_MODELS
from .model_catalog_data.volcengine import MODELS as VOLCENGINE_MODELS


MODEL_CATALOG: dict[str, list[str]] = {
    "openai": OPENAI_MODELS,
    "anthropic": ANTHROPIC_MODELS,
    "google": GOOGLE_MODELS,
    "volcengine": VOLCENGINE_MODELS,
    "aliyun": ALIYUN_MODELS,
    "tuzi-web": TUZI_WEB_MODELS,
    "tuzi-openai": TUZI_OPENAI_MODELS,
    "tuzi-google": TUZI_GOOGLE_MODELS,
    "tuzi-anthropic": TUZI_ANTHROPIC_MODELS,
}


SUPPORTED_TESTS: dict[str, list[str]] = {
    "openai": [
        "chat",
        "tools",
        "transcription",
        "image",
        "video",
        "audio",
        "embedding",
    ],
    "anthropic": ["chat", "tools"],
    "google": [
        "chat",
        "tools",
        "transcription",
        "image",
        "video",
        "audio",
        "embedding",
    ],
    "volcengine": ["chat", "image", "video", "embedding"],
    "aliyun": ["chat", "transcription", "image", "video", "audio", "embedding"],
    "tuzi-web": [
        "chat",
        "tools",
        "transcription",
        "image",
        "video",
        "audio",
        "embedding",
    ],
    "tuzi-openai": [
        "chat",
        "tools",
        "transcription",
        "image",
        "video",
        "audio",
        "embedding",
    ],
    "tuzi-google": [
        "chat",
        "tools",
        "transcription",
        "image",
        "video",
        "audio",
        "embedding",
    ],
    "tuzi-anthropic": ["chat", "tools"],
}
