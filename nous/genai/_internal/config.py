from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path


_ENV_PRIORITY = (".env.local", ".env.production", ".env.development", ".env.test")

_ENV_PREFIX = "NOUS_GENAI_"


def get_prefixed_env(name: str) -> str | None:
    return os.environ.get(f"{_ENV_PREFIX}{name}")


def _parse_env_line(line: str) -> tuple[str, str] | None:
    stripped = line.strip()
    if not stripped or stripped.startswith("#"):
        return None
    if "=" not in stripped:
        return None
    key, value = stripped.split("=", 1)
    key = key.strip()
    value = value.strip()
    if not key:
        return None
    if (value.startswith("'") and value.endswith("'")) or (
        value.startswith('"') and value.endswith('"')
    ):
        value = value[1:-1]
    return key, value


def load_env_files(root: str | Path | None = None) -> list[Path]:
    """
    Load env files by priority:
    `.env.local > .env.production > .env.development > .env.test`.

    Implementation: apply higher priority first without overriding existing env.
    """
    base = Path(root) if root is not None else Path.cwd()
    loaded: list[Path] = []
    for name in _ENV_PRIORITY:
        path = base / name
        if not path.is_file():
            continue
        loaded.append(path)
        for line in path.read_text(encoding="utf-8").splitlines():
            parsed = _parse_env_line(line)
            if parsed is None:
                continue
            key, value = parsed
            os.environ.setdefault(key, value)
    return loaded


@dataclass(frozen=True, slots=True)
class ProviderKeys:
    openai_api_key: str | None
    google_api_key: str | None
    anthropic_api_key: str | None
    aliyun_api_key: str | None
    volcengine_api_key: str | None
    tuzi_web_api_key: str | None
    tuzi_openai_api_key: str | None
    tuzi_google_api_key: str | None
    tuzi_anthropic_api_key: str | None


def get_provider_keys() -> ProviderKeys:
    return ProviderKeys(
        openai_api_key=get_prefixed_env("OPENAI_API_KEY") or os.environ.get("OPENAI_API_KEY"),
        google_api_key=get_prefixed_env("GOOGLE_API_KEY") or os.environ.get("GOOGLE_API_KEY"),
        anthropic_api_key=get_prefixed_env("ANTHROPIC_API_KEY") or os.environ.get("ANTHROPIC_API_KEY"),
        aliyun_api_key=get_prefixed_env("ALIYUN_API_KEY") or os.environ.get("ALIYUN_API_KEY"),
        volcengine_api_key=get_prefixed_env("VOLCENGINE_API_KEY") or os.environ.get("VOLCENGINE_API_KEY"),
        tuzi_web_api_key=get_prefixed_env("TUZI_WEB_API_KEY") or os.environ.get("TUZI_WEB_API_KEY"),
        tuzi_openai_api_key=get_prefixed_env("TUZI_OPENAI_API_KEY") or os.environ.get("TUZI_OPENAI_API_KEY"),
        tuzi_google_api_key=get_prefixed_env("TUZI_GOOGLE_API_KEY") or os.environ.get("TUZI_GOOGLE_API_KEY"),
        tuzi_anthropic_api_key=get_prefixed_env("TUZI_ANTHROPIC_API_KEY")
        or os.environ.get("TUZI_ANTHROPIC_API_KEY"),
    )


def get_default_timeout_ms() -> int:
    raw = get_prefixed_env("TIMEOUT_MS")
    if raw is None:
        return 120_000
    try:
        value = int(raw)
    except ValueError:
        return 120_000
    return max(1, value)
