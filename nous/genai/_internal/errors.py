from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class ErrorInfo:
    type: str
    message: str
    provider_code: str | None = None
    retryable: bool = False


class GenAIError(RuntimeError):
    def __init__(self, info: ErrorInfo):
        super().__init__(info.message)
        self.info = info


def auth_error(message: str, provider_code: str | None = None) -> GenAIError:
    return GenAIError(ErrorInfo(type="AuthError", message=message, provider_code=provider_code))


def rate_limit_error(message: str, provider_code: str | None = None) -> GenAIError:
    return GenAIError(
        ErrorInfo(
            type="RateLimitError",
            message=message,
            provider_code=provider_code,
            retryable=True,
        )
    )


def invalid_request_error(message: str, provider_code: str | None = None) -> GenAIError:
    return GenAIError(
        ErrorInfo(type="InvalidRequestError", message=message, provider_code=provider_code)
    )


def not_supported_error(message: str) -> GenAIError:
    return GenAIError(ErrorInfo(type="NotSupportedError", message=message))


def timeout_error(message: str) -> GenAIError:
    return GenAIError(ErrorInfo(type="TimeoutError", message=message, retryable=True))


def provider_error(message: str, provider_code: str | None = None, retryable: bool = False) -> GenAIError:
    return GenAIError(
        ErrorInfo(type="ProviderError", message=message, provider_code=provider_code, retryable=retryable)
    )

