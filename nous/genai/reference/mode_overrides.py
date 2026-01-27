from __future__ import annotations

from typing import TypedDict


class CapabilityOverride(TypedDict, total=False):
    supports_stream: bool
    supports_job: bool


CAPABILITY_OVERRIDES: dict[str, dict[str, CapabilityOverride]] = {
    # NOTE: Keep this table small and explicit.
    #
    # Purpose:
    # - Fix known capability inference mistakes from heuristic `Adapter.capabilities(model_id)`.
    # - Provide a reviewable, auditable source of truth for edge models.
    #
    # Keying:
    # - provider: normalized lower-case provider name (e.g. "tuzi-web")
    # - model_id: exact catalog model id string (case-sensitive)
    #
    # Example:
    # "tuzi-web": {
    #     "some-model-id": {"supports_stream": False, "supports_job": True},
    # },
}
