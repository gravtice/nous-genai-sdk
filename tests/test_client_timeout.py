import os
import unittest
from unittest.mock import patch


class TestClientTimeout(unittest.TestCase):
    def test_generate_passes_request_timeout_ms(self) -> None:
        from nous.genai.client import Client
        from nous.genai.types import (
            Capability,
            GenerateParams,
            GenerateRequest,
            GenerateResponse,
            Message,
            OutputSpec,
            Part,
        )

        class DummyAdapter:
            def __init__(self) -> None:
                self.last_timeout_ms: int | None = None

            def capabilities(self, model_id: str) -> Capability:  # noqa: ARG002
                return Capability(
                    input_modalities={"text"},
                    output_modalities={"text"},
                    supports_stream=False,
                    supports_job=False,
                )

            def generate(self, request: GenerateRequest, *, stream: bool):  # noqa: ARG002
                self.last_timeout_ms = request.params.timeout_ms
                return GenerateResponse(
                    id="r1",
                    provider="dummy",
                    model=request.model,
                    status="completed",
                    output=[Message(role="assistant", content=[Part.from_text("ok")])],
                )

        client = Client()
        adapter = DummyAdapter()
        req = GenerateRequest(
            model="dummy:demo",
            input=[Message(role="user", content=[Part.from_text("hi")])],
            output=OutputSpec(modalities=["text"]),
            params=GenerateParams(timeout_ms=999),
        )

        with patch.object(client, "_adapter", return_value=adapter):
            client.generate(req)

        self.assertEqual(adapter.last_timeout_ms, 999)

    def test_generate_defaults_timeout_ms_from_env(self) -> None:
        from nous.genai.client import Client
        from nous.genai.types import (
            Capability,
            GenerateRequest,
            GenerateResponse,
            Message,
            OutputSpec,
            Part,
        )

        class DummyAdapter:
            def __init__(self) -> None:
                self.last_timeout_ms: int | None = None

            def capabilities(self, model_id: str) -> Capability:  # noqa: ARG002
                return Capability(
                    input_modalities={"text"},
                    output_modalities={"text"},
                    supports_stream=False,
                    supports_job=False,
                )

            def generate(self, request: GenerateRequest, *, stream: bool):  # noqa: ARG002
                self.last_timeout_ms = request.params.timeout_ms
                return GenerateResponse(
                    id="r1",
                    provider="dummy",
                    model=request.model,
                    status="completed",
                    output=[Message(role="assistant", content=[Part.from_text("ok")])],
                )

        with patch.dict(os.environ, {"NOUS_GENAI_TIMEOUT_MS": "12345"}, clear=False):
            client = Client()
            adapter = DummyAdapter()
            req = GenerateRequest(
                model="dummy:demo",
                input=[Message(role="user", content=[Part.from_text("hi")])],
                output=OutputSpec(modalities=["text"]),
            )
            with patch.object(client, "_adapter", return_value=adapter):
                client.generate(req)

        self.assertEqual(adapter.last_timeout_ms, 12345)

    def test_generate_stream_passes_request_timeout_ms(self) -> None:
        from nous.genai.client import Client
        from nous.genai.types import (
            Capability,
            GenerateEvent,
            GenerateParams,
            GenerateRequest,
            Message,
            OutputSpec,
            Part,
        )

        class DummyAdapter:
            def __init__(self) -> None:
                self.last_timeout_ms: int | None = None

            def capabilities(self, model_id: str) -> Capability:  # noqa: ARG002
                return Capability(
                    input_modalities={"text"},
                    output_modalities={"text"},
                    supports_stream=True,
                    supports_job=False,
                )

            def generate(self, request: GenerateRequest, *, stream: bool):
                if not stream:
                    raise AssertionError("expected stream=True")
                self.last_timeout_ms = request.params.timeout_ms
                return iter([GenerateEvent(type="done")])

        client = Client()
        adapter = DummyAdapter()
        req = GenerateRequest(
            model="dummy:demo",
            input=[Message(role="user", content=[Part.from_text("hi")])],
            output=OutputSpec(modalities=["text"]),
            params=GenerateParams(timeout_ms=456),
        )

        with patch.object(client, "_adapter", return_value=adapter):
            list(client.generate_stream(req))

        self.assertEqual(adapter.last_timeout_ms, 456)
