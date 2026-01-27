import asyncio
import os
import tempfile
import unittest
from unittest.mock import patch


class TestAnthropicImageUrlDoesNotCrash(unittest.TestCase):
    def test_messages_body_passes_proxy_url_to_downloader(self) -> None:
        from nous.genai.providers.anthropic import AnthropicAdapter
        from nous.genai.types import GenerateRequest, Message, OutputSpec, Part, PartSourceUrl

        with tempfile.NamedTemporaryFile(prefix="genaisdk-test-", suffix=".bin", delete=False) as f:
            tmp_path = f.name
            f.write(b"123")

        def _fake_download_to_tempfile(*, url: str, timeout_ms: int | None, max_bytes: int | None, proxy_url: str | None):  # type: ignore[no-untyped-def]
            self.assertEqual(url, "http://example.invalid/x.png")
            self.assertEqual(proxy_url, "http://proxy.local")
            return tmp_path

        adapter = AnthropicAdapter(api_key="__test__", proxy_url="http://proxy.local")
        req = GenerateRequest(
            model="anthropic:claude-sonnet-4",
            input=[
                Message(
                    role="user",
                    content=[
                        Part(type="image", mime_type="image/png", source=PartSourceUrl(url="http://example.invalid/x.png"))
                    ],
                )
            ],
            output=OutputSpec(modalities=["text"]),
        )
        try:
            with patch("nous.genai.providers.anthropic.download_to_tempfile", side_effect=_fake_download_to_tempfile):
                body = adapter._messages_body(req, stream=False)  # type: ignore[attr-defined]
        finally:
            try:
                os.unlink(tmp_path)
            except OSError:
                pass
        self.assertIsInstance(body, dict)


class TestUrlDownloadPrivateHostDnsBlocked(unittest.TestCase):
    def test_download_to_file_blocks_private_dns(self) -> None:
        from nous.genai import GenAIError
        from nous.genai._internal.http import download_to_file

        def _fake_getaddrinfo(host: str, port: object, proto: int):  # type: ignore[no-untyped-def]
            self.assertEqual(host, "evil.example")
            return [
                (socket.AF_INET, socket.SOCK_STREAM, proto, "", ("10.0.0.1", 0)),
            ]

        with tempfile.TemporaryDirectory(prefix="genaisdk-test-") as d:
            out_path = os.path.join(d, "out.bin")
            with patch.dict(os.environ, {"NOUS_GENAI_ALLOW_PRIVATE_URLS": "0"}, clear=False):
                with patch("nous.genai._internal.http.socket.getaddrinfo") as mock_getaddrinfo:
                    import socket

                    mock_getaddrinfo.side_effect = _fake_getaddrinfo
                    with self.assertRaises(GenAIError) as cm:
                        download_to_file(url="http://evil.example/x", output_path=out_path)
        self.assertEqual(cm.exception.info.type, "InvalidRequestError")
        self.assertIn("private/loopback", cm.exception.info.message)


class TestOpenAIProviderOptionsNoOverride(unittest.TestCase):
    def test_images_provider_options_cannot_override_prompt(self) -> None:
        from nous.genai import GenAIError
        from nous.genai.providers.openai import OpenAIAdapter
        from nous.genai.types import GenerateRequest, Message, OutputSpec, Part

        adapter = OpenAIAdapter(api_key="__test__")
        req = GenerateRequest(
            model="openai:gpt-image-1",
            input=[Message(role="user", content=[Part.from_text("hi")])],
            output=OutputSpec(modalities=["image"]),
            provider_options={"openai": {"prompt": "override"}},
        )
        with self.assertRaises(GenAIError) as cm:
            adapter.generate(req, stream=False)
        self.assertEqual(cm.exception.info.type, "InvalidRequestError")
        self.assertIn("cannot override body.prompt", cm.exception.info.message)

    def test_transcribe_provider_options_cannot_override_model(self) -> None:
        from nous.genai import GenAIError
        from nous.genai.providers.openai import OpenAIAdapter
        from nous.genai.types import GenerateRequest, Message, OutputSpec, Part, PartSourceBytes

        adapter = OpenAIAdapter(api_key="__test__")
        req = GenerateRequest(
            model="openai:whisper-1",
            input=[
                Message(
                    role="user",
                    content=[
                        Part(
                            type="audio",
                            mime_type="audio/wav",
                            source=PartSourceBytes(data="AA==", encoding="base64"),
                        )
                    ],
                )
            ],
            output=OutputSpec(modalities=["text"]),
            provider_options={"openai": {"model": "override"}},
        )
        with self.assertRaises(GenAIError) as cm:
            adapter.generate(req, stream=False)
        self.assertEqual(cm.exception.info.type, "InvalidRequestError")
        self.assertIn("cannot override fields.model", cm.exception.info.message)


class TestClientInputModalitiesValidated(unittest.TestCase):
    def test_client_rejects_unsupported_input_modality(self) -> None:
        from nous.genai import GenAIError
        from nous.genai.client import Client
        from nous.genai.types import Capability, GenerateRequest, Message, OutputSpec, Part, PartSourceBytes

        class _DummyAdapter:
            def capabilities(self, model_id: str) -> Capability:  # noqa: ARG002
                return Capability(
                    input_modalities={"text"},
                    output_modalities={"text"},
                    supports_stream=False,
                    supports_job=False,
                )

            def generate(self, request: object, *, stream: bool):  # noqa: ARG002
                raise AssertionError("adapter.generate should not be called")

        client = Client()
        req = GenerateRequest(
            model="dummy:demo",
            input=[
                Message(
                    role="user",
                    content=[
                        Part(
                            type="audio",
                            mime_type="audio/wav",
                            source=PartSourceBytes(data="AA==", encoding="base64"),
                        )
                    ],
                )
            ],
            output=OutputSpec(modalities=["text"]),
        )
        with patch.object(client, "_adapter", return_value=_DummyAdapter()):
            with self.assertRaises(GenAIError) as cm:
                client.generate(req)
        self.assertEqual(cm.exception.info.type, "NotSupportedError")
        self.assertIn("requested input modalities not supported", cm.exception.info.message)


class TestPartValidation(unittest.TestCase):
    def test_image_requires_source(self) -> None:
        from nous.genai.types import Part

        with self.assertRaises(ValueError) as cm:
            Part(type="image", mime_type="image/png", source=None)
        self.assertIn("requires source", str(cm.exception))


class TestMcpArtifactByteLimit(unittest.TestCase):
    def test_generate_converts_base64_to_artifact_url_under_limit(self) -> None:
        try:
            from mcp.server.fastmcp import FastMCP  # noqa: F401
        except ModuleNotFoundError:
            self.skipTest("missing dependency: mcp")

        from nous.genai.mcp_server import build_server
        from nous.genai.types import Capability, GenerateResponse, Message, Part, PartSourceBytes

        resp = GenerateResponse(
            id="r1",
            provider="openai",
            model="openai:gpt-image-1",
            status="completed",
            output=[
                Message(
                    role="assistant",
                    content=[
                        Part(
                            type="image",
                            mime_type="image/png",
                            source=PartSourceBytes(data="MTIzNA==", encoding="base64"),
                        )
                    ],
                )
            ],
        )
        req_dict = {
            "model": "openai:gpt-image-1",
            "input": [{"role": "user", "content": [{"type": "text", "text": "x"}]}],
            "output": {"modalities": ["image"]},
        }

        with patch.dict(
            os.environ,
            {
                "NOUS_GENAI_MAX_INLINE_BASE64_CHARS": "4",
                "NOUS_GENAI_MAX_ARTIFACTS": "64",
                "NOUS_GENAI_MAX_ARTIFACT_BYTES": "1024",
                "NOUS_GENAI_MCP_PUBLIC_BASE_URL": "",
            },
            clear=False,
        ):
            adapter = type(
                "DummyAdapter",
                (),
                {
                    "capabilities": lambda self, _: Capability(
                        input_modalities={"text"},
                        output_modalities={"image"},
                        supports_stream=False,
                        supports_job=False,
                    ),
                    "generate": lambda self, *_args, **_kwargs: resp,
                },
            )()
            with patch("nous.genai.client.Client._adapter", return_value=adapter):
                server = build_server(host="127.0.0.1", port=7001)
                out = server.call_tool("generate", {"request": req_dict})
                if asyncio.iscoroutine(out):
                    out = asyncio.run(out)

        if isinstance(out, tuple) and len(out) == 2 and isinstance(out[1], dict):
            out = out[1]
        self.assertIsInstance(out, dict)
        part = out["output"][0]["content"][0]
        self.assertEqual(part["source"]["kind"], "url")
        self.assertIn("http://127.0.0.1:7001/artifact/", part["source"]["url"])

    def test_generate_keeps_base64_when_over_artifact_bytes_limit(self) -> None:
        try:
            from mcp.server.fastmcp import FastMCP  # noqa: F401
        except ModuleNotFoundError:
            self.skipTest("missing dependency: mcp")

        from nous.genai.mcp_server import build_server
        from nous.genai.types import Capability, GenerateResponse, Message, Part, PartSourceBytes

        resp = GenerateResponse(
            id="r1",
            provider="openai",
            model="openai:gpt-image-1",
            status="completed",
            output=[
                Message(
                    role="assistant",
                    content=[
                        Part(
                            type="image",
                            mime_type="image/png",
                            source=PartSourceBytes(data="MTIzNA==", encoding="base64"),
                        )
                    ],
                )
            ],
        )
        req_dict = {
            "model": "openai:gpt-image-1",
            "input": [{"role": "user", "content": [{"type": "text", "text": "x"}]}],
            "output": {"modalities": ["image"]},
        }

        with patch.dict(
            os.environ,
            {
                "NOUS_GENAI_MAX_INLINE_BASE64_CHARS": "4",
                "NOUS_GENAI_MAX_ARTIFACTS": "64",
                "NOUS_GENAI_MAX_ARTIFACT_BYTES": "3",
            },
            clear=False,
        ):
            adapter = type(
                "DummyAdapter",
                (),
                {
                    "capabilities": lambda self, _: Capability(
                        input_modalities={"text"},
                        output_modalities={"image"},
                        supports_stream=False,
                        supports_job=False,
                    ),
                    "generate": lambda self, *_args, **_kwargs: resp,
                },
            )()
            with patch("nous.genai.client.Client._adapter", return_value=adapter):
                server = build_server(host="127.0.0.1", port=7001)
                out = server.call_tool("generate", {"request": req_dict})
                if asyncio.iscoroutine(out):
                    out = asyncio.run(out)

        if isinstance(out, tuple) and len(out) == 2 and isinstance(out[1], dict):
            out = out[1]
        self.assertIsInstance(out, dict)
        part = out["output"][0]["content"][0]
        self.assertEqual(part["source"]["kind"], "bytes")
        self.assertEqual(part["source"]["encoding"], "base64")
