import asyncio
import logging
import os
import unittest
from unittest.mock import patch
from urllib.parse import urlparse


class TestMcpArtifactEviction(unittest.TestCase):
    def test_max_artifacts_evicts_oldest(self) -> None:
        try:
            from mcp.server.fastmcp import FastMCP  # noqa: F401
        except ModuleNotFoundError:
            self.skipTest("missing dependency: mcp")

        try:
            from starlette.testclient import TestClient
        except ModuleNotFoundError:
            self.skipTest("missing dependency: starlette")

        from nous.genai.mcp_server import build_server
        from nous.genai.types import (
            Capability,
            GenerateResponse,
            Message,
            Part,
            PartSourceBytes,
        )

        logging.getLogger("httpx").setLevel(logging.WARNING)
        logging.getLogger("httpcore").setLevel(logging.WARNING)

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
                "NOUS_GENAI_MAX_INLINE_BASE64_CHARS": "1",
                "NOUS_GENAI_MAX_ARTIFACTS": "1",
                "NOUS_GENAI_MAX_ARTIFACT_BYTES": "1024",
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

                out1 = server.call_tool("generate", {"request": req_dict})
                if asyncio.iscoroutine(out1):
                    out1 = asyncio.run(out1)
                if (
                    isinstance(out1, tuple)
                    and len(out1) == 2
                    and isinstance(out1[1], dict)
                ):
                    out1 = out1[1]

                out2 = server.call_tool("generate", {"request": req_dict})
                if asyncio.iscoroutine(out2):
                    out2 = asyncio.run(out2)
                if (
                    isinstance(out2, tuple)
                    and len(out2) == 2
                    and isinstance(out2[1], dict)
                ):
                    out2 = out2[1]

                self.assertIsInstance(out1, dict)
                self.assertIsInstance(out2, dict)

                url1 = out1["output"][0]["content"][0]["source"]["url"]
                url2 = out2["output"][0]["content"][0]["source"]["url"]
                artifact_id1 = url1.rsplit("/", 1)[-1]
                artifact_id2 = url2.rsplit("/", 1)[-1]
                self.assertNotEqual(artifact_id1, artifact_id2)

                app = server.streamable_http_app()
                client = TestClient(app)
                r2 = client.get(f"/artifact/{artifact_id2}")
                self.assertEqual(r2.status_code, 200)
                r1 = client.get(f"/artifact/{artifact_id1}")
                self.assertEqual(r1.status_code, 404)

    def test_bearer_signed_artifact_url_allows_headerless_download(self) -> None:
        try:
            from mcp.server.fastmcp import FastMCP  # noqa: F401
        except ModuleNotFoundError:
            self.skipTest("missing dependency: mcp")

        try:
            from starlette.testclient import TestClient
        except ModuleNotFoundError:
            self.skipTest("missing dependency: starlette")

        from nous.genai.mcp_server import _BearerAuthMiddleware, build_server
        from nous.genai.types import (
            Capability,
            GenerateResponse,
            Message,
            Part,
            PartSourceBytes,
        )

        logging.getLogger("httpx").setLevel(logging.WARNING)
        logging.getLogger("httpcore").setLevel(logging.WARNING)

        bearer = "test-token"
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
                "NOUS_GENAI_MAX_INLINE_BASE64_CHARS": "1",
                "NOUS_GENAI_MAX_ARTIFACTS": "4",
                "NOUS_GENAI_MAX_ARTIFACT_BYTES": "1024",
                "NOUS_GENAI_ARTIFACT_URL_TTL_SECONDS": "600",
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
                server = build_server(host="127.0.0.1", port=7001, bearer_token=bearer)
                out = server.call_tool("generate", {"request": req_dict})
                if asyncio.iscoroutine(out):
                    out = asyncio.run(out)
                if (
                    isinstance(out, tuple)
                    and len(out) == 2
                    and isinstance(out[1], dict)
                ):
                    out = out[1]

                url = out["output"][0]["content"][0]["source"]["url"]
                parsed = urlparse(url)
                self.assertEqual(parsed.path.split("/", 2)[1], "artifact")
                self.assertIn("sig=", parsed.query)
                self.assertIn("exp=", parsed.query)
                artifact_id = parsed.path.rsplit("/", 1)[-1]

                app = server.streamable_http_app()
                app.add_middleware(_BearerAuthMiddleware, token=bearer)
                client = TestClient(app)

                r_missing = client.get(f"/artifact/{artifact_id}")
                self.assertEqual(r_missing.status_code, 401)

                r_header = client.get(
                    f"/artifact/{artifact_id}",
                    headers={"Authorization": f"Bearer {bearer}"},
                )
                self.assertEqual(r_header.status_code, 200)

                r_signed = client.get(f"{parsed.path}?{parsed.query}")
                self.assertEqual(r_signed.status_code, 200)
