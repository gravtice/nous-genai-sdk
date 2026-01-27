import io
import unittest
from contextlib import redirect_stdout
from unittest.mock import patch


class TestCliVideoWaitDefault(unittest.TestCase):
    def test_cli_video_waits_by_default(self) -> None:
        import nous.genai.cli as cli
        from nous.genai.types import (
            Capability,
            GenerateRequest,
            GenerateResponse,
            Message,
            Part,
            PartSourceUrl,
        )

        class DummyClient:
            def __init__(self) -> None:
                self.last_request: GenerateRequest | None = None

            def capabilities(self, model: str) -> Capability:  # noqa: ARG002
                return Capability(
                    input_modalities={"text"},
                    output_modalities={"video"},
                    supports_stream=False,
                    supports_job=True,
                )

            def generate(self, request: GenerateRequest, *, stream: bool = False):  # noqa: ARG002
                self.last_request = request
                return GenerateResponse(
                    id="r1",
                    provider="google",
                    model=request.model,
                    status="completed",
                    output=[
                        Message(
                            role="assistant",
                            content=[
                                Part(
                                    type="video",
                                    mime_type="video/mp4",
                                    source=PartSourceUrl(
                                        url="https://example.com/video.mp4"
                                    ),
                                )
                            ],
                        )
                    ],
                )

        dummy = DummyClient()
        buf = io.StringIO()
        with patch.object(cli, "Client", return_value=dummy):
            with redirect_stdout(buf):
                cli.main(
                    [
                        "--model",
                        "google:veo-2.0-generate-001",
                        "--prompt",
                        "hi",
                    ]
                )

        self.assertIsNotNone(dummy.last_request)
        assert dummy.last_request is not None
        self.assertTrue(dummy.last_request.wait)
        self.assertIn("https://example.com/video.mp4", buf.getvalue())
