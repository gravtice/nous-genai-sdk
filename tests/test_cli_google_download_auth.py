import io
import unittest
from contextlib import redirect_stdout
from unittest.mock import patch


class TestCliGoogleDownloadAuth(unittest.TestCase):
    def test_cli_download_uses_adapter_headers_for_same_host(self) -> None:
        import nous.genai.cli as cli
        from nous.genai.types import (
            Capability,
            GenerateRequest,
            GenerateResponse,
            Message,
            Part,
            PartSourceUrl,
        )

        class DummyAdapter:
            base_url = "https://generativelanguage.googleapis.com"

            def _download_headers(self) -> dict[str, str]:
                return {"x-goog-api-key": "k"}

        class DummyClient:
            def __init__(self) -> None:
                self.last_request: GenerateRequest | None = None

            def _adapter(self, provider: str) -> DummyAdapter:  # noqa: ARG002
                return DummyAdapter()

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
                                        url="https://generativelanguage.googleapis.com/v1beta/files/f1:download?alt=media"
                                    ),
                                )
                            ],
                        )
                    ],
                )

        dummy = DummyClient()
        captured: dict[str, object] = {}

        def fake_download_to_file(**kwargs) -> None:
            captured.update(kwargs)

        buf = io.StringIO()
        with (
            patch.object(cli, "Client", return_value=dummy),
            patch.object(
                cli,
                "download_to_file",
                side_effect=fake_download_to_file,
            ),
        ):
            with redirect_stdout(buf):
                cli.main(
                    [
                        "--model",
                        "google:veo-2.0-generate-001",
                        "--prompt",
                        "hi",
                        "--output-path",
                        "out.mp4",
                    ]
                )

        self.assertEqual(
            captured.get("url"),
            "https://generativelanguage.googleapis.com/v1beta/files/f1:download?alt=media",
        )
        self.assertEqual(captured.get("output_path"), "out.mp4")
        self.assertEqual(captured.get("headers"), {"x-goog-api-key": "k"})
        self.assertIn("[OK] downloaded to out.mp4", buf.getvalue())
