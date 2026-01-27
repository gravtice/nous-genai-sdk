import tempfile
import unittest
from unittest.mock import patch


class TestClientProtectedUrlArtifacts(unittest.TestCase):
    def test_externalize_protected_url_to_artifact(self) -> None:
        from nous.genai.client import Client
        from nous.genai.types import GenerateResponse, Message, Part, PartSourceUrl

        class DummyStore:
            def __init__(self) -> None:
                self.items: list[tuple[bytes, str | None]] = []

            def put(self, data: bytes, mime_type: str | None) -> str | None:
                self.items.append((data, mime_type))
                return "a1"

            def url(self, artifact_id: str) -> str:
                return f"http://artifact/{artifact_id}"

        class DummyAdapter:
            base_url = "https://generativelanguage.googleapis.com"

            def _download_headers(self) -> dict[str, str]:
                return {"x-goog-api-key": "k"}

        def fake_download_to_tempfile(**_kwargs) -> str:
            with tempfile.NamedTemporaryFile(
                prefix="genaisdk-test-", delete=False
            ) as f:
                f.write(b"video-bytes")
                return f.name

        client = Client(artifact_store=DummyStore())
        resp = GenerateResponse(
            id="r1",
            provider="google",
            model="google:models/veo-2.0-generate-001",
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

        with patch(
            "nous.genai.client.download_to_tempfile",
            side_effect=fake_download_to_tempfile,
        ):
            out = client._externalize_protected_url_parts(
                resp, adapter=DummyAdapter(), timeout_ms=1
            )

        part = out.output[0].content[0]
        assert part.source is not None
        self.assertEqual(part.source.kind, "url")
        self.assertEqual(part.source.url, "http://artifact/a1")

    def test_download_to_file_adds_headers_only_for_same_host(self) -> None:
        from nous.genai.client import Client

        class DummyAdapter:
            base_url = "https://generativelanguage.googleapis.com"

            def _download_headers(self) -> dict[str, str]:
                return {"x-goog-api-key": "k"}

        client = Client()
        captured: dict[str, object] = {}

        def fake_download_to_file(**kwargs) -> None:
            captured.update(kwargs)

        with (
            patch.object(client, "_adapter", return_value=DummyAdapter()),
            patch(
                "nous.genai.client._download_to_file",
                side_effect=fake_download_to_file,
            ),
        ):
            client.download_to_file(
                provider="google",
                url="https://generativelanguage.googleapis.com/v1beta/files/f1:download?alt=media",
                output_path="out.mp4",
            )
        self.assertEqual(captured.get("headers"), {"x-goog-api-key": "k"})

        captured.clear()
        with (
            patch.object(client, "_adapter", return_value=DummyAdapter()),
            patch(
                "nous.genai.client._download_to_file",
                side_effect=fake_download_to_file,
            ),
        ):
            client.download_to_file(
                provider="google",
                url="https://example.com/video.mp4",
                output_path="out.mp4",
            )
        self.assertIsNone(captured.get("headers"))
