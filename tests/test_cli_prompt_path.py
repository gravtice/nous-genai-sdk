import io
import os
import tempfile
import unittest
from contextlib import redirect_stdout
from unittest.mock import patch


class TestCliPromptPath(unittest.TestCase):
    def test_prompt_path_reads_file_when_prompt_missing(self) -> None:
        import nous.genai.cli as cli
        from nous.genai.types import (
            Capability,
            GenerateRequest,
            GenerateResponse,
            Message,
            Part,
        )

        class DummyClient:
            def __init__(self) -> None:
                self.last_request: GenerateRequest | None = None

            def capabilities(self, model: str) -> Capability:  # noqa: ARG002
                return Capability(
                    input_modalities={"text"},
                    output_modalities={"text"},
                    supports_stream=False,
                    supports_job=False,
                )

            def generate(self, request: GenerateRequest, *, stream: bool = False):  # noqa: ARG002
                self.last_request = request
                return GenerateResponse(
                    id="r1",
                    provider="openai",
                    model=request.model,
                    status="completed",
                    output=[Message(role="assistant", content=[Part.from_text("ok")])],
                )

        dummy = DummyClient()
        with tempfile.NamedTemporaryFile("w", encoding="utf-8", delete=False) as f:
            f.write("hello from file")
            prompt_path = f.name

        try:
            buf = io.StringIO()
            with patch.object(cli, "Client", return_value=dummy):
                with redirect_stdout(buf):
                    cli.main(
                        ["--model", "openai:gpt-4o-mini", "--prompt-path", prompt_path]
                    )
            self.assertIsNotNone(dummy.last_request)
            assert dummy.last_request is not None
            self.assertEqual(
                dummy.last_request.input[0].content[0].text, "hello from file"
            )
        finally:
            try:
                os.unlink(prompt_path)
            except OSError:
                pass

    def test_prompt_takes_priority_over_prompt_path(self) -> None:
        import nous.genai.cli as cli
        from nous.genai.types import (
            Capability,
            GenerateRequest,
            GenerateResponse,
            Message,
            Part,
        )

        class DummyClient:
            def __init__(self) -> None:
                self.last_request: GenerateRequest | None = None

            def capabilities(self, model: str) -> Capability:  # noqa: ARG002
                return Capability(
                    input_modalities={"text"},
                    output_modalities={"text"},
                    supports_stream=False,
                    supports_job=False,
                )

            def generate(self, request: GenerateRequest, *, stream: bool = False):  # noqa: ARG002
                self.last_request = request
                return GenerateResponse(
                    id="r1",
                    provider="openai",
                    model=request.model,
                    status="completed",
                    output=[Message(role="assistant", content=[Part.from_text("ok")])],
                )

        with tempfile.NamedTemporaryFile("w", encoding="utf-8", delete=False) as f:
            f.write("should not be read")
            prompt_path = f.name
        try:
            os.unlink(prompt_path)
        except OSError:
            pass

        dummy = DummyClient()
        buf = io.StringIO()
        with patch.object(cli, "Client", return_value=dummy):
            with redirect_stdout(buf):
                cli.main(
                    [
                        "--model",
                        "openai:gpt-4o-mini",
                        "--prompt",
                        "cli prompt",
                        "--prompt-path",
                        prompt_path,
                    ]
                )

        self.assertIsNotNone(dummy.last_request)
        assert dummy.last_request is not None
        self.assertEqual(dummy.last_request.input[0].content[0].text, "cli prompt")
