import io
import unittest
from contextlib import redirect_stderr, redirect_stdout
from unittest.mock import patch


class TestCliTimeoutHint(unittest.TestCase):
    def test_cli_prints_hint_on_running_job(self) -> None:
        import nous.genai.cli as cli
        from nous.genai.types import (
            Capability,
            GenerateRequest,
            GenerateResponse,
            JobInfo,
        )

        class DummyClient:
            def capabilities(self, model: str) -> Capability:  # noqa: ARG002
                return Capability(
                    input_modalities={"text"},
                    output_modalities={"video"},
                    supports_stream=False,
                    supports_job=True,
                )

            def generate(self, request: GenerateRequest, *, stream: bool = False):  # noqa: ARG002
                return GenerateResponse(
                    id="r1",
                    provider="tuzi-openai",
                    model=request.model,
                    status="running",
                    job=JobInfo(job_id="jid", poll_after_ms=1_000),
                )

        out = io.StringIO()
        err = io.StringIO()
        with patch.object(cli, "Client", return_value=DummyClient()):
            with redirect_stdout(out), redirect_stderr(err):
                cli.main(
                    [
                        "--timeout-ms",
                        "1234",
                        "--model",
                        "tuzi-openai:sora-2",
                        "--prompt",
                        "hi",
                        "--output-path",
                        "out.mp4",
                    ]
                )

        self.assertEqual(out.getvalue().strip(), "jid")
        stderr = err.getvalue()
        self.assertIn("job_id", stderr)
        self.assertIn("--timeout-ms", stderr)
        self.assertIn("out.mp4", stderr)
