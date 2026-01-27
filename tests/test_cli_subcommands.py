import unittest
from contextlib import redirect_stderr
from contextlib import redirect_stdout
import io
from unittest.mock import patch


class TestCliSubcommands(unittest.TestCase):
    def test_mapping_subcommand(self) -> None:
        import nous.genai.cli as cli

        with patch.object(cli, "_print_mappings") as fn:
            cli.main(["mapping"])
            fn.assert_called_once()

    def test_model_sdk_subcommand(self) -> None:
        import nous.genai.cli as cli

        with patch.object(cli, "_print_sdk_supported") as fn:
            cli.main(["model", "sdk"])
            fn.assert_called_once()

    def test_model_available_requires_provider(self) -> None:
        import nous.genai.cli as cli

        with redirect_stderr(io.StringIO()):
            with self.assertRaises(SystemExit):
                cli.main(["model", "available"])

    def test_model_available_calls_handler(self) -> None:
        import nous.genai.cli as cli

        with patch.object(cli, "_print_available_models") as fn:
            cli.main(["model", "available", "--provider", "openai"])
            fn.assert_called_once()
            args, kwargs = fn.call_args
            self.assertEqual(args[0], "openai")
            self.assertIsNone(kwargs.get("timeout_ms"))

    def test_model_available_all_calls_handler(self) -> None:
        import nous.genai.cli as cli

        with patch.object(cli, "_print_all_available_models") as fn:
            cli.main(["model", "available", "--all"])
            fn.assert_called_once()

    def test_token_generate_prints_sk_token(self) -> None:
        import nous.genai.cli as cli

        buf = io.StringIO()
        with redirect_stdout(buf):
            cli.main(["token", "generate"])
        token = buf.getvalue().strip()
        self.assertTrue(token.startswith("sk-"))
        self.assertGreater(len(token), 10)
