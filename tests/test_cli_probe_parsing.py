import unittest


class TestCliProbeParsing(unittest.TestCase):
    def test_parse_probe_models_csv_and_dedup(self) -> None:
        from nous.genai.cli import _parse_probe_models

        out = _parse_probe_models("openai", "gpt-4o-mini, gpt-4o-mini , gpt-4")
        self.assertEqual(out, ["gpt-4o-mini", "gpt-4"])

    def test_parse_probe_models_accepts_provider_prefix(self) -> None:
        from nous.genai.cli import _parse_probe_models

        out = _parse_probe_models("openai", "openai:gpt-4o-mini,openai:gpt-4")
        self.assertEqual(out, ["gpt-4o-mini", "gpt-4"])

    def test_parse_probe_models_rejects_provider_mismatch(self) -> None:
        from nous.genai.cli import _parse_probe_models

        with self.assertRaises(SystemExit):
            _parse_probe_models("openai", "google:gemini-2.0-flash")
