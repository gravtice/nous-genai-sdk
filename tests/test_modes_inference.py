import unittest
from unittest.mock import patch


class TestModesInference(unittest.TestCase):
    def test_modes_for_common_openai_models(self) -> None:
        from nous.genai.reference import get_sdk_supported_models_for_provider

        rows = get_sdk_supported_models_for_provider("openai")
        by_model = {r["model"]: r for r in rows}

        self.assertEqual(by_model["openai:gpt-4o-mini"]["modes"], ["sync", "stream", "async"])
        self.assertEqual(by_model["openai:dall-e-3"]["modes"], ["sync", "async"])
        self.assertEqual(by_model["openai:sora-2"]["modes"], ["sync", "job", "async"])

    def test_overrides_can_force_job_and_change_order(self) -> None:
        from nous.genai.reference import get_sdk_supported_models_for_provider

        overrides = {"openai": {"gpt-4o-mini": {"supports_job": True}}}
        with patch("nous.genai.reference.catalog.CAPABILITY_OVERRIDES", overrides):
            rows = get_sdk_supported_models_for_provider("openai")

        by_model = {r["model"]: r for r in rows}
        self.assertEqual(by_model["openai:gpt-4o-mini"]["modes"], ["sync", "stream", "job", "async"])
        self.assertIn("可能返回 running(job)", " ".join(by_model["openai:gpt-4o-mini"]["notes"]))

    def test_overrides_can_disable_stream(self) -> None:
        from nous.genai.reference import get_sdk_supported_models_for_provider

        overrides = {"openai": {"gpt-4o-mini": {"supports_stream": False}}}
        with patch("nous.genai.reference.catalog.CAPABILITY_OVERRIDES", overrides):
            rows = get_sdk_supported_models_for_provider("openai")

        by_model = {r["model"]: r for r in rows}
        self.assertEqual(by_model["openai:gpt-4o-mini"]["modes"], ["sync", "async"])

