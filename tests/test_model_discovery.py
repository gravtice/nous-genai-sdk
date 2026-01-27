import unittest
from unittest.mock import patch


class TestModelDiscovery(unittest.TestCase):
    def test_client_does_not_expose_list_models_alias(self) -> None:
        from nous.genai.client import Client

        self.assertFalse(hasattr(Client, "list_models"))

    def test_get_sdk_supported_models_for_provider_matches_catalog(self) -> None:
        from nous.genai.reference.catalog import MODEL_CATALOG
        from nous.genai.reference import get_sdk_supported_models_for_provider

        expected = set(MODEL_CATALOG["openai"])

        rows = get_sdk_supported_models_for_provider("openai")
        got = {m["model_id"] for m in rows}
        self.assertEqual(got, expected)

    def test_list_available_models_is_intersection(self) -> None:
        from nous.genai.reference.catalog import MODEL_CATALOG
        from nous.genai.client import Client

        supported = MODEL_CATALOG["openai"][0]
        client = Client()
        with patch.object(client, "list_provider_models", return_value=[supported, "__unknown__"]):
            self.assertEqual(client.list_available_models("openai"), [supported])

    def test_list_unsupported_models_is_difference(self) -> None:
        from nous.genai.reference.catalog import MODEL_CATALOG
        from nous.genai.client import Client

        supported = MODEL_CATALOG["openai"][0]
        client = Client()
        with patch.object(client, "list_provider_models", return_value=[supported, "__unknown__"]):
            self.assertEqual(client.list_unsupported_models("openai"), ["__unknown__"])

    def test_list_stale_models_is_difference(self) -> None:
        from nous.genai.reference.catalog import MODEL_CATALOG
        from nous.genai.client import Client

        supported_set = set(MODEL_CATALOG["openai"])
        supported = sorted(supported_set)
        missing = supported[0]
        remote = supported[1:]

        client = Client()
        with patch.object(client, "list_provider_models", return_value=remote):
            self.assertEqual(client.list_stale_models("openai"), [missing])

    def test_list_all_available_models_is_qualified_and_sorted(self) -> None:
        from nous.genai.client import Client

        client = Client()
        with patch("nous.genai.reference.get_supported_providers", return_value=["b", "a"]):
            calls: list[tuple[str, int | None]] = []

            def _side_effect(provider: str, *, timeout_ms: int | None = None) -> list[str]:
                calls.append((provider, timeout_ms))
                if provider == "a":
                    return ["m1", "m2"]
                if provider == "b":
                    return ["x"]
                return []

            with patch.object(client, "list_available_models", side_effect=_side_effect):
                out = client.list_all_available_models(timeout_ms=123)

        self.assertEqual(calls, [("a", 123), ("b", 123)])
        self.assertEqual(out, ["a:m1", "a:m2", "b:x"])
