import unittest


class TestCapabilityFlags(unittest.TestCase):
    def test_openai_chat_supports_tools_and_json_schema(self) -> None:
        from nous.genai.providers.openai import OpenAIAdapter

        cap = OpenAIAdapter(api_key="__demo__").capabilities("gpt-4o-mini")
        self.assertEqual(cap.supports_tools, True)
        self.assertEqual(cap.supports_json_schema, True)

    def test_openai_image_does_not_support_tools_or_json_schema(self) -> None:
        from nous.genai.providers.openai import OpenAIAdapter

        cap = OpenAIAdapter(api_key="__demo__").capabilities("gpt-image-1")
        self.assertEqual(cap.supports_tools, False)
        self.assertEqual(cap.supports_json_schema, False)

    def test_gemini_chat_supports_tools_and_json_schema(self) -> None:
        from nous.genai.providers.gemini import GeminiAdapter

        cap = GeminiAdapter(api_key="__demo__").capabilities("gemini-1.5-flash")
        self.assertEqual(cap.supports_tools, True)
        self.assertEqual(cap.supports_json_schema, True)

    def test_anthropic_chat_supports_tools_but_not_json_schema(self) -> None:
        from nous.genai.providers.anthropic import AnthropicAdapter

        cap = AnthropicAdapter(api_key="__demo__").capabilities("claude-3-5-sonnet-20240620")
        self.assertEqual(cap.supports_tools, True)
        self.assertEqual(cap.supports_json_schema, False)

