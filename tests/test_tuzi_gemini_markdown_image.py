import unittest


class TestTuziGeminiMarkdownImage(unittest.TestCase):
    def test_tuzi_web_markdown_image_becomes_image_part(self) -> None:
        from nous.genai.providers.gemini import _gemini_part_to_parts

        parts = _gemini_part_to_parts(
            {"text": "\n![Image](https://example.com/a.png)\n"},
            provider_name="tuzi-web",
        )
        self.assertEqual(len(parts), 1)
        self.assertEqual(parts[0].type, "image")
        self.assertIsNotNone(parts[0].source)
        self.assertEqual(getattr(parts[0].source, "kind", None), "url")

    def test_google_markdown_image_stays_text(self) -> None:
        from nous.genai.providers.gemini import _gemini_part_to_parts

        parts = _gemini_part_to_parts(
            {"text": "\n![Image](https://example.com/a.png)\n"},
            provider_name="google",
        )
        self.assertEqual(len(parts), 1)
        self.assertEqual(parts[0].type, "text")
        self.assertIn("![Image](", parts[0].text or "")

    def test_tuzi_web_keeps_remaining_text(self) -> None:
        from nous.genai.providers.gemini import _gemini_part_to_parts

        parts = _gemini_part_to_parts(
            {"text": "prefix\\n![Image](https://example.com/a.png)\\nsuffix"},
            provider_name="tuzi-web",
        )
        self.assertEqual([p.type for p in parts], ["text", "image"])
        self.assertIn("prefix", parts[0].text or "")
        self.assertIn("suffix", parts[0].text or "")

