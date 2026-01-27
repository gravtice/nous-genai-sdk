import unittest
from unittest.mock import patch


class TestOpenAIResponsesStreaming(unittest.TestCase):
    def _request(self):
        from nous.genai.types import GenerateRequest, Message, OutputSpec, Part

        return GenerateRequest(
            model="openai:gpt-5.2",
            input=[Message(role="user", content=[Part.from_text("hi")])],
            output=OutputSpec(modalities=["text"]),
        )

    def test_error_event_raises_provider_error(self) -> None:
        from nous.genai import GenAIError
        from nous.genai.providers.openai import OpenAIAdapter

        req = self._request()
        adapter = OpenAIAdapter(
            api_key="__demo__",
            base_url="https://example.invalid/v1",
            provider_name="openai",
            chat_api="responses",
        )

        with patch("nous.genai.providers.openai.request_stream_json_sse") as request_stream_json_sse:
            request_stream_json_sse.return_value = iter(
                [
                    {"type": "response.output_text.delta", "delta": "hi"},
                    {
                        "type": "error",
                        "code": "ERR_SOMETHING",
                        "message": "Something went wrong",
                        "param": None,
                        "sequence_number": 1,
                    },
                ]
            )

            it = adapter.generate(req, stream=True)
            ev = next(it)
            self.assertEqual(ev.type, "output.text.delta")
            self.assertEqual(ev.data.get("delta"), "hi")

            with self.assertRaises(GenAIError) as cm:
                next(it)
            self.assertEqual(cm.exception.info.type, "ProviderError")
            self.assertEqual(cm.exception.info.provider_code, "ERR_SOMETHING")
            self.assertIn("Something went wrong", str(cm.exception))

    def test_failed_event_raises_provider_error(self) -> None:
        from nous.genai import GenAIError
        from nous.genai.providers.openai import OpenAIAdapter

        req = self._request()
        adapter = OpenAIAdapter(
            api_key="__demo__",
            base_url="https://example.invalid/v1",
            provider_name="openai",
            chat_api="responses",
        )

        with patch("nous.genai.providers.openai.request_stream_json_sse") as request_stream_json_sse:
            request_stream_json_sse.return_value = iter(
                [
                    {
                        "type": "response.failed",
                        "sequence_number": 1,
                        "response": {
                            "error": {"code": "server_error", "message": "The model failed to generate a response."}
                        },
                    }
                ]
            )

            it = adapter.generate(req, stream=True)
            with self.assertRaises(GenAIError) as cm:
                next(it)
            self.assertEqual(cm.exception.info.type, "ProviderError")
            self.assertEqual(cm.exception.info.provider_code, "server_error")
            self.assertIn("failed to generate", str(cm.exception))

    def test_incomplete_event_raises_provider_error(self) -> None:
        from nous.genai import GenAIError
        from nous.genai.providers.openai import OpenAIAdapter

        req = self._request()
        adapter = OpenAIAdapter(
            api_key="__demo__",
            base_url="https://example.invalid/v1",
            provider_name="openai",
            chat_api="responses",
        )

        with patch("nous.genai.providers.openai.request_stream_json_sse") as request_stream_json_sse:
            request_stream_json_sse.return_value = iter(
                [
                    {
                        "type": "response.incomplete",
                        "sequence_number": 1,
                        "response": {"incomplete_details": {"reason": "max_output_tokens"}},
                    }
                ]
            )

            it = adapter.generate(req, stream=True)
            with self.assertRaises(GenAIError) as cm:
                next(it)
            self.assertEqual(cm.exception.info.type, "ProviderError")
            self.assertEqual(cm.exception.info.provider_code, None)
            self.assertIn("incomplete", str(cm.exception))
            self.assertIn("max_output_tokens", str(cm.exception))

