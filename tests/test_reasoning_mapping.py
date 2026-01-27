import unittest
from unittest.mock import patch


class TestReasoningMapping(unittest.TestCase):
    def test_openai_chat_maps_reasoning_effort(self) -> None:
        from nous.genai.types import (
            GenerateParams,
            GenerateRequest,
            Message,
            OutputSpec,
            Part,
            ReasoningSpec,
        )
        from nous.genai.providers.openai import OpenAIAdapter

        req = GenerateRequest(
            model="openai:gpt-4o-mini",
            input=[Message(role="user", content=[Part.from_text("test")])],
            output=OutputSpec(modalities=["text"]),
            params=GenerateParams(reasoning=ReasoningSpec(effort="HIGH")),
        )

        with patch("nous.genai.providers.openai.request_json") as request_json:
            request_json.return_value = {
                "id": "chatcmpl_123",
                "choices": [{"message": {"content": "ok"}}],
            }
            adapter = OpenAIAdapter(
                api_key="__demo__",
                base_url="https://example.invalid/v1",
                provider_name="openai",
            )
            out = adapter.generate(req, stream=False)
            self.assertEqual(out.status, "completed")

            _, kwargs = request_json.call_args
            body = kwargs["json_body"]
            self.assertEqual(body.get("reasoning_effort"), "high")

    def test_openai_responses_maps_reasoning_effort(self) -> None:
        from nous.genai.types import (
            GenerateParams,
            GenerateRequest,
            Message,
            OutputSpec,
            Part,
            ReasoningSpec,
        )
        from nous.genai.providers.openai import OpenAIAdapter

        req = GenerateRequest(
            model="openai:gpt-5.2",
            input=[Message(role="user", content=[Part.from_text("test")])],
            output=OutputSpec(modalities=["text"]),
            params=GenerateParams(reasoning=ReasoningSpec(effort="low")),
        )

        with patch("nous.genai.providers.openai.request_json") as request_json:
            request_json.return_value = {
                "id": "resp_123",
                "status": "completed",
                "output": [
                    {
                        "type": "message",
                        "content": [{"type": "output_text", "text": "ok"}],
                    }
                ],
            }
            adapter = OpenAIAdapter(
                api_key="__demo__",
                base_url="https://example.invalid/v1",
                provider_name="openai",
                chat_api="responses",
            )
            out = adapter.generate(req, stream=False)
            self.assertEqual(out.status, "completed")

            _, kwargs = request_json.call_args
            body = kwargs["json_body"]
            self.assertEqual(body.get("reasoning"), {"effort": "low"})

    def test_anthropic_maps_reasoning_effort_to_budget(self) -> None:
        from nous.genai.types import (
            GenerateParams,
            GenerateRequest,
            Message,
            OutputSpec,
            OutputTextSpec,
            Part,
            ReasoningSpec,
        )
        from nous.genai.providers.anthropic import AnthropicAdapter

        req = GenerateRequest(
            model="anthropic:claude-sonnet-4-5",
            input=[Message(role="user", content=[Part.from_text("test")])],
            output=OutputSpec(
                modalities=["text"], text=OutputTextSpec(max_output_tokens=2048)
            ),
            params=GenerateParams(reasoning=ReasoningSpec(effort="minimal")),
        )

        with patch("nous.genai.providers.anthropic.request_json") as request_json:
            request_json.return_value = {
                "id": "msg_123",
                "content": [{"type": "text", "text": "ok"}],
            }
            adapter = AnthropicAdapter(
                api_key="__demo__",
                base_url="https://example.invalid",
                provider_name="anthropic",
            )
            out = adapter.generate(req, stream=False)
            self.assertEqual(out.status, "completed")

            _, kwargs = request_json.call_args
            body = kwargs["json_body"]
            self.assertEqual(
                body.get("thinking"), {"type": "enabled", "budget_tokens": 1024}
            )

    def test_gemini_maps_reasoning_effort_to_thinking_level_for_gemini3(self) -> None:
        from nous.genai.types import (
            GenerateParams,
            GenerateRequest,
            Message,
            OutputSpec,
            Part,
            ReasoningSpec,
        )
        from nous.genai.providers.gemini import GeminiAdapter

        req = GenerateRequest(
            model="google:gemini-3-pro-preview",
            input=[Message(role="user", content=[Part.from_text("test")])],
            output=OutputSpec(modalities=["text"]),
            params=GenerateParams(reasoning=ReasoningSpec(effort="high")),
        )

        with patch("nous.genai.providers.gemini.request_json") as request_json:
            request_json.return_value = {
                "candidates": [{"content": {"parts": [{"text": "ok"}]}}]
            }
            adapter = GeminiAdapter(
                api_key="__demo__",
                base_url="https://example.invalid",
                provider_name="google",
            )
            out = adapter.generate(req, stream=False)
            self.assertEqual(out.status, "completed")

            _, kwargs = request_json.call_args
            body = kwargs["json_body"]
            gen_cfg = body.get("generationConfig") or {}
            thinking_cfg = gen_cfg.get("thinkingConfig") or {}
            self.assertEqual(thinking_cfg.get("thinkingLevel"), "high")

    def test_gemini_maps_reasoning_effort_to_thinking_budget_for_gemini2(self) -> None:
        from nous.genai.types import (
            GenerateParams,
            GenerateRequest,
            Message,
            OutputSpec,
            Part,
            ReasoningSpec,
        )
        from nous.genai.providers.gemini import GeminiAdapter

        req = GenerateRequest(
            model="google:gemini-2.5-flash",
            input=[Message(role="user", content=[Part.from_text("test")])],
            output=OutputSpec(modalities=["text"]),
            params=GenerateParams(reasoning=ReasoningSpec(effort="low")),
        )

        with patch("nous.genai.providers.gemini.request_json") as request_json:
            request_json.return_value = {
                "candidates": [{"content": {"parts": [{"text": "ok"}]}}]
            }
            adapter = GeminiAdapter(
                api_key="__demo__",
                base_url="https://example.invalid",
                provider_name="google",
            )
            out = adapter.generate(req, stream=False)
            self.assertEqual(out.status, "completed")

            _, kwargs = request_json.call_args
            body = kwargs["json_body"]
            gen_cfg = body.get("generationConfig") or {}
            thinking_cfg = gen_cfg.get("thinkingConfig") or {}
            self.assertEqual(thinking_cfg.get("thinkingBudget"), 1024)

    def test_gemini_ignores_reasoning_for_unsupported_models(self) -> None:
        from nous.genai.types import (
            GenerateParams,
            GenerateRequest,
            Message,
            OutputSpec,
            Part,
            ReasoningSpec,
        )
        from nous.genai.providers.gemini import GeminiAdapter

        req = GenerateRequest(
            model="google:gemini-1.5-flash",
            input=[Message(role="user", content=[Part.from_text("test")])],
            output=OutputSpec(modalities=["text"]),
            params=GenerateParams(reasoning=ReasoningSpec(effort="high")),
        )

        with patch("nous.genai.providers.gemini.request_json") as request_json:
            request_json.return_value = {
                "candidates": [{"content": {"parts": [{"text": "ok"}]}}]
            }
            adapter = GeminiAdapter(
                api_key="__demo__",
                base_url="https://example.invalid",
                provider_name="google",
            )
            out = adapter.generate(req, stream=False)
            self.assertEqual(out.status, "completed")

            _, kwargs = request_json.call_args
            body = kwargs["json_body"]
            gen_cfg = body.get("generationConfig") or {}
            self.assertNotIn("thinkingConfig", gen_cfg)
