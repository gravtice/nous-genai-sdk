import json
import unittest
from unittest.mock import patch


class TestToolCalling(unittest.TestCase):
    def test_openai_chat_tools_and_tool_choice(self) -> None:
        from nous.genai.types import GenerateRequest, Message, OutputSpec, Part, Tool, ToolChoice
        from nous.genai.providers.openai import OpenAIAdapter

        req = GenerateRequest(
            model="openai:gpt-4o-mini",
            input=[
                Message(role="user", content=[Part.from_text("hi")]),
                Message(
                    role="assistant",
                    content=[
                        Part.tool_call(tool_call_id="call_1", name="sum", arguments={"a": 1, "b": 2}),
                    ],
                ),
                Message(
                    role="tool",
                    content=[Part.tool_result(tool_call_id="call_1", name="sum", result={"value": 3})],
                ),
            ],
            output=OutputSpec(modalities=["text"]),
            tools=[
                Tool(
                    name="sum",
                    description="sum two integers",
                    parameters={
                        "type": "object",
                        "properties": {"a": {"type": "integer"}, "b": {"type": "integer"}},
                        "required": ["a", "b"],
                    },
                    strict=True,
                )
            ],
            tool_choice=ToolChoice(mode="tool", name="sum"),
        )

        with patch("nous.genai.providers.openai.request_json") as request_json:
            request_json.return_value = {
                "id": "chatcmpl_123",
                "choices": [
                    {
                        "message": {
                            "tool_calls": [
                                {
                                    "id": "call_2",
                                    "type": "function",
                                    "function": {"name": "sum", "arguments": "{\"a\":2,\"b\":3}"},
                                }
                            ]
                        }
                    }
                ],
            }
            adapter = OpenAIAdapter(api_key="__demo__", base_url="https://example.invalid/v1", provider_name="openai")
            out = adapter.generate(req, stream=False)
            self.assertEqual(out.status, "completed")
            self.assertEqual(out.output[0].content[0].type, "tool_call")
            self.assertEqual(out.output[0].content[0].meta.get("tool_call_id"), "call_2")
            self.assertEqual(out.output[0].content[0].meta.get("name"), "sum")
            self.assertEqual(out.output[0].content[0].meta.get("arguments"), {"a": 2, "b": 3})

            _, kwargs = request_json.call_args
            body = kwargs["json_body"]
            self.assertEqual(body["tool_choice"]["function"]["name"], "sum")
            self.assertEqual(body["tools"][0]["function"]["name"], "sum")
            self.assertEqual(body["tools"][0]["function"]["strict"], True)

            msgs = body["messages"]
            self.assertEqual(msgs[1]["tool_calls"][0]["id"], "call_1")
            self.assertEqual(msgs[1]["tool_calls"][0]["function"]["arguments"], json.dumps({"a": 1, "b": 2}, separators=(",", ":"), ensure_ascii=False))
            self.assertEqual(msgs[2]["role"], "tool")
            self.assertEqual(msgs[2]["tool_call_id"], "call_1")
            self.assertEqual(msgs[2]["content"], json.dumps({"value": 3}, separators=(",", ":"), ensure_ascii=False))

    def test_openai_responses_tools_and_tool_output(self) -> None:
        from nous.genai.types import GenerateRequest, Message, OutputSpec, Part, Tool, ToolChoice
        from nous.genai.providers.openai import OpenAIAdapter

        req = GenerateRequest(
            model="openai:gpt-5.2",
            input=[
                Message(role="user", content=[Part.from_text("hi")]),
                Message(
                    role="tool",
                    content=[Part.tool_result(tool_call_id="call_1", name="sum", result={"value": 3})],
                ),
            ],
            output=OutputSpec(modalities=["text"]),
            tools=[Tool(name="sum", parameters={"type": "object"})],
            tool_choice=ToolChoice(mode="required"),
        )

        with patch("nous.genai.providers.openai.request_json") as request_json:
            request_json.return_value = {
                "id": "resp_123",
                "status": "completed",
                "output": [
                    {"type": "function_call", "call_id": "call_2", "name": "sum", "arguments": "{\"a\":2,\"b\":3}"},
                    {"type": "message", "content": [{"type": "output_text", "text": "ok"}]},
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
            self.assertEqual([p.type for p in out.output[0].content], ["tool_call", "text"])
            self.assertEqual(out.output[0].content[0].meta.get("tool_call_id"), "call_2")
            self.assertEqual(out.output[0].content[0].meta.get("arguments"), {"a": 2, "b": 3})

            _, kwargs = request_json.call_args
            body = kwargs["json_body"]
            self.assertEqual(body["tool_choice"], "required")
            self.assertEqual(body["tools"][0]["type"], "function")
            self.assertEqual(body["tools"][0]["name"], "sum")
            self.assertNotIn("strict", body["tools"][0])

            items = body["input"]
            self.assertEqual(items[1]["type"], "function_call_output")
            self.assertEqual(items[1]["call_id"], "call_1")
            self.assertEqual(items[1]["output"], json.dumps({"value": 3}, separators=(",", ":"), ensure_ascii=False))

    def test_tuzi_openai_responses_tool_choice_tool_falls_back_to_required(self) -> None:
        from nous.genai.types import GenerateRequest, Message, OutputSpec, Part, Tool, ToolChoice
        from nous.genai.providers.openai import OpenAIAdapter

        req = GenerateRequest(
            model="tuzi-openai:gpt-4o-mini",
            input=[Message(role="user", content=[Part.from_text("hi")])],
            output=OutputSpec(modalities=["text"]),
            tools=[Tool(name="ping", parameters={"type": "object", "properties": {}})],
            tool_choice=ToolChoice(mode="tool", name="ping"),
        )

        with patch("nous.genai.providers.openai.request_json") as request_json:
            request_json.return_value = {
                "id": "resp_123",
                "status": "completed",
                "output": [{"type": "message", "content": [{"type": "output_text", "text": "ok"}]}],
            }
            adapter = OpenAIAdapter(
                api_key="__demo__",
                base_url="https://example.invalid/v1",
                provider_name="tuzi-openai",
                chat_api="responses",
            )
            out = adapter.generate(req, stream=False)
            self.assertEqual(out.status, "completed")

            _, kwargs = request_json.call_args
            body = kwargs["json_body"]
            self.assertEqual(body["tool_choice"], "required")

    def test_gemini_tools_and_tool_config(self) -> None:
        from nous.genai.types import GenerateRequest, Message, OutputSpec, Part, Tool, ToolChoice
        from nous.genai.providers.gemini import GeminiAdapter

        req = GenerateRequest(
            model="google:gemini-2.5-flash",
            input=[
                Message(role="user", content=[Part.from_text("hi")]),
                Message(role="tool", content=[Part.tool_result(name="sum", result={"value": 3})]),
            ],
            output=OutputSpec(modalities=["text"]),
            tools=[Tool(name="sum", parameters={"type": "object"})],
            tool_choice=ToolChoice(mode="tool", name="sum"),
        )

        with patch("nous.genai.providers.gemini.request_json") as request_json:
            request_json.return_value = {
                "candidates": [{"content": {"parts": [{"functionCall": {"name": "sum", "args": {"a": 1, "b": 2}}}]}}]
            }
            adapter = GeminiAdapter(api_key="__demo__", base_url="https://example.invalid", provider_name="google")
            out = adapter.generate(req, stream=False)
            self.assertEqual(out.status, "completed")
            self.assertEqual(out.output[0].content[0].type, "tool_call")
            self.assertEqual(out.output[0].content[0].meta.get("name"), "sum")
            self.assertEqual(out.output[0].content[0].meta.get("arguments"), {"a": 1, "b": 2})

            _, kwargs = request_json.call_args
            body = kwargs["json_body"]
            self.assertEqual(body["tools"][0]["functionDeclarations"][0]["name"], "sum")
            self.assertEqual(body["toolConfig"]["functionCallingConfig"]["mode"], "ANY")
            self.assertEqual(body["toolConfig"]["functionCallingConfig"]["allowedFunctionNames"], ["sum"])

            contents = body["contents"]
            self.assertEqual(contents[1]["role"], "user")
            self.assertEqual(contents[1]["parts"][0]["functionResponse"]["name"], "sum")

    def test_anthropic_tools_and_tool_choice(self) -> None:
        from nous.genai.types import GenerateRequest, Message, OutputSpec, OutputTextSpec, Part, Tool, ToolChoice
        from nous.genai.providers.anthropic import AnthropicAdapter

        req = GenerateRequest(
            model="anthropic:claude-3-5-sonnet-20240620",
            input=[
                Message(role="user", content=[Part.from_text("hi")]),
                Message(
                    role="tool",
                    content=[Part.tool_result(tool_call_id="toolu_1", name="sum", result={"value": 3})],
                ),
            ],
            output=OutputSpec(modalities=["text"], text=OutputTextSpec(max_output_tokens=256)),
            tools=[Tool(name="sum", parameters={"type": "object"})],
            tool_choice=ToolChoice(mode="tool", name="sum"),
        )

        with patch("nous.genai.providers.anthropic.request_json") as request_json:
            request_json.return_value = {
                "id": "msg_123",
                "content": [{"type": "tool_use", "id": "toolu_2", "name": "sum", "input": {"a": 1, "b": 2}}],
            }
            adapter = AnthropicAdapter(api_key="__demo__", base_url="https://example.invalid", provider_name="anthropic")
            out = adapter.generate(req, stream=False)
            self.assertEqual(out.status, "completed")
            self.assertEqual(out.output[0].content[0].type, "tool_call")
            self.assertEqual(out.output[0].content[0].meta.get("tool_call_id"), "toolu_2")
            self.assertEqual(out.output[0].content[0].meta.get("arguments"), {"a": 1, "b": 2})

            _, kwargs = request_json.call_args
            body = kwargs["json_body"]
            self.assertEqual(body["tools"][0]["name"], "sum")
            self.assertEqual(body["tool_choice"], {"type": "tool", "name": "sum"})

            msgs = body["messages"]
            tool_msg = msgs[1]
            self.assertEqual(tool_msg["role"], "user")
            self.assertEqual(tool_msg["content"][0]["type"], "tool_result")
            self.assertEqual(tool_msg["content"][0]["tool_use_id"], "toolu_1")
