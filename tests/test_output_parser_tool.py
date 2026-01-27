import unittest
from dataclasses import dataclass
from typing import TypedDict


class TestOutputParserTool(unittest.TestCase):
    def test_build_output_parser_tool_accepts_python_types(self) -> None:
        from nous.genai._internal.errors import GenAIError
        from nous.genai.tools.output_parser import build_output_parser_tool

        tool = build_output_parser_tool({"type": "object", "properties": {"a": {"type": "integer"}}})
        self.assertIsInstance(tool.parameters, dict)

        try:
            from pydantic import BaseModel
        except ModuleNotFoundError:
            with self.assertRaises(GenAIError) as ctx:
                build_output_parser_tool(list[int])
            self.assertEqual(ctx.exception.info.type, "NotSupportedError")
            return

        class Model(BaseModel):
            a: int

        class TD(TypedDict):
            a: int

        @dataclass
        class DC:
            a: int

        cases = [
            ("pydantic.BaseModel", Model),
            ("TypedDict", TD),
            ("dataclass", DC),
            ("typing", list[int]),
        ]

        for name, schema_type in cases:
            with self.subTest(name=name):
                tool = build_output_parser_tool(schema_type)
                self.assertIsInstance(tool.parameters, dict)
                assert isinstance(tool.parameters, dict)
                props = tool.parameters.get("properties")
                self.assertIsInstance(props, dict)
                assert isinstance(props, dict)
                out_schema = props.get("output")
                self.assertIsInstance(out_schema, dict)

    def test_extract_output_from_response(self) -> None:
        from nous.genai._internal.errors import GenAIError
        from nous.genai.tools.output_parser import extract_output_from_response
        from nous.genai.types import GenerateResponse, Message, Part

        resp = GenerateResponse(
            id="r1",
            provider="dummy",
            model="dummy:demo",
            status="completed",
            output=[
                Message(
                    role="assistant",
                    content=[
                        Part.tool_call(
                            name="nous_output_parser",
                            arguments={"output": {"a": 1}},
                        )
                    ],
                )
            ],
        )
        self.assertEqual(extract_output_from_response(resp), {"a": 1})

        resp2 = GenerateResponse(
            id="r2",
            provider="dummy",
            model="dummy:demo",
            status="completed",
            output=[Message(role="assistant", content=[Part.from_text("no tool call")])],
        )
        with self.assertRaises(GenAIError):
            extract_output_from_response(resp2)

    def test_parse_output(self) -> None:
        from nous.genai._internal.errors import GenAIError
        from nous.genai.tools.output_parser import parse_output
        from nous.genai.types import Capability, GenerateRequest, GenerateResponse, Message, Part

        class DummyClient:
            def __init__(self, *, supports_tools: bool) -> None:
                self._supports_tools = supports_tools
                self.last_request: GenerateRequest | None = None

            def capabilities(self, model: str) -> Capability:
                return Capability(
                    input_modalities={"text"},
                    output_modalities={"text"},
                    supports_stream=False,
                    supports_job=False,
                    supports_tools=self._supports_tools,
                    supports_json_schema=False,
                )

            def generate(self, request: GenerateRequest):
                self.last_request = request
                return GenerateResponse(
                    id="r3",
                    provider="dummy",
                    model=request.model,
                    status="completed",
                    output=[
                        Message(
                            role="assistant",
                            content=[
                                Part.tool_call(
                                    name="nous_output_parser",
                                    arguments={"output": {"a": 1}},
                                )
                            ],
                        )
                    ],
                )

        schema = {"type": "object", "properties": {"a": {"type": "integer"}}, "required": ["a"]}
        c = DummyClient(supports_tools=True)
        out = parse_output(c, model="dummy:demo", text="hello", json_schema=schema)
        self.assertEqual(out, {"a": 1})
        self.assertIsNotNone(c.last_request)
        assert c.last_request is not None
        self.assertEqual(c.last_request.tool_choice.mode, "tool")
        self.assertEqual(c.last_request.tool_choice.name, "nous_output_parser")
        self.assertTrue(c.last_request.input[0].content[0].require_text().endswith("hello"))

        with self.assertRaises(GenAIError) as ctx:
            parse_output(c, model="dummy:demo", text="   ", json_schema=schema)
        self.assertEqual(ctx.exception.info.type, "InvalidRequestError")

        c2 = DummyClient(supports_tools=False)
        with self.assertRaises(GenAIError) as ctx2:
            parse_output(c2, model="dummy:demo", text="hello", json_schema=schema)
        self.assertEqual(ctx2.exception.info.type, "NotSupportedError")
