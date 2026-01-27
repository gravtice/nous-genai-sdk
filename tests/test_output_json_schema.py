import unittest
from dataclasses import dataclass
from typing import TypedDict
from unittest.mock import patch


class TestOutputTextJsonSchema(unittest.TestCase):
    def test_client_coerces_python_type_json_schema(self) -> None:
        try:
            from pydantic import BaseModel
        except ModuleNotFoundError:
            self.skipTest("missing dependency: pydantic")

        from nous.genai.client import Client
        from nous.genai.types import (
            Capability,
            GenerateRequest,
            GenerateResponse,
            Message,
            OutputSpec,
            OutputTextSpec,
            Part,
        )

        class Model(BaseModel):
            a: int

        class TD(TypedDict):
            a: int

        @dataclass
        class DC:
            a: int

        class DummyAdapter:
            def __init__(self) -> None:
                self.last_schema = None

            def capabilities(self, model_id: str) -> Capability:  # noqa: ARG002
                return Capability(
                    input_modalities={"text"},
                    output_modalities={"text"},
                    supports_stream=False,
                    supports_job=False,
                )

            def generate(self, request: GenerateRequest, *, stream: bool):  # noqa: ARG002
                spec = request.output.text
                if spec is None:
                    raise AssertionError("expected output.text to be set")
                self.last_schema = spec.json_schema
                if not isinstance(self.last_schema, dict):
                    raise AssertionError("expected json_schema to be coerced to dict")
                return GenerateResponse(
                    id="r1",
                    provider="dummy",
                    model=request.model,
                    status="completed",
                    output=[Message(role="assistant", content=[Part.from_text("ok")])],
                )

        client = Client()
        adapter = DummyAdapter()

        cases = [
            ("pydantic.BaseModel", Model),
            ("TypedDict", TD),
            ("dataclass", DC),
            ("typing", list[int]),
        ]
        for name, schema_type in cases:
            with self.subTest(name=name):
                req = GenerateRequest(
                    model="dummy:demo",
                    input=[Message(role="user", content=[Part.from_text("hi")])],
                    output=OutputSpec(
                        modalities=["text"],
                        text=OutputTextSpec(format="json", json_schema=schema_type),
                    ),
                )
                with patch.object(client, "_adapter", return_value=adapter):
                    client.generate(req)
                assert isinstance(adapter.last_schema, dict)

    def test_gemini_converts_json_schema_dict(self) -> None:
        try:
            from pydantic import BaseModel
        except ModuleNotFoundError:
            self.skipTest("missing dependency: pydantic")

        from nous.genai.providers.gemini import GeminiAdapter
        from nous.genai.types import (
            GenerateRequest,
            Message,
            OutputSpec,
            OutputTextSpec,
            Part,
        )

        class Child(BaseModel):
            x: int

        class Parent(BaseModel):
            child: Child
            b: str | None = None

        schema = Parent.model_json_schema()
        req = GenerateRequest(
            model="google:gemini-1.5-flash",
            input=[Message(role="user", content=[Part.from_text("hi")])],
            output=OutputSpec(
                modalities=["text"],
                text=OutputTextSpec(format="json", json_schema=schema),
            ),
        )

        adapter = GeminiAdapter(api_key="dummy")
        body = adapter._generate_body(req, model_name="gemini-1.5-flash")
        gen_cfg = body.get("generationConfig")
        if not isinstance(gen_cfg, dict):
            raise AssertionError("missing generationConfig")
        resp_schema = gen_cfg.get("responseSchema")
        if not isinstance(resp_schema, dict):
            raise AssertionError("missing responseSchema")

        self.assertEqual(resp_schema.get("type"), "OBJECT")
        props = resp_schema.get("properties")
        self.assertIsInstance(props, dict)
        assert isinstance(props, dict)
        self.assertIn("child", props)
        self.assertIn("b", props)

        child_schema = props["child"]
        self.assertIsInstance(child_schema, dict)
        assert isinstance(child_schema, dict)
        self.assertEqual(child_schema.get("type"), "OBJECT")
        self.assertIsInstance(child_schema.get("properties"), dict)

        b_schema = props["b"]
        self.assertIsInstance(b_schema, dict)
        assert isinstance(b_schema, dict)
        self.assertEqual(b_schema.get("type"), "STRING")
        self.assertEqual(b_schema.get("nullable"), True)

    def test_rejects_gemini_schema_dict_in_json_schema(self) -> None:
        from nous.genai._internal.errors import GenAIError
        from nous.genai.client import Client
        from nous.genai.providers.gemini import GeminiAdapter
        from nous.genai.types import (
            Capability,
            GenerateRequest,
            Message,
            OutputSpec,
            OutputTextSpec,
            Part,
        )

        gemini_schema = {"type": "OBJECT", "properties": {"a": {"type": "STRING"}}}

        req = GenerateRequest(
            model="dummy:demo",
            input=[Message(role="user", content=[Part.from_text("hi")])],
            output=OutputSpec(
                modalities=["text"],
                text=OutputTextSpec(format="json", json_schema=gemini_schema),
            ),
        )

        class DummyAdapter:
            def capabilities(self, model_id: str) -> Capability:  # noqa: ARG002
                return Capability(
                    input_modalities={"text"},
                    output_modalities={"text"},
                    supports_stream=False,
                    supports_job=False,
                )

            def generate(self, request: GenerateRequest, *, stream: bool):  # noqa: ARG002
                raise AssertionError("should fail before adapter.generate()")

        client = Client()
        with patch.object(client, "_adapter", return_value=DummyAdapter()):
            with self.assertRaises(GenAIError) as ctx:
                client.generate(req)
            self.assertEqual(ctx.exception.info.type, "InvalidRequestError")

        req_google = GenerateRequest(
            model="google:gemini-1.5-flash",
            input=[Message(role="user", content=[Part.from_text("hi")])],
            output=OutputSpec(
                modalities=["text"],
                text=OutputTextSpec(format="json", json_schema=gemini_schema),
            ),
        )
        adapter = GeminiAdapter(api_key="dummy")
        with self.assertRaises(GenAIError) as ctx2:
            adapter._generate_body(req_google, model_name="gemini-1.5-flash")
        self.assertEqual(ctx2.exception.info.type, "InvalidRequestError")
