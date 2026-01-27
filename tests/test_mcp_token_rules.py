import asyncio
import unittest
from unittest.mock import patch


def _call_tool(server, name: str, args: dict) -> dict:
    out = server.call_tool(name, args)
    if asyncio.iscoroutine(out):
        out = asyncio.run(out)
    if isinstance(out, tuple) and len(out) == 2 and isinstance(out[1], dict):
        out = out[1]
    if not isinstance(out, dict):
        raise AssertionError(f"unexpected tool output type: {type(out)}")
    return out


class TestMcpTokenRules(unittest.TestCase):
    def test_parse_bracket_rules_supports_models_and_providers(self) -> None:
        from nous.genai.mcp_server import _parse_mcp_token_scopes

        scopes = _parse_mcp_token_scopes(
            "t1: [openai google openai:gpt-4o-mini]; t2: [openai:gpt-4o-mini]"
        )
        self.assertEqual(set(scopes.keys()), {"t1", "t2"})

        t1 = scopes["t1"]
        self.assertIn("openai", t1.providers)
        self.assertIn("google", t1.providers)
        self.assertIn("openai", t1.providers_all)
        self.assertIn("google", t1.providers_all)

        t2 = scopes["t2"]
        self.assertIn("openai", t2.providers)
        self.assertNotIn("google", t2.providers)
        self.assertNotIn("openai", t2.providers_all)
        self.assertEqual(set(t2.models.get("openai") or set()), {"gpt-4o-mini"})

    def test_token_scopes_filter_providers_models_and_generate(self) -> None:
        try:
            from mcp.server.fastmcp import FastMCP  # noqa: F401
        except ModuleNotFoundError:
            self.skipTest("missing dependency: mcp")

        from nous.genai.mcp_server import (
            _REQUEST_TOKEN,
            _parse_mcp_token_scopes,
            build_server,
        )
        from nous.genai.types import GenerateResponse, Message, Part
        from mcp.server.fastmcp.exceptions import ToolError

        scopes = _parse_mcp_token_scopes(
            "t1: [openai google]; t2: [openai:gpt-4o-mini]"
        )

        def fake_adapter(_self, provider: str):
            return object()

        def fake_list_available_models(_self, provider: str, *, timeout_ms=None):  # noqa: ARG001
            if provider == "openai":
                return ["gpt-4o-mini", "gpt-4.1"]
            if provider == "google":
                return ["gemini-2.0-flash"]
            return []

        def fake_supported_for_provider(provider: str):
            if provider == "openai":
                return [
                    {
                        "model_id": "gpt-4o-mini",
                        "model": "openai:gpt-4o-mini",
                        "modes": ["sync"],
                        "input_modalities": ["text"],
                        "output_modalities": ["text"],
                    },
                    {
                        "model_id": "gpt-4.1",
                        "model": "openai:gpt-4.1",
                        "modes": ["sync"],
                        "input_modalities": ["text"],
                        "output_modalities": ["text"],
                    },
                ]
            if provider == "google":
                return [
                    {
                        "model_id": "gemini-2.0-flash",
                        "model": "google:gemini-2.0-flash",
                        "modes": ["sync"],
                        "input_modalities": ["text"],
                        "output_modalities": ["text"],
                    }
                ]
            return []

        resp = GenerateResponse(
            id="r1",
            provider="openai",
            model="openai:gpt-4o-mini",
            status="completed",
            output=[Message(role="assistant", content=[Part(type="text", text="ok")])],
        )

        with patch(
            "nous.genai.reference.get_model_catalog",
            return_value={"openai": [], "google": []},
        ):
            with patch(
                "nous.genai.reference.get_supported_providers",
                return_value=["openai", "google"],
            ):
                with patch(
                    "nous.genai.reference.get_sdk_supported_models_for_provider",
                    side_effect=fake_supported_for_provider,
                ):
                    with patch("nous.genai.client.Client._adapter", new=fake_adapter):
                        with patch(
                            "nous.genai.client.Client.list_available_models",
                            autospec=True,
                            side_effect=fake_list_available_models,
                        ) as list_mock:
                            with patch(
                                "nous.genai.client.Client.generate", return_value=resp
                            ) as gen_mock:
                                server = build_server(
                                    host="127.0.0.1", port=7001, token_scopes=scopes
                                )

                                ctx = _REQUEST_TOKEN.set("t2")
                                try:
                                    out = _call_tool(server, "list_providers", {})
                                    self.assertEqual(out["supported"], ["openai"])
                                    self.assertEqual(out["configured"], ["openai"])

                                    models = _call_tool(
                                        server,
                                        "list_available_models",
                                        {"provider": "openai"},
                                    )
                                    self.assertEqual(
                                        [m["model"] for m in models["models"]],
                                        ["openai:gpt-4o-mini"],
                                    )

                                    with self.assertRaises(ToolError):
                                        _call_tool(
                                            server,
                                            "list_available_models",
                                            {"provider": "google"},
                                        )

                                    all_models = _call_tool(
                                        server, "list_all_available_models", {}
                                    )
                                    self.assertEqual(
                                        [m["model"] for m in all_models["models"]],
                                        ["openai:gpt-4o-mini"],
                                    )
                                    providers = [
                                        c.args[1] for c in list_mock.call_args_list
                                    ]
                                    self.assertEqual(set(providers), {"openai"})

                                    req = {
                                        "model": "openai:gpt-4o-mini",
                                        "input": [
                                            {
                                                "role": "user",
                                                "content": [
                                                    {"type": "text", "text": "x"}
                                                ],
                                            }
                                        ],
                                        "output": {"modalities": ["text"]},
                                    }
                                    gen_out = _call_tool(
                                        server, "generate", {"request": req}
                                    )
                                    self.assertEqual(gen_out["status"], "completed")
                                    self.assertTrue(gen_mock.called)

                                    req_bad = dict(req)
                                    req_bad["model"] = "openai:gpt-4.1"
                                    with self.assertRaises(ToolError):
                                        _call_tool(
                                            server, "generate", {"request": req_bad}
                                        )
                                finally:
                                    _REQUEST_TOKEN.reset(ctx)
