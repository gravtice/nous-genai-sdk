import asyncio
import unittest
from unittest.mock import patch

try:
    from mcp.server.fastmcp.exceptions import ToolError  # noqa: F401
except ModuleNotFoundError as e:
    raise unittest.SkipTest("missing dependency: mcp") from e


def _call_tool(server, name: str, args: dict) -> object:
    out = server.call_tool(name, args)
    if asyncio.iscoroutine(out):
        out = asyncio.run(out)
    if isinstance(out, tuple) and len(out) == 2 and isinstance(out[1], dict):
        return out[1]
    return out


class TestMcpModelKeywordFilter(unittest.TestCase):
    def test_list_available_models_filters_by_keyword(self) -> None:
        from nous.genai.mcp_server import build_server

        with patch(
            "nous.genai.mcp_server.Client.list_available_models",
            return_value=["gpt-4o-mini", "gpt-image-1"],
        ):
            server = build_server(host="127.0.0.1", port=7001, model_keywords=["image"])
            out = _call_tool(server, "list_available_models", {"provider": "openai"})

        self.assertIsInstance(out, dict)
        models = [m["model"] for m in out["models"]]
        self.assertEqual(models, ["openai:gpt-image-1"])

    def test_list_all_available_models_filters_by_keyword_or(self) -> None:
        from nous.genai.mcp_server import build_server

        all_models = [
            "openai:gpt-4o-mini",
            "openai:gpt-image-1",
            "tuzi-openai:gpt-4o-mini",
            "tuzi-openai:gpt-image-1",
        ]
        with patch("nous.genai.mcp_server.Client.list_all_available_models", return_value=all_models):
            server = build_server(host="127.0.0.1", port=7001, model_keywords=["tuzi", "image"])
            out = _call_tool(server, "list_all_available_models", {})

        self.assertIsInstance(out, dict)
        models = [m["model"] for m in out["models"]]
        self.assertEqual(
            models,
            [
                "openai:gpt-image-1",
                "tuzi-openai:gpt-4o-mini",
                "tuzi-openai:gpt-image-1",
            ],
        )

    def test_generate_rejects_disallowed_model(self) -> None:
        from nous.genai.mcp_server import build_server

        req_dict = {
            "model": "openai:gpt-4o-mini",
            "input": [{"role": "user", "content": [{"type": "text", "text": "x"}]}],
            "output": {"modalities": ["text"]},
        }
        with patch("nous.genai.mcp_server.Client.generate", side_effect=AssertionError("should not call provider")):
            server = build_server(host="127.0.0.1", port=7001, model_keywords=["image"])
            with self.assertRaises(ToolError) as cm:
                _call_tool(server, "generate", {"request": req_dict})

        self.assertIn("model not allowed", str(cm.exception))
