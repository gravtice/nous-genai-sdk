import os
import unittest
from unittest.mock import patch


class TestMcpCliUrlResolution(unittest.TestCase):
    def test_default_host_port_builds_mcp_url(self) -> None:
        from nous.genai import mcp_cli

        with patch.dict(
            os.environ,
            {
                "NOUS_GENAI_MCP_URL": "",
                "NOUS_GENAI_MCP_BASE_URL": "",
                "NOUS_GENAI_MCP_PUBLIC_BASE_URL": "",
                "NOUS_GENAI_MCP_HOST": "127.0.0.1",
                "NOUS_GENAI_MCP_PORT": "6123",
            },
            clear=False,
        ):
            self.assertEqual(mcp_cli._mcp_url(), "http://127.0.0.1:6123/mcp")

    def test_explicit_mcp_url_takes_precedence(self) -> None:
        from nous.genai import mcp_cli

        with patch.dict(
            os.environ,
            {
                "NOUS_GENAI_MCP_URL": "http://example.com/prefix",
                "NOUS_GENAI_MCP_BASE_URL": "http://bad.example",
                "NOUS_GENAI_MCP_PUBLIC_BASE_URL": "http://bad.example",
                "NOUS_GENAI_MCP_HOST": "127.0.0.1",
                "NOUS_GENAI_MCP_PORT": "6001",
            },
            clear=False,
        ):
            self.assertEqual(mcp_cli._mcp_url(), "http://example.com/prefix/mcp")
