import unittest


class TestMcpServerTransports(unittest.TestCase):
    def test_build_http_app_exposes_streamable_and_sse(self) -> None:
        try:
            from mcp.server.fastmcp import FastMCP  # noqa: F401
        except ModuleNotFoundError:
            self.skipTest("missing dependency: mcp")

        try:
            from starlette.routing import Mount, Route  # noqa: F401
        except ModuleNotFoundError:
            self.skipTest("missing dependency: starlette")

        from nous.genai.mcp_server import build_http_app, build_server

        server = build_server(host="127.0.0.1", port=7001)
        app = build_http_app(server)

        paths = {getattr(route, "path", None) for route in app.router.routes}
        self.assertIn(server.settings.streamable_http_path, paths)
        self.assertIn(server.settings.sse_path, paths)
        self.assertIn(server.settings.message_path.rstrip("/"), paths)
