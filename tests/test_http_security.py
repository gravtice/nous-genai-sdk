import os
import tempfile
import unittest
from unittest.mock import patch


class _FakeHttpResponse:
    def __init__(self, *, status: int, headers: dict[str, str] | None = None, body: bytes = b"") -> None:
        self.status = status
        self._headers = {k.lower(): v for k, v in (headers or {}).items()}
        self._body = body
        self._offset = 0

    def getheader(self, name: str) -> str | None:
        return self._headers.get(name.lower())

    def read(self, n: int | None = None) -> bytes:
        if n is None:
            out = self._body[self._offset :]
            self._offset = len(self._body)
            return out
        if self._offset >= len(self._body):
            return b""
        end = min(len(self._body), self._offset + n)
        out = self._body[self._offset : end]
        self._offset = end
        return out


class _FakeHttpConnection:
    def __init__(self, resp: _FakeHttpResponse) -> None:
        self._resp = resp
        self.request_args: tuple[str, str] | None = None
        self.request_headers: dict[str, str] | None = None

    def request(self, method: str, path: str, body: object = None, headers: dict[str, str] | None = None):  # noqa: ARG002
        self.request_args = (method, path)
        self.request_headers = dict(headers or {})

    def getresponse(self) -> _FakeHttpResponse:
        return self._resp

    def close(self) -> None:
        return


class TestUrlDownloadPinsIpAgainstDnsRebinding(unittest.TestCase):
    def test_download_resolves_once_and_pins_connect_ip(self) -> None:
        from nous.genai._internal.http import download_to_file

        calls: dict[str, int] = {"rebind.example": 0}

        def _fake_getaddrinfo(host: str, port: object, proto: int):  # type: ignore[no-untyped-def]
            self.assertEqual(port, None)
            self.assertEqual(proto, socket.IPPROTO_TCP)
            if host != "rebind.example":
                raise AssertionError(f"unexpected host lookup: {host}")
            calls["rebind.example"] += 1
            if calls["rebind.example"] == 1:
                return [
                    (socket.AF_INET, socket.SOCK_STREAM, proto, "", ("93.184.216.34", 0)),
                ]
            return [
                (socket.AF_INET, socket.SOCK_STREAM, proto, "", ("127.0.0.1", 0)),
            ]

        def _fake_make_connection(parsed, timeout_s, *, proxy_url, connect_host=None, tls_server_hostname=None):  # type: ignore[no-untyped-def]
            self.assertIsNone(proxy_url)
            self.assertEqual(parsed.hostname, "rebind.example")
            self.assertEqual(connect_host, "93.184.216.34")
            self.assertEqual(tls_server_hostname, "rebind.example")
            body = b"ok"
            resp = _FakeHttpResponse(status=200, headers={"Content-Length": str(len(body))}, body=body)
            return _FakeHttpConnection(resp)

        import socket

        with tempfile.TemporaryDirectory(prefix="genaisdk-test-") as d:
            out_path = os.path.join(d, "out.bin")
            with patch("nous.genai._internal.http.socket.getaddrinfo", side_effect=_fake_getaddrinfo):
                with patch("nous.genai._internal.http._make_connection", side_effect=_fake_make_connection):
                    download_to_file(url="http://rebind.example/x", output_path=out_path, timeout_ms=100)

            with open(out_path, "rb") as f:
                self.assertEqual(f.read(), b"ok")

        self.assertEqual(calls["rebind.example"], 1)


class TestMakeConnectionProxyHttpsTarget(unittest.TestCase):
    def test_http_proxy_with_https_target_uses_httpsconnection(self) -> None:
        import urllib.parse
        import http.client

        from nous.genai._internal.http import _make_connection

        parsed = urllib.parse.urlparse("https://example.com/v1/models")
        conn = _make_connection(parsed, 0.1, proxy_url="http://proxy.local:8080")
        self.assertIsInstance(conn, http.client.HTTPSConnection)
        self.assertEqual(conn.host, "proxy.local")
        self.assertEqual(conn.port, 8080)
        self.assertEqual(getattr(conn, "_tunnel_host", None), "example.com")
        self.assertEqual(getattr(conn, "_tunnel_port", None), 443)


class TestUrlDownloadRedirectAndLimits(unittest.TestCase):
    def test_redirect_to_private_host_is_blocked(self) -> None:
        from nous.genai import GenAIError
        from nous.genai._internal.http import download_to_file

        import ipaddress

        def _fake_resolve(host: str):  # type: ignore[no-untyped-def]
            if host == "public.example":
                return ([ipaddress.ip_address("93.184.216.34")], False)
            if host == "private.example":
                return ([ipaddress.ip_address("127.0.0.1")], True)
            raise AssertionError(f"unexpected host: {host}")

        calls: list[str] = []

        def _fake_make_connection(parsed, timeout_s, *, proxy_url, connect_host=None, tls_server_hostname=None):  # type: ignore[no-untyped-def]
            self.assertIsNone(proxy_url)
            self.assertEqual(parsed.hostname, "public.example")
            self.assertEqual(connect_host, "93.184.216.34")
            self.assertEqual(tls_server_hostname, "public.example")
            calls.append("public.example")
            resp = _FakeHttpResponse(status=302, headers={"Location": "http://private.example/secret"})
            return _FakeHttpConnection(resp)

        with tempfile.TemporaryDirectory(prefix="genaisdk-test-") as d:
            out_path = os.path.join(d, "out.bin")
            with patch.dict(os.environ, {"NOUS_GENAI_ALLOW_PRIVATE_URLS": "0"}, clear=False):
                with patch("nous.genai._internal.http._resolve_url_host_ips", side_effect=_fake_resolve):
                    with patch("nous.genai._internal.http._make_connection", side_effect=_fake_make_connection):
                        with self.assertRaises(GenAIError) as cm:
                            download_to_file(url="http://public.example/start", output_path=out_path, timeout_ms=100)

        self.assertEqual(cm.exception.info.type, "InvalidRequestError")
        self.assertIn("private/loopback", cm.exception.info.message)
        self.assertEqual(calls, ["public.example"])

    def test_download_rejects_content_length_over_max_bytes(self) -> None:
        from nous.genai import GenAIError
        from nous.genai._internal.http import download_to_file

        import ipaddress

        def _fake_resolve(host: str):  # type: ignore[no-untyped-def]
            self.assertEqual(host, "public.example")
            return ([ipaddress.ip_address("93.184.216.34")], False)

        def _fake_make_connection(parsed, timeout_s, *, proxy_url, connect_host=None, tls_server_hostname=None):  # type: ignore[no-untyped-def]
            self.assertIsNone(proxy_url)
            self.assertEqual(parsed.hostname, "public.example")
            self.assertEqual(connect_host, "93.184.216.34")
            self.assertEqual(tls_server_hostname, "public.example")
            resp = _FakeHttpResponse(status=200, headers={"Content-Length": "4"}, body=b"test")
            return _FakeHttpConnection(resp)

        with tempfile.TemporaryDirectory(prefix="genaisdk-test-") as d:
            out_path = os.path.join(d, "out.bin")
            with patch("nous.genai._internal.http._resolve_url_host_ips", side_effect=_fake_resolve):
                with patch("nous.genai._internal.http._make_connection", side_effect=_fake_make_connection):
                    with self.assertRaises(GenAIError) as cm:
                        download_to_file(url="http://public.example/x", output_path=out_path, timeout_ms=100, max_bytes=3)

        self.assertEqual(cm.exception.info.type, "NotSupportedError")
        self.assertIn("url download too large", cm.exception.info.message)

    def test_download_rejects_body_exceeding_max_bytes(self) -> None:
        from nous.genai import GenAIError
        from nous.genai._internal.http import download_to_file

        import ipaddress

        def _fake_resolve(host: str):  # type: ignore[no-untyped-def]
            self.assertEqual(host, "public.example")
            return ([ipaddress.ip_address("93.184.216.34")], False)

        def _fake_make_connection(parsed, timeout_s, *, proxy_url, connect_host=None, tls_server_hostname=None):  # type: ignore[no-untyped-def]
            self.assertIsNone(proxy_url)
            self.assertEqual(parsed.hostname, "public.example")
            self.assertEqual(connect_host, "93.184.216.34")
            self.assertEqual(tls_server_hostname, "public.example")
            resp = _FakeHttpResponse(status=200, headers=None, body=b"test")
            return _FakeHttpConnection(resp)

        with tempfile.TemporaryDirectory(prefix="genaisdk-test-") as d:
            out_path = os.path.join(d, "out.bin")
            with patch("nous.genai._internal.http._resolve_url_host_ips", side_effect=_fake_resolve):
                with patch("nous.genai._internal.http._make_connection", side_effect=_fake_make_connection):
                    with self.assertRaises(GenAIError) as cm:
                        download_to_file(url="http://public.example/x", output_path=out_path, timeout_ms=100, max_bytes=3)
            self.assertFalse(os.path.exists(out_path))

        self.assertEqual(cm.exception.info.type, "NotSupportedError")
        self.assertIn("url download exceeded limit", cm.exception.info.message)

    def test_download_timeout_maps_to_timeout_error(self) -> None:
        from nous.genai import GenAIError
        from nous.genai._internal.http import download_to_file

        import ipaddress
        import socket

        def _fake_resolve(host: str):  # type: ignore[no-untyped-def]
            self.assertEqual(host, "public.example")
            return ([ipaddress.ip_address("93.184.216.34")], False)

        class _TimeoutConn:
            def request(self, method: str, path: str, body: object = None, headers: dict[str, str] | None = None):  # noqa: ARG002
                raise socket.timeout()

            def close(self) -> None:
                return

        def _fake_make_connection(parsed, timeout_s, *, proxy_url, connect_host=None, tls_server_hostname=None):  # type: ignore[no-untyped-def]
            self.assertIsNone(proxy_url)
            self.assertEqual(parsed.hostname, "public.example")
            self.assertEqual(connect_host, "93.184.216.34")
            self.assertEqual(tls_server_hostname, "public.example")
            return _TimeoutConn()

        with tempfile.TemporaryDirectory(prefix="genaisdk-test-") as d:
            out_path = os.path.join(d, "out.bin")
            with patch("nous.genai._internal.http._resolve_url_host_ips", side_effect=_fake_resolve):
                with patch("nous.genai._internal.http._make_connection", side_effect=_fake_make_connection):
                    with self.assertRaises(GenAIError) as cm:
                        download_to_file(url="http://public.example/x", output_path=out_path, timeout_ms=100, max_bytes=3)

        self.assertEqual(cm.exception.info.type, "TimeoutError")
