import unittest
from unittest.mock import patch


class _FakeResponse:
    def __init__(self, *, status: int, body: bytes) -> None:
        self.status = status
        self._body = body

    def read(self, n: int | None = None) -> bytes:  # noqa: ARG002
        return self._body


class _FakeConn:
    def __init__(self, resp: _FakeResponse) -> None:
        self._resp = resp

    def request(
        self,
        method: str,
        path: str,
        body: object = None,
        headers: dict[str, str] | None = None,
    ):  # noqa: ARG002
        return

    def getresponse(self) -> _FakeResponse:
        return self._resp

    def close(self) -> None:
        return


class TestHttpErrorMapping(unittest.TestCase):
    def test_request_json_maps_500_to_retryable_provider_error(self) -> None:
        from nous.genai import GenAIError
        from nous.genai._internal.http import request_json

        resp = _FakeResponse(status=500, body=b'{"message":"oops"}')

        def _fake_make_connection(
            parsed, timeout_s, *, proxy_url, connect_host=None, tls_server_hostname=None
        ):  # type: ignore[no-untyped-def]
            return _FakeConn(resp)

        with patch(
            "nous.genai._internal.http._make_connection",
            side_effect=_fake_make_connection,
        ):
            with self.assertRaises(GenAIError) as cm:
                request_json(method="GET", url="https://example.com/x")

        self.assertEqual(cm.exception.info.type, "ProviderError")
        self.assertTrue(cm.exception.info.retryable)

    def test_request_json_maps_429_to_rate_limit_error(self) -> None:
        from nous.genai import GenAIError
        from nous.genai._internal.http import request_json

        resp = _FakeResponse(status=429, body=b'{"message":"slow down"}')

        def _fake_make_connection(
            parsed, timeout_s, *, proxy_url, connect_host=None, tls_server_hostname=None
        ):  # type: ignore[no-untyped-def]
            return _FakeConn(resp)

        with patch(
            "nous.genai._internal.http._make_connection",
            side_effect=_fake_make_connection,
        ):
            with self.assertRaises(GenAIError) as cm:
                request_json(method="GET", url="https://example.com/x")

        self.assertEqual(cm.exception.info.type, "RateLimitError")
        self.assertTrue(cm.exception.info.retryable)

    def test_request_json_maps_network_error_to_retryable_provider_error(self) -> None:
        from nous.genai import GenAIError
        from nous.genai._internal.http import request_json

        class _NetErrConn:
            def request(
                self,
                method: str,
                path: str,
                body: object = None,
                headers: dict[str, str] | None = None,
            ):  # noqa: ARG002
                raise OSError("boom")

            def close(self) -> None:
                return

        def _fake_make_connection(
            parsed, timeout_s, *, proxy_url, connect_host=None, tls_server_hostname=None
        ):  # type: ignore[no-untyped-def]
            return _NetErrConn()

        with patch(
            "nous.genai._internal.http._make_connection",
            side_effect=_fake_make_connection,
        ):
            with self.assertRaises(GenAIError) as cm:
                request_json(method="GET", url="https://example.com/x")

        self.assertEqual(cm.exception.info.type, "ProviderError")
        self.assertTrue(cm.exception.info.retryable)
