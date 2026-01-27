from __future__ import annotations

import errno
import http.client
import ipaddress
import json
import os
import socket
import ssl
import tempfile
import urllib.parse
from base64 import b64encode
from dataclasses import dataclass
from typing import Any, Iterable, Iterator
from uuid import uuid4

from .config import get_default_timeout_ms, get_prefixed_env
from .errors import (
    auth_error,
    invalid_request_error,
    not_supported_error,
    provider_error,
    rate_limit_error,
    timeout_error,
)


def _timeout_seconds(timeout_ms: int | None) -> float:
    if timeout_ms is None:
        timeout_ms = get_default_timeout_ms()
    return max(0.001, timeout_ms / 1000.0)


def _env_truthy(name: str) -> bool:
    return os.environ.get(name) in {"1", "true", "TRUE", "yes", "YES"}


def _default_url_download_max_bytes() -> int:
    raw = get_prefixed_env("URL_DOWNLOAD_MAX_BYTES")
    if raw is None:
        return 128 * 1024 * 1024
    try:
        value = int(raw)
    except ValueError:
        return 128 * 1024 * 1024
    return max(1, value)


def _is_private_ip(ip: ipaddress.IPv4Address | ipaddress.IPv6Address) -> bool:
    return bool(
        ip.is_private
        or ip.is_loopback
        or ip.is_link_local
        or ip.is_multicast
        or ip.is_reserved
        or ip.is_unspecified
    )


def _resolve_host_ips(host: str) -> list[ipaddress.IPv4Address | ipaddress.IPv6Address]:
    out: list[ipaddress.IPv4Address | ipaddress.IPv6Address] = []
    try:
        infos = socket.getaddrinfo(host, None, proto=socket.IPPROTO_TCP)
    except OSError:
        return out
    seen: set[ipaddress.IPv4Address | ipaddress.IPv6Address] = set()
    for family, _, _, _, sockaddr in infos:
        if family == socket.AF_INET:
            addr = sockaddr[0]
        elif family == socket.AF_INET6:
            addr = sockaddr[0]
        else:
            continue
        try:
            ip = ipaddress.ip_address(addr)
        except ValueError:
            continue
        if not isinstance(ip, (ipaddress.IPv4Address, ipaddress.IPv6Address)):
            continue
        if ip in seen:
            continue
        seen.add(ip)
        out.append(ip)
    return out


def _is_private_host(host: str) -> bool:
    h = host.strip().lower().rstrip(".")
    if h in {"localhost"} or h.endswith(".localhost"):
        return True
    try:
        ip = ipaddress.ip_address(h)
    except ValueError:
        ip = None
    if isinstance(ip, (ipaddress.IPv4Address, ipaddress.IPv6Address)):
        return _is_private_ip(ip)
    for resolved in _resolve_host_ips(h):
        if _is_private_ip(resolved):
            return True
    return False


def _resolve_url_host_ips(host: str) -> tuple[list[ipaddress.IPv4Address | ipaddress.IPv6Address], bool]:
    """
    Resolve a URL host once and classify it as private/loopback.

    Returns: (resolved_ips, is_private)
    """
    h = host.strip().lower().rstrip(".")
    if h in {"localhost"} or h.endswith(".localhost"):
        return [], True
    try:
        ip = ipaddress.ip_address(h)
    except ValueError:
        ip = None
    if isinstance(ip, (ipaddress.IPv4Address, ipaddress.IPv6Address)):
        return [ip], _is_private_ip(ip)
    resolved = _resolve_host_ips(h)
    return resolved, any(_is_private_ip(x) for x in resolved)


def download_to_file(
    *,
    url: str,
    output_path: str,
    timeout_ms: int | None = None,
    max_bytes: int | None = None,
    headers: dict[str, str] | None = None,
    proxy_url: str | None = None,
) -> None:
    """
    Download URL to a local file with timeout/proxy support and a hard size limit.

    Security: by default rejects obvious private/loopback IP hosts unless `NOUS_GENAI_ALLOW_PRIVATE_URLS=1`.
    """
    effective_max = _default_url_download_max_bytes() if max_bytes is None else max_bytes
    if effective_max <= 0:
        raise invalid_request_error("max_bytes must be positive")

    cur = url
    initial_host: str | None = None
    for _ in range(5):
        parsed = urllib.parse.urlparse(cur)
        if parsed.scheme.lower() not in {"http", "https"}:
            raise invalid_request_error(f"unsupported url scheme: {parsed.scheme}")
        if not parsed.hostname:
            raise invalid_request_error(f"invalid url: {cur}")
        if initial_host is None:
            initial_host = parsed.hostname
        resolved, is_private = _resolve_url_host_ips(parsed.hostname)
        if is_private and not _env_truthy("NOUS_GENAI_ALLOW_PRIVATE_URLS"):
            raise invalid_request_error(
                "url host is private/loopback; set NOUS_GENAI_ALLOW_PRIVATE_URLS=1 to allow"
            )
        if not resolved:
            raise provider_error(f"dns resolution failed: {parsed.hostname}", retryable=True)
        connect_ip = str(resolved[0])

        path = _path_with_query(parsed)
        timeout_s = _timeout_seconds(timeout_ms)
        conn = _make_connection(
            parsed,
            timeout_s,
            proxy_url=proxy_url,
            connect_host=connect_ip,
            tls_server_hostname=parsed.hostname,
        )
        try:
            req_headers: dict[str, str] = {"Accept": "*/*"}
            if headers and initial_host and parsed.hostname.lower() == initial_host.lower():
                req_headers.update(headers)
            if proxy_url:
                target_port = parsed.port or (443 if parsed.scheme.lower() == "https" else 80)
                default_port = 443 if parsed.scheme.lower() == "https" else 80
                req_headers["Host"] = (
                    parsed.hostname
                    if target_port == default_port
                    else f"{parsed.hostname}:{target_port}"
                )
            conn.request("GET", path, headers=req_headers)
            resp = conn.getresponse()
            if resp.status in {301, 302, 303, 307, 308}:
                loc = resp.getheader("Location")
                if not loc:
                    raise provider_error("redirect response missing Location header")
                cur = urllib.parse.urljoin(cur, loc)
                continue
            if resp.status < 200 or resp.status >= 300:
                raw = resp.read(64 * 1024 + 1)
                _raise_for_status(resp.status, raw[:64 * 1024])

            raw_len = resp.getheader("Content-Length")
            if raw_len:
                try:
                    n = int(raw_len)
                except ValueError:
                    n = -1
                if n > effective_max:
                    raise not_supported_error(
                        f"url download too large ({n} > {effective_max}); set NOUS_GENAI_URL_DOWNLOAD_MAX_BYTES or use path/ref"
                    )

            out_dir = os.path.dirname(os.path.abspath(output_path)) or "."
            with tempfile.NamedTemporaryFile(prefix="genaisdk-dl-", dir=out_dir, delete=False) as tmp:
                tmp_path = tmp.name
            total = 0
            try:
                with open(tmp_path, "wb") as f:
                    while True:
                        chunk = resp.read(64 * 1024)
                        if not chunk:
                            break
                        total += len(chunk)
                        if total > effective_max:
                            raise not_supported_error(
                                f"url download exceeded limit ({total} > {effective_max}); set NOUS_GENAI_URL_DOWNLOAD_MAX_BYTES or use path/ref"
                            )
                        f.write(chunk)
            except Exception:
                try:
                    os.unlink(tmp_path)
                except OSError:
                    pass
                raise
            try:
                os.replace(tmp_path, output_path)
            except Exception:
                try:
                    os.unlink(tmp_path)
                except OSError:
                    pass
                raise
            return
        except (socket.timeout, TimeoutError):
            raise timeout_error("request timeout")
        except (ssl.SSLError, http.client.HTTPException, OSError) as e:
            raise provider_error(f"network error: {type(e).__name__}", retryable=True)
        finally:
            conn.close()

    raise provider_error("too many redirects", retryable=False)


def download_to_tempfile(
    *,
    url: str,
    timeout_ms: int | None = None,
    max_bytes: int | None = None,
    headers: dict[str, str] | None = None,
    suffix: str = "",
    proxy_url: str | None = None,
) -> str:
    with tempfile.NamedTemporaryFile(prefix="genaisdk-", suffix=suffix, delete=False) as f:
        tmp_path = f.name
    try:
        download_to_file(
            url=url,
            output_path=tmp_path,
            timeout_ms=timeout_ms,
            max_bytes=max_bytes,
            headers=headers,
            proxy_url=proxy_url,
        )
    except Exception:
        try:
            os.unlink(tmp_path)
        except OSError:
            pass
        raise
    return tmp_path


def _proxy_tunnel_headers(proxy: urllib.parse.ParseResult) -> dict[str, str] | None:
    user = proxy.username
    pw = proxy.password
    if user is None and pw is None:
        return None
    user = "" if user is None else user
    pw = "" if pw is None else pw
    token = b64encode(f"{user}:{pw}".encode("utf-8")).decode("ascii")
    return {"Proxy-Authorization": f"Basic {token}"}


class _PinnedHTTPConnection(http.client.HTTPConnection):
    def __init__(
        self,
        host: str,
        port: int,
        *,
        connect_host: str,
        timeout: float,
    ) -> None:
        super().__init__(host, port, timeout=timeout)
        self._connect_host = connect_host

    def connect(self) -> None:
        self.sock = self._create_connection(
            (self._connect_host, self.port),
            self.timeout,
            self.source_address,
        )
        try:
            self.sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
        except OSError as e:
            if e.errno != errno.ENOPROTOOPT:
                raise
        if self._tunnel_host:
            self._tunnel()


class _PinnedHTTPSConnection(http.client.HTTPSConnection):
    def __init__(
        self,
        host: str,
        port: int,
        *,
        connect_host: str,
        tls_server_hostname: str,
        timeout: float,
        context: ssl.SSLContext,
    ) -> None:
        super().__init__(host, port, timeout=timeout, context=context)
        self._connect_host = connect_host
        self._tls_server_hostname = tls_server_hostname

    def connect(self) -> None:
        self.sock = self._create_connection(
            (self._connect_host, self.port),
            self.timeout,
            self.source_address,
        )
        try:
            self.sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
        except OSError as e:
            if e.errno != errno.ENOPROTOOPT:
                raise
        if self._tunnel_host:
            self._tunnel()
        self.sock = self._context.wrap_socket(self.sock, server_hostname=self._tls_server_hostname)


def _make_connection(
    parsed: urllib.parse.ParseResult,
    timeout_s: float,
    *,
    proxy_url: str | None,
    connect_host: str | None = None,
    tls_server_hostname: str | None = None,
) -> http.client.HTTPConnection:
    scheme = parsed.scheme.lower()
    target_host = parsed.hostname
    if not target_host:
        raise invalid_request_error(f"invalid url: {parsed.geturl()}")

    target_port = parsed.port
    is_https = scheme == "https"
    if not is_https and scheme != "http":
        raise invalid_request_error(f"unsupported url scheme: {scheme}")

    target_connect_host = target_host if connect_host is None else connect_host
    tls_hostname = target_host if tls_server_hostname is None else tls_server_hostname

    if proxy_url:
        p = urllib.parse.urlparse(proxy_url)
        if not p.hostname:
            raise invalid_request_error(f"invalid proxy url: {proxy_url}")
        if p.scheme.lower() not in {"http", "https"}:
            raise invalid_request_error(f"unsupported proxy url scheme: {p.scheme}")
        proxy_port = p.port or (443 if p.scheme == "https" else 80)
        effective_target_port = target_port or (443 if is_https else 80)
        if is_https:
            ctx = ssl.create_default_context()
            conn = _PinnedHTTPSConnection(
                p.hostname,
                proxy_port,
                connect_host=p.hostname,
                tls_server_hostname=tls_hostname,
                timeout=timeout_s,
                context=ctx,
            )
        else:
            conn = http.client.HTTPConnection(p.hostname, proxy_port, timeout=timeout_s)
        conn.set_tunnel(target_connect_host, effective_target_port, headers=_proxy_tunnel_headers(p))
        return conn

    if is_https:
        ctx = ssl.create_default_context()
        effective_port = target_port or 443
        if target_connect_host != target_host or tls_hostname != target_host:
            return _PinnedHTTPSConnection(
                target_host,
                effective_port,
                connect_host=target_connect_host,
                tls_server_hostname=tls_hostname,
                timeout=timeout_s,
                context=ctx,
            )
        return http.client.HTTPSConnection(target_host, effective_port, timeout=timeout_s, context=ctx)
    effective_port = target_port or 80
    if target_connect_host != target_host:
        return _PinnedHTTPConnection(
            target_host,
            effective_port,
            connect_host=target_connect_host,
            timeout=timeout_s,
        )
    return http.client.HTTPConnection(target_host, effective_port, timeout=timeout_s)


def _path_with_query(parsed: urllib.parse.ParseResult) -> str:
    path = parsed.path or "/"
    if parsed.query:
        return f"{path}?{parsed.query}"
    return path


def _extract_error_message(body: bytes) -> tuple[str, str | None]:
    if not body:
        return "empty error body", None
    try:
        obj = json.loads(body)
    except Exception:
        text = body.decode("utf-8", errors="replace")
        return text[:2_000], None

    if isinstance(obj, dict):
        if isinstance(obj.get("error"), dict):
            err = obj["error"]
            msg = err.get("message") or err.get("detail") or str(err)
            code = err.get("code") or err.get("status") or None
            return str(msg)[:2_000], str(code) if code is not None else None
        msg = obj.get("message") or str(obj)
        code = obj.get("code") or obj.get("status") or None
        return str(msg)[:2_000], str(code) if code is not None else None

    return str(obj)[:2_000], None


def _raise_for_status(status: int, body: bytes) -> None:
    message, provider_code = _extract_error_message(body)
    if status in (401, 403):
        raise auth_error(message, provider_code=provider_code)
    if status == 429:
        raise rate_limit_error(message, provider_code=provider_code)
    if status in (400, 404, 409, 415, 422):
        raise invalid_request_error(message, provider_code=provider_code)
    if status in (408, 504):
        raise timeout_error(message)
    if 500 <= status <= 599:
        raise provider_error(message, provider_code=provider_code, retryable=True)
    raise provider_error(message, provider_code=provider_code, retryable=False)


@dataclass(frozen=True, slots=True)
class StreamingBody:
    content_type: str
    content_length: int
    chunks: Iterable[bytes]


def multipart_form_data_fields(*, fields: dict[str, str]) -> StreamingBody:
    boundary = uuid4().hex
    boundary_bytes = boundary.encode("ascii")
    crlf = b"\r\n"
    tail = b"--" + boundary_bytes + b"--" + crlf

    parts = []
    total = len(tail)
    for k, v in fields.items():
        part = (
            b"--"
            + boundary_bytes
            + crlf
            + f'Content-Disposition: form-data; name="{k}"'.encode("utf-8")
            + crlf
            + crlf
            + v.encode("utf-8")
            + crlf
        )
        parts.append(part)
        total += len(part)

    def _chunks() -> Iterator[bytes]:
        yield from parts
        yield tail

    return StreamingBody(
        content_type=f"multipart/form-data; boundary={boundary}",
        content_length=total,
        chunks=_chunks(),
    )


def multipart_form_data(
    *,
    fields: dict[str, str],
    file_field: str,
    file_path: str,
    filename: str,
    file_mime_type: str,
    chunk_size: int = 64 * 1024,
) -> StreamingBody:
    boundary = uuid4().hex
    boundary_bytes = boundary.encode("ascii")
    crlf = b"\r\n"

    def _field_part(name: str, value: str) -> bytes:
        header = (
            b"--"
            + boundary_bytes
            + crlf
            + f'Content-Disposition: form-data; name="{name}"'.encode("utf-8")
            + crlf
            + crlf
        )
        return header + value.encode("utf-8") + crlf

    def _file_preamble() -> bytes:
        return (
            b"--"
            + boundary_bytes
            + crlf
            + (
                f'Content-Disposition: form-data; name="{file_field}"; filename="{filename}"'
            ).encode("utf-8")
            + crlf
            + f"Content-Type: {file_mime_type}".encode("utf-8")
            + crlf
            + crlf
        )

    file_size = os.stat(file_path).st_size
    tail = b"--" + boundary_bytes + b"--" + crlf
    preamble = _file_preamble()

    total = len(tail) + len(preamble) + file_size + len(crlf)
    field_parts = []
    for k, v in fields.items():
        part = _field_part(k, v)
        field_parts.append(part)
        total += len(part)

    def _chunks() -> Iterator[bytes]:
        for part in field_parts:
            yield part
        yield preamble
        with open(file_path, "rb") as f:
            while True:
                chunk = f.read(chunk_size)
                if not chunk:
                    break
                yield chunk
        yield crlf
        yield tail

    return StreamingBody(
        content_type=f"multipart/form-data; boundary={boundary}",
        content_length=total,
        chunks=_chunks(),
    )


def multipart_form_data_json_and_file(
    *,
    metadata_field: str,
    metadata: dict[str, Any],
    file_field: str,
    file_path: str,
    filename: str,
    file_mime_type: str,
    chunk_size: int = 64 * 1024,
) -> StreamingBody:
    boundary = uuid4().hex
    boundary_bytes = boundary.encode("ascii")
    crlf = b"\r\n"

    meta_bytes = json.dumps(metadata, separators=(",", ":")).encode("utf-8")
    meta_part = (
        b"--"
        + boundary_bytes
        + crlf
        + f'Content-Disposition: form-data; name="{metadata_field}"'.encode("utf-8")
        + crlf
        + b"Content-Type: application/json; charset=utf-8"
        + crlf
        + crlf
        + meta_bytes
        + crlf
    )
    file_preamble = (
        b"--"
        + boundary_bytes
        + crlf
        + (
            f'Content-Disposition: form-data; name="{file_field}"; filename="{filename}"'
        ).encode("utf-8")
        + crlf
        + f"Content-Type: {file_mime_type}".encode("utf-8")
        + crlf
        + crlf
    )
    tail = b"--" + boundary_bytes + b"--" + crlf

    file_size = os.stat(file_path).st_size
    total = len(meta_part) + len(file_preamble) + file_size + len(crlf) + len(tail)

    def _chunks() -> Iterator[bytes]:
        yield meta_part
        yield file_preamble
        with open(file_path, "rb") as f:
            while True:
                chunk = f.read(chunk_size)
                if not chunk:
                    break
                yield chunk
        yield crlf
        yield tail

    return StreamingBody(
        content_type=f"multipart/form-data; boundary={boundary}",
        content_length=total,
        chunks=_chunks(),
    )


def request_json(
    *,
    method: str,
    url: str,
    headers: dict[str, str] | None = None,
    json_body: Any | None = None,
    timeout_ms: int | None = None,
    proxy_url: str | None = None,
) -> dict[str, Any]:
    body = None if json_body is None else json.dumps(json_body, separators=(",", ":")).encode("utf-8")
    req_headers = {"Accept": "application/json"}
    if body is not None:
        req_headers["Content-Type"] = "application/json"
        req_headers["Content-Length"] = str(len(body))
    if headers:
        req_headers.update(headers)

    parsed = urllib.parse.urlparse(url)
    path = _path_with_query(parsed)
    timeout_s = _timeout_seconds(timeout_ms)
    conn = _make_connection(parsed, timeout_s, proxy_url=proxy_url)
    try:
        conn.request(method.upper(), path, body=body, headers=req_headers)
        resp = conn.getresponse()
        raw = resp.read()
        if resp.status < 200 or resp.status >= 300:
            _raise_for_status(resp.status, raw)
        if not raw:
            return {}
        try:
            obj = json.loads(raw)
        except Exception:
            raise provider_error("invalid json response", retryable=True)
        if not isinstance(obj, dict):
            raise provider_error("invalid json response", retryable=True)
        return obj
    except (socket.timeout, TimeoutError):
        raise timeout_error("request timeout")
    except (ssl.SSLError, http.client.HTTPException, OSError) as e:
        raise provider_error(f"network error: {type(e).__name__}", retryable=True)
    finally:
        conn.close()


def request_bytes(
    *,
    method: str,
    url: str,
    headers: dict[str, str] | None = None,
    body: bytes | None = None,
    timeout_ms: int | None = None,
    proxy_url: str | None = None,
) -> bytes:
    req_headers: dict[str, str] = {}
    if body is not None:
        req_headers["Content-Length"] = str(len(body))
    if headers:
        req_headers.update(headers)

    parsed = urllib.parse.urlparse(url)
    path = _path_with_query(parsed)
    timeout_s = _timeout_seconds(timeout_ms)
    conn = _make_connection(parsed, timeout_s, proxy_url=proxy_url)
    try:
        conn.request(method.upper(), path, body=body, headers=req_headers)
        resp = conn.getresponse()
        raw = resp.read()
        if resp.status < 200 or resp.status >= 300:
            _raise_for_status(resp.status, raw)
        return raw
    except (socket.timeout, TimeoutError):
        raise timeout_error("request timeout")
    except (ssl.SSLError, http.client.HTTPException, OSError) as e:
        raise provider_error(f"network error: {type(e).__name__}", retryable=True)
    finally:
        conn.close()


def request_stream_json_sse(
    *,
    method: str,
    url: str,
    headers: dict[str, str] | None = None,
    json_body: Any | None = None,
    timeout_ms: int | None = None,
    proxy_url: str | None = None,
) -> Iterator[dict[str, Any]]:
    body = None if json_body is None else json.dumps(json_body, separators=(",", ":")).encode("utf-8")
    req_headers = {"Accept": "text/event-stream"}
    if body is not None:
        req_headers["Content-Type"] = "application/json"
        req_headers["Content-Length"] = str(len(body))
    if headers:
        req_headers.update(headers)

    parsed = urllib.parse.urlparse(url)
    path = _path_with_query(parsed)
    timeout_s = _timeout_seconds(timeout_ms)
    conn = _make_connection(parsed, timeout_s, proxy_url=proxy_url)

    try:
        conn.request(method.upper(), path, body=body, headers=req_headers)
        resp = conn.getresponse()
        if resp.status < 200 or resp.status >= 300:
            raw = resp.read()
            _raise_for_status(resp.status, raw)

        def _iter() -> Iterator[dict[str, Any]]:
            try:
                for ev in _iter_sse_events(resp):
                    if not ev.data:
                        continue
                    if ev.data == "[DONE]":
                        return
                    try:
                        obj = json.loads(ev.data)
                    except Exception:
                        raise provider_error(f"invalid sse json: {ev.data[:200]}")
                    if isinstance(obj, dict):
                        yield obj
            finally:
                conn.close()

        return _iter()
    except (socket.timeout, TimeoutError):
        conn.close()
        raise timeout_error("request timeout")
    except (ssl.SSLError, http.client.HTTPException, OSError) as e:
        conn.close()
        raise provider_error(f"network error: {type(e).__name__}", retryable=True)
    except Exception:
        conn.close()
        raise


def request_streaming_body_json(
    *,
    method: str,
    url: str,
    headers: dict[str, str] | None,
    body: StreamingBody,
    timeout_ms: int | None = None,
    proxy_url: str | None = None,
) -> dict[str, Any]:
    req_headers = {"Accept": "application/json"}
    req_headers["Content-Type"] = body.content_type
    req_headers["Content-Length"] = str(body.content_length)
    if headers:
        req_headers.update(headers)

    parsed = urllib.parse.urlparse(url)
    path = _path_with_query(parsed)
    timeout_s = _timeout_seconds(timeout_ms)
    conn = _make_connection(parsed, timeout_s, proxy_url=proxy_url)

    try:
        conn.putrequest(method.upper(), path)
        for k, v in req_headers.items():
            conn.putheader(k, v)
        conn.endheaders()
        for chunk in body.chunks:
            if chunk:
                conn.send(chunk)
        resp = conn.getresponse()
        raw = resp.read()
        if resp.status < 200 or resp.status >= 300:
            _raise_for_status(resp.status, raw)
        if not raw:
            return {}
        try:
            obj = json.loads(raw)
        except Exception:
            raise provider_error("invalid json response", retryable=True)
        if not isinstance(obj, dict):
            raise provider_error("invalid json response", retryable=True)
        return obj
    except (socket.timeout, TimeoutError):
        raise timeout_error("request timeout")
    except (ssl.SSLError, http.client.HTTPException, OSError) as e:
        raise provider_error(f"network error: {type(e).__name__}", retryable=True)
    finally:
        conn.close()


@dataclass(frozen=True, slots=True)
class SSEEvent:
    data: str
    event: str | None = None
    id: str | None = None
    retry: int | None = None


def request_stream_sse(
    *,
    method: str,
    url: str,
    headers: dict[str, str] | None = None,
    body: bytes | None = None,
    timeout_ms: int | None = None,
    proxy_url: str | None = None,
) -> Iterator[SSEEvent]:
    req_headers = {"Accept": "text/event-stream"}
    if body is not None:
        req_headers["Content-Length"] = str(len(body))
    if headers:
        req_headers.update(headers)

    parsed = urllib.parse.urlparse(url)
    path = _path_with_query(parsed)
    timeout_s = _timeout_seconds(timeout_ms)
    conn = _make_connection(parsed, timeout_s, proxy_url=proxy_url)

    try:
        conn.request(method.upper(), path, body=body, headers=req_headers)
        resp = conn.getresponse()
        if resp.status < 200 or resp.status >= 300:
            raw = resp.read()
            _raise_for_status(resp.status, raw)

        def _iter() -> Iterator[SSEEvent]:
            try:
                yield from _iter_sse_events(resp)
            finally:
                conn.close()

        return _iter()
    except (socket.timeout, TimeoutError):
        conn.close()
        raise timeout_error("request timeout")
    except (ssl.SSLError, http.client.HTTPException, OSError) as e:
        conn.close()
        raise provider_error(f"network error: {type(e).__name__}", retryable=True)
    except Exception:
        conn.close()
        raise


def _iter_sse_events(resp: http.client.HTTPResponse) -> Iterator[SSEEvent]:
    buffer: list[str] = []
    event_type: str | None = None
    event_id: str | None = None
    retry_ms: int | None = None
    while True:
        line = resp.readline()
        if not line:
            break
        text = line.decode("utf-8", errors="replace")
        text = text.rstrip("\n")
        if text.endswith("\r"):
            text = text[:-1]
        if not text:
            if buffer or event_type is not None or event_id is not None or retry_ms is not None:
                yield SSEEvent(
                    data="\n".join(buffer),
                    event=event_type,
                    id=event_id,
                    retry=retry_ms,
                )
                buffer.clear()
                event_type = None
                event_id = None
                retry_ms = None
            continue
        if text.startswith(":"):
            continue
        if ":" in text:
            field, value = text.split(":", 1)
            value = value[1:] if value.startswith(" ") else value
        else:
            field, value = text, ""
        if field == "data":
            buffer.append(value)
        elif field == "event":
            event_type = value or None
        elif field == "id":
            if "\x00" not in value:
                event_id = value
        elif field == "retry":
            try:
                retry_ms = int(value)
            except ValueError:
                continue
    if buffer or event_type is not None or event_id is not None or retry_ms is not None:
        yield SSEEvent(data="\n".join(buffer), event=event_type, id=event_id, retry=retry_ms)


def _iter_sse_data(resp: http.client.HTTPResponse) -> Iterator[str]:
    for ev in _iter_sse_events(resp):
        if ev.data:
            yield ev.data
