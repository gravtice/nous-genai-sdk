import unittest


class _FakeSseResponse:
    def __init__(self, lines: list[bytes]) -> None:
        self._lines = list(lines)
        self._idx = 0

    def readline(self) -> bytes:
        if self._idx >= len(self._lines):
            return b""
        line = self._lines[self._idx]
        self._idx += 1
        return line


class TestSseParser(unittest.TestCase):
    def test_parses_event_id_retry_and_multiline_data(self) -> None:
        from nous.genai._internal.http import SSEEvent, _iter_sse_events

        resp = _FakeSseResponse(
            [
                b": keepalive\n",
                b"event: message\n",
                b"id: abc123\n",
                b"retry: 5000\n",
                b'data: {"a":1}\n',
                b'data: {"b":2}\n',
                b"\n",
            ]
        )
        events = list(_iter_sse_events(resp))
        self.assertEqual(
            events,
            [
                SSEEvent(
                    data='{"a":1}\n{"b":2}',
                    event="message",
                    id="abc123",
                    retry=5000,
                )
            ],
        )

    def test_id_with_null_is_ignored(self) -> None:
        from nous.genai._internal.http import SSEEvent, _iter_sse_events

        resp = _FakeSseResponse(
            [
                b"id: ok\x00nope\n",
                b"data: {}\n",
                b"\n",
            ]
        )
        events = list(_iter_sse_events(resp))
        self.assertEqual(events, [SSEEvent(data="{}", event=None, id=None, retry=None)])

    def test_event_without_data_is_emitted_with_empty_data(self) -> None:
        from nous.genai._internal.http import SSEEvent, _iter_sse_events

        resp = _FakeSseResponse(
            [
                b"id: 1\n",
                b"\n",
            ]
        )
        events = list(_iter_sse_events(resp))
        self.assertEqual(events, [SSEEvent(data="", event=None, id="1", retry=None)])
