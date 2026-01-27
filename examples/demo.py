"""
nous-genai-sdk demo CLI (preset inputs, zero-parameter).

Usage:
  uv run examples/demo.py image --model tuzi-openai:dall-e-3
  uv run examples/demo.py tts --model tuzi-openai:tts-1
  uv run examples/demo.py chat --model tuzi-openai:gpt-4o-mini
  uv run examples/demo.py transcribe --model tuzi-openai:whisper-1
  uv run examples/demo.py embed --model tuzi-openai:text-embedding-3-small

See also:
  uv run examples/demo.py --help
"""

from __future__ import annotations

import argparse
import base64
import binascii
import json
from dataclasses import asdict
from pathlib import Path
from typing import Iterable

from nous.genai import (
    Client,
    GenAIError,
    GenerateRequest,
    Message,
    OutputAudioSpec,
    OutputEmbeddingSpec,
    OutputImageSpec,
    OutputSpec,
    OutputTextSpec,
    Part,
    PartSourceBytes,
    PartSourcePath,
    PartSourceRef,
    PartSourceUrl,
)
from nous.genai.tools import parse_output
from nous.genai.types import detect_mime_type


_DEMO_DIR = Path(__file__).resolve().parent
_DEMO_IMAGE_PATH = _DEMO_DIR / "demo_image.png"
_DEMO_AUDIO_PATH = _DEMO_DIR / "demo_tts.mp3"

_PRESET_IMAGE_PROMPT = "A cute cat, high quality photo, natural lighting."
_PRESET_CHAT_PROMPT = "用中文描述这张图，限制 3 句话。"
_PRESET_TTS_TEXT = "你好，这是一个语音转写测试。"
_PRESET_EMBED_TEXT = "hello world"
_PRESET_CHAT_OUTPUT_SCHEMA = {
    "type": "object",
    "properties": {"description": {"type": "string"}},
    "required": ["description"],
    "additionalProperties": False,
}


def _iter_parts(resp) -> Iterable[Part]:
    for msg in resp.output:
        for part in msg.content:
            yield part


def _print_text(resp) -> None:
    chunks: list[str] = []
    for part in _iter_parts(resp):
        if part.type == "text" and isinstance(part.text, str):
            chunks.append(part.text)
    if chunks:
        print("".join(chunks))
        return
    print(json.dumps(asdict(resp), ensure_ascii=False, indent=2))


def _write_binary(client: Client, *, provider: str, part: Part, out_path: str) -> None:
    src = part.source
    if src is None:
        raise SystemExit("unexpected output: missing source")

    if isinstance(src, PartSourceUrl):
        if not src.url:
            raise SystemExit("unexpected output: url is empty")
        client.download_to_file(provider=provider, url=src.url, output_path=out_path)
        print(f"wrote: {out_path}")
        return

    if isinstance(src, PartSourceBytes):
        if src.encoding == "base64":
            if not src.data:
                raise SystemExit("unexpected output: base64 data is empty")
            try:
                raw = base64.b64decode(src.data, validate=True)
            except (binascii.Error, ValueError) as e:
                raise SystemExit(f"unexpected output: invalid base64: {e}") from None
        else:
            raw = src.data
        if not isinstance(raw, (bytes, bytearray)) or not raw:
            raise SystemExit("unexpected output: bytes data is empty")
        Path(out_path).write_bytes(bytes(raw))
        print(f"wrote: {out_path}")
        return

    if isinstance(src, PartSourceRef):
        provider = src.provider.strip() if isinstance(src.provider, str) else ""
        rid = src.id.strip() if isinstance(src.id, str) else ""
        ref = f"{provider}:{rid}" if provider and rid else rid or provider
        if not ref:
            raise SystemExit("unexpected output: ref is empty")
        raise SystemExit(f"output is ref (cannot write file here): {ref}")

    raise SystemExit(
        f"unexpected output source type: {type(src).__name__} (expected url/bytes)"
    )


def _generate_demo_image(client: Client, *, model: str, out_path: Path) -> None:
    req = GenerateRequest(
        model=model,
        input=[Message(role="user", content=[Part.from_text(_PRESET_IMAGE_PROMPT)])],
        output=OutputSpec(modalities=["image"], image=OutputImageSpec(n=1)),
    )
    resp = client.generate(req)
    part = next(
        (p for p in _iter_parts(resp) if p.type == "image" and p.source is not None),
        None,
    )
    if part is None:
        raise SystemExit("missing image output")
    _write_binary(client, provider=resp.provider, part=part, out_path=str(out_path))


def _cmd_image(args: argparse.Namespace) -> int:
    client = Client()
    _generate_demo_image(client, model=args.model, out_path=_DEMO_IMAGE_PATH)
    return 0


def _generate_demo_tts(client: Client, *, model: str, out_path: Path) -> None:
    audio = OutputAudioSpec(voice="alloy", format="mp3")
    req = GenerateRequest(
        model=model,
        input=[Message(role="user", content=[Part.from_text(_PRESET_TTS_TEXT)])],
        output=OutputSpec(modalities=["audio"], audio=audio),
    )
    resp = client.generate(req)
    part = next(
        (p for p in _iter_parts(resp) if p.type == "audio" and p.source is not None),
        None,
    )
    if part is None:
        raise SystemExit("missing audio output")
    _write_binary(client, provider=resp.provider, part=part, out_path=str(out_path))


def _cmd_tts(args: argparse.Namespace) -> int:
    client = Client()
    _generate_demo_tts(client, model=args.model, out_path=_DEMO_AUDIO_PATH)
    return 0


def _cmd_transcribe(args: argparse.Namespace) -> int:
    client = Client()
    if not _DEMO_AUDIO_PATH.is_file():
        raise SystemExit(
            f"missing preset audio: {_DEMO_AUDIO_PATH.name}; "
            f'run: uv run examples/demo.py tts --model "<tts_model>" '
            f'(e.g. "tuzi-openai:tts-1")'
        )
    mime = detect_mime_type(str(_DEMO_AUDIO_PATH)) or "application/octet-stream"
    audio = Part(
        type="audio", mime_type=mime, source=PartSourcePath(path=str(_DEMO_AUDIO_PATH))
    )

    req = GenerateRequest(
        model=args.model,
        input=[Message(role="user", content=[audio])],
        output=OutputSpec(modalities=["text"]),
    )
    resp = client.generate(req)
    _print_text(resp)
    return 0


def _cmd_chat(args: argparse.Namespace) -> int:
    client = Client()
    cap = client.capabilities(args.model)
    if not cap.supports_json_schema and not cap.supports_tools:
        raise SystemExit(
            "this model does not support structured output (no json_schema/tools)"
        )
    if not _DEMO_IMAGE_PATH.is_file():
        raise SystemExit(
            f"missing preset image: {_DEMO_IMAGE_PATH.name}; "
            f'run: uv run examples/demo.py image --model "<image_model>" '
            f'(e.g. "tuzi-openai:dall-e-3")'
        )
    mime = detect_mime_type(str(_DEMO_IMAGE_PATH)) or "application/octet-stream"
    parts: list[Part] = [
        Part.from_text(_PRESET_CHAT_PROMPT),
        Part(
            type="image",
            mime_type=mime,
            source=PartSourcePath(path=str(_DEMO_IMAGE_PATH)),
        ),
    ]

    if cap.supports_json_schema:
        req = GenerateRequest(
            model=args.model,
            input=[Message(role="user", content=parts)],
            output=OutputSpec(
                modalities=["text"],
                text=OutputTextSpec(
                    format="json", json_schema=_PRESET_CHAT_OUTPUT_SCHEMA
                ),
            ),
        )
        resp = client.generate(req)
        text = ""
        for part in _iter_parts(resp):
            if part.type == "text" and isinstance(part.text, str):
                text += part.text
        obj = json.loads(text)
        print(json.dumps(obj, ensure_ascii=False, indent=2))
        return 0

    req_plain = GenerateRequest(
        model=args.model,
        input=[Message(role="user", content=parts)],
        output=OutputSpec(modalities=["text"]),
    )
    resp_plain = client.generate(req_plain)
    plain_text = ""
    for part in _iter_parts(resp_plain):
        if part.type == "text" and isinstance(part.text, str):
            plain_text += part.text
    plain_text = plain_text.strip()
    if not plain_text:
        raise SystemExit("missing text output (step1)")

    out = parse_output(
        client,
        model=args.model,
        text=plain_text,
        json_schema=_PRESET_CHAT_OUTPUT_SCHEMA,
    )
    print(json.dumps(out, ensure_ascii=False, indent=2))
    return 0


def _cmd_embed(args: argparse.Namespace) -> int:
    client = Client()
    req = GenerateRequest(
        model=args.model,
        input=[Message(role="user", content=[Part.from_text(_PRESET_EMBED_TEXT)])],
        output=OutputSpec(modalities=["embedding"], embedding=OutputEmbeddingSpec()),
    )
    resp = client.generate(req)
    part = next(
        (
            p
            for p in _iter_parts(resp)
            if p.type == "embedding" and p.embedding is not None
        ),
        None,
    )
    if part is None:
        raise SystemExit("missing embedding output")
    vec = part.embedding or []
    print(f"dims: {len(vec)}")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="nous-genai-sdk demo (preset inputs)")
    sub = parser.add_subparsers(dest="cmd", required=True)

    p_chat = sub.add_parser("chat", help="Chat with preset prompt + preset image")
    p_chat.add_argument("--model", required=True, help='e.g. "tuzi-openai:gpt-4o-mini"')
    p_chat.set_defaults(_run=_cmd_chat)

    p_image = sub.add_parser(
        "image", help=f"Generate preset image -> {_DEMO_IMAGE_PATH.name}"
    )
    p_image.add_argument("--model", required=True, help='e.g. "tuzi-openai:dall-e-3"')
    p_image.set_defaults(_run=_cmd_image)

    p_tts = sub.add_parser(
        "tts", help=f"Generate preset speech -> {_DEMO_AUDIO_PATH.name}"
    )
    p_tts.add_argument("--model", required=True, help='e.g. "tuzi-openai:tts-1"')
    p_tts.set_defaults(_run=_cmd_tts)

    p_tr = sub.add_parser(
        "transcribe", help=f"Transcribe preset speech ({_DEMO_AUDIO_PATH.name})"
    )
    p_tr.add_argument("--model", required=True, help='e.g. "tuzi-openai:whisper-1"')
    p_tr.set_defaults(_run=_cmd_transcribe)

    p_embed = sub.add_parser("embed", help="Embed preset text")
    p_embed.add_argument(
        "--model", required=True, help='e.g. "tuzi-openai:text-embedding-3-small"'
    )
    p_embed.set_defaults(_run=_cmd_embed)

    args = parser.parse_args(argv)
    try:
        return int(args._run(args))
    except GenAIError as e:
        raise SystemExit(f"{e.info.type}: {e.info.message}") from None


if __name__ == "__main__":
    raise SystemExit(main())
