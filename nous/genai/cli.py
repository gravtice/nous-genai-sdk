from __future__ import annotations

import argparse
import base64
import json
import secrets
import sys
import threading
import time
import urllib.parse
from collections.abc import Callable
from dataclasses import replace
from typing import TypeVar

from .client import Client
from ._internal.errors import GenAIError
from ._internal.http import download_to_file
from .reference import (
    get_model_catalog,
    get_parameter_mappings,
    get_sdk_supported_models,
)
from .types import (
    GenerateRequest,
    Message,
    OutputAudioSpec,
    OutputEmbeddingSpec,
    OutputImageSpec,
    OutputSpec,
    OutputTextSpec,
    OutputVideoSpec,
    Part,
    PartSourceBytes,
    PartSourcePath,
    PartSourceUrl,
    detect_mime_type,
)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        prog="genai", description="nous-genai-sdk CLI (minimal)"
    )
    sub = parser.add_subparsers(dest="command")

    model = sub.add_parser("model", help="Model discovery and support listing")
    model_sub = model.add_subparsers(dest="model_command", required=True)

    model_sub.add_parser("sdk", help="List SDK curated models")

    pm = model_sub.add_parser(
        "provider", help="List remotely available model ids for a provider"
    )
    pm.add_argument(
        "--provider", required=True, help="Provider (e.g. openai/google/tuzi-openai)"
    )
    pm.add_argument("--timeout-ms", type=int, help="Timeout budget in milliseconds")

    av = model_sub.add_parser(
        "available",
        help="List available models (sdk ∩ provider) with capabilities",
    )
    av_scope = av.add_mutually_exclusive_group(required=True)
    av_scope.add_argument(
        "--provider", help="Provider (e.g. openai/google/tuzi-openai)"
    )
    av_scope.add_argument(
        "--all", action="store_true", help="List across all providers"
    )
    av.add_argument("--timeout-ms", type=int, help="Timeout budget in milliseconds")

    um = model_sub.add_parser(
        "unsupported",
        help="List provider-available but not in SDK catalog models",
    )
    um.add_argument("--provider", help="Provider (omit to scan all catalog providers)")
    um.add_argument("--timeout-ms", type=int, help="Timeout budget in milliseconds")

    st = model_sub.add_parser(
        "stale",
        help="List stale model ids (sdk catalog - provider) for a provider",
    )
    st.add_argument(
        "--provider", required=True, help="Provider (e.g. openai/google/tuzi-openai)"
    )
    st.add_argument("--timeout-ms", type=int, help="Timeout budget in milliseconds")

    sub.add_parser("mapping", help="Print parameter mapping table")

    token = sub.add_parser("token", help="Token utilities")
    token_sub = token.add_subparsers(dest="token_command", required=True)
    token_sub.add_parser("generate", help="Generate a new token (sk-...)")

    parser.add_argument(
        "--model", default="openai:gpt-4o-mini", help='Model like "openai:gpt-4o-mini"'
    )
    parser.add_argument(
        "--protocol",
        choices=["chat_completions", "responses"],
        help='OpenAI chat protocol override ("chat_completions" or "responses")',
    )
    parser.add_argument("--prompt", help="Text prompt")
    parser.add_argument(
        "--prompt-path",
        help="Read prompt text from a file (lower priority than --prompt)",
    )
    parser.add_argument("--image-path", help="Input image file path")
    parser.add_argument("--audio-path", help="Input audio file path")
    parser.add_argument("--video-path", help="Input video file path")
    parser.add_argument("--output-path", help="Write output to file (text/json/binary)")
    parser.add_argument("--ouput-path", dest="output_path", help=argparse.SUPPRESS)
    parser.add_argument(
        "--timeout-ms",
        type=int,
        help="Timeout budget in milliseconds (overrides NOUS_GENAI_TIMEOUT_MS)",
    )

    probe = sub.add_parser("probe", help="Probe model modalities/modes for a provider")
    probe.add_argument("--provider", required=True, help="Provider (e.g. tuzi-web)")
    probe.add_argument(
        "--model", help="Comma-separated model ids (or provider:model_id)"
    )
    probe.add_argument(
        "--all",
        action="store_true",
        help="Probe all SDK-supported models for the provider",
    )
    probe.add_argument(
        "--timeout-ms",
        type=int,
        help="Timeout budget in milliseconds (overrides NOUS_GENAI_TIMEOUT_MS)",
    )

    args = parser.parse_args(argv)
    timeout_ms: int | None = getattr(args, "timeout_ms", None)
    if timeout_ms is not None and timeout_ms < 1:
        raise SystemExit("--timeout-ms must be >= 1")

    if args.command == "mapping":
        try:
            _print_mappings()
        except BrokenPipeError:
            return
        return

    if args.command == "token":
        cmd = str(getattr(args, "token_command", "") or "").strip()
        if cmd == "generate":
            print(_generate_token())
            return
        raise SystemExit(f"unknown token subcommand: {cmd}")

    if args.command == "model":
        try:
            cmd = str(getattr(args, "model_command", "") or "").strip()
            if cmd == "sdk":
                _print_sdk_supported()
                return
            if cmd == "provider":
                _print_provider_models(str(args.provider), timeout_ms=timeout_ms)
                return
            if cmd == "available":
                if bool(getattr(args, "all", False)):
                    _print_all_available_models(timeout_ms=timeout_ms)
                    return
                _print_available_models(str(args.provider), timeout_ms=timeout_ms)
                return
            if cmd == "unsupported":
                if getattr(args, "provider", None):
                    _print_unsupported_models(str(args.provider), timeout_ms=timeout_ms)
                    return
                _print_unsupported(timeout_ms=timeout_ms)
                return
            if cmd == "stale":
                _print_stale_models(str(args.provider), timeout_ms=timeout_ms)
                return
            raise SystemExit(f"unknown model subcommand: {cmd}")
        except BrokenPipeError:
            return

    if args.command == "probe":
        try:
            raise SystemExit(_run_probe(args, timeout_ms=timeout_ms))
        except BrokenPipeError:
            return

    provider, model_id = _split_model(args.model)
    prompt = args.prompt
    if prompt is None and args.prompt_path:
        try:
            with open(args.prompt_path, "r", encoding="utf-8") as f:
                prompt = f.read()
        except OSError as e:
            raise SystemExit(f"cannot read --prompt-path: {e}") from None
    client = Client()
    _apply_protocol_override(client, provider=provider, protocol=args.protocol)

    cap = client.capabilities(args.model)
    output = _infer_output_spec(provider=provider, model_id=model_id, cap=cap)

    parts = _build_input_parts(
        prompt=prompt,
        image_path=args.image_path,
        audio_path=args.audio_path,
        video_path=args.video_path,
        input_modalities=set(cap.input_modalities),
        output_modalities=set(output.modalities),
        provider=provider,
        model_id=model_id,
    )
    req = GenerateRequest(
        model=args.model,
        input=[Message(role="user", content=parts)],
        output=output,
        wait=True,
    )
    if timeout_ms is not None:
        req = replace(req, params=replace(req.params, timeout_ms=timeout_ms))

    try:
        wait_spinner = bool(req.wait) and bool(getattr(cap, "supports_job", False))
        show_progress = wait_spinner and sys.stderr.isatty()
        resp, elapsed_s = _run_with_spinner(
            lambda: client.generate(req),
            enabled=show_progress,
            label="等待任务完成",
        )
        if resp.status != "completed":
            if resp.job and resp.job.job_id:
                print(resp.job.job_id)
                if resp.status == "running":
                    effective_timeout_ms = timeout_ms
                    if effective_timeout_ms is None:
                        effective_timeout_ms = getattr(
                            client, "_default_timeout_ms", None
                        )
                    timeout_note = (
                        f"{effective_timeout_ms}ms"
                        if isinstance(effective_timeout_ms, int)
                        else "timeout"
                    )
                    print(
                        f"[INFO] 任务仍在运行（等待 {elapsed_s:.1f}s，可能已超时 {timeout_note}）；已返回 job_id。"
                        "可增大 --timeout-ms 或设置 NOUS_GENAI_TIMEOUT_MS 后重试。",
                        file=sys.stderr,
                    )
                    if args.output_path:
                        print(
                            f"[INFO] 未写入输出文件：{args.output_path}",
                            file=sys.stderr,
                        )
                return
            raise SystemExit(f"[FAIL]: request status={resp.status}")
        if not resp.output:
            raise SystemExit("[FAIL]: missing output")
        _write_response(
            resp.output[0].content,
            output=output,
            output_path=args.output_path,
            timeout_ms=timeout_ms,
            download_auth=_download_auth(client, provider=provider),
        )
        if show_progress:
            print(f"[INFO] 完成，用时 {elapsed_s:.1f}s", file=sys.stderr)
    except GenAIError as e:
        code = f" ({e.info.provider_code})" if e.info.provider_code else ""
        retryable = " retryable" if e.info.retryable else ""
        raise SystemExit(
            f"[FAIL]{code}{retryable}: {e.info.type}: {e.info.message}"
        ) from None


_DEFAULT_VIDEO_URL = (
    "https://interactive-examples.mdn.mozilla.net/media/cc0-videos/flower.mp4"
)


def _run_probe(args: argparse.Namespace, *, timeout_ms: int | None) -> int:
    from .client import _normalize_provider
    from .reference import get_sdk_supported_models_for_provider

    provider = _normalize_provider(str(args.provider))
    if not provider:
        raise SystemExit("--provider must be non-empty")

    if bool(args.all) == bool(args.model):
        raise SystemExit('probe requires exactly one of: "--model" or "--all"')

    if args.all:
        rows = get_sdk_supported_models_for_provider(provider)
        model_ids = [
            str(r["model_id"])
            for r in rows
            if isinstance(r, dict) and isinstance(r.get("model_id"), str)
        ]
    else:
        model_ids = _parse_probe_models(provider, str(args.model))

    if not model_ids:
        raise SystemExit(f"no models to probe for provider={provider}")

    client = Client()
    totals = {"ok": 0, "fail": 0, "skip": 0}

    for model_id in model_ids:
        model = f"{provider}:{model_id}"
        print(f"== {model} ==")
        try:
            cap = client.capabilities(model)
        except GenAIError as e:
            print(f"[FAIL] capabilities: {e.info.type}: {e.info.message}")
            totals["fail"] += 1
            print()
            continue

        modes = _probe_modes_for(cap)
        print(
            "declared:"
            f" modes={','.join(modes)}"
            f" in={','.join(sorted(cap.input_modalities))}"
            f" out={','.join(sorted(cap.output_modalities))}"
        )

        results = _probe_model(
            client,
            provider=provider,
            model_id=model_id,
            cap=cap,
            timeout_ms=timeout_ms,
        )
        for k in totals.keys():
            totals[k] += results[k]

        status = "OK" if results["fail"] == 0 else "FAIL"
        print(
            f"result: {status} (ok={results['ok']} fail={results['fail']} skip={results['skip']})"
        )
        print()

    print(f"[SUMMARY] ok={totals['ok']} fail={totals['fail']} skip={totals['skip']}")
    return 0 if totals["fail"] == 0 else 1


_T = TypeVar("_T")


def _run_with_spinner(
    fn: Callable[[], _T], *, enabled: bool, label: str
) -> tuple[_T, float]:
    start = time.perf_counter()
    if not enabled or not sys.stderr.isatty():
        out = fn()
        return out, time.perf_counter() - start

    done = threading.Event()
    result: dict[str, _T] = {}
    error: dict[str, BaseException] = {}

    def _worker() -> None:
        try:
            result["value"] = fn()
        except BaseException as e:  # noqa: BLE001
            error["exc"] = e
        finally:
            done.set()

    t = threading.Thread(target=_worker, name="genai-cli-wait", daemon=True)
    t.start()

    if done.wait(0.25):
        t.join()
        exc = error.get("exc")
        if exc is not None:
            raise exc
        if "value" not in result:
            raise RuntimeError("missing result value")
        return result["value"], time.perf_counter() - start

    frames = ["⠋", "⠙", "⠹", "⠸", "⠼", "⠴", "⠦", "⠧", "⠇", "⠏"]
    i = 0
    try:
        while not done.wait(0.1):
            frame = frames[i % len(frames)]
            i += 1
            elapsed = time.perf_counter() - start
            sys.stderr.write(f"\r{frame} {label}... {elapsed:5.1f}s")
            sys.stderr.flush()
    finally:
        sys.stderr.write("\r" + (" " * 64) + "\r")
        sys.stderr.flush()
        t.join()

    exc = error.get("exc")
    if exc is not None:
        raise exc
    if "value" not in result:
        raise RuntimeError("missing result value")
    return result["value"], time.perf_counter() - start


def _parse_probe_models(provider: str, value: str) -> list[str]:
    from .client import _normalize_provider

    p = _normalize_provider(provider)
    items = [x.strip() for x in value.split(",")]
    out: list[str] = []
    seen: set[str] = set()
    for raw in items:
        if not raw:
            continue
        if ":" in raw:
            pp, mid = raw.split(":", 1)
            pp = _normalize_provider(pp)
            mid = mid.strip()
            if pp != p:
                raise SystemExit(f"model provider mismatch: expected {p}, got {pp}")
            raw = mid
        if not raw:
            continue
        if raw in seen:
            continue
        seen.add(raw)
        out.append(raw)
    return out


def _probe_modes_for(cap) -> list[str]:
    modes: list[str] = ["sync"]
    if cap.supports_stream:
        modes.append("stream")
    if cap.supports_job:
        modes.append("job")
    modes.append("async")
    return modes


def _probe_model(
    client: Client,
    *,
    provider: str,
    model_id: str,
    cap,
    timeout_ms: int | None,
) -> dict[str, int]:
    results = {"ok": 0, "fail": 0, "skip": 0}

    out_modalities = sorted(cap.output_modalities)
    for i, out_modality in enumerate(out_modalities):
        ok = _probe_output_modality(
            client,
            provider=provider,
            model_id=model_id,
            cap=cap,
            out_modality=out_modality,
            timeout_ms=timeout_ms,
            probe_job=bool(cap.supports_job) and i == 0,
        )
        _accumulate(results, ok)

    for in_modality in sorted(set(cap.input_modalities) - {"text"}):
        ok = _probe_input_modality(
            client,
            provider=provider,
            model_id=model_id,
            cap=cap,
            in_modality=in_modality,
            timeout_ms=timeout_ms,
        )
        _accumulate(results, ok)

    if cap.supports_stream:
        ok = _probe_stream_mode(
            client,
            provider=provider,
            model_id=model_id,
            cap=cap,
            timeout_ms=timeout_ms,
        )
        _accumulate(results, ok)

    return results


def _accumulate(totals: dict[str, int], outcome: dict[str, int]) -> None:
    for k, v in outcome.items():
        totals[k] += v


def _probe_output_modality(
    client: Client,
    *,
    provider: str,
    model_id: str,
    cap,
    out_modality: str,
    timeout_ms: int | None,
    probe_job: bool,
) -> dict[str, int]:
    label = f"output:{out_modality}"
    try:
        req = _build_probe_request(
            provider=provider,
            model_id=model_id,
            cap=cap,
            out_modality=out_modality,
            in_modality=None,
            timeout_ms=timeout_ms,
            force_wait=False if cap.supports_job else None,
        )
        resp = client.generate(req)
        _validate_probe_response(resp, expected_out=out_modality)
    except GenAIError as e:
        return _probe_fail(label, e)
    except SystemExit as e:
        return _probe_fail(label, e)
    except Exception as e:
        return _probe_fail(label, e)
    out = _probe_ok(label)
    if not probe_job:
        return out
    if resp.status != "running":
        _accumulate(out, _probe_skip("mode:job", f"response status={resp.status}"))
        return out
    _accumulate(out, _probe_ok("mode:job"))
    return out


def _probe_input_modality(
    client: Client,
    *,
    provider: str,
    model_id: str,
    cap,
    in_modality: str,
    timeout_ms: int | None,
) -> dict[str, int]:
    out_modality = (
        "text"
        if "text" in set(cap.output_modalities)
        else sorted(cap.output_modalities)[0]
    )
    label = f"input:{in_modality}"
    try:
        req = _build_probe_request(
            provider=provider,
            model_id=model_id,
            cap=cap,
            out_modality=out_modality,
            in_modality=in_modality,
            timeout_ms=timeout_ms,
            force_wait=False if cap.supports_job else None,
        )
        resp = client.generate(req)
        _validate_probe_response(resp, expected_out=out_modality)
    except GenAIError as e:
        return _probe_fail(label, e)
    except SystemExit as e:
        return _probe_fail(label, e)
    except Exception as e:
        return _probe_fail(label, e)
    return _probe_ok(label)


def _probe_stream_mode(
    client: Client,
    *,
    provider: str,
    model_id: str,
    cap,
    timeout_ms: int | None,
) -> dict[str, int]:
    label = "mode:stream"
    if "text" not in set(cap.output_modalities):
        return _probe_skip(label, "stream probe requires text output")
    try:
        req = _build_probe_request(
            provider=provider,
            model_id=model_id,
            cap=cap,
            out_modality="text",
            in_modality=None,
            timeout_ms=timeout_ms,
        )
        deltas = 0
        for ev in client.generate_stream(req):
            if ev.type == "output.text.delta":
                delta = ev.data.get("delta")
                if isinstance(delta, str) and delta:
                    deltas += 1
                    break
            if ev.type == "done":
                break
        if deltas == 0:
            raise SystemExit("no output.text.delta received")
    except GenAIError as e:
        return _probe_fail(label, e)
    except SystemExit as e:
        return _probe_fail(label, e)
    except Exception as e:
        return _probe_fail(label, e)
    return _probe_ok(label)


def _build_probe_request(
    *,
    provider: str,
    model_id: str,
    cap,
    out_modality: str,
    in_modality: str | None,
    timeout_ms: int | None,
    force_wait: bool | None = None,
) -> GenerateRequest:
    model = f"{provider}:{model_id}"
    wait = True
    if force_wait is not None:
        wait = force_wait
    elif out_modality == "video":
        wait = False

    output = _output_spec_for_modality(
        provider=provider, model_id=model_id, modality=out_modality
    )
    parts = _probe_input_parts(
        provider=provider,
        model_id=model_id,
        cap=cap,
        out_modality=out_modality,
        in_modality=in_modality,
    )
    req = GenerateRequest(
        model=model,
        input=[Message(role="user", content=parts)],
        output=output,
        wait=wait,
    )
    if timeout_ms is not None:
        req = replace(req, params=replace(req.params, timeout_ms=timeout_ms))
    return req


def _probe_input_parts(
    *,
    provider: str,
    model_id: str,
    cap,
    out_modality: str,
    in_modality: str | None,
) -> list[Part]:
    prompt = _probe_prompt(out_modality=out_modality, in_modality=in_modality)
    if out_modality == "embedding":
        return [Part.from_text(prompt)]

    if in_modality is None:
        if set(cap.input_modalities) == {"audio"}:
            return [
                Part(
                    type="audio",
                    mime_type="audio/wav",
                    source=PartSourceBytes(data=_probe_wav_bytes()),
                )
            ]
        return [Part.from_text(prompt)]

    if in_modality == "text":
        return [Part.from_text(prompt)]
    if in_modality == "image":
        return [Part.from_text(prompt), _probe_image_part()]
    if in_modality == "audio":
        text_meta: dict[str, object] = {}
        if set(cap.input_modalities) == {"audio"} and out_modality == "text":
            text_meta = {"transcription_prompt": True}
        parts = [
            Part(
                type="audio",
                mime_type="audio/wav",
                source=PartSourceBytes(data=_probe_wav_bytes()),
            )
        ]
        if prompt:
            parts.insert(0, Part(type="text", text=prompt, meta=text_meta))
        return parts
    if in_modality == "video":
        return [
            Part.from_text(prompt),
            Part(
                type="video",
                mime_type="video/mp4",
                source=PartSourceUrl(url=_DEFAULT_VIDEO_URL),
            ),
        ]
    raise SystemExit(f"unknown input modality: {in_modality}")


def _probe_prompt(*, out_modality: str, in_modality: str | None) -> str:
    if out_modality == "embedding":
        return "hello"
    if out_modality == "image":
        return "生成一张简单的红色方块图。"
    if out_modality == "audio":
        return "用中文说：你好。"
    if out_modality == "video":
        return "生成一个简单短视频：一只猫在草地上奔跑。"
    if in_modality == "image":
        return "请用一句话描述这张图。"
    if in_modality == "audio":
        return "请转写音频内容。"
    if in_modality == "video":
        return "请用一句话描述视频内容。"
    return "只回复：pong"


def _output_spec_for_modality(
    *, provider: str, model_id: str, modality: str
) -> OutputSpec:
    if modality == "embedding":
        return OutputSpec(modalities=["embedding"], embedding=OutputEmbeddingSpec())
    if modality == "image":
        return OutputSpec(modalities=["image"], image=OutputImageSpec(n=1))
    if modality == "audio":
        voice, language = _probe_audio_voice(provider=provider, model_id=model_id)
        return OutputSpec(
            modalities=["audio"],
            audio=OutputAudioSpec(voice=voice, language=language, format="mp3"),
        )
    if modality == "video":
        duration = _probe_video_duration(provider=provider, model_id=model_id)
        return OutputSpec(
            modalities=["video"],
            video=OutputVideoSpec(duration_sec=duration, aspect_ratio="16:9"),
        )
    if modality == "text":
        return OutputSpec(modalities=["text"], text=OutputTextSpec(format="text"))
    raise SystemExit(f"unknown output modality: {modality}")


def _probe_audio_voice(*, provider: str, model_id: str) -> tuple[str, str | None]:
    mid_l = model_id.lower().strip()
    if (
        provider in {"google", "tuzi-google"}
        or mid_l.startswith(("gemini-", "gemma-"))
        or "native-audio" in mid_l
    ):
        return ("Kore", "en-US")
    if provider == "aliyun":
        return ("Cherry", "zh-CN")
    return ("alloy", None)


def _probe_video_duration(*, provider: str, model_id: str) -> int:
    mid_l = model_id.lower().strip()
    if mid_l.startswith("sora-"):
        return 4
    if mid_l.startswith("veo-"):
        return 5
    if provider.startswith("tuzi"):
        return 10
    return 4


def _probe_image_part() -> Part:
    return Part(
        type="image",
        mime_type="image/png",
        source=PartSourceBytes(data=_probe_png_bytes()),
    )


def _probe_png_bytes() -> bytes:
    import struct
    import zlib

    width = 64
    height = 64

    # RGB red square (unfiltered scanlines).
    row = b"\x00" + (b"\xff\x00\x00" * width)
    raw = row * height
    compressed = zlib.compress(raw, level=6)

    def chunk(typ: bytes, data: bytes) -> bytes:
        crc = zlib.crc32(typ)
        crc = zlib.crc32(data, crc)
        return (
            struct.pack(">I", len(data))
            + typ
            + data
            + struct.pack(">I", crc & 0xFFFFFFFF)
        )

    signature = b"\x89PNG\r\n\x1a\n"
    ihdr = struct.pack(">IIBBBBB", width, height, 8, 2, 0, 0, 0)
    return (
        signature
        + chunk(b"IHDR", ihdr)
        + chunk(b"IDAT", compressed)
        + chunk(b"IEND", b"")
    )


def _probe_wav_bytes() -> bytes:
    import io
    import wave

    buf = io.BytesIO()
    with wave.open(buf, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(16_000)
        wf.writeframes(b"\x00\x00" * 16_000)  # 1s silence
    return buf.getvalue()


def _validate_probe_response(resp, *, expected_out: str) -> None:
    if resp.status == "running":
        if not resp.job or not resp.job.job_id:
            raise SystemExit("running response missing job info")
        return
    if resp.status != "completed":
        raise SystemExit(f"unexpected status: {resp.status}")
    if not resp.output:
        raise SystemExit("missing output")
    parts = [p for m in resp.output for p in m.content]
    if expected_out == "text":
        if not any(p.type == "text" and isinstance(p.text, str) for p in parts):
            raise SystemExit("missing text output part")
        return
    if expected_out == "embedding":
        if not any(
            p.type == "embedding" and isinstance(p.embedding, list) for p in parts
        ):
            raise SystemExit("missing embedding output part")
        return
    if expected_out in {"image", "audio", "video"}:
        if not any(p.type == expected_out and p.source is not None for p in parts):
            raise SystemExit(f"missing {expected_out} output part")
        return
    raise SystemExit(f"unknown expected output modality: {expected_out}")


def _probe_ok(label: str) -> dict[str, int]:
    print(f"[OK]   {label}")
    return {"ok": 1, "fail": 0, "skip": 0}


def _probe_skip(label: str, reason: str) -> dict[str, int]:
    print(f"[SKIP] {label}: {reason}")
    return {"ok": 0, "fail": 0, "skip": 1}


def _probe_fail(label: str, err: BaseException) -> dict[str, int]:
    if isinstance(err, GenAIError):
        msg = f"{err.info.type}: {err.info.message}"
    else:
        msg = str(err) or err.__class__.__name__
    print(f"[FAIL] {label}: {msg}")
    return {"ok": 0, "fail": 1, "skip": 0}


def _print_sdk_supported() -> None:
    models = get_sdk_supported_models()
    by_provider: dict[str, list[dict]] = {}
    for m in models:
        by_provider.setdefault(m["provider"], []).append(m)

    for p in sorted(by_provider.keys()):
        print(f"== {p} ==")
        for m in sorted(by_provider[p], key=lambda x: (x["category"], x["model_id"])):
            inp = ",".join(m["input_modalities"])
            out = ",".join(m["output_modalities"])
            modes = ",".join(m["modes"])
            print(
                f"{m['category']:13} {m['model']:45} modes={modes:18} in={inp:18} out={out:18}"
            )
        print()


def _print_provider_models(provider: str, *, timeout_ms: int | None) -> None:
    from .client import _normalize_provider

    client = Client()
    p = _normalize_provider(provider)
    for model_id in sorted(
        client.list_provider_models(provider, timeout_ms=timeout_ms)
    ):
        print(f"{p}:{model_id}")


def _print_available_models(provider: str, *, timeout_ms: int | None) -> None:
    from .client import _normalize_provider
    from .reference import get_sdk_supported_models_for_provider

    client = Client()
    p = _normalize_provider(provider)
    rows = get_sdk_supported_models_for_provider(p)
    by_model_id = {m["model_id"]: m for m in rows}
    for model_id in client.list_available_models(provider, timeout_ms=timeout_ms):
        m = by_model_id.get(model_id)
        if m is None:
            print(f"{p}:{model_id}")
            continue
        inp = ",".join(m["input_modalities"])
        out = ",".join(m["output_modalities"])
        modes = ",".join(m["modes"])
        print(f"{m['model']:45} modes={modes:18} in={inp:18} out={out:18}")


def _print_all_available_models(*, timeout_ms: int | None) -> None:
    models = get_sdk_supported_models()
    by_model: dict[str, dict] = {m["model"]: m for m in models}
    client = Client()
    for model in client.list_all_available_models(timeout_ms=timeout_ms):
        m = by_model.get(model)
        if m is None:
            print(model)
            continue
        inp = ",".join(m["input_modalities"])
        out = ",".join(m["output_modalities"])
        modes = ",".join(m["modes"])
        print(f"{model:45} modes={modes:18} in={inp:18} out={out:18}")


def _print_unsupported_models(provider: str, *, timeout_ms: int | None) -> None:
    from .client import _normalize_provider

    client = Client()
    p = _normalize_provider(provider)
    for model_id in client.list_unsupported_models(provider, timeout_ms=timeout_ms):
        print(f"{p}:{model_id}")


def _print_stale_models(provider: str, *, timeout_ms: int | None) -> None:
    from .client import _normalize_provider

    client = Client()
    p = _normalize_provider(provider)
    for model_id in client.list_stale_models(provider, timeout_ms=timeout_ms):
        print(f"{p}:{model_id}")


def _print_unsupported(*, timeout_ms: int | None) -> None:
    supported: dict[str, set[str]] = {}
    catalog = get_model_catalog()
    for provider, model_ids in catalog.items():
        supported[provider] = {m for m in model_ids if isinstance(m, str) and m}

    client = Client()
    for provider in sorted(catalog.keys()):
        remote = set(client.list_provider_models(provider, timeout_ms=timeout_ms))
        if not remote:
            continue
        unknown = sorted(remote - supported.get(provider, set()))
        if not unknown:
            continue
        print(f"== {provider} ==")
        for model_id in unknown:
            print(f"{provider}:{model_id}")
        print()


def _print_mappings() -> None:
    items = get_parameter_mappings()
    items = sorted(
        items,
        key=lambda x: (
            x["provider"],
            x["protocol"],
            x["operation"],
            x["from"],
            x["to"],
        ),
    )
    cur = None
    for m in items:
        key = (m["provider"], m["protocol"])
        if key != cur:
            if cur is not None:
                print()
            cur = key
            print(f"== {m['provider']} (protocol={m['protocol']}) ==")
        note = f"  # {m['notes']}" if m.get("notes") else ""
        print(f"{m['operation']:14} {m['from']:55} -> {m['to']}{note}")


def _split_model(model: str) -> tuple[str, str]:
    if ":" not in model:
        raise SystemExit('model must be "{provider}:{model_id}"')
    provider, model_id = model.split(":", 1)
    provider = provider.strip().lower()
    model_id = model_id.strip()
    if not provider or not model_id:
        raise SystemExit('model must be "{provider}:{model_id}"')
    return provider, model_id


def _apply_protocol_override(
    client: Client, *, provider: str, protocol: str | None
) -> None:
    if not protocol:
        return
    if provider != "openai":
        raise SystemExit("--protocol only applies to provider=openai")
    if client._openai is None:
        raise SystemExit("NOUS_GENAI_OPENAI_API_KEY/OPENAI_API_KEY not configured")
    client._openai = replace(client._openai, chat_api=protocol)


def _infer_output_spec(*, provider: str, model_id: str, cap) -> OutputSpec:
    out = set(cap.output_modalities)
    if out == {"embedding"}:
        return OutputSpec(modalities=["embedding"], embedding=OutputEmbeddingSpec())
    if out == {"image"}:
        return OutputSpec(modalities=["image"], image=OutputImageSpec(n=1))
    if out == {"audio"}:
        if provider == "google":
            audio = OutputAudioSpec(voice="Kore", language="en-US")
        elif provider == "aliyun":
            audio = OutputAudioSpec(voice="Cherry", language="zh-CN", format="wav")
        else:
            audio = OutputAudioSpec(voice="alloy", format="mp3")
        return OutputSpec(modalities=["audio"], audio=audio)
    if out == {"video"}:
        duration = 4
        if provider.startswith("tuzi"):
            duration = 10
            if model_id.lower().startswith("sora-"):
                duration = 4
        if provider == "google" and model_id.lower().startswith("veo-"):
            duration = 5
        return OutputSpec(
            modalities=["video"],
            video=OutputVideoSpec(duration_sec=duration, aspect_ratio="16:9"),
        )
    if "text" in out:
        return OutputSpec(modalities=["text"], text=OutputTextSpec(format="text"))
    raise SystemExit(f"cannot infer output for model={provider}:{model_id}")


def _build_input_parts(
    *,
    prompt: str | None,
    image_path: str | None,
    audio_path: str | None,
    video_path: str | None,
    input_modalities: set[str],
    output_modalities: set[str],
    provider: str,
    model_id: str,
) -> list[Part]:
    if output_modalities == {"embedding"}:
        if image_path or audio_path or video_path:
            raise SystemExit("embedding only supports text input")
        if not prompt:
            raise SystemExit("embedding requires --prompt")
        return [Part.from_text(prompt)]

    if output_modalities == {"image"}:
        if audio_path or video_path:
            raise SystemExit(
                "image generation does not take --audio-path/--video-path input"
            )
        if image_path and "image" not in input_modalities:
            raise SystemExit(
                "image generation does not take --image-path input for this model"
            )
        if not prompt:
            raise SystemExit("image generation requires --prompt")

    if output_modalities == {"audio"}:
        if image_path or audio_path or video_path:
            raise SystemExit(
                "TTS does not take --image-path/--audio-path/--video-path input"
            )
        if not prompt:
            raise SystemExit("audio generation requires --prompt")
        return [Part.from_text(prompt)]

    if output_modalities == {"video"}:
        if image_path or audio_path or video_path:
            raise SystemExit(
                "video generation does not take --image-path/--audio-path/--video-path input"
            )
        if not prompt:
            raise SystemExit("video generation requires --prompt")
        return [Part.from_text(prompt)]

    parts: list[Part] = []
    if prompt:
        meta = {}
        if provider == "openai" and (
            model_id == "whisper-1" or "-transcribe" in model_id
        ):
            meta = {"transcription_prompt": True}
        parts.append(Part(type="text", text=prompt, meta=meta))

    if image_path:
        mime = detect_mime_type(image_path)
        if not mime or not mime.startswith("image/"):
            raise SystemExit(f"cannot detect image mime type for: {image_path}")
        parts.append(
            Part(type="image", mime_type=mime, source=PartSourcePath(path=image_path))
        )
    if audio_path:
        mime = detect_mime_type(audio_path)
        if not mime or not mime.startswith("audio/"):
            raise SystemExit(f"cannot detect audio mime type for: {audio_path}")
        parts.append(
            Part(type="audio", mime_type=mime, source=PartSourcePath(path=audio_path))
        )
    if video_path:
        mime = detect_mime_type(video_path)
        if not mime or not mime.startswith("video/"):
            raise SystemExit(f"cannot detect video mime type for: {video_path}")
        parts.append(
            Part(type="video", mime_type=mime, source=PartSourcePath(path=video_path))
        )

    if not parts:
        raise SystemExit(
            "missing input: provide --prompt and/or --image-path/--audio-path/--video-path"
        )
    return parts


def _run_stream_text(
    client: Client, req: GenerateRequest, *, timeout_ms: int | None
) -> str:
    if timeout_ms is not None:
        req = replace(req, params=replace(req.params, timeout_ms=timeout_ms))
    chunks: list[str] = []
    for ev in client.generate_stream(req):
        if ev.type != "output.text.delta":
            continue
        delta = ev.data.get("delta")
        if isinstance(delta, str) and delta:
            chunks.append(delta)
    return "".join(chunks)


def _write_text(text: str, *, output_path: str | None) -> None:
    if output_path:
        with open(output_path, "w", encoding="utf-8") as f:
            f.write(text)
        print(f"[OK] wrote {output_path}")
        return
    print(text)


def _guess_ext(mime: str | None) -> str:
    if not mime:
        return ""
    m = mime.lower()
    if m == "image/png":
        return ".png"
    if m in {"image/jpeg", "image/jpg"}:
        return ".jpg"
    if m == "image/webp":
        return ".webp"
    if m in {"audio/mpeg", "audio/mp3"}:
        return ".mp3"
    if m in {"audio/wav", "audio/wave"}:
        return ".wav"
    if m in {"audio/mp4", "audio/m4a"}:
        return ".m4a"
    if m == "video/mp4":
        return ".mp4"
    if m == "video/quicktime":
        return ".mov"
    return ""


def _download_with_headers(
    url: str,
    output_path: str,
    *,
    timeout_ms: int | None,
    headers: dict[str, str] | None,
) -> None:
    download_to_file(
        url=url,
        output_path=output_path,
        timeout_ms=timeout_ms,
        max_bytes=None,
        headers=headers,
    )


def _download_auth(
    client: object, *, provider: str
) -> tuple[dict[str, str], set[str]] | None:
    adapter_getter = getattr(client, "_adapter", None)
    if not callable(adapter_getter):
        return None
    try:
        adapter = adapter_getter(provider)
    except Exception:
        return None
    header_fn = getattr(adapter, "_download_headers", None)
    if not callable(header_fn):
        return None
    try:
        raw = header_fn()
    except Exception:
        return None
    if not isinstance(raw, dict):
        return None
    headers: dict[str, str] = {}
    for k, v in raw.items():
        if isinstance(k, str) and k and isinstance(v, str) and v:
            headers[k] = v
    if not headers:
        return None

    base_url = getattr(adapter, "base_url", None)
    if not isinstance(base_url, str) or not base_url:
        return None
    host = urllib.parse.urlparse(base_url).hostname
    if not isinstance(host, str) or not host:
        return None
    return headers, {host.lower()}


def _write_response(
    parts: list[Part],
    *,
    output: OutputSpec,
    output_path: str | None,
    timeout_ms: int | None,
    download_auth: tuple[dict[str, str], set[str]] | None,
) -> None:
    modalities = set(output.modalities)
    if modalities == {"text"}:
        text = parts[0].text or ""
        _write_text(text, output_path=output_path)
        return

    if modalities == {"embedding"}:
        vec = parts[0].embedding or []
        if output_path:
            with open(output_path, "w", encoding="utf-8") as f:
                json.dump(vec, f, ensure_ascii=False)
            print(f"[OK] wrote {output_path} (dims={len(vec)})")
            return
        head = ", ".join([f"{x:.6f}" for x in vec[:8]])
        print(f"dims={len(vec)} [{head}{', ...' if len(vec) > 8 else ''}]")
        return

    if modalities in ({"image"}, {"audio"}, {"video"}):
        p = next((x for x in parts if x.type in modalities), None)
        if not p:
            raise SystemExit("missing binary output part")
        if not p.source:
            raise SystemExit("missing output source")
        if p.source.kind == "url":
            if output_path:
                headers: dict[str, str] | None = None
                if download_auth is not None:
                    allowed = download_auth[1]
                    host = urllib.parse.urlparse(p.source.url).hostname
                    if isinstance(host, str) and host and host.lower() in allowed:
                        headers = download_auth[0]
                _download_with_headers(
                    p.source.url,
                    output_path,
                    timeout_ms=timeout_ms,
                    headers=headers,
                )
                print(f"[OK] downloaded to {output_path}")
            else:
                print(p.source.url)
            return
        if p.source.kind == "ref":
            if output_path:
                raise SystemExit(
                    "cannot write ref output; provider-specific download required"
                )
            ref = (
                f"{p.source.provider}:{p.source.id}"
                if p.source.provider
                else p.source.id
            )
            print(ref)
            return
        if p.source.kind != "bytes":
            raise SystemExit(f"unsupported output source kind: {p.source.kind}")
        data: bytes
        if p.source.encoding == "base64":
            raw_b64 = p.source.data
            if not isinstance(raw_b64, str) or not raw_b64:
                raise SystemExit("invalid base64 output") from None
            try:
                data = base64.b64decode(raw_b64)
            except Exception:
                raise SystemExit("invalid base64 output") from None
        else:
            raw = p.source.data
            if isinstance(raw, bytearray):
                raw = bytes(raw)
            if not isinstance(raw, bytes):
                raise SystemExit(f"invalid bytes output (encoding={p.source.encoding})")
            data = raw
        path = output_path or f"genai_output{_guess_ext(p.mime_type)}"
        with open(path, "wb") as f:
            f.write(data)
        print(f"[OK] wrote {path} ({p.mime_type}, {len(data)} bytes)")
        return

    raise SystemExit(f"unsupported output modalities: {output.modalities}")


def _generate_token() -> str:
    return f"sk-{secrets.token_urlsafe(32)}"


if __name__ == "__main__":
    main()
