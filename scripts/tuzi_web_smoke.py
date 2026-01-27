from __future__ import annotations

import json
import sys
import time
from dataclasses import asdict
from datetime import datetime
from pathlib import Path
from typing import Any

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from nous.genai import (  # noqa: E402
    Client,
    GenAIError,
    GenerateParams,
    GenerateRequest,
    Message,
    OutputAudioSpec,
    OutputEmbeddingSpec,
    OutputImageSpec,
    OutputSpec,
    OutputTextSpec,
    OutputVideoSpec,
    Part,
    PartSourcePath,
    PartSourceUrl,
)

_SAMPLE_IMAGE_URL = "https://wiki.tu-zi.com/code/cat_1.png"


def _capabilities_dict(cap) -> dict[str, Any]:
    return {
        "input_modalities": sorted(cap.input_modalities),
        "output_modalities": sorted(cap.output_modalities),
        "supports_stream": bool(cap.supports_stream),
        "supports_job": bool(cap.supports_job),
    }


def _group_for_capabilities(cap) -> str:
    inp = set(cap.input_modalities)
    out = set(cap.output_modalities)
    if out == {"embedding"}:
        return "embedding"
    if inp == {"audio"} and out == {"text"}:
        return "transcription"
    if out == {"image"}:
        return "image"
    if out == {"audio"}:
        return "audio"
    if out == {"video"}:
        return "video"
    if "text" in out:
        return "text"
    return "unknown"


def _sanitize_tsv(text: str) -> str:
    return " ".join(str(text).replace("\t", " ").splitlines()).strip()


def _iter_parts(resp):
    for msg in resp.output:
        for part in msg.content:
            yield part


def _extract_text_preview(resp, *, max_chars: int) -> str | None:
    chunks: list[str] = []
    for part in _iter_parts(resp):
        if part.type == "text" and isinstance(part.text, str):
            chunks.append(part.text)
    if not chunks:
        return None
    text = "".join(chunks)
    if len(text) <= max_chars:
        return text
    return text[:max_chars]


def _extract_embedding_dims(resp) -> int | None:
    for part in _iter_parts(resp):
        if part.type == "embedding" and isinstance(part.embedding, list):
            return len(part.embedding)
    return None


def _extract_first_binary_part(resp, kind: str) -> dict[str, Any] | None:
    for part in _iter_parts(resp):
        if part.type != kind or part.source is None:
            continue
        src = part.source
        if src.kind == "url":
            return {"kind": "url", "mime_type": part.mime_type, "url": src.url}
        if src.kind == "bytes":
            if getattr(src, "encoding", None) == "base64" and isinstance(src.data, str):
                return {
                    "kind": "bytes",
                    "encoding": "base64",
                    "mime_type": part.mime_type,
                    "base64_chars": len(src.data),
                }
            if isinstance(src.data, (bytes, bytearray)):
                return {
                    "kind": "bytes",
                    "mime_type": part.mime_type,
                    "bytes": len(src.data),
                }
            return {
                "kind": "bytes",
                "mime_type": part.mime_type,
                "data_type": type(src.data).__name__,
            }
        if src.kind == "ref":
            ref = f"{src.provider}:{src.id}" if src.provider else src.id
            return {"kind": "ref", "mime_type": part.mime_type, "ref": ref}
        if src.kind == "path":
            return {"kind": "path", "mime_type": part.mime_type, "path": src.path}
        return {"kind": str(getattr(src, "kind", "")), "mime_type": part.mime_type}
    return None


def _ensure_sample_wav(path: Path) -> str:
    if path.is_file():
        return str(path)

    import math
    import struct
    import wave

    path.parent.mkdir(parents=True, exist_ok=True)
    sample_rate = 16_000
    duration_sec = 0.5
    freq_hz = 440.0
    amp = 0.2
    frames = int(sample_rate * duration_sec)

    with wave.open(str(path), "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        for i in range(frames):
            val = int(
                amp * 32767.0 * math.sin(2.0 * math.pi * freq_hz * i / sample_rate)
            )
            wf.writeframes(struct.pack("<h", val))

    return str(path)


def _ensure_sample_png(path: Path) -> str:
    if path.is_file():
        return str(path)

    import base64

    path.parent.mkdir(parents=True, exist_ok=True)
    data = base64.b64decode(
        "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mP8/wIAAgMBAp9G5xkAAAAASUVORK5CYII="
    )
    path.write_bytes(data)
    return str(path)


def _request_for_group(
    group: str, *, sample_wav_path: str
) -> tuple[GenerateRequest, dict[str, Any], int]:
    if group == "image":
        timeout_ms = 45_000
        output = OutputSpec(modalities=["image"], image=OutputImageSpec(n=1))
        prompt = "A red square on a white background."
        wait = True
    elif group == "audio":
        timeout_ms = 30_000
        output = OutputSpec(
            modalities=["audio"], audio=OutputAudioSpec(voice="alloy", format="mp3")
        )
        prompt = "Say OK."
        wait = True
    elif group == "video":
        timeout_ms = 150_000
        output = OutputSpec(
            modalities=["video"],
            video=OutputVideoSpec(duration_sec=10, aspect_ratio="16:9"),
        )
        prompt = "A 10 second video of a red square on a white background."
        wait = False
    elif group == "embedding":
        timeout_ms = 12_000
        output = OutputSpec(modalities=["embedding"], embedding=OutputEmbeddingSpec())
        prompt = "hello"
        wait = True
    elif group == "transcription":
        timeout_ms = 30_000
        output = OutputSpec(modalities=["text"], text=OutputTextSpec(format="text"))
        prompt = "Transcribe this audio."
        wait = True
    else:
        timeout_ms = 12_000
        output = OutputSpec(modalities=["text"], text=OutputTextSpec(format="text"))
        prompt = "Only reply: OK"
        wait = True

    parts = [Part.from_text(prompt)]
    if group == "transcription":
        parts.append(
            Part(
                type="audio",
                mime_type="audio/wav",
                source=PartSourcePath(path=sample_wav_path),
            )
        )

    req = GenerateRequest(
        model="",
        input=[Message(role="user", content=parts)],
        output=output,
        wait=wait,
    )
    meta = {
        "output_modalities": list(output.modalities),
        "wait": wait,
        "timeout_ms": timeout_ms,
    }
    return req, meta, timeout_ms


def main() -> int:
    client = Client()
    if getattr(client, "_tuzi_web", None) is None:
        raise SystemExit(
            "TUZI_WEB_API_KEY not configured (NOUS_GENAI_TUZI_WEB_API_KEY/TUZI_WEB_API_KEY)"
        )

    from nous.genai.reference import get_model_catalog

    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    out_dir = Path("build")
    out_dir.mkdir(parents=True, exist_ok=True)
    base = f"tuzi-web-smoke-{stamp}"
    report_path = out_dir / f"{base}.jsonl"
    failures_path = out_dir / f"{base}.failures.tsv"
    summary_path = out_dir / f"{base}.summary.json"

    models = [
        m
        for m in get_model_catalog().get("tuzi-web", [])
        if isinstance(m, str) and m.strip()
    ]
    seen: set[str] = set()
    models = [m for m in models if m not in seen and not seen.add(m)]

    t0 = time.time()
    pass_n = 0
    fail_n = 0
    group_counts: dict[str, int] = {
        "text": 0,
        "image": 0,
        "audio": 0,
        "video": 0,
        "unknown": 0,
    }
    failures: list[dict[str, str]] = []
    sample_wav_path = _ensure_sample_wav(out_dir / "smoke-sample.wav")
    sample_png_path = _ensure_sample_png(out_dir / "smoke-sample.png")

    with report_path.open("w", encoding="utf-8") as out:
        for model_id in models:
            model = f"tuzi-web:{model_id}"
            row: dict[str, Any] = {"model": model}

            t1 = time.time()
            try:
                cap = client.capabilities(model)
                cap_dict = _capabilities_dict(cap)
                group = _group_for_capabilities(cap)
                group_counts[group] = group_counts.get(group, 0) + 1

                base_req, req_meta, timeout_ms = _request_for_group(
                    group, sample_wav_path=sample_wav_path
                )
                wait = base_req.wait
                if cap.supports_job:
                    wait = False
                req_meta["wait"] = wait

                input_msgs = base_req.input
                if group == "video" and "image" in set(cap.input_modalities):
                    if model_id.lower().startswith(("pika-", "runway-")):
                        src = PartSourceUrl(url=_SAMPLE_IMAGE_URL)
                    else:
                        src = PartSourcePath(path=sample_png_path)
                    img = Part(type="image", mime_type="image/png", source=src)
                    msg0 = base_req.input[0]
                    input_msgs = [Message(role=msg0.role, content=[*msg0.content, img])]

                req = GenerateRequest(
                    model=model,
                    input=input_msgs,
                    output=base_req.output,
                    params=GenerateParams(timeout_ms=timeout_ms),
                    wait=wait,
                )

                row.update(
                    {"capabilities": cap_dict, "group": group, "request": req_meta}
                )
                resp = client.generate(req)
                row["seconds"] = round(time.time() - t1, 3)
                row["response"] = {
                    "status": resp.status,
                    "provider": resp.provider,
                    "model": resp.model,
                }

                if resp.status not in {"completed", "running"}:
                    err = getattr(resp, "error", None)
                    if err is not None:
                        row["error"] = asdict(err)
                    row["result"] = {"ok": False}
                    fail_n += 1
                    failures.append(
                        {
                            "model": model,
                            "group": group,
                            "error_type": getattr(err, "type", "UnknownError"),
                            "provider_code": getattr(err, "provider_code", "") or "",
                            "message": getattr(err, "message", "") or "",
                        }
                    )
                else:
                    result: dict[str, Any] = {"ok": True}
                    if getattr(resp, "job", None) is not None and getattr(
                        resp.job, "job_id", None
                    ):
                        result["job"] = resp.job.job_id
                    if resp.status == "completed":
                        if group in {"text", "transcription"}:
                            preview = _extract_text_preview(resp, max_chars=200)
                            if preview is not None:
                                result["text_preview"] = preview
                        elif group == "embedding":
                            dims = _extract_embedding_dims(resp)
                            if dims is not None:
                                result["dims"] = dims
                        elif group in {"image", "audio", "video"}:
                            info = _extract_first_binary_part(resp, group)
                            if info is not None:
                                result.update(info)
                    row["result"] = result
                    pass_n += 1
            except GenAIError as e:
                row["seconds"] = round(time.time() - t1, 3)
                row["error"] = asdict(e.info)
                row["result"] = {"ok": False}
                fail_n += 1
                failures.append(
                    {
                        "model": model,
                        "group": row.get("group", "unknown"),
                        "error_type": e.info.type,
                        "provider_code": e.info.provider_code or "",
                        "message": e.info.message,
                    }
                )
            except Exception as e:
                row["seconds"] = round(time.time() - t1, 3)
                row["error"] = {
                    "type": type(e).__name__,
                    "message": str(e),
                    "provider_code": None,
                    "retryable": False,
                }
                row["result"] = {"ok": False}
                fail_n += 1
                failures.append(
                    {
                        "model": model,
                        "group": row.get("group", "unknown"),
                        "error_type": type(e).__name__,
                        "provider_code": "",
                        "message": str(e),
                    }
                )

            out.write(json.dumps(row, ensure_ascii=False) + "\n")
            out.flush()

    summary = {
        "provider": "tuzi-web",
        "total": len(models),
        "pass": pass_n,
        "fail": fail_n,
        "seconds": round(time.time() - t0, 3),
        "report": str(report_path),
        "group_counts": group_counts,
    }
    summary_path.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )

    with failures_path.open("w", encoding="utf-8") as f:
        f.write("model\tgroup\terror_type\tprovider_code\tmessage\n")
        for item in failures:
            f.write(
                "\t".join(
                    [
                        _sanitize_tsv(item["model"]),
                        _sanitize_tsv(item["group"]),
                        _sanitize_tsv(item["error_type"]),
                        _sanitize_tsv(item["provider_code"]),
                        _sanitize_tsv(item["message"]),
                    ]
                )
                + "\n"
            )

    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
