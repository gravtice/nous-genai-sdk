from __future__ import annotations

import argparse
import json
import sys
import time
from collections.abc import Iterator
from datetime import datetime
from pathlib import Path
from typing import Any

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from nous.genai import (
    Client,
    GenAIError,
    GenerateParams,
    GenerateRequest,
    Message,
    OutputAudioSpec,
    OutputSpec,
    OutputTextSpec,
    OutputVideoSpec,
    Part,
)
from nous.genai.reference import get_sdk_supported_models_for_provider, get_supported_providers


def _is_configured(client: Client, provider: str) -> bool:
    try:
        client._adapter(provider)  # type: ignore[attr-defined]
    except GenAIError:
        return False
    return True


def _unique_nonempty(values: list[str]) -> list[str]:
    seen: set[str] = set()
    out: list[str] = []
    for v in values:
        s = v.strip()
        if not s or s in seen:
            continue
        seen.add(s)
        out.append(s)
    return out


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="modes_probe",
        description="Probe catalog model modes (stream/job) for configured providers.",
    )
    parser.add_argument(
        "providers",
        nargs="*",
        help="Optional providers to probe (default: all configured supported providers).",
    )
    return parser.parse_args(argv)


def _probe_stream(client: Client, model: str, *, timeout_ms: int) -> tuple[bool, str | None]:
    req = GenerateRequest(
        model=model,
        input=[Message(role="user", content=[Part.from_text("Only reply: OK")])],
        output=OutputSpec(modalities=["text"], text=OutputTextSpec(format="text")),
        params=GenerateParams(timeout_ms=timeout_ms),
        wait=True,
    )
    try:
        out = client.generate(req, stream=True)
        if not isinstance(out, Iterator):
            return False, "provider returned non-stream response"
        for ev in out:
            if ev.type == "done":
                return True, None
        return True, None
    except Exception as e:
        return False, f"{type(e).__name__}: {e}"


def _probe_job_text(client: Client, model: str, *, timeout_ms: int) -> tuple[bool, str | None]:
    req = GenerateRequest(
        model=model,
        input=[Message(role="user", content=[Part.from_text("Only reply: OK")])],
        output=OutputSpec(modalities=["text"], text=OutputTextSpec(format="text")),
        params=GenerateParams(timeout_ms=timeout_ms),
        wait=False,
    )
    try:
        out = client.generate(req, stream=False)
        if isinstance(out, Iterator):
            return False, "provider returned stream response"
        if out.status != "running":
            return False, f"unexpected status: {out.status}"
        if not out.job or not out.job.job_id:
            return False, "missing job_id"
        return True, None
    except Exception as e:
        return False, f"{type(e).__name__}: {e}"


def _probe_job_audio(client: Client, model: str, *, timeout_ms: int) -> tuple[bool, str | None]:
    req = GenerateRequest(
        model=model,
        input=[Message(role="user", content=[Part.from_text("Say OK.")])],
        output=OutputSpec(modalities=["audio"], audio=OutputAudioSpec(voice="alloy", format="mp3")),
        params=GenerateParams(timeout_ms=timeout_ms),
        wait=False,
    )
    try:
        out = client.generate(req, stream=False)
        if isinstance(out, Iterator):
            return False, "provider returned stream response"
        if out.status != "running":
            return False, f"unexpected status: {out.status}"
        if not out.job or not out.job.job_id:
            return False, "missing job_id"
        return True, None
    except Exception as e:
        return False, f"{type(e).__name__}: {e}"


def _probe_job_video(
    client: Client,
    model: str,
    *,
    timeout_ms: int,
    video_duration_sec: int,
) -> tuple[bool, str | None]:
    provider, _model_id = model.split(":", 1)
    if provider == "google":
        duration_sec = 5
    elif provider.startswith("tuzi-"):
        duration_sec = 5
    else:
        duration_sec = video_duration_sec

    req = GenerateRequest(
        model=model,
        input=[Message(role="user", content=[Part.from_text("A short video of a red square.")])],
        output=OutputSpec(
            modalities=["video"],
            video=OutputVideoSpec(duration_sec=duration_sec, aspect_ratio="16:9"),
        ),
        params=GenerateParams(timeout_ms=timeout_ms),
        wait=False,
    )
    try:
        out = client.generate(req, stream=False)
        if isinstance(out, Iterator):
            return False, "provider returned stream response"
        if out.status != "running":
            return False, f"unexpected status: {out.status}"
        if not out.job or not out.job.job_id:
            return False, "missing job_id"
        return True, None
    except Exception as e:
        return False, f"{type(e).__name__}: {e}"


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    client = Client()

    supported = [p.strip().lower() for p in get_supported_providers()]
    supported_set = set(supported)

    requested = _unique_nonempty([p.strip().lower() for p in args.providers])
    if requested:
        unknown = sorted(set(requested) - supported_set)
        if unknown:
            raise SystemExit(f"unknown providers: {', '.join(unknown)}")
        not_configured = [p for p in requested if not _is_configured(client, p)]
        if not_configured:
            raise SystemExit(f"providers not configured: {', '.join(not_configured)}")
        providers = requested
    else:
        providers = [p for p in supported if _is_configured(client, p)]
    if not providers:
        raise SystemExit("no configured providers found (check .env.local and NOUS_GENAI_*_API_KEY)")

    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    out_dir = Path("build")
    out_dir.mkdir(parents=True, exist_ok=True)
    report_path = out_dir / f"modes-probe-{stamp}.jsonl"
    summary_path = out_dir / f"modes-probe-{stamp}.summary.json"

    rows_n = 0
    mismatches: list[dict[str, Any]] = []
    t0 = time.time()

    with report_path.open("w", encoding="utf-8") as f:
        for provider in sorted(providers):
            sdk_rows = get_sdk_supported_models_for_provider(provider)
            by_model_id = {r["model_id"]: r for r in sdk_rows}
            available = client.list_available_models(provider, timeout_ms=15_000)

            for model_id in available:
                row = by_model_id.get(model_id)
                if row is None:
                    continue
                model = row["model"]
                expected_modes: list[str] = list(row["modes"])

                expected_stream = "stream" in expected_modes
                expected_job = "job" in expected_modes

                stream_ok = None
                stream_error = None
                if "text" in row["output_modalities"]:
                    stream_ok, stream_error = _probe_stream(client, model, timeout_ms=20_000)

                job_ok = None
                job_error = None
                if expected_job:
                    out_mods = set(row["output_modalities"])
                    if out_mods == {"video"}:
                        job_ok, job_error = _probe_job_video(client, model, timeout_ms=60_000, video_duration_sec=4)
                    elif out_mods == {"audio"}:
                        job_ok, job_error = _probe_job_audio(client, model, timeout_ms=60_000)
                    elif out_mods == {"text"}:
                        job_ok, job_error = _probe_job_text(client, model, timeout_ms=60_000)

                suggested_override: dict[str, bool] = {}
                if stream_ok is True and not expected_stream:
                    suggested_override["supports_stream"] = True
                if stream_ok is False and expected_stream:
                    suggested_override["supports_stream"] = False
                if job_ok is False and expected_job:
                    suggested_override["supports_job"] = False

                if suggested_override:
                    mismatches.append(
                        {
                            "model": model,
                            "expected_modes": expected_modes,
                            "stream_ok": stream_ok,
                            "job_ok": job_ok,
                            "suggested_override": suggested_override,
                        }
                    )

                f.write(
                    json.dumps(
                        {
                            "provider": provider,
                            "model": model,
                            "model_id": model_id,
                            "category": row["category"],
                            "expected_modes": expected_modes,
                            "stream_ok": stream_ok,
                            "stream_error": stream_error,
                            "job_ok": job_ok,
                            "job_error": job_error,
                            "suggested_override": suggested_override or None,
                        },
                        ensure_ascii=False,
                    )
                    + "\n"
                )
                rows_n += 1

    summary = {
        "providers": sorted(providers),
        "rows": rows_n,
        "mismatches": mismatches,
        "seconds": round(time.time() - t0, 3),
        "report_path": str(report_path),
    }
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    print(f"Wrote: {report_path}")
    print(f"Wrote: {summary_path}")
    if mismatches:
        print(f"Mismatches: {len(mismatches)} (see summary)")
        return 2
    print("OK: no mismatches")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
