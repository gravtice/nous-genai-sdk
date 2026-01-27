from __future__ import annotations

import io
import json
import sys
import time
from contextlib import redirect_stderr, redirect_stdout
from datetime import datetime
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from nous.genai import Client
from nous.genai.cli import main as genai_main
from nous.genai.reference import get_sdk_supported_models_for_provider

_PROMPT = "A short video of a red square on a white background."
_TIMEOUT_MS = 600_000


def _sanitize_tsv(text: str) -> str:
    return " ".join(str(text).replace("\t", " ").splitlines()).strip()


def _system_exit_message(e: SystemExit) -> str:
    code = getattr(e, "code", None)
    if isinstance(code, str):
        return code.strip()
    return ""


def main() -> int:
    client = Client()
    if getattr(client, "_tuzi_openai", None) is None:
        raise SystemExit("TUZI_OPENAI_API_KEY not configured (NOUS_GENAI_TUZI_OPENAI_API_KEY/TUZI_OPENAI_API_KEY)")

    rows = get_sdk_supported_models_for_provider("tuzi-openai")
    video_ids = [r["model_id"] for r in rows if r.get("category") == "video"]
    if not video_ids:
        raise SystemExit("tuzi-openai video catalog empty")

    remote = set(client.list_provider_models("tuzi-openai", timeout_ms=15_000))

    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    out_dir = Path("build")
    out_dir.mkdir(parents=True, exist_ok=True)
    base = f"tuzi-openai-video-cli-smoke-{stamp}"
    report_path = out_dir / f"{base}.jsonl"
    failures_path = out_dir / f"{base}.failures.tsv"
    summary_path = out_dir / f"{base}.summary.json"

    pass_n = 0
    fail_n = 0
    failures: list[dict[str, str]] = []
    t0 = time.time()

    with report_path.open("w", encoding="utf-8") as out:
        for model_id in video_ids:
            model = f"tuzi-openai:{model_id}"
            output_path = out_dir / f"{base}.{model_id}.mp4"
            row: dict[str, object] = {
                "model": model,
                "listed": model_id in remote,
                "prompt": _PROMPT,
                "timeout_ms": _TIMEOUT_MS,
                "output_path": str(output_path),
            }

            t1 = time.time()
            stdout = io.StringIO()
            stderr = io.StringIO()
            try:
                with redirect_stdout(stdout), redirect_stderr(stderr):
                    genai_main(
                        [
                            "--model",
                            model,
                            "--prompt",
                            _PROMPT,
                            "--timeout-ms",
                            str(_TIMEOUT_MS),
                            "--output-path",
                            str(output_path),
                        ]
                    )
                row["seconds"] = round(time.time() - t1, 3)
                out_text = stdout.getvalue().strip()
                err = stderr.getvalue().strip()
                if out_text:
                    row["stdout"] = out_text
                if err:
                    row["stderr"] = err

                if not output_path.is_file():
                    raise RuntimeError("missing output file")
                size = output_path.stat().st_size
                if size < 1:
                    raise RuntimeError("empty output file")

                row["result"] = {"ok": True, "bytes": size}
                pass_n += 1
            except SystemExit as e:
                row["seconds"] = round(time.time() - t1, 3)
                msg = _system_exit_message(e)
                out_text = stdout.getvalue().strip()
                err = stderr.getvalue().strip()
                if out_text:
                    row["stdout"] = out_text
                if err:
                    row["stderr"] = err
                row["error"] = {"type": "SystemExit", "message": msg}
                row["result"] = {"ok": False}
                fail_n += 1
                failures.append(
                    {
                        "model": model,
                        "listed": "1" if model_id in remote else "0",
                        "output_path": str(output_path),
                        "message": msg or err or "SystemExit",
                    }
                )
            except Exception as e:
                row["seconds"] = round(time.time() - t1, 3)
                out_text = stdout.getvalue().strip()
                err = stderr.getvalue().strip()
                if out_text:
                    row["stdout"] = out_text
                if err:
                    row["stderr"] = err
                row["error"] = {"type": type(e).__name__, "message": str(e)}
                row["result"] = {"ok": False}
                fail_n += 1
                failures.append(
                    {
                        "model": model,
                        "listed": "1" if model_id in remote else "0",
                        "output_path": str(output_path),
                        "message": f"{type(e).__name__}: {e}",
                    }
                )

            out.write(json.dumps(row, ensure_ascii=False) + "\n")
            out.flush()

    summary = {
        "provider": "tuzi-openai",
        "category": "video",
        "prompt": _PROMPT,
        "timeout_ms": _TIMEOUT_MS,
        "total": len(video_ids),
        "pass": pass_n,
        "fail": fail_n,
        "seconds": round(time.time() - t0, 3),
        "report": str(report_path),
        "failures": str(failures_path),
        "summary": str(summary_path),
    }
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    with failures_path.open("w", encoding="utf-8") as f:
        f.write("model\tlisted\toutput_path\tmessage\n")
        for item in failures:
            f.write(
                "\t".join(
                    [
                        _sanitize_tsv(item["model"]),
                        _sanitize_tsv(item["listed"]),
                        _sanitize_tsv(item["output_path"]),
                        _sanitize_tsv(item["message"]),
                    ]
                )
                + "\n"
            )

    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
