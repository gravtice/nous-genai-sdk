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

from nous.genai import Client  # noqa: E402
from nous.genai.cli import main as genai_main  # noqa: E402
from nous.genai.reference import get_sdk_supported_models_for_provider  # noqa: E402

_PROMPT = "Only reply: OK"


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
        raise SystemExit(
            "TUZI_OPENAI_API_KEY not configured (NOUS_GENAI_TUZI_OPENAI_API_KEY/TUZI_OPENAI_API_KEY)"
        )

    rows = get_sdk_supported_models_for_provider("tuzi-openai")
    chat_ids = [r["model_id"] for r in rows if r.get("category") == "chat"]
    if not chat_ids:
        raise SystemExit("tuzi-openai chat catalog empty")

    remote = set(client.list_provider_models("tuzi-openai", timeout_ms=15_000))

    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    out_dir = Path("build")
    out_dir.mkdir(parents=True, exist_ok=True)
    base = f"tuzi-openai-chat-cli-smoke-{stamp}"
    report_path = out_dir / f"{base}.jsonl"
    failures_path = out_dir / f"{base}.failures.tsv"
    summary_path = out_dir / f"{base}.summary.json"

    pass_n = 0
    fail_n = 0
    failures: list[dict[str, str]] = []
    t0 = time.time()

    with report_path.open("w", encoding="utf-8") as out:
        for model_id in chat_ids:
            model = f"tuzi-openai:{model_id}"
            row: dict[str, object] = {"model": model, "listed": model_id in remote}

            t1 = time.time()
            stdout = io.StringIO()
            stderr = io.StringIO()
            try:
                with redirect_stdout(stdout), redirect_stderr(stderr):
                    genai_main(["--model", model, "--prompt", _PROMPT])
                row["seconds"] = round(time.time() - t1, 3)
                text = stdout.getvalue().strip()
                err = stderr.getvalue().strip()
                if err:
                    row["stderr"] = err
                row["result"] = {"ok": True, "text_preview": text[:200]}
                pass_n += 1
            except SystemExit as e:
                row["seconds"] = round(time.time() - t1, 3)
                msg = _system_exit_message(e)
                err = stderr.getvalue().strip()
                if err:
                    row["stderr"] = err
                row["error"] = {"type": "SystemExit", "message": msg}
                row["result"] = {"ok": False}
                fail_n += 1
                failures.append(
                    {
                        "model": model,
                        "listed": "1" if model_id in remote else "0",
                        "message": msg or err or "SystemExit",
                    }
                )
            except Exception as e:
                row["seconds"] = round(time.time() - t1, 3)
                err = stderr.getvalue().strip()
                if err:
                    row["stderr"] = err
                row["error"] = {"type": type(e).__name__, "message": str(e)}
                row["result"] = {"ok": False}
                fail_n += 1
                failures.append(
                    {
                        "model": model,
                        "listed": "1" if model_id in remote else "0",
                        "message": f"{type(e).__name__}: {e}",
                    }
                )

            out.write(json.dumps(row, ensure_ascii=False) + "\n")
            out.flush()

    summary = {
        "provider": "tuzi-openai",
        "category": "chat",
        "prompt": _PROMPT,
        "total": len(chat_ids),
        "pass": pass_n,
        "fail": fail_n,
        "seconds": round(time.time() - t0, 3),
        "report": str(report_path),
        "failures": str(failures_path),
        "summary": str(summary_path),
    }
    summary_path.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )

    with failures_path.open("w", encoding="utf-8") as f:
        f.write("model\tlisted\tmessage\n")
        for item in failures:
            f.write(
                "\t".join(
                    [
                        _sanitize_tsv(item["model"]),
                        _sanitize_tsv(item["listed"]),
                        _sanitize_tsv(item["message"]),
                    ]
                )
                + "\n"
            )

    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
