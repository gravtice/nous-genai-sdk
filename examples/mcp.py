from __future__ import annotations

import sys
from pathlib import Path


def main(argv: list[str] | None = None) -> int:
    # Avoid shadowing the real `mcp` package with this file name (`examples/mcp.py`).
    script_dir = Path(__file__).resolve().parent
    sys.path = [p for p in sys.path if Path(p or ".").resolve() != script_dir]

    from nous.genai.mcp_cli import main as cli_main

    return cli_main(argv)


if __name__ == "__main__":
    raise SystemExit(main())

