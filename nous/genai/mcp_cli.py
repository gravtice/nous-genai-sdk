from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
from pathlib import Path
from typing import Any

from ._internal.config import load_env_files


def _env(name: str) -> str | None:
    return os.environ.get(name)


def _env_int(name: str, default: int) -> int:
    raw = _env(name)
    if raw is None:
        return default
    try:
        return int(raw)
    except ValueError:
        return default


def _mcp_url() -> str:
    base = (_env("NOUS_GENAI_MCP_URL") or "").strip()
    if base:
        return _ensure_mcp_path(base)

    base = (_env("NOUS_GENAI_MCP_BASE_URL") or "").strip()
    if not base:
        base = (_env("NOUS_GENAI_MCP_PUBLIC_BASE_URL") or "").strip()
    if base:
        return _ensure_mcp_path(base)

    host = (_env("NOUS_GENAI_MCP_HOST") or "").strip() or "127.0.0.1"
    port = _env_int("NOUS_GENAI_MCP_PORT", 6001)
    if host in {"0.0.0.0", "::"}:
        host = "127.0.0.1"
    return f"http://{host}:{port}/mcp"

def _ensure_mcp_path(base: str) -> str:
    stripped = base.rstrip("/")
    if stripped.endswith("/mcp"):
        return stripped
    return f"{stripped}/mcp"


def _print_json(obj: Any, *, indent: int | None = 2) -> None:
    print(json.dumps(obj, ensure_ascii=False, indent=indent))


def _write_json(obj: Any, path: str, *, indent: int | None = 2) -> None:
    text = json.dumps(obj, ensure_ascii=False, indent=indent) + "\n"
    Path(path).write_text(text, encoding="utf-8")


def _summarize_schema(schema: dict[str, Any] | None) -> str:
    if not isinstance(schema, dict) or not schema:
        return "none"
    required = schema.get("required")
    props = schema.get("properties")
    parts: list[str] = []
    if isinstance(required, list):
        parts.append(f"required={len(required)}")
    if isinstance(props, dict):
        parts.append(f"properties={len(props)}")
    title = schema.get("title")
    if isinstance(title, str) and title:
        parts.append(f"title={title}")
    return ", ".join(parts) if parts else "ok"


async def _run(args: argparse.Namespace) -> int:
    loaded_envs = load_env_files()
    url = _mcp_url()
    bearer = (args.bearer_token or _env("NOUS_GENAI_MCP_BEARER_TOKEN") or "").strip()

    from mcp import ClientSession
    from mcp.client.streamable_http import streamable_http_client
    from mcp.shared._httpx_utils import create_mcp_http_client

    if args.cmd == "env":
        _print_json(
            {
                "cwd": str(Path.cwd()),
                "loaded_env_files": [str(p) for p in loaded_envs],
                "mcp_url": url,
                "bearer_auth": bool(bearer),
            }
        )
        return 0

    headers = {"Authorization": f"Bearer {bearer}"} if bearer else None
    async with create_mcp_http_client(headers=headers) as http_client:
        async with streamable_http_client(url, http_client=http_client) as (read_stream, write_stream, _):
            async with ClientSession(read_stream, write_stream) as session:
                await session.initialize()

                if args.cmd == "tools":
                    tools_result = await session.list_tools()
                    if args.json:
                        indent = None if args.compact else 2
                        if args.name:
                            tool = next((t for t in tools_result.tools if t.name == args.name), None)
                            if tool is None:
                                raise ValueError(f"tool not found: {args.name}")
                            data = tool.model_dump(mode="json", exclude_none=True)
                        else:
                            data = tools_result.model_dump(mode="json", exclude_none=True)
                        if args.out:
                            _write_json(data, args.out, indent=indent)
                        else:
                            _print_json(data, indent=indent)
                        return 0

                    if args.out:
                        raise ValueError("--out requires --json")
                    if args.compact:
                        raise ValueError("--compact requires --json")

                    tools = tools_result.tools
                    if args.name:
                        tool = next((t for t in tools if t.name == args.name), None)
                        if tool is None:
                            raise ValueError(f"tool not found: {args.name}")
                        tools = [tool]

                    for tool in tools:
                        print(f"- {tool.name}")
                        desc = getattr(tool, "description", None)
                        if isinstance(desc, str):
                            desc = " ".join(desc.split())
                        if desc:
                            print(f"  description:  {desc}")
                        print(f"  inputSchema:  {_summarize_schema(tool.inputSchema)}")
                        print(f"  outputSchema: {_summarize_schema(tool.outputSchema)}")
                        if args.full:
                            print("  inputSchema JSON:")
                            _print_json(tool.inputSchema or {})
                            print("  outputSchema JSON:")
                            _print_json(tool.outputSchema or {})
                    return 0

                if args.cmd == "templates":
                    tmpl_result = await session.list_resource_templates()
                    for t in tmpl_result.resourceTemplates:
                        print(f"- {t.uriTemplate} ({t.mimeType or 'unknown'})")
                    return 0

                if args.cmd == "resources":
                    res_result = await session.list_resources()
                    for r in res_result.resources:
                        print(f"- {r.uri} ({r.mimeType or 'unknown'})")
                    return 0

                if args.cmd == "read":
                    read_result = await session.read_resource(args.uri)
                    for item in read_result.contents:
                        if getattr(item, "text", None) is not None:
                            text = item.text
                            if args.max_chars > 0 and len(text) > args.max_chars:
                                text = text[: args.max_chars] + f"... ({len(item.text)} chars)"
                            print(text)
                            continue
                        blob = getattr(item, "blob", None)
                        if isinstance(blob, str):
                            print(f"<blob base64: {len(blob)} chars>")
                            continue
                        print(f"<resource: {item.uri}>")
                    return 0

                if args.cmd == "call":
                    tool_args: dict[str, Any] | None = None
                    if args.args_file:
                        tool_args = json.loads(Path(args.args_file).read_text(encoding="utf-8"))
                    elif args.args:
                        tool_args = json.loads(args.args)

                    result = await session.call_tool(args.name, arguments=tool_args)
                    if result.structuredContent is not None:
                        _print_json(result.structuredContent, indent=None if args.compact else 2)
                    else:
                        _print_json(
                            result.model_dump(mode="json", exclude_none=True),
                            indent=None if args.compact else 2,
                        )

                    return 1 if result.isError else 0

                raise ValueError(f"unknown cmd: {args.cmd}")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="genai-mcp-cli",
        description="MCP client for debugging nous-genai-sdk MCP server (Streamable HTTP)",
    )
    parser.add_argument(
        "--bearer-token",
        dest="bearer_token",
        help="HTTP Authorization Bearer token (or set NOUS_GENAI_MCP_BEARER_TOKEN).",
    )
    sub = parser.add_subparsers(dest="cmd")

    sub.add_parser("env", help="Print resolved env + MCP URL")

    p_tools = sub.add_parser("tools", help="List tools (or dump JSON with --json)")
    p_tools.add_argument("name", nargs="?", help="Tool name (optional, e.g. generate)")
    p_tools.add_argument("--full", action="store_true", help="Print full JSON schemas")
    p_tools.add_argument("--json", action="store_true", help="Dump JSON output (list_tools or tool)")
    p_tools.add_argument("--out", help="Write JSON to file instead of stdout (requires --json)")
    p_tools.add_argument("--compact", action="store_true", help="Compact JSON output (requires --json)")

    sub.add_parser("templates", help="List resource templates")
    sub.add_parser("resources", help="List concrete resources")

    p_read = sub.add_parser("read", help="Read resource by URI")
    p_read.add_argument("uri", help='URI like "genaisdk://artifact/{id}"')
    p_read.add_argument("--max-chars", type=int, default=2000, help="Max chars to print for text resources")

    p_call = sub.add_parser("call", help="Call a tool with JSON arguments")
    p_call.add_argument("name", help="Tool name (e.g. list_providers, list_available_models, generate)")
    p_call.add_argument("--args", help='JSON string (e.g. \'{"provider":"openai"}\')')
    p_call.add_argument("--args-file", help="Path to JSON file with tool arguments")
    p_call.add_argument("--compact", action="store_true", help="Compact JSON output")

    args = parser.parse_args(argv)
    if not args.cmd:
        args.cmd = "tools"

    try:
        return asyncio.run(_run(args))
    except BrokenPipeError:
        return 0
    except KeyboardInterrupt:
        return 130
    except Exception as e:
        print(f"error: {e}", file=sys.stderr)
        return 1


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
