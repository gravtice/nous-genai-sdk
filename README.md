# nous-genai-sdk

![CI](https://github.com/gravtice/nous-genai-sdk/actions/workflows/ci.yml/badge.svg)
![Python](https://img.shields.io/badge/python-≥3.12-blue)
![License](https://img.shields.io/badge/license-Apache--2.0-green)

中文文档：`readme_zh.md`

Unified, standardized Python GenAI SDK: one interface (`Client.generate()` / `generate_stream()`) + a consistent `GenerateRequest/GenerateResponse` schema across providers and modalities.

## Features

- **Multi-provider**: OpenAI, Google (Gemini), Anthropic (Claude), Aliyun (DashScope/Bailian), Volcengine (Doubao/Ark), Tuzi
- **Multimodal**: text/image/audio/video input and output (model-dependent)
- **Unified API**: a single `Client.generate()` for all providers
- **Streaming**: `generate_stream()` for incremental output
- **Tool calling**: function tools (model/provider-dependent)
- **JSON Schema output**: structured output (model/provider-dependent)
- **MCP Server**: Streamable HTTP and SSE transport
- **Security**: SSRF protection, DNS pinning, download limits, Bearer token auth (MCP)

## Installation

```bash
pip install nous-genai-sdk
```

For development:

```bash
pip install -e .
# or (recommended)
uv sync
```

## Configuration (Zero-parameter)

SDK/CLI/MCP loads env files automatically with priority (high → low):

`.env.local > .env.production > .env.development > .env.test`

Process env vars override `.env.*` (the loader uses `os.environ.setdefault()`).

Minimal `.env.local` (OpenAI only):

```bash
NOUS_GENAI_OPENAI_API_KEY=...
NOUS_GENAI_TIMEOUT_MS=120000
```

See `docs/CONFIGURATION.md` or copy `.env.example` to `.env.local`.

## Quickstart

### Text generation

```python
from nous.genai import Client, GenerateRequest, Message, OutputSpec, Part

client = Client()
resp = client.generate(
    GenerateRequest(
        model="openai:gpt-4o-mini",
        input=[Message(role="user", content=[Part.from_text("Hello!")])],
        output=OutputSpec(modalities=["text"]),
    )
)
print(resp.output[0].content[0].text)
```

### Streaming

```python
import sys
from nous.genai import Client, GenerateRequest, Message, OutputSpec, Part

client = Client()
req = GenerateRequest(
    model="openai:gpt-4o-mini",
    input=[Message(role="user", content=[Part.from_text("Tell me a joke")])],
    output=OutputSpec(modalities=["text"]),
)
for ev in client.generate_stream(req):
    if ev.type == "output.text.delta":
        sys.stdout.write(str(ev.data.get("delta", "")))
        sys.stdout.flush()
print()
```

### Image understanding

```python
from nous.genai import Client, GenerateRequest, Message, OutputSpec, Part, PartSourcePath
from nous.genai.types import detect_mime_type

path = "./cat.png"
mime = detect_mime_type(path) or "application/octet-stream"

client = Client()
resp = client.generate(
    GenerateRequest(
        model="openai:gpt-4o-mini",
        input=[
            Message(
                role="user",
                content=[
                    Part.from_text("Describe this image"),
                    Part(type="image", mime_type=mime, source=PartSourcePath(path=path)),
                ],
            )
        ],
        output=OutputSpec(modalities=["text"]),
    )
)
print(resp.output[0].content[0].text)
```

### List available models

```python
from nous.genai import Client

client = Client()
print(client.list_all_available_models())
```

## Providers

| Provider | Notes |
|----------|------|
| `openai` | GPT-4, DALL·E, Whisper, TTS |
| `google` | Gemini, Imagen, Veo |
| `anthropic` | Claude |
| `aliyun` | DashScope / Bailian (OpenAI-compatible + AIGC) |
| `volcengine` | Ark / Doubao (OpenAI-compatible) |
| `tuzi-web` / `tuzi-openai` / `tuzi-google` / `tuzi-anthropic` | Tuzi adapters |

## Binary output

Binary `Part.source` is a tagged union:

- **Input**: `bytes/path/base64/url/ref` (MCP forbids `bytes/path`)
- **Output**: `url/base64/ref` (SDK does not auto-download to disk)

If you need to write to file, see `examples/demo.py` (`_write_binary()`), or reuse `Client.download_to_file()` for the built-in safe downloader.

## CLI & MCP Server

```bash
# CLI
uv run genai --model openai:gpt-4o-mini --prompt "Hello"
uv run genai model available --all

# MCP Server
uv run genai-mcp-server                    # Streamable HTTP: /mcp, SSE: /sse
uv run genai-mcp-cli tools                 # Debug CLI
```

## Security

- **SSRF protection**: rejects private/loopback URLs by default (`NOUS_GENAI_ALLOW_PRIVATE_URLS=1` to allow)
- **DNS pinning**: mitigates DNS rebinding
- **Download limit**: 128MiB per URL by default (`NOUS_GENAI_URL_DOWNLOAD_MAX_BYTES`)
- **Bearer token auth**: for MCP server
- **Token rules**: fine-grained access control

## Testing

```bash
uv run pytest tests/ -v
```

## Docs

- [Configuration](docs/CONFIGURATION.md)
- [Architecture](docs/ARCHITECTURE_DESIGN.md)
- [MCP Server Security Review](docs/MCP_SERVER_CLI_SECURITY_REVIEW.md)
- [Contributing](CONTRIBUTING.md)
- [Changelog](CHANGELOG.md)

## License

[Apache-2.0](LICENSE)
