# 配置项清单（运行时）

本文基于当前代码（`nous/` + CLI + MCP）整理“可配置项”以及每项支持的设置方式，避免把设计文档里的未来项混进来。

## 1) 配置加载与优先级（零参数）

自动加载：以下入口都会调用 `nous.genai._internal.config.load_env_files()`：

- `Client()`（`nous/genai/client.py`）
- `python -m nous.genai` / `genai`（`nous/genai/cli.py`）
- `genai-mcp-server` / `genai-mcp-cli`（`nous/genai/mcp_server.py` / `nous/genai/mcp_cli.py`）

环境文件优先级（高 → 低）：

1. `.env.local`
2. `.env.production`
3. `.env.development`
4. `.env.test`

覆盖规则：`load_env_files()` 用的是 `os.environ.setdefault()`，因此不会覆盖进程已存在的环境变量：

`进程环境变量 > .env.local > .env.production > .env.development > .env.test > 代码默认值`

## 2) 通用环境变量（NOUS_GENAI_*）

| 配置项 | 默认值 | 作用 | 支持的设置方式 |
| --- | --- | --- | --- |
| `NOUS_GENAI_TIMEOUT_MS` | `120000`（非法值回退到默认；小于 1 会被钳制到 1） | 默认超时预算（毫秒）。用于 HTTP 请求与部分轮询预算。 | `.env.*` / 进程环境变量；可被 Python/MCP/CLI 覆盖（见第 6 节）。 |
| `NOUS_GENAI_URL_DOWNLOAD_MAX_BYTES` | `134217728`（128MiB；非法值回退到默认；小于 1 会被钳制到 1） | URL 下载硬限制（主要用于 CLI/demo 下载与 URL→临时文件路径）。 | `.env.*` / 进程环境变量；也可在代码里调用 `download_to_file(..., max_bytes=...)` 覆盖。 |
| `NOUS_GENAI_ALLOW_PRIVATE_URLS` | 默认不允许 | 允许下载私网/loopback URL（SSRF 防护开关）。truthy 取值：`1/true/TRUE/yes/YES`。 | 仅 `.env.*` / 进程环境变量。 |
| `NOUS_GENAI_TRANSPORT` | 空（默认不启用 MCP 限制） | 传输模式标记。为 `mcp`（或兼容值 `sse`）时会禁止 `Part.source.kind == "path"`，并禁止 raw bytes（`kind=="bytes"` 且 `encoding!="base64"`），因为远端无法访问本地文件/无法直接传二进制。 | 通常由 `genai-mcp-server` 启动时强制设置为 `mcp`；也可手动通过 `.env.*` / 进程环境变量设置。 |

## 3) Provider 配置（密钥与 Base URL）

### 3.1 Provider API Key（推荐 NOUS 前缀，也兼容原生变量名）

`Client()` 读取顺序：优先 `NOUS_GENAI_*`，否则回退到对应 provider 的原生变量名。

| Provider | 推荐变量名 | 兼容变量名 |
| --- | --- | --- |
| OpenAI | `NOUS_GENAI_OPENAI_API_KEY` | `OPENAI_API_KEY` |
| Google（AI Studio / Gemini Developer API） | `NOUS_GENAI_GOOGLE_API_KEY` | `GOOGLE_API_KEY` |
| Anthropic | `NOUS_GENAI_ANTHROPIC_API_KEY` | `ANTHROPIC_API_KEY` |
| Aliyun（OpenAI-compatible） | `NOUS_GENAI_ALIYUN_API_KEY` | `ALIYUN_API_KEY` |
| Volcengine（OpenAI-compatible） | `NOUS_GENAI_VOLCENGINE_API_KEY` | `VOLCENGINE_API_KEY` |
| Tuzi（web key） | `NOUS_GENAI_TUZI_WEB_API_KEY` | `TUZI_WEB_API_KEY` |
| Tuzi（OpenAI 协议专用 key） | `NOUS_GENAI_TUZI_OPENAI_API_KEY` | `TUZI_OPENAI_API_KEY` |
| Tuzi（Gemini 协议专用 key） | `NOUS_GENAI_TUZI_GOOGLE_API_KEY` | `TUZI_GOOGLE_API_KEY` |
| Tuzi（Anthropic 协议专用 key） | `NOUS_GENAI_TUZI_ANTHROPIC_API_KEY` | `TUZI_ANTHROPIC_API_KEY` |

设置方式：`.env.*` / 进程环境变量。

### 3.2 Provider Base URL 覆盖（仅对部分 provider 暴露为环境变量）

说明：

- OpenAI / Google / Anthropic 的 `base_url` 在 `Client()` 入口未暴露为 env 配置（使用适配器内置默认值）。
- Aliyun / Volcengine / Tuzi 为 OpenAI-compatible 或多协议聚合，`Client()` 提供了 env 覆盖点。

| 配置项 | 默认值 | 作用 | 支持的设置方式 |
| --- | --- | --- | --- |
| `ALIYUN_OAI_BASE_URL` | `https://dashscope.aliyuncs.com/compatible-mode/v1` | Aliyun OpenAI-compatible Base URL。 | `.env.*` / 进程环境变量。 |
| `VOLCENGINE_OAI_BASE_URL` | `https://ark.cn-beijing.volces.com/api/v3` | Volcengine OpenAI-compatible Base URL。 | `.env.*` / 进程环境变量。 |
| `TUZI_BASE_URL` | `https://api.tu-zi.com` | Tuzi 的 base host（用于拼默认的各协议 base_url）。 | `.env.*` / 进程环境变量。 |
| `TUZI_OAI_BASE_URL` | `${TUZI_BASE_URL}/v1` | Tuzi OpenAI-compatible Base URL。 | `.env.*` / 进程环境变量。 |
| `TUZI_GOOGLE_BASE_URL` | `${TUZI_BASE_URL}` | Tuzi Gemini v1beta Base URL（代码里会自行拼路径）。 | `.env.*` / 进程环境变量。 |
| `TUZI_ANTHROPIC_BASE_URL` | `${TUZI_BASE_URL}` | Tuzi Anthropic Base URL（代码里会自行拼路径）。 | `.env.*` / 进程环境变量。 |

## 4) MCP Server（genai-mcp-server）配置

该 Server 会同时暴露两种 HTTP transport：

- Streamable HTTP：`/mcp`（推荐）
- SSE：`/sse` + `/messages/`（兼容保留）

### 4.1 Server 监听与鉴权

| 配置项 | 默认值 | 作用 | 支持的设置方式 |
| --- | --- | --- | --- |
| `NOUS_GENAI_MCP_HOST` | `127.0.0.1` | MCP HTTP Server bind host。 | `.env.*` / 进程环境变量；或代码调用 `build_server(host=...)`。 |
| `NOUS_GENAI_MCP_PORT` | `6001`（server 内会钳制到 1..65535） | MCP HTTP Server bind port。 | `.env.*` / 进程环境变量；或代码调用 `build_server(port=...)`。 |
| `NOUS_GENAI_MCP_BEARER_TOKEN` | 空 | 若设置，则要求所有 HTTP 请求带 `Authorization: Bearer <token>`。 | `.env.*` / 进程环境变量；或 CLI `genai-mcp-server --bearer-token ...`（CLI 优先）。 |
| `NOUS_GENAI_MCP_TOKEN_RULES` | 空 | 多 token 鉴权 + 白名单：按 token 配置允许的 provider / model（model 用 `{provider}:{model_id}`）。与 `NOUS_GENAI_MCP_BEARER_TOKEN` / `--bearer-token` 互斥。 | `.env.*` / 进程环境变量。 |
| `NOUS_GENAI_MCP_PUBLIC_BASE_URL` | 自动推导 | 用于生成对外可访问的 artifact URL（`/artifact/{id}`）。 | `.env.*` / 进程环境变量。 |
| `NOUS_GENAI_ARTIFACT_URL_TTL_SECONDS` | `600` | 当启用 MCP 鉴权（`NOUS_GENAI_MCP_BEARER_TOKEN` 或 `NOUS_GENAI_MCP_TOKEN_RULES`）时，artifact URL 会自动附带短期签名（`?exp=...&sig=...`），以兼容“不带 header 直接下载”的客户端；该值控制签名有效期（秒）。 | `.env.*` / 进程环境变量。 |

`NOUS_GENAI_MCP_TOKEN_RULES` 格式（示例；分号分隔多个 token）：

`token1: [openai google]; token2: [openai:gpt-4o-mini openai:gpt-4.1]`

### 4.2 Artifact（大体积 base64 输出）内存上限

| 配置项 | 默认值 | 作用 | 支持的设置方式 |
| --- | --- | --- | --- |
| `NOUS_GENAI_MAX_INLINE_BASE64_CHARS` | `4096`（小于 0 会当作 0） | base64 内联阈值：超过则尽量转为 server 内 artifact URL。`0` 表示尽可能全部转 URL。 | `.env.*` / 进程环境变量。 |
| `NOUS_GENAI_MAX_ARTIFACTS` | `64`（小于 1 会当作 1） | artifact 个数上限（LRU 淘汰）。 | `.env.*` / 进程环境变量。 |
| `NOUS_GENAI_MAX_ARTIFACT_BYTES` | `67108864`（64MiB；小于 0 会当作 0） | artifact 总内存字节上限（LRU 淘汰）。`0` 表示禁用 artifact 存储。 | `.env.*` / 进程环境变量。 |

### 4.3 MCP Server CLI 参数

| 参数 | 默认值 | 作用 |
| --- | --- | --- |
| `--proxy <url>` | 空 | provider 请求使用的 HTTP proxy（显式传入，不依赖 `HTTP_PROXY`）。 |
| `--bearer-token <token>` | 空 | 覆盖/替代 `NOUS_GENAI_MCP_BEARER_TOKEN`（与 `NOUS_GENAI_MCP_TOKEN_RULES` 互斥）。 |
| `--model-keyword <kw>` | 空 | 只暴露包含该子串的模型（大小写不敏感；可重复/逗号分隔）。 |

## 5) MCP CLI（genai-mcp-cli）配置

### 5.1 MCP URL / Base URL 解析优先级

说明：`genai-mcp-cli` 使用 Streamable HTTP transport（连接 `/mcp`）。

`genai-mcp-cli` 连接目标按以下优先级解析（高 → 低）：

1. `NOUS_GENAI_MCP_URL`（可直接给完整 `/mcp` URL；若不以 `/mcp` 结尾会自动补上）
2. `NOUS_GENAI_MCP_BASE_URL`（同上：自动补 `/mcp`）
3. `NOUS_GENAI_MCP_PUBLIC_BASE_URL`（同上：自动补 `/mcp`）
4. `NOUS_GENAI_MCP_HOST` + `NOUS_GENAI_MCP_PORT`（host 为 `0.0.0.0/::` 时会替换为 `127.0.0.1`）

### 5.2 MCP CLI 参数

| 参数 | 默认值 | 作用 |
| --- | --- | --- |
| `--bearer-token <token>` | 空 | 覆盖/替代 `NOUS_GENAI_MCP_BEARER_TOKEN`。 |
| `read --max-chars <n>` | `2000` | 读取 text resource 时最多打印多少字符（调试用）。 |

## 6) Python / MCP 请求级参数（`GenerateRequest`）

这些属于“单次调用参数”，不是全局环境变量；主要通过 Python 代码或 MCP `generate` 工具的 `request` JSON 设置。

### 6.1 `GenerateParams`（`request.params.*`）

| 字段 | 作用 | 支持的设置方式 |
| --- | --- | --- |
| `temperature` | 随机性控制（provider 兼容时生效）。 | Python：`GenerateRequest(params=...)`；MCP：`generate.request.params.temperature`。 |
| `top_p` | 采样控制（provider 兼容时生效）。 | Python；MCP。 |
| `seed` | 随机种子（provider 兼容时生效）。 | Python；MCP。 |
| `max_output_tokens` | 输出 token 上限（provider 兼容时生效）。 | Python；MCP。 |
| `stop` | 停止词列表（provider 兼容时生效）。 | Python；MCP。 |
| `timeout_ms` | 单次调用超时预算（毫秒）。 | Python：`GenerateRequest.params.timeout_ms`；CLI：`genai --timeout-ms`（会写入 `request.params.timeout_ms`）；MCP：`generate.request.params.timeout_ms`；全局默认见 `NOUS_GENAI_TIMEOUT_MS`。 |
| `idempotency_key` | 幂等键（provider 兼容时作为请求头/字段传递）。 | Python；MCP。 |
| `reasoning.effort` | 推理强度（provider 兼容时生效）。 | Python；MCP。 |

### 6.2 其他常用字段

| 字段 | 作用 | 支持的设置方式 |
| --- | --- | --- |
| `request.wait` | `True`：等待 job 完成；`False`：返回 `job_id`（若模型为异步 job）。 | Python；MCP。`genai` CLI 默认 `wait=True`（包含视频）；如遇 `status=running`，通常是超时，调大 `--timeout-ms` 或 `NOUS_GENAI_TIMEOUT_MS`。 |
| `request.output.*` | 输出模态与相关参数（text/image/audio/video/embedding）。 | Python；MCP。`genai` CLI 会按模型能力自动推断。 |
| `request.tools` / `request.tool_choice` | Tool calling（provider 兼容时生效）。 | Python；MCP。 |
| `request.provider_options` | provider 私有扩展参数（例如 Google 的 `generationConfig`）。 | Python；MCP。 |
