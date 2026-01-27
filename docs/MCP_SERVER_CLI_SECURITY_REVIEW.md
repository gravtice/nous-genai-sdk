# MCP Server / CLI 安全与资源审查（简版）

## 范围

- `nous/genai/mcp_server.py`：MCP Server（artifact 存储/路由、工具封装）
- `nous/genai/cli.py`：本地 CLI（生成/下载/列模型）
- `nous/genai/_internal/http.py`：统一 HTTP 层（URL 下载、SSE、代理、超时）

## 默认安全姿态（结论）

- 默认绑定 `127.0.0.1`（`NOUS_GENAI_MCP_HOST` 未设置时），适合作为本机工具服务；不建议直接暴露到公网。
- 可选启用 `NOUS_GENAI_MCP_BEARER_TOKEN`（或 CLI `--bearer-token`），或使用 `NOUS_GENAI_MCP_TOKEN_RULES` 配置多 token 白名单鉴权，为所有 HTTP 端点加一层共享密钥鉴权。
- URL 下载默认拒绝 private/loopback，并限制最大字节数；支持显式开关放开（`NOUS_GENAI_ALLOW_PRIVATE_URLS=1`）。
- artifact 存储有数量/总字节上限与 LRU 淘汰；避免无限增长占满内存。

## 关键风险点与对应措施

### 1) URL 下载（SSRF / 内网探测 / 资源消耗）

- 风险：URL 可控时可能下载私网地址、重定向绕过、DNS rebinding（检查与连接不一致）、超大文件/无超时卡死。
- 措施：
  - 默认拦截 private/loopback host（可显式放开）。
  - 最大字节数硬限制（`NOUS_GENAI_URL_DOWNLOAD_MAX_BYTES`）。
  - 解析一次并固定 IP（IP pinning）避免 DNS rebinding（TOCTOU）。
  - 统一超时（默认取 `NOUS_GENAI_TIMEOUT_MS`）。

### 2) SSE（流式协议兼容性/恢复）

- 风险：仅消费 `data:` 会丢失 `event/id/retry`，影响恢复/重放与协议兼容。
- 措施：完整解析 `event/id/retry` 并保留元数据（提供 `SSEEvent`）。

### 3) MCP artifact（资源压力/泄漏面）

- 风险：大 base64 输出导致内存增长；artifact URL 若服务暴露到公网可能泄漏生成内容。
- 措施：
  - `NOUS_GENAI_MCP_MAX_ARTIFACTS` 与 `NOUS_GENAI_MCP_MAX_ARTIFACT_BYTES` 限制 + LRU 淘汰。
  - 启用 MCP 鉴权（`NOUS_GENAI_MCP_BEARER_TOKEN` / `--bearer-token` 或 `NOUS_GENAI_MCP_TOKEN_RULES`）后，artifact 下载也受保护；同时 artifact URL 会附带短期签名参数（`?exp=...&sig=...`）以兼容不自动携带 header 的客户端（例如某些 UI/工具链的自动下载）。 

### 4) 代理（可达性/一致性）

- 风险：代理接入不一致导致 URL 下载与 provider 请求走不同网络路径，排障困难；HTTPS 目标经 HTTP proxy 可能失败。
- 措施：统一走内部 HTTP 层，修正 HTTPS 目标经 HTTP proxy 的连接方式（CONNECT + TLS to target）。
