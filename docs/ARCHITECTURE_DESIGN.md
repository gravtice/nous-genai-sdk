# GenAISDK Architecture Design

> 目标：提供一个**单一 endpoint** 的 Python SDK，用统一的请求/响应模型支持端到端多模态（文本/图片/音频/视频）输入输出，并对接多家生成式 AI 提供商。  
> 本文只做设计，不包含实现代码。

## 0. 文档约定

- 术语：
  - **Endpoint**：对 SDK 使用者暴露的主调用入口（本 SDK 只有一个：`generate()`）。
  - **Provider**：上游模型/平台提供方（OpenAI、Anthropic、Vertex AI、OpenRouter、ElevenLabs 等）。
  - **Model**：某 provider 下的具体模型标识。
  - **Asset**：图片/音频/视频/文件等二进制输入输出内容。
- 规范性词汇：
  - **MUST/必须**：强制要求，否则视为不兼容。
  - **SHOULD/应该**：推荐做法，允许例外但需说明原因。
  - **MAY/可以**：可选能力。

## 1. 背景与问题

业界常见 SDK/网关（如 LiteLLM）通常按模态或任务拆分多个 endpoint（chat、images、audio、video）。这在“端到端多模态模型”逐渐成为主流时，会导致：

- 使用者需要在多个 endpoint 之间切换，难以表达“同一上下文里混合多模态输入并产出多模态结果”的真实需求。
- Provider 的产品形态差异（同步/异步 Job、是否需要上传、是否支持流式）泄漏到业务代码。

本 SDK 的核心思路：**对外只提供一个生成 endpoint**，内部通过适配器把不同 provider 的多端点/Job/上传差异屏蔽掉，同时保持行为可预期、可控成本。

## 2. 设计目标与非目标（KISS）

### 2.1 目标

- 对外只有一个主入口：`Client.generate(...)`（以及可选的 stream 交付形态，语义仍是同一 endpoint）。
- 输入/输出统一表达：同一套 `Message + Part` 结构覆盖文本/图片/音频/视频。
- 显式能力边界：模型不支持的输入/输出组合必须明确报错，不做“偷偷多次调用拼结果”。
- Provider 扩展性：新增 provider 只需要实现最小适配器接口，不侵入核心抽象。
- 资源处理安全：大文件不默认读入内存；输出媒体默认返回引用（URL/ref），下载由用户显式触发。

### 2.2 非目标

- 不内置 Agent/RAG/多模型自动编排流水线。
- 不承诺所有 provider 都支持任意模态输入/输出；以能力矩阵为准。
- 不为“统一”牺牲可解释性：`provider_options` 作为显式逃生口，而不是把 provider 私有行为硬塞进公共字段。

## 3. 对外 API 形态（Python）

### 3.1 Client

- `Client()`：零参数创建，自动加载配置（见第 8 节）。
- `generate(request) -> GenerateResponse`
- `generate(request, stream=True) -> Iterator[GenerateEvent]`
  - 也可以提供 `generate_stream(request)` 作为薄封装，但不引入第二套语义。

### 3.2 关键行为

- 同步能力：能在单次调用中返回完整结果的 provider/model，`status="completed"`。
- 异步 Job：视频等长任务常见。`generate()` 支持两种交付：
  - 默认 `wait=True`（受 `timeout_ms` 约束）：尽量等待完成；超时则返回 `status="running"`。
  - `wait=False`：立即返回 `status="running"` + `job` 信息。
- 流式：若 provider 支持流式输出，`stream=True` 返回事件迭代器，且必须与非流式共享同一 `GenerateRequest`（参考 Gemini 的 `generateContent/streamGenerateContent`）；若不支持则必须明确报错（不隐式降级）。

## 4. 统一数据模型

### 4.1 Part（内容片段）

所有输入输出内容都由 `Part` 表达。最小字段：

- `type`: `"text" | "image" | "audio" | "video" | "file" | "tool_call" | "tool_result"`
- `mime_type`: 可选，但对二进制内容**应该**提供（如 `image/png`、`audio/wav`、`video/mp4`）
- `source`: 四选一（越简单越好）
  - `{kind:"bytes", data: <bytes>}`（小内容；大视频不推荐）
  - `{kind:"path", path:"..."}`
  - `{kind:"url", url:"https://..."}`
  - `{kind:"ref", provider:"openai", id:"..."}`（provider 文件引用；可能是 ID，也可能是 URI，如 Gemini 的 `file_uri`）
- `text`: 仅当 `type="text"` 时存在
- `meta`: 可选（宽高、时长、采样率、hash、语言等）

设计约束：

- SDK **必须**能在不把大文件读入内存的情况下工作（使用 `path/url/ref`）。
- 输出媒体 **默认**使用 `url/ref`，是否下载由调用方决定（可提供辅助方法，但不改变主响应结构）。

设计参考（Google Gemini `generateContent`）：

- `parts[].text` ↔ `Part{type:"text"}`
- `parts[].inline_data{mime_type,data}` ↔ `Part{source.kind="bytes", mime_type=...}`（`data` 为二进制内容的编码形式）
- `parts[].file_data{mime_type,file_uri}` ↔ `Part{source.kind="ref", id="<file_uri>", mime_type=...}`

### 4.2 Message

- `role`: `"system" | "user" | "assistant" | "tool"`
- `content`: `Part[]`

说明：

- `system` 用于全局指令（部分 provider 使用独立字段，如 Gemini 的 `systemInstruction`；适配器负责映射；不支持则降级）。
- `tool_call/tool_result` 为可选能力：若 model/provider 支持工具调用，可在统一结构里表达；否则禁止透传并明确报错。

### 4.3 GenerateRequest（规范化请求）

最小字段建议：

- `model`: `"{provider}:{model_id}"`
  - 例：`openai:gpt-4.1`、`anthropic:claude-3-7-sonnet`、`openrouter:anthropic/claude-3.5-sonnet`
- `input`: `Message[]`
- `output`:
  - `modalities`: `["text", "image", "audio", "video"]`（可多选）
  - 可选按模态细化（只保留最小公分母；其余走 `provider_options`）：
    - `text`: `{format:"text"|"json", json_schema?:object, max_output_tokens?:int}`
    - `image`: `{n?:int, size?:string, format?:string}`
    - `audio`: `{voice?:string, format?:string, language?:string}`
    - `video`: `{duration_sec?:int, aspect_ratio?:string, fps?:int, format?:string}`
  - 说明：对 Gemini，`output.modalities` 可映射到 `responseModalities`，音频相关配置可映射到 `speechConfig`（voice/format 等）。
- `params`（跨 provider 尽量统一）：
  - `temperature?:float`
  - `top_p?:float`
  - `seed?:int`
  - `max_output_tokens?:int`
  - `stop?:string[]`
  - `timeout_ms?:int`
  - `idempotency_key?:string`（如果 provider 支持则使用；否则仅用于本地去重/日志关联）
- `wait?:bool`（默认 true；控制 Job 行为）
- `provider_options?: { [provider_name: string]: object }`

关键约束：

- 若 `output.modalities` 含多个模态，而目标 model 不支持“单次生成多模态输出”，SDK **必须**直接报错（不做自动链式调用）。
- `provider_options` 是显式逃生口：SDK **不得**把 provider 私有字段塞进公共 `params/output` 里。

### 4.4 GenerateResponse（统一响应）

- `id`: SDK 生成的请求 ID（用于日志/追踪）
- `provider`: 实际执行的 provider
- `model`: 实际执行的 model
- `status`: `"completed" | "running" | "failed" | "canceled"`
- `output`: `Message[]`（完成时必须存在；失败/运行中可为空）
- `usage`: 统一的使用统计（能拿到多少算多少）
  - 例：`input_tokens/output_tokens/total_tokens`、`seconds`、`image_count`、`video_seconds`、`cost_estimate`（若可得）
- `job`: 仅当 `status="running"` 时存在
  - `job_id`: provider 侧任务 ID
  - `poll_after_ms`: 建议轮询间隔
  - `expires_at`: 可选
  - 说明：`job` 是数据，不是新 endpoint；后续操作由同一个 `Client` 方法完成（实现阶段定义，如 `wait(job_id)`/`poll(job_id)`）。
- `error`: 仅当 `status="failed"` 时存在（统一错误模型见第 7 节）

### 4.5 GenerateEvent（流式事件）

保持最小事件集，避免“发明一套新协议”：

- `type`（建议）：
  - `output.text.delta`
  - `output.audio.chunk`
  - `output.image.ref`
  - `output.video.ref`
  - `tool.call`
  - `error`
  - `done`
- `data`: 事件载荷（文本增量/音频字节块/媒体 ref 或 url/错误信息等）

## 5. Provider 适配器架构

### 5.1 核心原则

- Provider 的“多端点世界”（chat/images/tts/video、上传、Job、轮询）全部封装在适配器内部。
- 核心层只处理：标准化、路由、资产准备、错误/重试、日志与事件流。

### 5.2 适配器最小接口（建议）

`Adapter` 是概念接口；实现上**不要求**“一个 provider 一份重复代码”。同协议的 provider（如大量 OpenAI-compatible）应该共享同一套协议实现，通过配置差异化（base_url、鉴权、路径、feature flags）。

每个 provider 至少要有一个可路由的 `Adapter` 实例（或配置化实例），必须提供：

- `capabilities(model_id) -> Capability`
  - 描述该 model 的输入/输出模态、是否支持流式、是否 Job、大小限制等。
- `prepare(request: GenerateRequest) -> ProviderPlan`
  - 校验能力（或由核心统一校验）
  - 资产处理（必要时上传，生成 `ref`）
  - 选择 provider 内部具体 endpoint（若拆分）
- `execute(plan: ProviderPlan, stream: bool) -> ProviderRawResponse | Iterator[ProviderRawEvent]`
- `parse(raw) -> GenerateResponse | Iterator[GenerateEvent]`

说明：

- `ProviderPlan` 是内部结构，不暴露给用户；其存在是为了把“准备阶段”和“执行阶段”分开（便于复用/测试/日志）。
- 核心层必须禁止“隐式多次调用”行为：一次 `generate()` 对应一次 provider 侧“生成动作”（允许：必要的上传与 Job 轮询）。

### 5.3 Model 命名与路由

- 统一 `model` 形式：`"{provider}:{model_id}"`
- provider 示例：
  - `openai:...`
  - `anthropic:...`
  - `openrouter:...`（`model_id` 通常包含 `/`，如 `anthropic/claude-3.5-sonnet`）
  - `vertexai:...`（或 `google:...`，需在实现阶段固定命名）
  - `elevenlabs:...`
- 解析规则必须简单、可预测：第一个 `:` 左边是 provider，其余原样作为 `model_id`。

### 5.4 协议复用（OpenAI-compatible 等）

现实情况：很多 provider 宣称“支持 OpenAI API 协议”，但通常只覆盖其中一部分（例如只支持 `chat.completions`，不支持 `responses`/files/tools，或参数/错误码/流式事件有差异）。因此需要区分：

- **Provider**：计费主体/鉴权/域名/限流/模型清单的归属（路由维度）
- **Protocol**：HTTP 请求/响应的形状与语义（代码复用维度）

建议实现方式（保持 KISS）：

- 维护一个 `provider registry`：把 provider 映射到协议与连接配置
  - `protocol`: `openai_compat | anthropic | gemini | replicate | runway | ...`
  - `base_url`、鉴权方式、必要的固定 query/header
  - `quirks`（少量 feature flags）：仅用于处理已知不兼容点（比如 Azure OpenAI 的路径与 `api-version`）
- 同一协议只实现一次解析/序列化/流式解码逻辑；不同 provider 通过 registry 配置出“实例”。

约束（避免把“兼容”当成幻觉）：

- **不能**因为协议兼容就假设能力兼容：`capabilities(model_id)` 仍然以 provider + model 为粒度声明。
- 遇到“协议字段存在但语义不同”的情况，优先走 `provider_options` 显式透传，公共字段不强行吸收。

### 5.5 对接方式（REST 优先）

对接 provider 时默认采用 **REST/HTTP 直连**，而不是依赖各家官方 SDK。

原因（工程性优先）：

- 依赖更小：避免引入大量 SDK 与传递依赖，降低版本冲突与供应链风险。
- 行为一致：超时、重试、代理、日志、流式解码（SSE/chunk）等由 SDK 统一控制，而不是分散在不同 SDK 的默认行为里。
- 兼容更强：所谓 “OpenAI-compatible” 本质是协议兼容，用 REST 更容易做精确适配与差异隔离（见 5.4）。
- 可观测性更好：统一的 request/response 记录与错误归一化更直接。

例外（尽量少）：

- 仅当 provider 没有稳定 REST、或鉴权/上传流程强依赖特定库时，才引入**最小**辅助依赖（例如 OAuth2/签名计算），但仍以“自己发 HTTP 请求”为主，不把整套官方 SDK 作为硬依赖。

## 6. 资产处理（上传/引用/下载）

### 6.1 输入资产

SDK 必须支持以下输入形式：

- `url`：优先推荐（大文件、云存储、可复用）
- `path`：本地文件（由 SDK 读取并按 provider 要求上传或内联）
- `bytes`：仅限小文件；必须有大小阈值（实现阶段定义）
- `ref`：已经上传过的 provider 文件引用（跳过上传）

### 6.2 上传策略（在适配器内实现）

按 provider 能力选择，优先级：

1. provider 支持 URL 直接读取：直接传 URL（不上传）
2. provider 提供文件 API：上传得到 `ref`
3. 只能内联：小文件可 base64/bytes 内联（需阈值与明确告警）

示例：Gemini 的 `files.upload` 产出 `file_uri`，在 `parts[].file_data.file_uri` 中引用；对应本 SDK 的 `source.kind="ref"`。

### 6.3 输出资产

- 默认返回 `url/ref`（而不是内联 bytes）。
- 可在实现阶段提供辅助方法（例如 `client.download(part, to_path=...)`），但不改变 `GenerateResponse` 的统一结构。

## 7. 错误模型与重试

### 7.1 统一错误类型（建议）

- `AuthError`：鉴权失败（401/403）
- `RateLimitError`：限流（429）
- `InvalidRequestError`：参数/能力不匹配/输入非法（400/422）
- `NotSupportedError`：请求组合超出模型能力（必须清晰指明缺失能力）
- `TimeoutError`：超时
- `ProviderError`：provider 侧 5xx 或不可归类错误

`GenerateResponse.error` 建议包含：

- `type`（上述枚举）
- `message`
- `provider_code`（若可得）
- `retryable: bool`

### 7.2 重试原则

- 默认重试只对“安全且可解释”的场景启用（429、部分 5xx、短暂网络错误）。
- 若 provider 支持幂等键，优先使用 `idempotency_key`。
- 不对明显非幂等的生成动作做激进重试（避免重复计费/重复生成）。

## 8. 配置与密钥（零参数、文件优先）

### 8.1 配置文件优先级

自动加载优先级（从高到低）：

1. `.env.local`
2. `.env.production`
3. `.env.development`
4. `.env.test`

### 8.2 推荐环境变量命名（示例）

通用：

- `NOUS_GENAI_TIMEOUT_MS`
- `NOUS_GENAI_LOG_LEVEL`
- 代理：通过 `Client(proxy_url=...)` 或 `genai-mcp-server --proxy ...` 显式指定（不使用 `HTTP_PROXY/http_proxy` 以避免影响宿主进程）

Provider：

- `NOUS_GENAI_OPENAI_API_KEY`
- `NOUS_GENAI_ANTHROPIC_API_KEY`
- `NOUS_GENAI_OPENROUTER_API_KEY`
- `NOUS_GENAI_GOOGLE_API_KEY` / `NOUS_GENAI_VERTEXAI_*`
- `NOUS_GENAI_AZURE_OPENAI_*`
- `NOUS_GENAI_ELEVENLABS_API_KEY`

约束：

- `.env.local` 必须被 gitignore（不提交）。
- SDK 默认行为必须“零参数可跑”，但前提是必要密钥已配置。

## 9. Provider 接入策略（按类型分组）

本节不承诺具体 API 细节（实现阶段以各 provider 官方文档为准），只定义“应该如何映射到统一模型”。

### 9.1 A 类：统一对话式/Parts 风格（最接近端到端多模态）

目标：尽可能直接映射 `Message + Part`，减少信息损失。

- OpenAI（部分模型支持多模态输入；输出能力依模型而定）
- Google Gemini（AI Studio / Vertex AI）（典型为多模态输入；输出能力依模型而定）
- Azure OpenAI（通常为 OpenAI 兼容形态，但能力取决于部署与版本）

### 9.2 B 类：对话模型 + 独立图像/音频/视频端点（“多端点世界”）

策略：适配器内部根据 `output.modalities` 选择具体端点；若请求组合在 provider 内无法单次完成，直接 `NotSupportedError`。

候选（示例，按实际能力调整）：

- Anthropic（目前以文本输出为主；视觉输入在部分模型可用；多模态输出通常不成立）
- DeepSeek、xAI、MiniMax 等（多为文本/视觉对话为主，图像/音频/视频若存在通常是独立产品线）
- 阿里云、火山引擎（多产品形态并存，常见“按能力拆端点”）

### 9.3 C 类：媒体生成平台（高概率异步 Job）

策略：`generate()` 支持 `status="running"`；适配器封装轮询拿 `url/ref`。

- Runway
- Kling AI
- 其他视频生成/编辑平台

### 9.4 D 类：托管/市场类（模型高度异构）

策略：公共字段只覆盖最小公分母；其余通过 `provider_options` 原样透传，并在 `capabilities()` 中尽量提供“保守能力声明”。

- Replicate
- OpenRouter（聚合路由到多家模型；很多情况下是 OpenAI 兼容 API，但模型能力取决于具体路由目标）

### 9.5 E 类：语音平台（TTS/STT）

策略：把 TTS/STT 都视为 `generate()` 的一种：输入/输出模态决定路由。

- ElevenLabs（TTS：text->audio；STT：audio->text）
- IndexTTS（若作为独立 provider，映射同上）

## 10. 能力声明（Capability）

SDK 必须提供“能力可查询/可校验”的机制（至少内部要有；是否对外开放 API 可在实现阶段决定）。

最小 `Capability` 字段建议：

- `input_modalities: set[str]`
- `output_modalities: set[str]`
- `supports_stream: bool`
- `supports_job: bool`
- `supports_tools: bool`（可选）
- `supports_json_schema: bool`（可选）
- `limits`（可选，保守即可）：最大 tokens、最大文件大小、最大视频时长等

校验规则（必须）：

- 请求输入/输出模态必须是 capability 子集，否则 `NotSupportedError`。

## 11. 演进与兼容性

- 公共字段保持稳定，小步增加：新增能力优先在 `provider_options` 试水，成熟后再抽象到公共字段。
- 严禁为“统一”引入不必要的中间层/复杂配置系统；宁可在能力不足时明确失败。
- 版本策略（建议）：语义化版本；破坏性变更必须升级 MAJOR，并提供迁移说明。

## 12. 参考映射（Google Gemini GenerateContent）

Gemini 的 `generateContent` 是“单一请求模型 + 多模态 parts”的典型实现，本 SDK 可以直接借鉴其“同构输入/输出”的形态，并做 provider-agnostic 抽象：

- `GenerateRequest.input` ↔ Gemini `contents[]`（每条 `Message` 对应一条 `content`，`Message.content` ↔ `parts[]`）
- `role=system` ↔ Gemini 顶层 `systemInstruction`（适配器抽取并映射；若有多个 system message，合并策略需明确）
- `Part.source.kind="bytes"` ↔ Gemini `inline_data`
- `Part.source.kind="ref"` ↔ Gemini `file_data.file_uri`（通过 `files.upload` 获得）
- `GenerateRequest.output.modalities` ↔ Gemini `responseModalities`
- `GenerateRequest.output.audio` ↔ Gemini `speechConfig`
- `stream=True` ↔ Gemini `streamGenerateContent`（请求体结构一致，仅交付方式不同）

差异处理（保持 KISS）：

- Gemini 响应可能返回 `candidates[]`；MVP 阶段 SDK 默认选择首个/最佳候选作为 `GenerateResponse.output`，其余候选不保证暴露（可作为后续扩展）。
