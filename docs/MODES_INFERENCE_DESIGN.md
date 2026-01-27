# Model `modes` 推断准确性设计

> 目标：让 SDK 对外暴露的模型 `modes`（CLI/MCP 输出）稳定、可解释、尽量准确。  
> 本文只做设计，不包含实现代码。

## 1. 背景

当前 SDK 会在以下入口返回模型的 `modes` 字段：

    - CLI：`genai model sdk` / `genai model available --provider <provider>` / `genai model available --all`
- MCP Server：`list_available_models` / `list_all_available_models`

它们用于“调用方/工具”快速判断：一个 `provider:model_id` 支持哪些调用形态（例如是否支持 `stream`、是否可能返回 `job`）。

## 2. 当前问题

### 2.1 `modes` 计算逻辑过于简化

现状：`modes` 主要由 `Capability.supports_stream` 推导，且只在 chat 场景里插入 `stream`；`supports_job` 没有被反映到 `modes`，导致 video/部分 async task 模型在列表里看不出“可能返回 running(job)”的事实。

### 2.2 能力推断依赖字符串启发式，误报难避免

多 provider（尤其是“聚合/兼容层”）的 `capabilities(model_id)` 本质上是基于 `model_id` 的启发式；这类推断不可避免会出现：

- 误报：宣称支持 `stream`/`job`，实际不支持
- 漏报：实际支持 `stream`/`job`，但未识别出来

这会让 CLI/MCP 输出误导上层逻辑（例如自动选择 stream 或选择 wait 策略）。

## 3. 目标与非目标（KISS）

### 3.1 目标

- `modes` 字段语义清晰、可解释（文档里定义严格）
- 对 curated catalog（`MODEL_CATALOG`）里的模型，`modes` 推断尽量准确
- 不引入运行时探测（避免网络/延迟/不确定性）
- 保持实现简单：默认走现有能力推断，少量显式 override 兜底

### 3.2 非目标

- 不对“provider 远端返回但不在 catalog 里”的未知模型做完整推断
- 不新增复杂的通用 Job 轮询 API（当前仅暴露 `job_id` 即可）
- 不引入基于 CLI 参数的覆盖（遵循零参数/文件优先原则）

## 4. `modes` 定义（必须统一）

`modes` 是一个 SDK 视角的能力标识列表，面向“工具/集成方”，不是 provider 官方术语。

固定枚举：

- `sync`：支持 `Client.generate(request, stream=False)`（一次调用返回一个 `GenerateResponse`；可能是 `completed` 或 `running`）
- `stream`：支持 `Client.generate(request, stream=True)` / `Client.generate_stream(...)`（返回事件迭代器）
- `job`：该模型可能产生异步任务：`GenerateResponse.status="running"` 且包含 `job.job_id`
  - `request.wait=True` 时 SDK 可能在超时预算内等待完成；超时仍可能返回 `running(job)`
  - `request.wait=False` 时 SDK 应尽快返回 `running(job)`
- `async`：支持 `Client.generate_async(request)`（对 `sync` 的 asyncio 包装；仅覆盖非流式）

约束：

- `async` 仅代表 SDK 层异步调用形态，不代表 provider 支持“真正异步 API”
- `stream` 与 `async` 不做组合枚举（不提供 async-stream 语义）

## 5. 数据源与优先级（核心方案）

坚持“三层数据源 + 明确优先级”，避免把推断逻辑越堆越复杂。

### 5.1 数据源 1：显式 Overrides（最高优先级）

维护一份极小的“已知例外”表（按 `provider:model_id` 精确匹配），只覆盖 `supports_stream` / `supports_job` 等少量字段。

目的：修正启发式误报/漏报，并提供可审计的变更入口。

约束：

- Overrides 只允许“变更支持与否”，不引入复杂规则引擎
- Overrides 只覆盖踩坑模型，数量应可控（几十级别）

### 5.2 数据源 2：离线验证报告（用于更新 Overrides）

提供一个离线 probe/烟测流程：

- 对 catalog 模型做最小请求（短 prompt、短超时、低成本）
- 分别验证：
  - `stream`：`stream=True` 是否能正常产生事件并结束
  - `job`：`wait=False` 是否返回 `running + job_id`
- 输出 JSONL/TSV 报告，供人工 review 后更新 Overrides

说明：验证报告不是运行时依赖，只是证据来源。

### 5.3 数据源 3：默认能力推断（最低优先级）

继续使用现有 `Adapter.capabilities(model_id) -> Capability`（启发式/协议规则）作为默认路径。

## 6. `modes` 计算规则（确定性、可测试）

输入：`Capability`（应用 Overrides 后）。

输出：`modes: list[str]`，顺序固定，便于稳定展示与比对。

固定顺序：

1. `sync`（恒有）
2. `stream`（当 `supports_stream=True`）
3. `job`（当 `supports_job=True`）
4. `async`（恒有）

注意：

- 不再用“category==chat”作为 `stream` 的硬条件；以 `supports_stream` 为准
- `job` 不取代 `sync`：它只是告知“可能是 running(job) 的交付形态”

## 7. 变更面与兼容性

- CLI/MCP 返回的 `modes` 新增 `job`（向后兼容扩展：旧调用方只要不做严格相等判断就不会炸）
- 文档/注释同步更新，明确 `job` 语义
- 若某些调用方把 `modes` 当成“唯一可用调用方法集合”，应在集成侧放宽到“包含即可”判断

## 8. 验收标准（可量化）

- 对 catalog 内模型：
  - `supports_stream=True` 的模型，`modes` 必须包含 `stream`
  - `supports_job=True` 的模型，`modes` 必须包含 `job`

## 9. 实施步骤（最小闭环）

1. 引入 Overrides（先空表也可）
2. 调整 `modes` 计算逻辑：支持 `job`，并用固定顺序输出
3. 增加单测：覆盖 `job`/`stream` 的 `modes` 推断规则与 Overrides
4. 添加离线 probe 脚本：产出报告，驱动 Overrides 迭代
