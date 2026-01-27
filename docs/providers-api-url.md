# Providers API URL（按生成模态）

> 说明：按 SDK 的能力类型整理（`text / embedding / image / audio / video / transcription`）。  
> 如果某厂商的官方文档用“统一参考页”覆盖多模态，则不同模态可能会指向同一个文档页。

> 注意：本文档包含“未接入 provider”的资料收集，不代表 SDK 已支持。当前 SDK 已接入：OpenAI / Anthropic / Google AI Studio / 阿里云 / 火山引擎 / Tuzi。

## OpenAI

- Base URL：`https://api.openai.com/v1`

| 模态 | 官方 API 文档 |
| --- | --- |
| `text` | `https://platform.openai.com/docs/api-reference/responses` |
| `embedding` | `https://platform.openai.com/docs/api-reference/embeddings` |
| `image` | `https://platform.openai.com/docs/api-reference/images` |
| `audio` | `https://platform.openai.com/docs/api-reference/audio` |
| `video` | `https://platform.openai.com/docs/api-reference/videos` |

## Anthropic（Claude）

- Base URL：`https://api.anthropic.com`

| 模态 | 官方 API 文档 |
| --- | --- |
| `text` | `https://platform.claude.com/docs/en/intro` |

## Tuzi（兔子API，多协议：OpenAI-compatible / Gemini v1beta / Anthropic）

- OpenAI-compatible Base URL：`https://api.tu-zi.com/v1`
- Gemini v1beta Base URL：`https://api.tu-zi.com/v1beta`
- 官方文档（总览）：`https://tuzi-api.apifox.cn/7324323m0`

| 模态 | 官方 API 文档 |
| --- | --- |
| `text`（OpenAI） | `https://tuzi-api.apifox.cn/343647063e0` |
| `embedding`（OpenAI） | `https://tuzi-api.apifox.cn/343647065e0` |
| `image`（OpenAI） | `https://tuzi-api.apifox.cn/343647071e0` |
| `audio`（OpenAI） | `https://tuzi-api.apifox.cn/343647060e0` |
| `text/image/audio`（Gemini generateContent） | `https://tuzi-api.apifox.cn/346157733e0` |
| `image`（Gemini image generation example） | `https://tuzi-api.apifox.cn/346162482e0` |
| `text`（Anthropic /v1/messages） | `https://tuzi-api.apifox.cn/346380647e0` |

## Google Vertex AI（Gemini on Vertex）

| 模态 | 官方 API 文档 |
| --- | --- |
| `text` | `https://docs.cloud.google.com/vertex-ai/generative-ai/docs/model-reference/inference` |
| `embedding` | `https://docs.cloud.google.com/vertex-ai/generative-ai/docs/model-reference/inference` |
| `image` | `https://docs.cloud.google.com/vertex-ai/generative-ai/docs/model-reference/inference` |
| `audio` | `https://docs.cloud.google.com/vertex-ai/generative-ai/docs/model-reference/inference` |
| `video` | `https://docs.cloud.google.com/vertex-ai/generative-ai/docs/model-reference/inference` |

- REST Reference（总入口）：`https://docs.cloud.google.com/vertex-ai/generative-ai/docs/reference/rest`

## Google AI Studio（Gemini Developer API）

- Base URL（REST）：`https://generativelanguage.googleapis.com/v1beta`
- AI Studio 入口：`https://ai.google.dev/aistudio`

| 模态 | 官方 API 文档 |
| --- | --- |
| `text` | `https://ai.google.dev/api/generate-content` |
| `embedding` | `https://ai.google.dev/api/embeddings` |
| `image` | `https://ai.google.dev/api/generate-content#ImageConfig` |
| `audio` | `https://ai.google.dev/api/generate-content#SpeechConfig` |
| `video` | `https://ai.google.dev/api/models#method:-models.predictlongrunning` |

## 未接入（资料收集，仅供参考）

## Azure（Azure OpenAI / Foundry Models）

| 模态 | 官方 API 文档 |
| --- | --- |
| `text` | `https://learn.microsoft.com/en-us/azure/ai-foundry/openai/reference?view=foundry-classic` |
| `embedding` | `https://learn.microsoft.com/en-us/azure/ai-foundry/openai/reference?view=foundry-classic` |
| `image` | `https://learn.microsoft.com/en-us/azure/ai-foundry/openai/reference?view=foundry-classic` |
| `audio` | `https://learn.microsoft.com/en-us/azure/ai-foundry/openai/reference?view=foundry-classic` |

## 阿里云（百炼 / Model Studio / DashScope）

- OpenAI-compatible Base URL：`https://dashscope.aliyuncs.com/compatible-mode/v1`

| 模态 | 官方 API 文档 |
| --- | --- |
| `text` | `https://help.aliyun.com/zh/model-studio/model-api-reference/` |
| `embedding` | `https://help.aliyun.com/zh/model-studio/model-api-reference/` |
| `image` | `https://help.aliyun.com/zh/model-studio/model-api-reference/` |
| `audio` | `https://help.aliyun.com/zh/model-studio/qwen-tts-api` |
| `transcription` | `https://help.aliyun.com/zh/model-studio/qwen-asr-api-reference` |
| `video` | `https://help.aliyun.com/zh/model-studio/model-api-reference/` |

- DashScope Apps API 参考：`https://help.aliyun.com/zh/model-studio/dashscope-api-reference/`

## 火山引擎（火山方舟 Ark / 豆包大模型）

- OpenAI-compatible Base URL：`https://ark.cn-beijing.volces.com/api/v3`

| 模态 | 官方 API 文档 |
| --- | --- |
| `text` | `https://www.volcengine.com/docs/82379/1494384` |
| `embedding` | `https://www.volcengine.com/docs/82379/1399008` |
| `image` | `https://www.volcengine.com/docs/82379/1399008` |
| `video` | `https://www.volcengine.com/docs/82379/1399008` |

## DeepSeek

| 模态 | 官方 API 文档 |
| --- | --- |
| `text` | `https://api-docs.deepseek.com/` |
| `embedding` | `https://api-docs.deepseek.com/` |

## MiniMax

| 模态 | 官方 API 文档 |
| --- | --- |
| `text` | `https://platform.minimax.io/docs/guides/quickstart` |

## Runway

| 模态 | 官方 API 文档 |
| --- | --- |
| `video` | `https://docs.dev.runwayml.com/` |

## Replicate

> 说明：Replicate 的 HTTP API 是“统一入口”，具体模态取决于你调用的模型。

| 模态 | 官方 API 文档 |
| --- | --- |
| `text` | `https://replicate.com/docs/reference/http` |
| `image` | `https://replicate.com/docs/reference/http` |
| `audio` | `https://replicate.com/docs/reference/http` |
| `video` | `https://replicate.com/docs/reference/http` |

## xAI（Grok）

| 模态 | 官方 API 文档 |
| --- | --- |
| `text` | `https://docs.x.ai/docs/api-reference` |

## Kling AI（可灵）

| 模态 | 官方 API 文档 |
| --- | --- |
| `video` | `https://docs.qingque.cn/d/home/eZQArO-0RpjbQMpf5DPa-w8Rp?identityId=1oEER8VjdS8` |

## IndexTTS

| 模态 | 官方 API 文档 |
| --- | --- |
| `audio` | `https://github.com/index-tts/index-tts` |

## EventLabs / ElevenLabs

| 模态 | 官方 API 文档 |
| --- | --- |
| `audio` | `https://elevenlabs.io/docs/api-reference/introduction` |
