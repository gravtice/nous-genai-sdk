# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [0.1.0] - 2026-01-27

### Added

- Initial release of nous-genai-sdk
- Unified `Client.generate()` API for multi-provider, multi-modal generation
- Support for multiple providers:
  - OpenAI (GPT-4, DALL-E, Whisper, TTS)
  - Google (Gemini, Imagen, Veo)
  - Anthropic (Claude)
  - Aliyun (DashScope / Bailian)
  - Volcengine (Doubao)
  - Tuzi (web/openai/google/anthropic protocols)
- Multi-modal support:
  - Text input/output
  - Image input/output (understanding & generation)
  - Audio input/output (transcription & TTS)
  - Video input/output (understanding & generation)
  - Embedding generation
- Streaming support via `generate_stream()`
- Async job support for long-running tasks (video generation)
- Tool calling support (function calling)
- JSON schema output support
- MCP Server with Streamable HTTP and SSE transports
- CLI tool (`genai`) for quick testing
- Security features:
  - SSRF protection (private/loopback URL blocking)
  - DNS pinning to prevent rebinding attacks
  - URL download size limits
  - Bearer token authentication for MCP
  - Token rules for fine-grained access control
- Zero-config design with `.env` file auto-loading
- Comprehensive test suite (103 tests)

### Security

- Default rejection of private/loopback URLs (configurable via `NOUS_GENAI_ALLOW_PRIVATE_URLS`)
- URL download hard limit (default 128MiB, configurable via `NOUS_GENAI_URL_DOWNLOAD_MAX_BYTES`)
- MCP artifact memory limits with LRU eviction
- Signed artifact URLs for authenticated access

[Unreleased]: https://github.com/gravtice/nous-genai-sdk/compare/v0.1.0...HEAD
[0.1.0]: https://github.com/gravtice/nous-genai-sdk/releases/tag/v0.1.0
