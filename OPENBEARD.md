# OpenBeard

A fork of [Google's Gemini CLI](https://github.com/google-gemini/gemini-cli) that works with any OpenAI-compatible API endpoint. CLI command: `beard`.

## Quick Start

```bash
# Clone
git clone https://github.com/beardthelion/openbeard.git
cd openbeard

# Install dependencies
npm install

# Build
npm run build

# Run (defaults are embedded, no env vars needed)
npm start
```

## Configuration

Defaults are embedded for development. Override via environment variables or `~/.openbeard/settings.json`:

```bash
# Use a different endpoint
export OPENAI_BASE_URL=https://your-api.com/v1
export OPENAI_API_KEY=your-key
export OPENAI_MODEL=your-model
```

Or via CLI flags:
```bash
beard --base-url https://your-api.com/v1 --api-key your-key --model your-model
```

## What's Different from Upstream

**Core changes:**
- `OpenAICompatibleContentGenerator` translates OpenAI-compatible API calls to the CLI's internal Gemini format
- `OPENAI_COMPATIBLE` auth type with `openaiBaseUrl`, `openaiApiKey`, `openaiModel` config
- `openaiTranslator.ts` handles bidirectional Gemini <-> OpenAI message/tool/response conversion
- Default config: OpenGateway + mimo-v2.5-pro

**Streaming fixes:**
- `stream_options: { include_usage: true }` for accurate token stats
- Usage-only chunks captured (not dropped by early return)
- Context calibration with real `promptTokenCount` via `TokenGroundTruthEvent`

**Reasoning support:**
- `reasoning_content` deltas from MiMo/DeepSeek-style models flow through as Gemini `thought` parts
- Full accumulated buffer emitted (not deltas) to match Gemini native behavior
- UI's existing ThinkingMessage component handles display automatically

**Tool control:**
- Gemini `toolConfig.mode` translated to OpenAI `tool_choice`:
  - `NONE` -> `"none"`
  - `ANY` -> `"required"` (or specific function for single name)
  - `AUTO` -> omitted (default)
- `toolConfig` hooks (BeforeToolSelection) merged with mode precedence

**Token accuracy:**
- `cached_tokens` from OpenAI `prompt_tokens_details` -> `cachedContentTokenCount`
- `reasoning_tokens` from `completion_tokens_details` -> `thoughtsTokenCount`
- Both streaming and non-streaming paths covered

## Architecture

**Streaming stack:** SDK -> LoggingContentGenerator -> GeminiChat (retry) -> Turn (events) -> GeminiClient (orchestrator)

**Translation layer:** `openaiTranslator.ts` contains:
- `geminiContentsToOpenAIMessages` - history conversion
- `geminiToolsToOpenAITools` - tool schema conversion
- `geminiConfigToOpenAIParams` - generation config conversion
- `openaiResponseToGeminiResponse` - non-streaming response conversion
- `openaiStreamChunkToGeminiResponse` - streaming chunk conversion

**Settings:** 4 scopes: schema defaults -> system defaults -> user (`~/.openbeard/settings.json`) -> workspace (`.gemini/settings.json`) -> system overrides. Supports env var interpolation.

## Build Process

Uses esbuild for bundling (`node esbuild.js`), not just tsc. Entry point is `bundle/gemini.js`. For type-checking only: `npx tsc --noEmit --pretty`. After code changes: `npm run build`.

## Known Limitations

- Token count estimates used for proactive context checks (though real counts flow through for calibration)
- No embedding support
- Default 1M context window (matches most models)
- `thoughtSignature` fields from Gemini history are safely ignored (translator extracts only specific fields)

## License

MIT (inherited from upstream Gemini CLI's Apache 2.0, with OpenBeard modifications)
