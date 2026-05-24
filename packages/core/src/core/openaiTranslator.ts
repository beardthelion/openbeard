/**
 * @license
 * Copyright 2025 Google LLC
 * SPDX-License-Identifier: Apache-2.0
 */

import {
  type Content,
  type Part,
  type FunctionCall,
  type Tool,
  type GenerateContentConfig,
  GenerateContentResponse,
  type Candidate,
  FinishReason,
} from '@google/genai';

// --- OpenAI types (minimal, matching the Chat Completions API) ---

export interface OpenAIMessage {
  role: 'system' | 'user' | 'assistant' | 'tool';
  content?: string | null | OpenAIContentPart[];
  reasoning_content?: string | null;
  tool_calls?: OpenAIToolCall[];
  tool_call_id?: string;
}

export type OpenAIContentPart =
  | { type: 'text'; text: string }
  | { type: 'image_url'; image_url: { url: string } };

export interface OpenAIToolCall {
  id: string;
  type: 'function';
  function: {
    name: string;
    arguments: string;
  };
}

export interface OpenAITool {
  type: 'function';
  function: {
    name: string;
    description?: string;
    parameters?: Record<string, unknown>;
  };
}

export interface OpenAIRequestParams {
  model: string;
  messages: OpenAIMessage[];
  temperature?: number;
  top_p?: number;
  max_tokens?: number;
  stop?: string[];
  stream?: boolean;
  stream_options?: { include_usage: boolean };
  tools?: OpenAITool[];
  tool_choice?: 'auto' | 'none' | 'required' | { type: 'function'; function: { name: string } };
  response_format?: { type: string };
  seed?: number;
  presence_penalty?: number;
  frequency_penalty?: number;
}

export interface OpenAIUsage {
  prompt_tokens: number;
  completion_tokens: number;
  total_tokens: number;
  prompt_tokens_details?: {
    cached_tokens?: number;
  };
  completion_tokens_details?: {
    reasoning_tokens?: number;
  };
}

export interface OpenAIChatCompletion {
  id: string;
  object: string;
  created: number;
  model: string;
  choices: OpenAIChoice[];
  usage?: OpenAIUsage;
}

export interface OpenAIChoice {
  index: number;
  message: OpenAIMessage;
  finish_reason: string | null;
}

export interface OpenAIStreamChunk {
  id: string;
  object: string;
  created: number;
  model: string;
  choices: OpenAIStreamChoice[];
  usage?: OpenAIUsage;
}

export interface OpenAIStreamChoice {
  index: number;
  delta: OpenAIStreamDelta;
  finish_reason: string | null;
}

export interface OpenAIStreamDelta {
  role?: string;
  content?: string | null;
  reasoning_content?: string | null;
  tool_calls?: OpenAIStreamToolCall[];
}

export interface OpenAIStreamToolCall {
  index: number;
  id?: string;
  type?: 'function';
  function?: {
    name?: string;
    arguments?: string;
  };
}

// --- Accumulator state for streaming ---

export interface StreamAccumulatorState {
  textBuffer: string;
  thoughtBuffer: string;
  toolCallBuffers: Map<
    number,
    { id: string; name: string; argsBuffer: string }
  >;
  finished: boolean;
  finishReason: string | null;
  role: string;
  responseId: string;
  model: string;
  yieldedFinish: boolean;
  usage?: OpenAIUsage;
}

export function createStreamAccumulator(): StreamAccumulatorState {
  return {
    textBuffer: '',
    thoughtBuffer: '',
    toolCallBuffers: new Map(),
    finished: false,
    finishReason: null,
    role: 'model',
    responseId: '',
    model: '',
    yieldedFinish: false,
  };
}

// --- Gemini -> OpenAI translation ---

let toolCallIdCounter = 0;

/**
 * Generate a unique tool call ID for OpenAI format.
 */
function generateToolCallId(): string {
  return `call_${Date.now()}_${++toolCallIdCounter}`;
}

/**
 * Extract system instruction text from Gemini's ContentUnion type.
 */
function extractSystemInstruction(
  systemInstruction?: string | Content,
): string | undefined {
  if (!systemInstruction) return undefined;
  if (typeof systemInstruction === 'string') return systemInstruction;
  if (systemInstruction.parts) {
    return systemInstruction.parts
      .filter((p) => p.text && !p.thought)
      .map((p) => p.text)
      .join('\n');
  }
  return undefined;
}

/**
 * Convert Gemini Content[] to OpenAI messages[].
 */
export function geminiContentsToOpenAIMessages(
  contents: Content[],
  systemInstruction?: string | Content,
): OpenAIMessage[] {
  const messages: OpenAIMessage[] = [];

  // Add system message first if present
  const sysText = extractSystemInstruction(systemInstruction);
  if (sysText) {
    messages.push({ role: 'system', content: sysText });
  }

  for (const content of contents) {
    const role = content.role;
    const parts = content.parts ?? [];

    if (role === 'user') {
      // Separate function responses from text/image parts
      const functionResponseParts = parts.filter((p) => p.functionResponse);
      const otherParts = parts.filter((p) => !p.functionResponse);

      // Function responses become tool messages
      for (const part of functionResponseParts) {
        if (part.functionResponse) {
          const fr = part.functionResponse;
          const output = fr.response
            ? typeof fr.response === 'string'
              ? fr.response
              : JSON.stringify(fr.response)
            : '';
          messages.push({
            role: 'tool',
            tool_call_id: fr.id || '',
            content: output,
          });
        }
      }

      // Text and image parts become user messages
      if (otherParts.length > 0) {
        const hasImages = otherParts.some(
          (p) => p.inlineData || p.fileData,
        );

        if (hasImages) {
          // Use array format for multimodal content
          const contentParts: OpenAIContentPart[] = [];
          for (const part of otherParts) {
            if (part.text && !part.thought) {
              contentParts.push({ type: 'text', text: part.text });
            } else if (part.inlineData) {
              contentParts.push({
                type: 'image_url',
                image_url: {
                  url: `data:${part.inlineData.mimeType || 'application/octet-stream'};base64,${part.inlineData.data || ''}`,
                },
              });
            } else if (part.fileData?.fileUri) {
              contentParts.push({
                type: 'image_url',
                image_url: { url: part.fileData.fileUri },
              });
            }
          }
          if (contentParts.length > 0) {
            messages.push({ role: 'user', content: contentParts });
          }
        } else {
          // Simple text content
          const textParts = otherParts
            .filter((p) => p.text && !p.thought)
            .map((p) => p.text!)
            .join('\n');
          if (textParts) {
            messages.push({ role: 'user', content: textParts });
          }
        }
      }
    } else if (role === 'model') {
      const textParts = parts
        .filter((p) => p.text && !p.thought)
        .map((p) => p.text!)
        .join('\n');
      const functionCallParts = parts.filter((p) => p.functionCall);

      if (functionCallParts.length > 0) {
        const toolCalls: OpenAIToolCall[] = functionCallParts.map((p) => {
          const fc = p.functionCall!;
          return {
            id: fc.id || generateToolCallId(),
            type: 'function' as const,
            function: {
              name: fc.name || '',
              arguments: JSON.stringify(fc.args ?? {}),
            },
          };
        });
        messages.push({
          role: 'assistant',
          ...(textParts ? { content: textParts } : {}),
          tool_calls: toolCalls,
        });
      } else if (textParts) {
        messages.push({ role: 'assistant', content: textParts });
      }
    }
  }

  return messages;
}

/**
 * Convert Gemini Tool[] to OpenAI tools[].
 */
export function geminiToolsToOpenAITools(tools?: Tool[]): OpenAITool[] | undefined {
  if (!tools || tools.length === 0) return undefined;

  const openaiTools: OpenAITool[] = [];
  for (const tool of tools) {
    if (tool.functionDeclarations) {
      for (const fd of tool.functionDeclarations) {
        // Gemini CLI stores schemas in parametersJsonSchema, while the
        // @google/genai SDK's FunctionDeclaration has both `parameters`
        // (Schema type) and `parametersJsonSchema`. Prefer the latter since
        // it's what the Gemini CLI tools actually populate.
        const schema =
          (fd.parametersJsonSchema as Record<string, unknown>) ||
          (fd.parameters as Record<string, unknown>);
        openaiTools.push({
          type: 'function',
          function: {
            name: fd.name || '',
            description: fd.description || '',
            ...(schema && { parameters: schema }),
          },
        });
      }
    }
  }

  return openaiTools.length > 0 ? openaiTools : undefined;
}

/**
 * Convert Gemini GenerateContentConfig to OpenAI request params.
 */
export function geminiConfigToOpenAIParams(
  config?: GenerateContentConfig,
): Partial<OpenAIRequestParams> {
  if (!config) return {};

  const params: Partial<OpenAIRequestParams> = {};

  if (config.temperature !== undefined) params.temperature = config.temperature;
  if (config.topP !== undefined) params.top_p = config.topP;
  if (config.maxOutputTokens !== undefined)
    params.max_tokens = config.maxOutputTokens;
  if (config.stopSequences) params.stop = config.stopSequences;
  if (config.seed !== undefined) params.seed = config.seed;
  if (config.presencePenalty !== undefined)
    params.presence_penalty = config.presencePenalty;
  if (config.frequencyPenalty !== undefined)
    params.frequency_penalty = config.frequencyPenalty;

  if (config.responseMimeType === 'application/json') {
    params.response_format = { type: 'json_object' };
  }

  // Translate Gemini toolConfig to OpenAI tool_choice
  const toolConfig = config.toolConfig;
  if (toolConfig?.functionCallingConfig) {
    const fcc = toolConfig.functionCallingConfig;
    const mode = fcc.mode;
    if (mode === 'NONE') {
      params.tool_choice = 'none';
    } else if (mode === 'ANY') {
      if (fcc.allowedFunctionNames?.length === 1) {
        params.tool_choice = {
          type: 'function',
          function: { name: fcc.allowedFunctionNames[0] },
        };
      } else {
        params.tool_choice = 'required';
      }
    }
    // AUTO is the OpenAI default, so we skip it
  }

  return params;
}

// --- OpenAI -> Gemini translation ---

/**
 * Map OpenAI finish_reason to Gemini FinishReason.
 */
export function mapOpenAIFinishReason(
  openaiReason: string | null,
): FinishReason {
  switch (openaiReason) {
    case 'stop':
      return FinishReason.STOP;
    case 'length':
      return FinishReason.MAX_TOKENS;
    case 'tool_calls':
      return FinishReason.STOP;
    case 'content_filter':
      return FinishReason.SAFETY;
    default:
      return FinishReason.STOP;
  }
}

/**
 * Convert an OpenAI ChatCompletion response to a Gemini GenerateContentResponse.
 */
export function openaiResponseToGeminiResponse(
  response: OpenAIChatCompletion,
): GenerateContentResponse {
  const choice = response.choices?.[0];
  const parts: Part[] = [];

  if (choice?.message) {
    // Handle reasoning_content as Gemini thought parts
    if (choice.message.reasoning_content) {
      parts.push({ text: choice.message.reasoning_content, thought: true });
    }
    if (choice.message.content) {
      if (typeof choice.message.content === 'string') {
        parts.push({ text: choice.message.content });
      } else if (Array.isArray(choice.message.content)) {
        // Extract text from content parts
        const text = choice.message.content
          .filter((p) => p.type === 'text')
          .map((p) => (p as { type: 'text'; text: string }).text)
          .join('\n');
        if (text) parts.push({ text });
      }
    }
    if (choice.message.tool_calls) {
      for (const tc of choice.message.tool_calls) {
        let args: Record<string, unknown> = {};
        try {
          args = JSON.parse(tc.function.arguments);
        } catch {
          // If arguments aren't valid JSON, pass as-is
          args = { raw: tc.function.arguments };
        }
        parts.push({
          functionCall: {
            id: tc.id,
            name: tc.function.name,
            args,
          } as FunctionCall,
        });
      }
    }
  }

  const candidate: Candidate = {
    content: {
      role: 'model',
      parts,
    },
    finishReason: choice ? mapOpenAIFinishReason(choice.finish_reason) : undefined,
    index: 0,
  };

  const geminiResponse = new GenerateContentResponse();
  geminiResponse.candidates = [candidate];
  geminiResponse.usageMetadata = response.usage
    ? {
        promptTokenCount: response.usage.prompt_tokens,
        candidatesTokenCount: response.usage.completion_tokens,
        totalTokenCount: response.usage.total_tokens,
        ...(response.usage.prompt_tokens_details?.cached_tokens !== undefined && {
          cachedContentTokenCount: response.usage.prompt_tokens_details.cached_tokens,
        }),
        ...(response.usage.completion_tokens_details?.reasoning_tokens !== undefined && {
          thoughtsTokenCount: response.usage.completion_tokens_details.reasoning_tokens,
        }),
      }
    : undefined;
  geminiResponse.responseId = response.id;
  geminiResponse.modelVersion = response.model;

  return geminiResponse;
}

/**
 * Process an OpenAI stream chunk and update the accumulator state.
 * Returns a GenerateContentResponse if there's new content to yield, or null.
 */
export function openaiStreamChunkToGeminiResponse(
  chunk: OpenAIStreamChunk,
  state: StreamAccumulatorState,
): GenerateContentResponse | null {
  // Update state from chunk — must happen before the early return so that
  // usage-only chunks (empty choices, sent when stream_options.include_usage
  // is true) still capture token counts.
  if (chunk.id) state.responseId = chunk.id;
  if (chunk.model) state.model = chunk.model;
  if (chunk.usage) state.usage = chunk.usage;

  const choice = chunk.choices?.[0];
  if (!choice) return null;

  const delta = choice.delta;
  let newDeltaText: string | null = null;
  let newDeltaThought: string | null = null;

  // Accumulate reasoning_content as thoughts (MiMo / DeepSeek style)
  if (delta?.reasoning_content) {
    state.thoughtBuffer += delta.reasoning_content;
    newDeltaThought = delta.reasoning_content;
  }

  // Accumulate text and capture the delta
  if (delta?.content) {
    state.textBuffer += delta.content;
    newDeltaText = delta.content;
  }

  // Accumulate tool calls
  if (delta?.tool_calls) {
    for (const tc of delta.tool_calls) {
      const idx = tc.index;
      let buffer = state.toolCallBuffers.get(idx);
      if (!buffer) {
        buffer = { id: '', name: '', argsBuffer: '' };
        state.toolCallBuffers.set(idx, buffer);
      }
      if (tc.id) buffer.id = tc.id;
      if (tc.function?.name) buffer.name = tc.function.name;
      if (tc.function?.arguments) {
        buffer.argsBuffer += tc.function.arguments;
      }
    }
  }

  // Check for finish
  if (choice.finish_reason) {
    state.finished = true;
    state.finishReason = choice.finish_reason;
  }

  // Build response parts - yield only the DELTA, not the full buffer
  const parts: Part[] = [];

  // Emit reasoning_content as Gemini thought parts (thought: true).
  // Use the full accumulated buffer to match Gemini's native behavior
  // where each chunk contains the complete thought text so far.
  if (newDeltaThought && state.thoughtBuffer) {
    parts.push({ text: state.thoughtBuffer, thought: true });
  }

  if (newDeltaText) {
    parts.push({ text: newDeltaText });
  }

  // Only include tool calls when the stream is finished (they need complete arguments)
  if (state.finished && state.toolCallBuffers.size > 0) {
    for (const [, buffer] of state.toolCallBuffers) {
      let args: Record<string, unknown> = {};
      try {
        args = JSON.parse(buffer.argsBuffer);
      } catch {
        args = { raw: buffer.argsBuffer };
      }
      parts.push({
        functionCall: {
          id: buffer.id || undefined,
          name: buffer.name,
          args,
        } as FunctionCall,
      });
    }
  }

  // If we're yielding a finished response, clear the state so the [DONE]
  // handler in parseSSEStream doesn't double-yield the same tool calls.
  if (parts.length > 0 && state.finished) {
    state.yieldedFinish = true;
    state.toolCallBuffers.clear();
  }

  // Don't yield if there's nothing new UNLESS we just finished
  // (the finishReason is critical for downstream validation)
  if (parts.length === 0 && !state.finished) return null;
  if (parts.length === 0 && !state.finished && state.toolCallBuffers.size === 0) return null;

  const candidate: Candidate = {
    content: {
      role: 'model',
      parts,
    },
    finishReason: state.finished
      ? mapOpenAIFinishReason(state.finishReason)
      : undefined,
    index: 0,
  };

  const geminiResponse = new GenerateContentResponse();
  geminiResponse.candidates = [candidate];
  geminiResponse.usageMetadata = state.usage
    ? {
        promptTokenCount: state.usage.prompt_tokens,
        candidatesTokenCount: state.usage.completion_tokens,
        totalTokenCount: state.usage.total_tokens,
        ...(state.usage.prompt_tokens_details?.cached_tokens !== undefined && {
          cachedContentTokenCount: state.usage.prompt_tokens_details.cached_tokens,
        }),
        ...(state.usage.completion_tokens_details?.reasoning_tokens !== undefined && {
          thoughtsTokenCount: state.usage.completion_tokens_details.reasoning_tokens,
        }),
      }
    : undefined;
  geminiResponse.responseId = state.responseId || undefined;
  geminiResponse.modelVersion = state.model || undefined;

  return geminiResponse;
}

/**
 * Estimate token count from text (rough approximation).
 * Most OpenAI-compatible endpoints don't expose a tokenization API.
 */
export function estimateTokenCount(text: string): number {
  // Rough approximation: ~4 characters per token for English
  return Math.ceil(text.length / 4);
}
