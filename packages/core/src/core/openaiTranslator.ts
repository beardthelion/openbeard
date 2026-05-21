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
  tools?: OpenAITool[];
  response_format?: { type: string };
  seed?: number;
  presence_penalty?: number;
  frequency_penalty?: number;
}

export interface OpenAIChatCompletion {
  id: string;
  object: string;
  created: number;
  model: string;
  choices: OpenAIChoice[];
  usage?: {
    prompt_tokens: number;
    completion_tokens: number;
    total_tokens: number;
  };
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
  usage?: {
    prompt_tokens: number;
    completion_tokens: number;
    total_tokens: number;
  };
}

export interface OpenAIStreamChoice {
  index: number;
  delta: OpenAIStreamDelta;
  finish_reason: string | null;
}

export interface OpenAIStreamDelta {
  role?: string;
  content?: string | null;
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
  toolCallBuffers: Map<
    number,
    { id: string; name: string; argsBuffer: string }
  >;
  finished: boolean;
  finishReason: string | null;
  role: string;
  responseId: string;
  model: string;
  usage?: {
    prompt_tokens: number;
    completion_tokens: number;
    total_tokens: number;
  };
}

export function createStreamAccumulator(): StreamAccumulatorState {
  return {
    textBuffer: '',
    toolCallBuffers: new Map(),
    finished: false,
    finishReason: null,
    role: 'model',
    responseId: '',
    model: '',
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
          content: textParts || null,
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
        openaiTools.push({
          type: 'function',
          function: {
            name: fd.name || '',
            description: fd.description || '',
            parameters: (fd.parameters as Record<string, unknown>) || undefined,
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
  const choice = chunk.choices?.[0];
  if (!choice) return null;

  // Update state from chunk
  if (chunk.id) state.responseId = chunk.id;
  if (chunk.model) state.model = chunk.model;
  if (chunk.usage) state.usage = chunk.usage;

  const delta = choice.delta;
  let hasNewContent = false;

  // Accumulate text
  if (delta?.content) {
    state.textBuffer += delta.content;
    hasNewContent = true;
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
        hasNewContent = true;
      }
    }
  }

  // Check for finish
  if (choice.finish_reason) {
    state.finished = true;
    state.finishReason = choice.finish_reason;
    hasNewContent = true;
  }

  if (!hasNewContent) return null;

  // Build response parts from current state
  const parts: Part[] = [];

  if (state.textBuffer) {
    parts.push({ text: state.textBuffer });
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
