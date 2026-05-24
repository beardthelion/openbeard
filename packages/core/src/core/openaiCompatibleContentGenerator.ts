/**
 * @license
 * Copyright 2025 Google LLC
 * SPDX-License-Identifier: Apache-2.0
 */

import type {
  CountTokensResponse,
  GenerateContentResponse,
  GenerateContentParameters,
  CountTokensParameters,
  EmbedContentResponse,
  EmbedContentParameters,
  ContentListUnion,
  ContentUnion,
  Content,
  Part,
  PartUnion,
} from '@google/genai';
import { HttpProxyAgent } from 'http-proxy-agent';
import { HttpsProxyAgent } from 'https-proxy-agent';
import type { ContentGenerator } from './contentGenerator.js';
import type { LlmRole } from '../telemetry/llmRole.js';
import {
  geminiContentsToOpenAIMessages,
  geminiToolsToOpenAITools,
  geminiConfigToOpenAIParams,
  openaiResponseToGeminiResponse,
  openaiStreamChunkToGeminiResponse,
  createStreamAccumulator,
  type OpenAIChatCompletion,
  type OpenAIStreamChunk,
  type OpenAIRequestParams,
} from './openaiTranslator.js';
import { estimateTokenCountSync } from '../utils/tokenCalculation.js';

export interface OpenAICompatibleContentGeneratorOptions {
  baseUrl: string;
  apiKey: string;
  model: string;
  headers?: Record<string, string>;
  proxy?: string;
}

/**
 * ContentGenerator implementation that translates between Gemini's internal
 * types and the OpenAI Chat Completions API format, allowing the Gemini CLI
 * to work with any OpenAI-compatible endpoint (OpenAI, Ollama, vLLM, LiteLLM, etc.).
 */
export class OpenAICompatibleContentGenerator implements ContentGenerator {
  private readonly baseUrl: string;
  private readonly apiKey: string;
  private readonly model: string;
  private readonly headers: Record<string, string>;
  private readonly proxy?: string;

  constructor(options: OpenAICompatibleContentGeneratorOptions) {
    // Normalize base URL: strip trailing slash
    this.baseUrl = options.baseUrl.replace(/\/+$/, '');
    this.apiKey = options.apiKey;
    this.model = options.model;
    this.headers = options.headers ?? {};
    this.proxy = options.proxy;
  }

  async generateContent(
    request: GenerateContentParameters,
    _userPromptId: string,
    _role: LlmRole,
  ): Promise<GenerateContentResponse> {
    const body = this.buildRequestBody(request, false);
    const response = await this.doFetch(body);
    const data = (await response.json()) as OpenAIChatCompletion;
    return openaiResponseToGeminiResponse(data);
  }

  async generateContentStream(
    request: GenerateContentParameters,
    _userPromptId: string,
    _role: LlmRole,
  ): Promise<AsyncGenerator<GenerateContentResponse>> {
    const body = this.buildRequestBody(request, true);
    const response = await this.doFetch(body);
    return this.parseSSEStream(response);
  }

  async countTokens(
    request: CountTokensParameters,
  ): Promise<CountTokensResponse> {
    const parts: Part[] = [];
    const contents = toContents(request.contents);
    for (const content of contents) {
      for (const part of content.parts ?? []) {
        parts.push(part);
      }
    }
    if (request.config?.systemInstruction) {
      const si = toContent(request.config.systemInstruction);
      for (const part of si.parts ?? []) {
        parts.push(part);
      }
    }
    return { totalTokens: estimateTokenCountSync(parts) };
  }

  async embedContent(
    _request: EmbedContentParameters,
  ): Promise<EmbedContentResponse> {
    throw new Error(
      'Embedding is not supported for OpenAI-compatible endpoints. ' +
        'Use a Gemini-native endpoint for embedding features.',
    );
  }

  private buildRequestBody(
    request: GenerateContentParameters,
    stream: boolean,
  ): OpenAIRequestParams {
    const contents = toContents(request.contents);
    const config = request.config;

    // Convert systemInstruction from ContentUnion to string | Content
    const systemInstruction = config?.systemInstruction
      ? toContent(config.systemInstruction)
      : undefined;

    const messages = geminiContentsToOpenAIMessages(
      contents,
      systemInstruction,
    );

    const tools = geminiToolsToOpenAITools(
      config?.tools as import('@google/genai').Tool[] | undefined,
    );
    const configParams = geminiConfigToOpenAIParams(config);

    const body: OpenAIRequestParams = {
      model: this.model,
      messages,
      ...configParams,
      ...(tools && { tools }),
      ...(stream && { stream: true, stream_options: { include_usage: true } }),
    };

    return body;
  }

  private async doFetch(body: OpenAIRequestParams): Promise<Response> {
    const url = `${this.baseUrl}/chat/completions`;

    const headers: Record<string, string> = {
      'Content-Type': 'application/json',
      'Accept-Encoding': 'identity',
      ...this.headers,
    };

    if (this.apiKey) {
      headers['Authorization'] = `Bearer ${this.apiKey}`;
    }

    // Set up proxy agent if configured
    const proxyUrl = this.proxy?.trim();
    let agent: HttpProxyAgent<string> | HttpsProxyAgent<string> | undefined;
    if (proxyUrl) {
      agent = this.baseUrl.startsWith('http://')
        ? new HttpProxyAgent(proxyUrl)
        : new HttpsProxyAgent(proxyUrl);
    }

    const fetchOptions: RequestInit = {
      method: 'POST',
      headers,
      body: JSON.stringify(body),
      ...(agent && { agent } as unknown as RequestInit),
    };

    const response = await fetch(url, fetchOptions);

    if (!response.ok) {
      const errorText = await response.text().catch(() => 'Unknown error');
      const hint = this.getErrorHint(response.status);
      throw new Error(
        `OpenAI API error ${response.status} ${response.statusText}: ${errorText}${hint}`,
      );
    }

    return response;
  }

  private getErrorHint(status: number): string {
    switch (status) {
      case 401:
        return '\n\nCheck your API key: set OPENAI_API_KEY or update ~/.openbeard/settings.json';
      case 403:
        return '\n\nAccess denied. Your API key may lack permissions for this model.';
      case 404:
        return `\n\nEndpoint not found. Verify OPENAI_BASE_URL (${this.baseUrl}) and model (${this.model}) are correct.`;
      case 429:
        return '\n\nRate limited. Wait a moment and try again, or check your plan limits.';
      case 500:
      case 502:
      case 503:
        return '\n\nServer error. The API endpoint may be temporarily unavailable.';
      default:
        return '';
    }
  }

  private async *parseSSEStream(
    response: Response,
  ): AsyncGenerator<GenerateContentResponse> {
    const reader = response.body?.getReader();
    if (!reader) {
      throw new Error('Response body is null, cannot parse SSE stream');
    }

    const decoder = new TextDecoder();
    const state = createStreamAccumulator();
    let buffer = '';

    try {
      while (true) {
        const { done, value } = await reader.read();
        if (done) break;

        buffer += decoder.decode(value, { stream: true });
        const lines = buffer.split('\n');
        // Keep the last incomplete line in the buffer
        buffer = lines.pop() ?? '';

        for (const line of lines) {
          const trimmed = line.trim();
          if (!trimmed || trimmed.startsWith(':')) continue; // skip empty lines and comments
          if (trimmed === 'data: [DONE]') {
            // Only yield a final response if the finish hasn't already been
            // yielded from the actual stream chunk (which happens when a
            // single chunk carries both tool_calls and finish_reason).
            if (!state.yieldedFinish) {
              if (state.textBuffer || state.thoughtBuffer || state.toolCallBuffers.size > 0) {
                const finalResponse = openaiStreamChunkToGeminiResponse(
                  {
                    id: state.responseId,
                    object: 'chat.completion.chunk',
                    created: 0,
                    model: state.model,
                    choices: [
                      {
                        index: 0,
                        delta: {},
                        finish_reason: state.finishReason || 'stop',
                      },
                    ],
                    ...(state.usage && { usage: state.usage }),
                  },
                  state,
                );
                if (finalResponse) yield finalResponse;
              }
            } else if (state.usage) {
              // The finish chunk was already yielded (before usage arrived),
              // but we now have usage data. Yield a minimal synthetic response
              // carrying usageMetadata so the Finished event reaches the
              // calibration system (TokenGroundTruthEvent).
              const usageResponse = openaiStreamChunkToGeminiResponse(
                {
                  id: state.responseId,
                  object: 'chat.completion.chunk',
                  created: 0,
                  model: state.model,
                  choices: [
                    {
                      index: 0,
                      delta: {},
                      finish_reason: state.finishReason || 'stop',
                    },
                  ],
                  usage: state.usage,
                },
                state,
              );
              if (usageResponse) yield usageResponse;
            }
            return;
          }

          if (trimmed.startsWith('data: ')) {
            const jsonStr = trimmed.slice(6);
            try {
              const chunk = JSON.parse(jsonStr) as OpenAIStreamChunk;
              const geminiResponse = openaiStreamChunkToGeminiResponse(
                chunk,
                state,
              );
              if (geminiResponse) {
                yield geminiResponse;
              }
            } catch {
              // Skip malformed JSON chunks
            }
          }
        }
      }

      // Process any remaining buffer
      if (buffer.trim()) {
        const trimmed = buffer.trim();
        if (
          trimmed.startsWith('data: ') &&
          trimmed !== 'data: [DONE]'
        ) {
          try {
            const chunk = JSON.parse(trimmed.slice(6)) as OpenAIStreamChunk;
            const geminiResponse = openaiStreamChunkToGeminiResponse(
              chunk,
              state,
            );
            if (geminiResponse) yield geminiResponse;
          } catch {
            // Skip malformed JSON
          }
        }
      }

      // Final yield if stream ended without [DONE]
      if (!state.yieldedFinish) {
        if (state.textBuffer || state.thoughtBuffer || state.toolCallBuffers.size > 0) {
          const finalResponse = openaiStreamChunkToGeminiResponse(
            {
              id: state.responseId,
              object: 'chat.completion.chunk',
              created: 0,
              model: state.model,
              choices: [
                {
                  index: 0,
                  delta: {},
                  finish_reason: state.finishReason || 'stop',
                },
              ],
              ...(state.usage && { usage: state.usage }),
            },
            state,
          );
          if (finalResponse) yield finalResponse;
        }
      } else if (state.usage) {
        // Finish already yielded — emit usage-only response for calibration
        const usageResponse = openaiStreamChunkToGeminiResponse(
          {
            id: state.responseId,
            object: 'chat.completion.chunk',
            created: 0,
            model: state.model,
            choices: [
              {
                index: 0,
                delta: {},
                finish_reason: state.finishReason || 'stop',
              },
            ],
            usage: state.usage,
          },
          state,
        );
        if (usageResponse) yield usageResponse;
      }
    } finally {
      reader.releaseLock();
    }
  }
}

// --- Content normalization helpers ---
// These handle Gemini's ContentListUnion / ContentUnion types which can be
// Content, Content[], Part, Part[], string, or string[].

function isContent(obj: unknown): obj is Content {
  return (
    typeof obj === 'object' &&
    obj !== null &&
    'parts' in obj &&
    'role' in obj
  );
}

function toPart(part: PartUnion): Part {
  if (typeof part === 'string') {
    return { text: part };
  }
  return part;
}

function toContent(content: ContentUnion): Content {
  if (typeof content === 'string') {
    return { role: 'user', parts: [{ text: content }] };
  }
  if (Array.isArray(content)) {
    return { role: 'user', parts: content.map(toPart) };
  }
  if (isContent(content)) {
    return content;
  }
  // It's a Part
  return { role: 'user', parts: [toPart(content as PartUnion)] };
}

function toContents(contents: ContentListUnion): Content[] {
  if (Array.isArray(contents)) {
    return contents.map((c) => {
      if (typeof c === 'string') {
        return { role: 'user', parts: [{ text: c }] } as Content;
      }
      if (isContent(c)) {
        return c;
      }
      // It's a PartUnion[]
      return { role: 'user', parts: (c as PartUnion[]).map(toPart) } as Content;
    });
  }
  if (typeof contents === 'string') {
    return [{ role: 'user', parts: [{ text: contents }] }];
  }
  if (isContent(contents)) {
    return [contents];
  }
  // Single Part
  return [{ role: 'user', parts: [contents as Part] }];
}
