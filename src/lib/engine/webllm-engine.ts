'use client';

import type { MLCEngine } from '@mlc-ai/web-llm';
import type { GenerateOptions, GenerationDelta, LocalEngine, ProgressHandler } from './types';

/**
 * WebLLM reports real prefill/decode rates alongside the OpenAI-shaped usage
 * block, under a non-standard `extra` key. Both are optional at the type level
 * because older engine versions omit them.
 */
interface WebLlmUsage {
  prompt_tokens?: number;
  completion_tokens?: number;
  extra?: {
    decode_tokens_per_s?: number;
    prefill_tokens_per_s?: number;
    time_to_first_token_s?: number;
  };
}

/**
 * WebLLM (MLC) backend — compiles the model to WebGPU shaders and runs entirely
 * in the tab. Weights are cached in the browser's Cache Storage, so the second
 * load of a model is near-instant and works offline.
 */
export class WebLlmEngine implements LocalEngine {
  readonly backend = 'webllm' as const;

  private constructor(
    private engine: MLCEngine,
    readonly modelId: string,
  ) {}

  static async load(
    modelId: string,
    onProgress: ProgressHandler,
    signal?: AbortSignal,
  ): Promise<WebLlmEngine> {
    // Dynamic import keeps the ~1 MB WebLLM bundle out of the initial payload
    // and off the static-export prerender path, where `navigator` is undefined.
    const webllm = await import('@mlc-ai/web-llm');

    if (signal?.aborted) throw new DOMException('Aborted', 'AbortError');

    const engine = await webllm.CreateMLCEngine(modelId, {
      initProgressCallback: (report) => {
        onProgress(report.progress, report.text);
      },
      logLevel: 'WARN',
    });

    if (signal?.aborted) {
      await engine.unload();
      throw new DOMException('Aborted', 'AbortError');
    }

    return new WebLlmEngine(engine, modelId);
  }

  async *generate(options: GenerateOptions): AsyncGenerator<GenerationDelta> {
    const { sampling } = options;
    const started = performance.now();

    const stream = await this.engine.chat.completions.create({
      messages: options.messages as Parameters<
        typeof this.engine.chat.completions.create
      >[0]['messages'],
      stream: true,
      stream_options: { include_usage: true },
      temperature: sampling.temperature,
      top_p: sampling.topP,
      max_tokens: sampling.maxTokens,
      presence_penalty: sampling.presencePenalty,
      frequency_penalty: sampling.frequencyPenalty,
      ...(sampling.seed !== null ? { seed: sampling.seed } : {}),
    });

    let text = '';
    let firstTokenAt: number | null = null;
    let usage: WebLlmUsage | null = null;

    for await (const chunk of stream) {
      if (options.signal?.aborted) {
        await this.engine.interruptGenerate();
        break;
      }

      const delta = chunk.choices?.[0]?.delta?.content ?? '';
      if (delta) {
        if (firstTokenAt === null) firstTokenAt = performance.now();
        text += delta;
        yield { delta, text, done: false };
      }

      // The final chunk carries usage when `include_usage` is set.
      if (chunk.usage) {
        usage = chunk.usage as WebLlmUsage;
      }
    }

    const finished = performance.now();
    const completionTokens =
      usage?.completion_tokens ?? Math.max(1, Math.round(text.length / 4));
    const decodeMs = Math.max(1, finished - (firstTokenAt ?? started));

    yield {
      delta: '',
      text,
      done: true,
      stats: {
        promptTokens: usage?.prompt_tokens ?? 0,
        completionTokens,
        ttftMs: usage?.extra?.time_to_first_token_s
          ? usage.extra.time_to_first_token_s * 1000
          : (firstTokenAt ?? finished) - started,
        tokensPerSecond:
          usage?.extra?.decode_tokens_per_s ?? (completionTokens / decodeMs) * 1000,
        totalMs: finished - started,
        backend: 'webllm',
      },
    };
  }

  /** Drop the KV cache between evaluation cases so runs do not contaminate. */
  async resetChat() {
    try {
      await this.engine.resetChat();
    } catch {
      /* non-fatal */
    }
  }

  async unload() {
    try {
      await this.engine.unload();
    } catch {
      /* already gone */
    }
  }
}

/** Delete a cached model's weights from the browser so disk can be reclaimed. */
export async function evictModel(modelId: string): Promise<void> {
  const webllm = await import('@mlc-ai/web-llm');
  await webllm.deleteModelAllInfoInCache(modelId);
}

/** True when this model's weights are already in Cache Storage. */
export async function isModelCached(modelId: string): Promise<boolean> {
  try {
    const webllm = await import('@mlc-ai/web-llm');
    return await webllm.hasModelInCache(modelId);
  } catch {
    return false;
  }
}
