'use client';

import type { GenerateOptions, GenerationDelta, LocalEngine, ProgressHandler } from './types';

/**
 * llama.cpp-via-WebAssembly backend.
 *
 * This is the fallback for browsers with no usable WebGPU adapter. It is
 * meaningfully slower than WebLLM — CPU SIMD instead of GPU shaders — but it
 * runs real GGUF files, which means the quantization formats planned elsewhere
 * in this studio are the exact same artifacts this engine executes.
 *
 * The WASM binaries are pulled from jsDelivr so the app stays a pure static
 * export with nothing to host. Multi-threading needs `SharedArrayBuffer`, which
 * needs cross-origin isolation headers; without them we fall back to the
 * single-threaded build automatically.
 */

const WLLAMA_VERSION = '2.4.0';
const CDN = `https://cdn.jsdelivr.net/npm/@wllama/wllama@${WLLAMA_VERSION}/esm`;

const WASM_PATHS = {
  'single-thread/wllama.wasm': `${CDN}/single-thread/wllama.wasm`,
  'multi-thread/wllama.wasm': `${CDN}/multi-thread/wllama.wasm`,
};

/** Small GGUF models that download fast enough to be worth offering in a tab. */
export interface WllamaModelOption {
  id: string;
  label: string;
  repo: string;
  file: string;
  sizeMb: number;
  contextLength: number;
}

export const WLLAMA_MODELS: WllamaModelOption[] = [
  {
    id: 'llama-3.2-1b-q4km',
    label: 'Llama 3.2 1B Instruct · Q4_K_M',
    repo: 'bartowski/Llama-3.2-1B-Instruct-GGUF',
    file: 'Llama-3.2-1B-Instruct-Q4_K_M.gguf',
    sizeMb: 808,
    contextLength: 4096,
  },
  {
    id: 'llama-3.2-1b-q8',
    label: 'Llama 3.2 1B Instruct · Q8_0',
    repo: 'bartowski/Llama-3.2-1B-Instruct-GGUF',
    file: 'Llama-3.2-1B-Instruct-Q8_0.gguf',
    sizeMb: 1320,
    contextLength: 4096,
  },
  {
    id: 'llama-3.2-3b-q4km',
    label: 'Llama 3.2 3B Instruct · Q4_K_M',
    repo: 'bartowski/Llama-3.2-3B-Instruct-GGUF',
    file: 'Llama-3.2-3B-Instruct-Q4_K_M.gguf',
    sizeMb: 2020,
    contextLength: 4096,
  },
];

type WllamaInstance = {
  loadModelFromHF(
    repo: string,
    file: string,
    config?: Record<string, unknown>,
  ): Promise<void>;
  createChatCompletion(
    messages: { role: string; content: string }[],
    options: Record<string, unknown>,
  ): Promise<string>;
  exit(): Promise<void>;
};

export class WllamaEngine implements LocalEngine {
  readonly backend = 'wllama' as const;

  private constructor(
    private wllama: WllamaInstance,
    readonly modelId: string,
  ) {}

  static async load(
    optionId: string,
    onProgress: ProgressHandler,
    signal?: AbortSignal,
  ): Promise<WllamaEngine> {
    const option = WLLAMA_MODELS.find((m) => m.id === optionId) ?? WLLAMA_MODELS[0];

    // The published entry point is the built ESM bundle; the package root is
    // raw TypeScript and will not resolve through webpack.
    const mod = (await import(
      /* webpackIgnore: false */ '@wllama/wllama/esm/index.js'
    )) as { Wllama: new (paths: unknown, config?: unknown) => WllamaInstance };

    const threads = Math.min(
      8,
      Math.max(1, Math.floor((navigator.hardwareConcurrency || 4) / 2)),
    );

    const wllama = new mod.Wllama(WASM_PATHS, {
      // Without cross-origin isolation SharedArrayBuffer is unavailable and
      // the multi-threaded build cannot start.
      allowOffline: true,
      suppressNativeLog: true,
    });

    onProgress(0, `Downloading ${option.file} (~${option.sizeMb} MB)…`);

    await wllama.loadModelFromHF(option.repo, option.file, {
      n_threads: threads,
      n_ctx: option.contextLength,
      progressCallback: ({ loaded, total }: { loaded: number; total: number }) => {
        if (signal?.aborted) return;
        const pct = total > 0 ? loaded / total : 0;
        onProgress(
          pct,
          `Downloading ${option.file} — ${(loaded / 1e6).toFixed(0)} / ${(total / 1e6).toFixed(0)} MB`,
        );
      },
    });

    if (signal?.aborted) {
      await wllama.exit();
      throw new DOMException('Aborted', 'AbortError');
    }

    onProgress(1, 'Model ready');
    return new WllamaEngine(wllama, option.id);
  }

  async *generate(options: GenerateOptions): AsyncGenerator<GenerationDelta> {
    const started = performance.now();
    let firstTokenAt: number | null = null;
    let text = '';

    // wllama's completion API is callback-based, so bridge it into an async
    // iterator with a small queue rather than buffering the whole response.
    const queue: string[] = [];
    let resolveNext: (() => void) | null = null;
    let finished = false;
    let failure: unknown = null;

    const push = (chunk: string) => {
      queue.push(chunk);
      resolveNext?.();
      resolveNext = null;
    };

    const completion = this.wllama
      .createChatCompletion(options.messages, {
        nPredict: options.sampling.maxTokens,
        sampling: {
          temp: options.sampling.temperature,
          top_p: options.sampling.topP,
          penalty_repeat: options.sampling.repetitionPenalty,
          ...(options.sampling.seed !== null ? { seed: options.sampling.seed } : {}),
        },
        onNewToken: (_token: number, _piece: Uint8Array, currentText: string) => {
          if (firstTokenAt === null) firstTokenAt = performance.now();
          const delta = currentText.slice(text.length);
          text = currentText;
          if (delta) push(delta);
        },
        abortSignal: options.signal,
      })
      .then(() => {
        finished = true;
        resolveNext?.();
        resolveNext = null;
      })
      .catch((err: unknown) => {
        failure = err;
        finished = true;
        resolveNext?.();
        resolveNext = null;
      });

    let emitted = '';
    while (!finished || queue.length > 0) {
      if (queue.length === 0) {
        await new Promise<void>((resolve) => {
          resolveNext = resolve;
        });
        continue;
      }
      const delta = queue.shift()!;
      emitted += delta;
      yield { delta, text: emitted, done: false };
    }

    await completion;
    if (failure && !options.signal?.aborted) {
      throw failure;
    }

    const end = performance.now();
    const completionTokens = Math.max(1, Math.round(emitted.length / 4));
    const decodeMs = Math.max(1, end - (firstTokenAt ?? started));

    yield {
      delta: '',
      text: emitted,
      done: true,
      stats: {
        promptTokens: Math.round(
          options.messages.reduce((n, m) => n + m.content.length, 0) / 4,
        ),
        completionTokens,
        ttftMs: (firstTokenAt ?? end) - started,
        tokensPerSecond: (completionTokens / decodeMs) * 1000,
        totalMs: end - started,
        backend: 'wllama',
      },
    };
  }

  async unload() {
    try {
      await this.wllama.exit();
    } catch {
      /* already gone */
    }
  }
}
