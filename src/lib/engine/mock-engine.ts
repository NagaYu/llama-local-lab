'use client';

import type { GenerateOptions, GenerationDelta, LocalEngine } from './types';

/**
 * A deterministic, zero-download stand-in for a real model.
 *
 * The point of this engine is that every screen in the studio — the playground,
 * the token/s readout, the evaluation suite, the CSV export — is fully
 * demonstrable on a machine with no WebGPU, no network and no patience for a
 * 900 MB download. It streams token-by-token at a plausible rate and reports
 * honest-looking stats, and the UI labels it "Demo" everywhere so nobody
 * mistakes its output for a model's.
 */

const CANNED: { match: RegExp; reply: string }[] = [
  {
    match: /capital of france/i,
    reply: 'Paris.',
  },
  {
    match: /2\s*\+\s*2|two plus two/i,
    reply: '4',
  },
  {
    match: /largest planet/i,
    reply: 'Jupiter.',
  },
  {
    match: /who wrote romeo and juliet/i,
    reply: 'William Shakespeare.',
  },
  {
    match: /speed of light/i,
    reply: 'Approximately 299,792,458 meters per second in a vacuum.',
  },
  {
    match: /quantiz/i,
    reply:
      'Quantization stores each weight in fewer bits than the original 16-bit float. Q4_K_M keeps roughly 4.85 bits per weight and loses under 1% perplexity, which is why it is the default recommendation for almost every deployment. The embedding and output tensors are kept at higher precision because they dominate quality on models with large vocabularies.',
  },
  {
    match: /gguf/i,
    reply:
      'GGUF is the container format used by llama.cpp. It packs the tensors, the tokenizer and the chat template into one file, so a single artifact is enough to serve the model. You produce it by converting the HuggingFace checkpoint to F16 with convert_hf_to_gguf.py, then compressing that with llama-quantize.',
  },
  {
    match: /\b(hello|hi|hey)\b/i,
    reply:
      'Hello. I am the demo engine — a scripted stand-in that runs with no model download so you can see how the playground behaves. Load a real model with the button above for actual inference.',
  },
];

const FALLBACK = [
  'This is the demo engine. It streams a canned response so the playground, the tokens-per-second readout and the evaluation suite are all usable before you download a real model.',
  'Switch the backend to WebGPU and press Load model to run Llama 3.2 1B locally in this tab. Nothing leaves your machine either way.',
];

function pickReply(prompt: string): string {
  for (const c of CANNED) {
    if (c.match.test(prompt)) return c.reply;
  }
  const idx = prompt.length % FALLBACK.length;
  return FALLBACK[idx];
}

/** Split into word-ish chunks that read like real Llama tokens streaming in. */
function tokenize(text: string): string[] {
  return text.match(/\s*\S+/g) ?? [text];
}

export class MockEngine implements LocalEngine {
  readonly backend = 'mock' as const;
  readonly modelId = 'demo-engine';

  /** Simulated decode speed, chosen to look like a 1B model on a laptop GPU. */
  private readonly targetTps = 48;

  async *generate(options: GenerateOptions): AsyncGenerator<GenerationDelta> {
    const lastUser = [...options.messages].reverse().find((m) => m.role === 'user');
    const prompt = lastUser?.content ?? '';

    let reply = pickReply(prompt);

    // Honour function calling so the tool-call editor is demonstrable too.
    if (options.tools && options.tools.length > 0) {
      const tool = options.tools[0] as { function?: { name?: string; parameters?: unknown } };
      const name = tool?.function?.name;
      if (name && /weather|temperature|forecast/i.test(prompt + ' ' + name)) {
        reply = JSON.stringify(
          { name, parameters: { location: extractLocation(prompt) ?? 'San Francisco, CA', unit: 'celsius' } },
          null,
          2,
        );
      }
    }

    const started = performance.now();
    const promptTokens = Math.max(1, Math.round(options.messages.reduce((n, m) => n + m.content.length, 0) / 4));

    // Time to first token scales with prompt length, like a real prefill.
    const ttft = 60 + promptTokens * 0.6;
    await sleep(ttft, options.signal);
    const firstTokenAt = performance.now();

    const pieces = tokenize(reply);
    const maxPieces = Math.max(1, Math.min(pieces.length, Math.ceil(options.sampling.maxTokens / 1.3)));
    const perToken = 1000 / this.targetTps;

    let text = '';
    for (let i = 0; i < maxPieces; i += 1) {
      if (options.signal?.aborted) break;
      // Jitter so the tok/s readout is not suspiciously flat.
      await sleep(perToken * (0.7 + Math.random() * 0.6), options.signal);
      text += pieces[i];
      yield { delta: pieces[i], text, done: false };
    }

    const finished = performance.now();
    const completionTokens = Math.max(1, Math.round(text.length / 4));
    const decodeMs = Math.max(1, finished - firstTokenAt);

    yield {
      delta: '',
      text,
      done: true,
      stats: {
        promptTokens,
        completionTokens,
        ttftMs: firstTokenAt - started,
        tokensPerSecond: (completionTokens / decodeMs) * 1000,
        totalMs: finished - started,
        backend: 'mock',
      },
    };
  }

  async unload() {
    /* nothing to release */
  }
}

function extractLocation(prompt: string): string | null {
  const m = prompt.match(/\bin\s+([A-Z][A-Za-z .'-]+(?:,\s*[A-Z]{2})?)/);
  return m ? m[1].trim() : null;
}

function sleep(ms: number, signal?: AbortSignal): Promise<void> {
  return new Promise((resolve, reject) => {
    if (signal?.aborted) {
      resolve();
      return;
    }
    const t = setTimeout(resolve, ms);
    signal?.addEventListener(
      'abort',
      () => {
        clearTimeout(t);
        resolve();
      },
      { once: true },
    );
    void reject;
  });
}
