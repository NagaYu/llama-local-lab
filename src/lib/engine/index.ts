'use client';

import { useCallback, useEffect, useRef, useState } from 'react';
import type { EngineBackend, EngineStatus } from '@/lib/types';
import { detectCapability } from './capability';
import { MockEngine } from './mock-engine';
import { WebLlmEngine } from './webllm-engine';
import { WllamaEngine } from './wllama-engine';
import type { GenerateOptions, GenerationDelta, GpuCapability, LocalEngine } from './types';

export * from './types';
export * from './tools';
export { detectCapability, canFitModel } from './capability';
export { WLLAMA_MODELS } from './wllama-engine';
export { evictModel, isModelCached } from './webllm-engine';

/**
 * Single shared engine for the whole app.
 *
 * The playground and the evaluation suite both want to run the model, and
 * loading two copies of a multi-hundred-megabyte set of weights into WebGPU is
 * not an option. Keeping the instance at module scope means switching tabs
 * mid-session does not evict it, and every consumer sees the same status.
 */
let current: LocalEngine | null = null;
let currentKey: string | null = null;
let loadPromise: Promise<LocalEngine> | null = null;

type Listener = (status: EngineStatus) => void;
const listeners = new Set<Listener>();
let status: EngineStatus = { phase: 'idle' };

function setStatus(next: EngineStatus) {
  status = next;
  listeners.forEach((l) => l(next));
}

function keyFor(backend: EngineBackend, modelId: string) {
  return `${backend}:${modelId}`;
}

export async function loadEngine(
  backend: EngineBackend,
  modelId: string,
  signal?: AbortSignal,
): Promise<LocalEngine> {
  const key = keyFor(backend, modelId);
  if (current && currentKey === key) return current;
  if (loadPromise && currentKey === key) return loadPromise;

  if (current) {
    await current.unload();
    current = null;
  }

  currentKey = key;
  const started = performance.now();
  setStatus({ phase: 'loading', progress: 0, text: 'Preparing…' });

  const onProgress = (progress: number, text: string) => {
    setStatus({ phase: 'loading', progress, text });
  };

  loadPromise = (async () => {
    if (backend === 'mock') {
      // A short simulated load so the progress UI is exercised in demo mode.
      for (const p of [0.25, 0.6, 1]) {
        await new Promise((r) => setTimeout(r, 90));
        onProgress(p, p < 1 ? 'Warming up demo engine…' : 'Demo engine ready');
      }
      return new MockEngine();
    }
    if (backend === 'wllama') {
      return WllamaEngine.load(modelId, onProgress, signal);
    }
    return WebLlmEngine.load(modelId, onProgress, signal);
  })();

  try {
    const engine = await loadPromise;
    current = engine;
    setStatus({
      phase: 'ready',
      modelId: engine.modelId,
      backend: engine.backend,
      loadMs: performance.now() - started,
    });
    return engine;
  } catch (err) {
    current = null;
    currentKey = null;
    const message = err instanceof Error ? err.message : String(err);
    if (err instanceof DOMException && err.name === 'AbortError') {
      setStatus({ phase: 'idle' });
    } else {
      setStatus({ phase: 'error', message: humanizeLoadError(message) });
    }
    throw err;
  } finally {
    loadPromise = null;
  }
}

function humanizeLoadError(message: string): string {
  if (/out of memory|OOM|allocation/i.test(message)) {
    return `The GPU ran out of memory loading this model. Try a smaller model or close other WebGPU tabs. (${message})`;
  }
  if (/fetch|network|Failed to load|NetworkError/i.test(message)) {
    return `Could not download the model weights. Check your connection, or use the demo engine to explore the studio offline. (${message})`;
  }
  if (/WebGPU|adapter|device/i.test(message)) {
    return `WebGPU could not start. Try the WASM backend or the demo engine. (${message})`;
  }
  return message;
}

export async function unloadEngine() {
  if (current) {
    await current.unload();
  }
  current = null;
  currentKey = null;
  setStatus({ phase: 'idle' });
}

export function getEngine(): LocalEngine | null {
  return current;
}

/**
 * Run a generation against the loaded engine.
 * Yields deltas; the final yield carries the stats.
 */
export async function* generate(options: GenerateOptions): AsyncGenerator<GenerationDelta> {
  const engine = current;
  if (!engine) throw new Error('No engine loaded.');
  setStatus({ phase: 'generating' });
  try {
    for await (const delta of engine.generate(options)) {
      yield delta;
    }
  } finally {
    if (current) {
      setStatus({
        phase: 'ready',
        modelId: current.modelId,
        backend: current.backend,
        loadMs: 0,
      });
    }
  }
}

/** Clear conversational state between evaluation cases. */
export async function resetChat() {
  if (current instanceof WebLlmEngine) {
    await current.resetChat();
  }
}

/* -------------------------------------------------------------------------- */
/*  React bindings                                                             */
/* -------------------------------------------------------------------------- */

export function useEngineStatus(): EngineStatus {
  const [value, setValue] = useState<EngineStatus>(status);
  useEffect(() => {
    setValue(status);
    listeners.add(setValue);
    return () => {
      listeners.delete(setValue);
    };
  }, []);
  return value;
}

export function useCapability(): { capability: GpuCapability | null; checking: boolean } {
  const [capability, setCapability] = useState<GpuCapability | null>(null);
  const [checking, setChecking] = useState(true);

  useEffect(() => {
    let cancelled = false;
    detectCapability()
      .then((cap) => {
        if (!cancelled) setCapability(cap);
      })
      .finally(() => {
        if (!cancelled) setChecking(false);
      });
    return () => {
      cancelled = true;
    };
  }, []);

  return { capability, checking };
}

export interface UseEngine {
  status: EngineStatus;
  ready: boolean;
  busy: boolean;
  load: (backend: EngineBackend, modelId: string) => Promise<void>;
  unload: () => Promise<void>;
  cancelLoad: () => void;
  backend: EngineBackend | null;
  modelId: string | null;
}

export function useEngine(): UseEngine {
  const engineStatus = useEngineStatus();
  const abortRef = useRef<AbortController | null>(null);

  const load = useCallback(async (backend: EngineBackend, modelId: string) => {
    abortRef.current?.abort();
    const controller = new AbortController();
    abortRef.current = controller;
    try {
      await loadEngine(backend, modelId, controller.signal);
    } catch {
      // Status already reflects the failure; swallow so callers do not need
      // a try/catch around every button handler.
    }
  }, []);

  const cancelLoad = useCallback(() => {
    abortRef.current?.abort();
    abortRef.current = null;
  }, []);

  useEffect(
    () => () => {
      abortRef.current?.abort();
    },
    [],
  );

  const active = getEngine();

  return {
    status: engineStatus,
    ready: engineStatus.phase === 'ready' || engineStatus.phase === 'generating',
    busy: engineStatus.phase === 'loading' || engineStatus.phase === 'generating',
    load,
    unload: unloadEngine,
    cancelLoad,
    backend: active?.backend ?? null,
    modelId: active?.modelId ?? null,
  };
}
