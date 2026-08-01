import type { ChatMessage, EngineBackend, GenerationStats, SamplingParams } from '@/lib/types';

export interface GenerateOptions {
  messages: { role: 'system' | 'user' | 'assistant'; content: string }[];
  sampling: SamplingParams;
  /** Tool definitions in OpenAI function-calling shape. */
  tools?: unknown[];
  signal?: AbortSignal;
}

export interface GenerationDelta {
  /** Incremental text since the previous delta. */
  delta: string;
  /** Full text accumulated so far. */
  text: string;
  done: boolean;
  stats?: GenerationStats;
}

export interface LocalEngine {
  readonly backend: EngineBackend;
  readonly modelId: string;
  generate(options: GenerateOptions): AsyncGenerator<GenerationDelta>;
  unload(): Promise<void>;
}

export type ProgressHandler = (progress: number, text: string) => void;

export interface GpuCapability {
  webgpu: boolean;
  wasm: boolean;
  /** Human-readable adapter description, when the browser exposes one. */
  adapter?: string;
  /** Maximum single buffer the adapter will allocate, in bytes. */
  maxBufferSize?: number;
  /** Maximum total storage buffer binding size, in bytes. */
  maxStorageBufferBindingSize?: number;
  /** Set when WebGPU is unavailable — shown verbatim in the UI. */
  reason?: string;
  /** True when SharedArrayBuffer is available (needed for wllama multi-thread). */
  crossOriginIsolated: boolean;
  hardwareConcurrency: number;
}

export type { ChatMessage, EngineBackend, GenerationStats, SamplingParams };
