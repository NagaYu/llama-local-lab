/**
 * Shared domain types for Llama Local Lab.
 *
 * Everything here is pure data — no React, no browser APIs — so it can be
 * imported from server components, client components and web workers alike.
 */

/* -------------------------------------------------------------------------- */
/*  Models                                                                     */
/* -------------------------------------------------------------------------- */

export type ModelFamily =
  | 'Llama 4'
  | 'Llama 3.3'
  | 'Llama 3.2'
  | 'Llama 3.1'
  | 'Llama 3'
  | 'Llama Guard'
  | 'Code Llama';

export type ModelModality = 'text' | 'multimodal';

export interface ModelArchitecture {
  /** Transformer block count. Drives KV-cache size. */
  layers: number;
  /** Model dimension (a.k.a. d_model / hidden_size). */
  hiddenSize: number;
  /** Query head count. */
  attentionHeads: number;
  /** Key/value head count (GQA). Equals attentionHeads for MHA models. */
  kvHeads: number;
  /** Vocabulary size — Llama 3+ uses a 128k tokenizer. */
  vocabSize: number;
  /**
   * True when `lm_head` reuses `token_embd` (Llama 3.2 1B/3B do; 8B and up
   * do not). Halves the embedding contribution to file size.
   */
  tiedEmbeddings?: boolean;
  /** Set for mixture-of-experts models such as Llama 4. */
  moe?: {
    experts: number;
    activeExperts: number;
    /** Billions of parameters actually used per token. */
    activeParamsB: number;
  };
}

export interface LlamaModel {
  /** Stable slug used in URLs and localStorage. */
  id: string;
  /** HuggingFace repo id, e.g. `meta-llama/Llama-3.3-70B-Instruct`. */
  repoId: string;
  name: string;
  family: ModelFamily;
  /** Total parameters in billions. */
  paramsB: number;
  /** Maximum context window in tokens. */
  contextLength: number;
  modality: ModelModality;
  license: string;
  licenseUrl?: string;
  /** True when the HF repo requires accepting the Llama community license. */
  gated: boolean;
  releasedAt: string;
  description: string;
  architecture: ModelArchitecture;
  /** Languages officially supported by Meta for this checkpoint. */
  languages?: string[];
  /** Populated at runtime from the HuggingFace API; undefined when offline. */
  hub?: HubStats;
}

export interface HubStats {
  downloads?: number;
  likes?: number;
  lastModified?: string;
  pipelineTag?: string;
  tags?: string[];
  /** Where the numbers came from, so the UI can be honest about staleness. */
  source: 'huggingface' | 'bundled';
  fetchedAt?: string;
}

/* -------------------------------------------------------------------------- */
/*  Quantization                                                               */
/* -------------------------------------------------------------------------- */

export type QuantTier = 'lossless' | 'recommended' | 'aggressive' | 'extreme';

export interface QuantFormat {
  /** llama-quantize type name, e.g. `Q4_K_M`. */
  id: string;
  label: string;
  /** Effective bits per weight for the transformer blocks. */
  bpw: number;
  /**
   * Average bits per weight across `token_embd` + `output` when those are two
   * separate tensors (Llama 3.1 8B and up). `token_embd` is a pure lookup and
   * quantizes near the model's nominal rate; `output` is a full matmul against
   * the vocabulary and is kept at Q6_K, so the pair averages out in between.
   */
  embedBpw: number;
  /**
   * Bits per weight for the single shared tensor on tied-embedding models
   * (Llama 3.2 1B and 3B). When `lm_head` reuses `token_embd`, llama.cpp keeps
   * the whole table at output precision — it does not follow the blocks down to
   * 2 or 3 bits. Ignoring this under-predicted Llama 3.2 1B at Q2_K by 10%.
   */
  outputBpw: number;
  /**
   * Wikitext-2 perplexity delta versus F16 measured on LLaMA-7B
   * (llama.cpp reference table). Used to derive a quality-loss estimate.
   */
  pplDelta: number;
  /** Decode throughput relative to Q4_K_M on the same hardware. */
  speedFactor: number;
  tier: QuantTier;
  /** True for the i-quants, which need an importance matrix to build. */
  needsImatrix: boolean;
  notes: string;
}

export interface QuantEstimate {
  format: QuantFormat;
  /** GGUF file size in bytes. */
  fileSizeBytes: number;
  /** Weights + KV cache + compute buffers, in bytes. */
  totalRamBytes: number;
  kvCacheBytes: number;
  overheadBytes: number;
  /** Percent quality degradation versus F16, derived from perplexity delta. */
  qualityLossPct: number;
  /** Estimated single-stream decode speed in tokens/sec on the selected GPU. */
  tokensPerSecond: number | null;
  fits: boolean;
  /** Fits only if some layers are offloaded to CPU. */
  partialFit: boolean;
  /** How many of the model's layers fit in VRAM. */
  gpuLayers: number;
}

export type KvCachePrecision = 'f16' | 'q8_0' | 'q4_0';

export interface QuantPlannerConfig {
  modelId: string;
  /** VRAM budget in GB. */
  vramGb: number;
  /** Context length to size the KV cache for. */
  contextLength: number;
  kvPrecision: KvCachePrecision;
  /** Reserve headroom for the OS / display, in GB. */
  reservedGb: number;
  gpuId: string | null;
  onlyFitting: boolean;
}

/* -------------------------------------------------------------------------- */
/*  Hardware & cost                                                            */
/* -------------------------------------------------------------------------- */

export type GpuVendor = 'NVIDIA' | 'AMD' | 'Apple';
export type GpuClass = 'datacenter' | 'workstation' | 'consumer' | 'unified';

export interface Gpu {
  id: string;
  name: string;
  vendor: GpuVendor;
  class: GpuClass;
  /** Usable VRAM (or unified memory) in GB. */
  vramGb: number;
  /** Peak memory bandwidth in GB/s — the dominant term for decode speed. */
  bandwidthGbs: number;
  /**
   * Fraction of peak bandwidth the inference kernels actually reach, relative
   * to a well-tuned CUDA deployment on Ada/Hopper (1.0). Metal and ROCm leave
   * real throughput on the table, and modelling them at CUDA efficiency
   * over-predicted Apple Silicon by roughly 50%.
   */
  bandwidthEfficiency?: number;
  /** Dense FP16/BF16 tensor throughput in TFLOPS. Bounds prefill + batching. */
  fp16Tflops: number;
  tdpWatts: number;
  /** Street price for the self-hosted amortization model, in USD. */
  msrpUsd?: number;
  releaseYear: number;
}

export type CostProviderKind = 'cloud-ondemand' | 'cloud-spot' | 'self-hosted';

export interface CostProvider {
  id: string;
  name: string;
  kind: CostProviderKind;
  /** Free-form note rendered under the provider column. */
  note: string;
  url: string;
}

/** Hourly USD price for one GPU on one provider. `null` = not offered. */
export type GpuPriceTable = Record<string, Record<string, number | null>>;

export interface SelfHostAssumptions {
  /** Years over which hardware capex is amortized. */
  amortizationYears: number;
  /** Electricity price in USD per kWh. */
  electricityUsdPerKwh: number;
  /** Datacenter/host overhead multiplier applied to GPU power (PUE + CPU + NIC). */
  overheadFactor: number;
  /** Fraction of wall-clock hours the box is actually serving, 0–1. */
  utilization: number;
  /** Extra fixed monthly cost: colo, bandwidth, ops. USD. */
  fixedMonthlyUsd: number;
}

export interface CostInputs {
  modelId: string;
  quantId: string;
  gpuId: string;
  /** Output tokens generated per day. */
  outputTokensPerDay: number;
  /** Input (prompt) tokens processed per day. */
  inputTokensPerDay: number;
  /** Concurrent requests the server batches together. */
  batchSize: number;
  /** 0–1. Fraction of provisioned hours the GPU is actually busy. */
  targetUtilization: number;
  selfHost: SelfHostAssumptions;
}

export interface ThroughputEstimate {
  /** Single-stream decode tokens/sec. */
  decodeSingle: number;
  /** Aggregate decode tokens/sec across the batch. */
  decodeBatched: number;
  /** Prefill tokens/sec (compute bound). */
  prefill: number;
  /** True when the model does not fit in this GPU's memory. */
  oom: boolean;
  memoryUtilizationPct: number;
}

export interface CostResult {
  provider: CostProvider;
  hourlyUsd: number | null;
  gpuHoursPerMonth: number;
  monthlyUsd: number | null;
  /** USD per million output tokens. */
  usdPerMillionOutput: number | null;
  breakdown?: {
    capexUsd: number;
    powerUsd: number;
    fixedUsd: number;
  };
  unavailableReason?: string;
}

/* -------------------------------------------------------------------------- */
/*  GGUF command generator                                                     */
/* -------------------------------------------------------------------------- */

export type CommandPlatform = 'linux' | 'macos' | 'windows';
export type CommandRunner = 'llama.cpp' | 'ollama' | 'lm-studio' | 'python';

export interface CommandStep {
  id: string;
  title: string;
  description: string;
  language: 'bash' | 'powershell' | 'dockerfile' | 'text' | 'python' | 'json';
  code: string;
  /** Rendered as a caution callout under the block. */
  warning?: string;
}

export interface CommandRecipe {
  id: string;
  title: string;
  summary: string;
  steps: CommandStep[];
}

export interface GgufConfig {
  modelId: string;
  quantId: string;
  platform: CommandPlatform;
  useImatrix: boolean;
  /** Number of threads passed to llama-quantize / llama-cli. */
  threads: number;
  /** Local directory the HF checkpoint is downloaded into. */
  workDir: string;
  /** Name the model is registered under in Ollama. */
  ollamaName: string;
  contextLength: number;
  gpuLayers: number;
}

/* -------------------------------------------------------------------------- */
/*  Playground                                                                 */
/* -------------------------------------------------------------------------- */

export interface WebLlmModelOption {
  /** WebLLM `model_id` from the prebuilt app config. */
  id: string;
  label: string;
  paramsB: number;
  /** Approximate VRAM required, in MB, as reported by WebLLM. */
  vramMb: number;
  contextLength: number;
  /** True for models we have verified against the bundled WebLLM version. */
  recommended: boolean;
  family: string;
}

export type EngineBackend = 'webllm' | 'wllama' | 'mock';

export type EngineStatus =
  | { phase: 'idle' }
  | { phase: 'checking' }
  | { phase: 'unsupported'; reason: string }
  | { phase: 'loading'; progress: number; text: string }
  | { phase: 'ready'; modelId: string; backend: EngineBackend; loadMs: number }
  | { phase: 'generating' }
  | { phase: 'error'; message: string };

export interface ChatMessage {
  id: string;
  role: 'system' | 'user' | 'assistant';
  content: string;
  /** Populated on assistant turns once generation settles. */
  stats?: GenerationStats;
  /** Tool call the model emitted, when function calling is enabled. */
  toolCall?: { name: string; arguments: string };
}

export interface GenerationStats {
  promptTokens: number;
  completionTokens: number;
  /** Time to first token, milliseconds. */
  ttftMs: number;
  /** Decode speed in tokens/sec. */
  tokensPerSecond: number;
  totalMs: number;
  backend: EngineBackend;
}

export interface SamplingParams {
  temperature: number;
  topP: number;
  maxTokens: number;
  repetitionPenalty: number;
  presencePenalty: number;
  frequencyPenalty: number;
  seed: number | null;
}

export interface PlaygroundConfig {
  modelId: string;
  systemPrompt: string;
  sampling: SamplingParams;
  functionCallingEnabled: boolean;
  /** Raw JSON text for the tools array — kept as a string so invalid JSON is editable. */
  toolsJson: string;
}

/* -------------------------------------------------------------------------- */
/*  Evaluation suite                                                           */
/* -------------------------------------------------------------------------- */

export type GraderKind = 'exact-match' | 'contains' | 'fuzzy' | 'llm-judge' | 'regex';

export interface EvalCase {
  id: string;
  question: string;
  expected: string;
  /** Optional per-case system prompt override. */
  system?: string;
  tags?: string[];
}

export interface EvalCaseResult {
  caseId: string;
  question: string;
  expected: string;
  actual: string;
  score: number;
  passed: boolean;
  grader: GraderKind;
  /** LLM-as-judge rationale, when that grader is used. */
  rationale?: string;
  latencyMs: number;
  tokensPerSecond: number;
  completionTokens: number;
  error?: string;
}

export interface EvalRun {
  id: string;
  startedAt: string;
  finishedAt?: string;
  modelId: string;
  backend: EngineBackend;
  grader: GraderKind;
  sampling: SamplingParams;
  systemPrompt: string;
  results: EvalCaseResult[];
  summary: EvalSummary;
}

export interface EvalSummary {
  total: number;
  passed: number;
  accuracy: number;
  meanScore: number;
  meanLatencyMs: number;
  meanTokensPerSecond: number;
}

/* -------------------------------------------------------------------------- */
/*  Persistence                                                                */
/* -------------------------------------------------------------------------- */

export interface PersistedState {
  planner: QuantPlannerConfig;
  cost: CostInputs;
  gguf: GgufConfig;
  playground: PlaygroundConfig;
  evalCases: EvalCase[];
  evalRuns: EvalRun[];
  priceOverrides: GpuPriceTable;
  throughputOverrides: Record<string, number>;
}
