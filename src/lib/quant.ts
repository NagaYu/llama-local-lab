import { KV_BYTES, type QuantFormatWithEmbed } from '@/data/quants';
import { BASELINE_PPL } from '@/data/quants';
import type {
  Gpu,
  KvCachePrecision,
  LlamaModel,
  QuantEstimate,
  QuantPlannerConfig,
  ThroughputEstimate,
} from '@/lib/types';

export const GIB = 1024 ** 3;
export const GB = 1e9;

/**
 * Memory-bandwidth utilization achieved by a well-tuned inference server during
 * decode. llama.cpp on CUDA lands around 0.70–0.80; vLLM is similar. 0.72 is a
 * conservative middle that reproduces published tokens/sec for Llama 3.1 8B on
 * an RTX 4090 (~155 tok/s at Q4_K_M) and Llama 3.3 70B on an A100 80GB (~35).
 */
const DECODE_MBU = 0.72;

/** Model FLOPs utilization during prefill — the compute-bound phase. */
const PREFILL_MFU = 0.35;

/** MFU ceiling during batched decode, which never reaches prefill efficiency. */
const DECODE_MFU = 0.25;

/** i-quants trade dequantization cost for size; k-quants are the baseline. */
function dequantEfficiency(quant: QuantFormatWithEmbed): number {
  if (quant.id.startsWith('IQ')) return 0.88;
  if (quant.id === 'F16' || quant.id === 'Q8_0' || quant.id.endsWith('_0')) return 1.02;
  return 1.0;
}

/**
 * Achieved memory bandwidth for one GPU and quantization format.
 *
 * Two multipliers stack on the datasheet number: how much of peak the backend's
 * kernels reach (CUDA on Ada/Hopper is the 1.0 reference; Metal and ROCm are
 * materially lower), and how much the format's dequantization costs.
 */
function achievedBandwidth(gpu: Gpu, quant: QuantFormatWithEmbed): number {
  return (
    gpu.bandwidthGbs *
    1e9 *
    DECODE_MBU *
    (gpu.bandwidthEfficiency ?? 1) *
    dequantEfficiency(quant)
  );
}

/* -------------------------------------------------------------------------- */
/*  Parameter accounting                                                       */
/* -------------------------------------------------------------------------- */

export interface ParamSplit {
  /** Parameters in `token_embd` + `output`, which quantize at higher precision. */
  embedParams: number;
  /** Everything else — the transformer blocks. */
  nonEmbedParams: number;
  /** Parameters in the `output` head alone; it is read in full every token. */
  outputHeadParams: number;
  totalParams: number;
}

/**
 * Split a model's parameters into embedding and transformer weights.
 *
 * This matters more than it looks: Llama 3's 128k vocabulary makes the
 * embedding tensors ~21% of Llama 3.2 1B. `llama-quantize` keeps them at higher
 * precision than the blocks, so a flat `params × bits ÷ 8` estimate is off by
 * double digits on small models. Splitting them brings the estimate to within
 * ~1% of published GGUF file sizes.
 */
export function splitParams(model: LlamaModel): ParamSplit {
  const { vocabSize, hiddenSize, tiedEmbeddings } = model.architecture;
  const oneTable = vocabSize * hiddenSize;
  const tables = tiedEmbeddings ? 1 : 2;
  const embedParams = oneTable * tables;
  const totalParams = model.paramsB * 1e9;
  return {
    embedParams: Math.min(embedParams, totalParams * 0.6),
    nonEmbedParams: Math.max(totalParams - embedParams, totalParams * 0.4),
    outputHeadParams: oneTable,
    totalParams,
  };
}

/**
 * Effective bits per weight for a model's embedding tensors.
 *
 * Two tables (untied) average a cheaply-quantized lookup with a Q6_K output
 * head. One shared table (tied) is *all* output head, so it stays at the higher
 * precision even when the blocks drop to 2 bits — modelling it as the average
 * under-predicted Llama 3.2 1B at Q2_K by 10%.
 */
function embedBpwFor(model: LlamaModel, quant: QuantFormatWithEmbed): number {
  return model.architecture.tiedEmbeddings ? quant.outputBpw : quant.embedBpw;
}

/** GGUF file size in bytes for a given model + quantization format. */
export function ggufSizeBytes(model: LlamaModel, quant: QuantFormatWithEmbed): number {
  const { embedParams, nonEmbedParams } = splitParams(model);
  const weights =
    (nonEmbedParams * quant.bpw) / 8 + (embedParams * embedBpwFor(model, quant)) / 8;
  // GGUF metadata, tokenizer and tensor alignment padding.
  const metadata = 8 * 1024 * 1024 + model.architecture.vocabSize * 12;
  return weights + metadata;
}

/**
 * Bytes actually pulled from memory to decode one token.
 *
 * The embedding table is a lookup — only one row is touched — but the output
 * head is a full matmul against the vocabulary, so it counts. For MoE models
 * only the routed experts are read, which is why Llama 4 Scout decodes like a
 * 17B model despite weighing 109B.
 */
export function bytesReadPerToken(model: LlamaModel, quant: QuantFormatWithEmbed): number {
  const { nonEmbedParams, outputHeadParams, totalParams } = splitParams(model);
  const moe = model.architecture.moe;
  const activeFraction = moe ? Math.min(1, (moe.activeParamsB * 1e9) / totalParams) : 1;
  const blocks = (nonEmbedParams * activeFraction * quant.bpw) / 8;
  const head = (outputHeadParams * embedBpwFor(model, quant)) / 8;
  return blocks + head;
}

/** Parameters involved in the forward pass — drives compute-bound throughput. */
export function activeParams(model: LlamaModel): number {
  const moe = model.architecture.moe;
  return moe ? moe.activeParamsB * 1e9 : model.paramsB * 1e9;
}

/* -------------------------------------------------------------------------- */
/*  Memory                                                                     */
/* -------------------------------------------------------------------------- */

/**
 * KV-cache size in bytes.
 *
 * `2 × layers × kv_heads × head_dim × context × bytes_per_element`, where the
 * leading 2 covers the separate K and V tensors. Grouped-query attention means
 * `kv_heads` is 8 for every modern Llama, which is the only reason 128k context
 * is affordable at all.
 */
export function kvCacheBytes(
  model: LlamaModel,
  contextLength: number,
  precision: KvCachePrecision,
): number {
  const { layers, hiddenSize, attentionHeads, kvHeads } = model.architecture;
  const headDim = hiddenSize / attentionHeads;
  const kvDim = kvHeads * headDim;
  return 2 * layers * kvDim * contextLength * KV_BYTES[precision];
}

/**
 * CUDA context, compute buffers and graph allocations. Scales with context and
 * model width; the constants are fitted to llama.cpp's reported buffer sizes.
 */
export function overheadBytes(model: LlamaModel, contextLength: number): number {
  const widthFactor = model.architecture.hiddenSize / 4096;
  const ctxFactor = contextLength / 1024;
  const compute = 0.1 * GIB * ctxFactor * widthFactor;
  const runtime = 0.45 * GIB;
  return Math.min(compute, 6 * GIB) + runtime;
}

/** Percent perplexity increase versus F16, used as the quality-loss proxy. */
export function qualityLossPct(quant: QuantFormatWithEmbed): number {
  return (quant.pplDelta / BASELINE_PPL) * 100;
}

/** Coarse letter grade so the table is scannable without reading numbers. */
export function qualityGrade(lossPct: number): { grade: string; tone: 'good' | 'ok' | 'warn' | 'bad' } {
  if (lossPct < 0.1) return { grade: 'A+', tone: 'good' };
  if (lossPct < 0.5) return { grade: 'A', tone: 'good' };
  if (lossPct < 1.5) return { grade: 'B', tone: 'ok' };
  if (lossPct < 4) return { grade: 'C', tone: 'warn' };
  if (lossPct < 10) return { grade: 'D', tone: 'warn' };
  return { grade: 'F', tone: 'bad' };
}

/* -------------------------------------------------------------------------- */
/*  Throughput                                                                 */
/* -------------------------------------------------------------------------- */

export function estimateThroughput(
  model: LlamaModel,
  quant: QuantFormatWithEmbed,
  gpu: Gpu,
  opts: { contextLength: number; kvPrecision: KvCachePrecision; batchSize: number },
): ThroughputEstimate {
  const fileBytes = ggufSizeBytes(model, quant);
  const kv = kvCacheBytes(model, opts.contextLength, opts.kvPrecision) * opts.batchSize;
  const overhead = overheadBytes(model, opts.contextLength);
  const required = fileBytes + kv + overhead;
  const capacity = gpu.vramGb * GIB;

  const readPerToken = bytesReadPerToken(model, quant);
  const decodeSingle = achievedBandwidth(gpu, quant) / readPerToken;

  const flops = 2 * activeParams(model);
  const prefill = (gpu.fp16Tflops * 1e12 * PREFILL_MFU) / flops;
  const decodeComputeCeiling = (gpu.fp16Tflops * 1e12 * DECODE_MFU) / flops;

  // Batching amortizes the weight read across requests, but KV traffic and
  // scheduling overhead keep it sublinear.
  const batchScaling = Math.pow(Math.max(1, opts.batchSize), 0.85);
  const decodeBatched = Math.min(decodeSingle * batchScaling, decodeComputeCeiling);

  return {
    decodeSingle,
    decodeBatched: Math.max(decodeBatched, decodeSingle),
    prefill,
    oom: required > capacity,
    memoryUtilizationPct: (required / capacity) * 100,
  };
}

/* -------------------------------------------------------------------------- */
/*  Planner                                                                    */
/* -------------------------------------------------------------------------- */

export function estimateQuant(
  model: LlamaModel,
  quant: QuantFormatWithEmbed,
  config: QuantPlannerConfig,
  gpu: Gpu | null,
): QuantEstimate {
  const fileSizeBytes = ggufSizeBytes(model, quant);
  const kv = kvCacheBytes(model, config.contextLength, config.kvPrecision);
  const overhead = overheadBytes(model, config.contextLength);
  const totalRamBytes = fileSizeBytes + kv + overhead;

  const budget = Math.max(0, (config.vramGb - config.reservedGb) * GIB);
  const fits = totalRamBytes <= budget;

  // How much of the model can live on the GPU once the KV cache and runtime
  // buffers have taken their share. Everything else streams from system RAM.
  const { nonEmbedParams, embedParams } = splitParams(model);
  const blockBytes = (nonEmbedParams * quant.bpw) / 8;
  const perLayerBytes = blockBytes / model.architecture.layers;
  const embedBytes = (embedParams * embedBpwFor(model, quant)) / 8;
  const forWeights = budget - kv - overhead - embedBytes;
  const gpuLayers = fits
    ? model.architecture.layers
    : Math.max(0, Math.min(model.architecture.layers, Math.floor(forWeights / perLayerBytes)));

  const tokensPerSecond = gpu
    ? achievedBandwidth(gpu, quant) / bytesReadPerToken(model, quant)
    : null;

  // A partial fit still runs, just slowly: throughput collapses toward PCIe
  // bandwidth for the offloaded layers.
  const offloadRatio = gpuLayers / model.architecture.layers;
  const adjustedTps =
    tokensPerSecond === null
      ? null
      : fits
        ? tokensPerSecond
        : tokensPerSecond * Math.max(0.04, Math.pow(offloadRatio, 3));

  return {
    format: quant,
    fileSizeBytes,
    totalRamBytes,
    kvCacheBytes: kv,
    overheadBytes: overhead,
    qualityLossPct: qualityLossPct(quant),
    tokensPerSecond: adjustedTps,
    fits,
    partialFit: !fits && gpuLayers > 0,
    gpuLayers,
  };
}

/** Largest context that fits in the remaining VRAM after weights and overhead. */
export function maxContextFor(
  model: LlamaModel,
  quant: QuantFormatWithEmbed,
  vramGb: number,
  precision: KvCachePrecision,
): number {
  const { layers, hiddenSize, attentionHeads, kvHeads } = model.architecture;
  const headDim = hiddenSize / attentionHeads;
  const perToken = 2 * layers * kvHeads * headDim * KV_BYTES[precision];
  const weights = ggufSizeBytes(model, quant);

  // Overhead itself grows with context, so solve iteratively rather than once.
  let ctx = 1024;
  for (let i = 0; i < 40; i += 1) {
    const budget = vramGb * GIB - weights - overheadBytes(model, ctx);
    if (budget <= 0) return 0;
    const next = budget / perToken;
    if (Math.abs(next - ctx) / Math.max(ctx, 1) < 0.01) break;
    ctx = next;
  }
  return Math.max(0, Math.floor(Math.min(ctx, model.contextLength)));
}

/** The best-quality quant that still fits, or null when nothing does. */
export function recommendQuant(
  model: LlamaModel,
  quants: QuantFormatWithEmbed[],
  config: QuantPlannerConfig,
  gpu: Gpu | null,
): QuantEstimate | null {
  const fitting = quants
    .map((q) => estimateQuant(model, q, config, gpu))
    .filter((e) => e.fits)
    .sort((a, b) => a.qualityLossPct - b.qualityLossPct);
  return fitting[0] ?? null;
}
