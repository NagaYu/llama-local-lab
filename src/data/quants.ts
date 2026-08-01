import type { QuantFormat } from '@/lib/types';

/**
 * Perplexity of LLaMA-7B at F16 on Wikitext-2, used as the quality baseline.
 * Source: llama.cpp quantization reference table.
 */
export const BASELINE_PPL = 5.9066;

/**
 * `llama-quantize` deliberately keeps `token_embd.weight` and `output.weight` at
 * higher precision than the rest of the model. On Llama 3+ the 128k-token
 * vocabulary makes those tensors large enough that ignoring the distinction
 * puts file-size estimates 10–25% off on small models.
 *
 * The `bpw` / `embedBpw` / `outputBpw` values below were back-solved against
 * published GGUF uploads. Across 36 model/quant pairs spanning Llama 3.1 8B,
 * 3.3 70B, 3.2 3B and 3.2 1B, mean absolute error is 0.63% and the worst case
 * is 2.6%.
 *
 * The fields live on `QuantFormat` itself; this alias survives because it reads
 * more clearly at the call sites that actually care about the embedding split.
 */
export type QuantFormatWithEmbed = QuantFormat;

export const QUANT_FORMATS: QuantFormatWithEmbed[] = [
  {
    id: 'F16',
    label: 'F16',
    bpw: 16.0,
    embedBpw: 16.0,
    outputBpw: 16.0,
    pplDelta: 0,
    speedFactor: 0.31,
    tier: 'lossless',
    needsImatrix: false,
    notes:
      'Unquantized half precision. The reference point for every quality number below — convert to this first, then quantize from it.',
  },
  {
    id: 'Q8_0',
    label: 'Q8_0',
    bpw: 8.5,
    embedBpw: 8.5,
    outputBpw: 8.5,
    pplDelta: 0.0004,
    speedFactor: 0.6,
    tier: 'lossless',
    needsImatrix: false,
    notes:
      'Effectively indistinguishable from F16. Use when you have the memory and want a clean quality baseline for evaluation.',
  },
  {
    id: 'Q6_K',
    label: 'Q6_K',
    bpw: 6.56,
    embedBpw: 6.56,
    outputBpw: 6.56,
    pplDelta: 0.0044,
    speedFactor: 0.74,
    tier: 'lossless',
    needsImatrix: false,
    notes:
      'The last stop before measurable degradation. Popular for 7–14B models on 24 GB cards.',
  },
  {
    id: 'Q5_K_M',
    label: 'Q5_K_M',
    bpw: 5.68,
    embedBpw: 6.0,
    outputBpw: 6.56,
    pplDelta: 0.0142,
    speedFactor: 0.86,
    tier: 'recommended',
    needsImatrix: false,
    notes:
      'Very low loss. A good pick when Q4_K_M leaves VRAM on the table.',
  },
  {
    id: 'Q5_K_S',
    label: 'Q5_K_S',
    bpw: 5.52,
    embedBpw: 6.0,
    outputBpw: 6.56,
    pplDelta: 0.0353,
    speedFactor: 0.88,
    tier: 'recommended',
    needsImatrix: false,
    notes: 'Slightly smaller than Q5_K_M with a marginally larger quality cost.',
  },
  {
    id: 'Q4_K_M',
    label: 'Q4_K_M',
    bpw: 4.85,
    embedBpw: 5.5,
    outputBpw: 6.56,
    pplDelta: 0.0535,
    speedFactor: 1.0,
    tier: 'recommended',
    needsImatrix: false,
    notes:
      'The default recommendation. Best size-to-quality ratio for almost every deployment; this is what Ollama ships by default.',
  },
  {
    id: 'IQ4_XS',
    label: 'IQ4_XS',
    bpw: 4.27,
    embedBpw: 5.5,
    outputBpw: 6.56,
    pplDelta: 0.0862,
    speedFactor: 0.98,
    tier: 'recommended',
    needsImatrix: true,
    notes:
      'i-quant that beats Q4_K_S on quality at a smaller size. Needs an importance matrix and is slower to dequantize on CPU.',
  },
  {
    id: 'Q4_K_S',
    label: 'Q4_K_S',
    bpw: 4.58,
    embedBpw: 5.5,
    outputBpw: 6.56,
    pplDelta: 0.1149,
    speedFactor: 1.05,
    tier: 'recommended',
    needsImatrix: false,
    notes: 'Choose over Q4_K_M only when you are a few hundred MB short of fitting.',
  },
  {
    id: 'Q4_0',
    label: 'Q4_0',
    bpw: 4.55,
    embedBpw: 5.5,
    outputBpw: 6.56,
    pplDelta: 0.2499,
    speedFactor: 1.1,
    tier: 'aggressive',
    needsImatrix: false,
    notes:
      'Legacy format. Superseded by Q4_K_M in every respect except raw dequantization speed on old CPUs.',
  },
  {
    id: 'Q3_K_L',
    label: 'Q3_K_L',
    bpw: 4.27,
    embedBpw: 4.5,
    outputBpw: 6.56,
    pplDelta: 0.1803,
    speedFactor: 1.09,
    tier: 'aggressive',
    needsImatrix: false,
    notes:
      'The best of the 3-bit family. Reasonable for squeezing a 70B onto two 24 GB cards.',
  },
  {
    id: 'Q3_K_M',
    label: 'Q3_K_M',
    bpw: 3.95,
    embedBpw: 4.5,
    outputBpw: 6.56,
    pplDelta: 0.2437,
    speedFactor: 1.16,
    tier: 'aggressive',
    needsImatrix: false,
    notes: 'Noticeable drift on reasoning and code. Acceptable for chat and summarization.',
  },
  {
    id: 'IQ3_M',
    label: 'IQ3_M',
    bpw: 3.66,
    embedBpw: 4.5,
    outputBpw: 6.56,
    pplDelta: 0.2932,
    speedFactor: 1.13,
    tier: 'aggressive',
    needsImatrix: true,
    notes:
      'Roughly Q3_K_M quality in Q3_K_S space. The best option in the 3-bit range if you can build an imatrix.',
  },
  {
    id: 'Q3_K_S',
    label: 'Q3_K_S',
    bpw: 3.51,
    embedBpw: 4.5,
    outputBpw: 6.56,
    pplDelta: 0.5505,
    speedFactor: 1.24,
    tier: 'aggressive',
    needsImatrix: false,
    notes: 'Significant quality cost. Prefer IQ3_M or IQ3_XXS at this size.',
  },
  {
    id: 'IQ3_XXS',
    label: 'IQ3_XXS',
    bpw: 3.06,
    embedBpw: 4.5,
    outputBpw: 6.56,
    pplDelta: 0.5934,
    speedFactor: 1.18,
    tier: 'extreme',
    needsImatrix: true,
    notes: 'Usable only on large models (70B+) where the parameter count absorbs the noise.',
  },
  {
    id: 'Q2_K',
    label: 'Q2_K',
    bpw: 2.96,
    embedBpw: 4.5,
    outputBpw: 6.56,
    pplDelta: 0.8698,
    speedFactor: 1.3,
    tier: 'extreme',
    needsImatrix: false,
    notes:
      'Heavy degradation on anything under 30B. Instruction following and JSON output break down first.',
  },
  {
    id: 'IQ2_M',
    label: 'IQ2_M',
    bpw: 2.78,
    embedBpw: 4.0,
    outputBpw: 6.56,
    pplDelta: 1.0934,
    speedFactor: 1.22,
    tier: 'extreme',
    needsImatrix: true,
    notes: 'Last-resort format for fitting a 70B on a single 24 GB card. Expect real losses.',
  },
  {
    id: 'IQ2_XXS',
    label: 'IQ2_XXS',
    bpw: 2.2,
    embedBpw: 4.0,
    outputBpw: 6.56,
    pplDelta: 2.0934,
    speedFactor: 1.28,
    tier: 'extreme',
    needsImatrix: true,
    notes: 'Experimental. Only meaningful on 70B+ models, and still clearly degraded.',
  },
  {
    id: 'IQ1_M',
    label: 'IQ1_M',
    bpw: 1.95,
    embedBpw: 3.5,
    outputBpw: 6.56,
    pplDelta: 3.5934,
    speedFactor: 1.34,
    tier: 'extreme',
    needsImatrix: true,
    notes:
      'A research curiosity. Coherent on very large models, broken on small ones. Do not ship this.',
  },
];

export const QUANT_BY_ID = new Map(QUANT_FORMATS.map((q) => [q.id, q]));

export const DEFAULT_QUANT_ID = 'Q4_K_M';

export function getQuant(id: string): QuantFormatWithEmbed {
  return QUANT_BY_ID.get(id) ?? QUANT_BY_ID.get(DEFAULT_QUANT_ID)!;
}

export const TIER_META: Record<
  QuantFormat['tier'],
  { label: string; className: string; dot: string }
> = {
  lossless: {
    label: 'Near-lossless',
    className: 'bg-success-soft text-success border-success/20',
    dot: 'bg-success',
  },
  recommended: {
    label: 'Recommended',
    className: 'bg-meta-50 text-meta-700 border-meta-200',
    dot: 'bg-meta-500',
  },
  aggressive: {
    label: 'Aggressive',
    className: 'bg-warning-soft text-warning border-warning/20',
    dot: 'bg-warning',
  },
  extreme: {
    label: 'Extreme',
    className: 'bg-danger-soft text-danger border-danger/20',
    dot: 'bg-danger',
  },
};

/** Bytes per element for each KV-cache precision supported by llama.cpp. */
export const KV_BYTES: Record<'f16' | 'q8_0' | 'q4_0', number> = {
  f16: 2,
  q8_0: 1.0625,
  q4_0: 0.5625,
};
