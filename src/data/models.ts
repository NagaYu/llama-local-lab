import type { LlamaModel, WebLlmModelOption } from '@/lib/types';

const LLAMA4_LICENSE = 'Llama 4 Community License';
const LLAMA33_LICENSE = 'Llama 3.3 Community License';
const LLAMA32_LICENSE = 'Llama 3.2 Community License';
const LLAMA31_LICENSE = 'Llama 3.1 Community License';

/**
 * Bundled model catalog.
 *
 * This is the offline source of truth so the whole studio works with no network
 * access. At runtime `lib/hf.ts` layers live download/like counts on top from
 * the public HuggingFace API — the specs below are never overwritten, because
 * the API does not expose architecture details reliably.
 */
export const MODELS: LlamaModel[] = [
  {
    id: 'llama-4-scout',
    repoId: 'meta-llama/Llama-4-Scout-17B-16E-Instruct',
    name: 'Llama 4 Scout 17B-16E',
    family: 'Llama 4',
    paramsB: 109,
    contextLength: 10_485_760,
    modality: 'multimodal',
    license: LLAMA4_LICENSE,
    licenseUrl: 'https://github.com/meta-llama/llama-models/blob/main/models/llama4/LICENSE',
    gated: true,
    releasedAt: '2025-04-05',
    description:
      'Natively multimodal mixture-of-experts model. 16 experts, 17B active parameters, and a 10M-token context window — the longest of any open-weights model. Designed to fit on a single H100 at Int4.',
    architecture: {
      layers: 48,
      hiddenSize: 5120,
      attentionHeads: 40,
      kvHeads: 8,
      vocabSize: 202_048,
      moe: { experts: 16, activeExperts: 1, activeParamsB: 17 },
    },
    languages: ['en', 'ar', 'de', 'es', 'fr', 'hi', 'id', 'it', 'pt', 'th', 'tl', 'vi'],
  },
  {
    id: 'llama-4-maverick',
    repoId: 'meta-llama/Llama-4-Maverick-17B-128E-Instruct',
    name: 'Llama 4 Maverick 17B-128E',
    family: 'Llama 4',
    paramsB: 400,
    contextLength: 1_048_576,
    modality: 'multimodal',
    license: LLAMA4_LICENSE,
    licenseUrl: 'https://github.com/meta-llama/llama-models/blob/main/models/llama4/LICENSE',
    gated: true,
    releasedAt: '2025-04-05',
    description:
      'The flagship Llama 4 release. 128 experts with 17B active parameters, tuned for high-quality chat, reasoning and image understanding. Targets a single H100 DGX host.',
    architecture: {
      layers: 48,
      hiddenSize: 5120,
      attentionHeads: 40,
      kvHeads: 8,
      vocabSize: 202_048,
      moe: { experts: 128, activeExperts: 1, activeParamsB: 17 },
    },
    languages: ['en', 'ar', 'de', 'es', 'fr', 'hi', 'id', 'it', 'pt', 'th', 'tl', 'vi'],
  },
  {
    id: 'llama-3-3-70b',
    repoId: 'meta-llama/Llama-3.3-70B-Instruct',
    name: 'Llama 3.3 70B Instruct',
    family: 'Llama 3.3',
    paramsB: 70.6,
    contextLength: 131_072,
    modality: 'text',
    license: LLAMA33_LICENSE,
    licenseUrl: 'https://github.com/meta-llama/llama-models/blob/main/models/llama3_3/LICENSE',
    gated: true,
    releasedAt: '2024-12-06',
    description:
      'Instruction-tuned text model that matches Llama 3.1 405B quality on most benchmarks at a fraction of the serving cost. Still the default choice for self-hosted dense inference.',
    architecture: {
      layers: 80,
      hiddenSize: 8192,
      attentionHeads: 64,
      kvHeads: 8,
      vocabSize: 128_256,
    },
    languages: ['en', 'de', 'fr', 'it', 'pt', 'hi', 'es', 'th'],
  },
  {
    id: 'llama-3-1-405b',
    repoId: 'meta-llama/Llama-3.1-405B-Instruct',
    name: 'Llama 3.1 405B Instruct',
    family: 'Llama 3.1',
    paramsB: 405.9,
    contextLength: 131_072,
    modality: 'text',
    license: LLAMA31_LICENSE,
    gated: true,
    releasedAt: '2024-07-23',
    description:
      'The largest dense open-weights Llama. Frontier-class quality, but needs a multi-node deployment or aggressive quantization to serve.',
    architecture: {
      layers: 126,
      hiddenSize: 16_384,
      attentionHeads: 128,
      kvHeads: 8,
      vocabSize: 128_256,
    },
    languages: ['en', 'de', 'fr', 'it', 'pt', 'hi', 'es', 'th'],
  },
  {
    id: 'llama-3-1-70b',
    repoId: 'meta-llama/Llama-3.1-70B-Instruct',
    name: 'Llama 3.1 70B Instruct',
    family: 'Llama 3.1',
    paramsB: 70.6,
    contextLength: 131_072,
    modality: 'text',
    license: LLAMA31_LICENSE,
    gated: true,
    releasedAt: '2024-07-23',
    description:
      'The 3.1 generation 70B. Superseded by Llama 3.3 70B, which has the same architecture and better instruction tuning.',
    architecture: {
      layers: 80,
      hiddenSize: 8192,
      attentionHeads: 64,
      kvHeads: 8,
      vocabSize: 128_256,
    },
    languages: ['en', 'de', 'fr', 'it', 'pt', 'hi', 'es', 'th'],
  },
  {
    id: 'llama-3-1-8b',
    repoId: 'meta-llama/Llama-3.1-8B-Instruct',
    name: 'Llama 3.1 8B Instruct',
    family: 'Llama 3.1',
    paramsB: 8.03,
    contextLength: 131_072,
    modality: 'text',
    license: LLAMA31_LICENSE,
    gated: true,
    releasedAt: '2024-07-23',
    description:
      'The workhorse. Fits comfortably on a single consumer GPU at Q4_K_M and runs at conversational speed on a laptop.',
    architecture: {
      layers: 32,
      hiddenSize: 4096,
      attentionHeads: 32,
      kvHeads: 8,
      vocabSize: 128_256,
    },
    languages: ['en', 'de', 'fr', 'it', 'pt', 'hi', 'es', 'th'],
  },
  {
    id: 'llama-3-2-3b',
    repoId: 'meta-llama/Llama-3.2-3B-Instruct',
    name: 'Llama 3.2 3B Instruct',
    family: 'Llama 3.2',
    paramsB: 3.21,
    contextLength: 131_072,
    modality: 'text',
    license: LLAMA32_LICENSE,
    gated: true,
    releasedAt: '2024-09-25',
    description:
      'Edge-class model with tied embeddings. Strong summarization and rewriting for its size; runs in a browser tab via WebGPU.',
    architecture: {
      layers: 28,
      hiddenSize: 3072,
      attentionHeads: 24,
      kvHeads: 8,
      vocabSize: 128_256,
      tiedEmbeddings: true,
    },
    languages: ['en', 'de', 'fr', 'it', 'pt', 'hi', 'es', 'th'],
  },
  {
    id: 'llama-3-2-1b',
    repoId: 'meta-llama/Llama-3.2-1B-Instruct',
    name: 'Llama 3.2 1B Instruct',
    family: 'Llama 3.2',
    paramsB: 1.24,
    contextLength: 131_072,
    modality: 'text',
    license: LLAMA32_LICENSE,
    gated: true,
    releasedAt: '2024-09-25',
    description:
      'The smallest official Llama. Roughly 20% of its parameters are the 128k-token embedding table, which is why naive size formulas get it wrong.',
    architecture: {
      layers: 16,
      hiddenSize: 2048,
      attentionHeads: 32,
      kvHeads: 8,
      vocabSize: 128_256,
      tiedEmbeddings: true,
    },
    languages: ['en', 'de', 'fr', 'it', 'pt', 'hi', 'es', 'th'],
  },
  {
    id: 'llama-3-2-11b-vision',
    repoId: 'meta-llama/Llama-3.2-11B-Vision-Instruct',
    name: 'Llama 3.2 11B Vision',
    family: 'Llama 3.2',
    paramsB: 10.6,
    contextLength: 131_072,
    modality: 'multimodal',
    license: LLAMA32_LICENSE,
    gated: true,
    releasedAt: '2024-09-25',
    description:
      'Llama 3.1 8B plus a cross-attention vision adapter. Document understanding and chart reasoning on a single 24 GB card.',
    architecture: {
      layers: 40,
      hiddenSize: 4096,
      attentionHeads: 32,
      kvHeads: 8,
      vocabSize: 128_256,
    },
    languages: ['en'],
  },
  {
    id: 'llama-3-2-90b-vision',
    repoId: 'meta-llama/Llama-3.2-90B-Vision-Instruct',
    name: 'Llama 3.2 90B Vision',
    family: 'Llama 3.2',
    paramsB: 88.6,
    contextLength: 131_072,
    modality: 'multimodal',
    license: LLAMA32_LICENSE,
    gated: true,
    releasedAt: '2024-09-25',
    description:
      'The large vision model of the 3.2 generation. Needs 2× 80 GB at FP16 or a single 80 GB card at Q4.',
    architecture: {
      layers: 100,
      hiddenSize: 8192,
      attentionHeads: 64,
      kvHeads: 8,
      vocabSize: 128_256,
    },
    languages: ['en'],
  },
  {
    id: 'llama-guard-3-8b',
    repoId: 'meta-llama/Llama-Guard-3-8B',
    name: 'Llama Guard 3 8B',
    family: 'Llama Guard',
    paramsB: 8.03,
    contextLength: 131_072,
    modality: 'text',
    license: LLAMA31_LICENSE,
    gated: true,
    releasedAt: '2024-07-23',
    description:
      'Safety classifier fine-tuned from Llama 3.1 8B. Run it alongside your generator to label prompts and responses against the MLCommons taxonomy.',
    architecture: {
      layers: 32,
      hiddenSize: 4096,
      attentionHeads: 32,
      kvHeads: 8,
      vocabSize: 128_256,
    },
    languages: ['en'],
  },
  {
    id: 'llama-guard-3-1b',
    repoId: 'meta-llama/Llama-Guard-3-1B',
    name: 'Llama Guard 3 1B',
    family: 'Llama Guard',
    paramsB: 1.5,
    contextLength: 131_072,
    modality: 'text',
    license: LLAMA32_LICENSE,
    gated: true,
    releasedAt: '2024-09-25',
    description:
      'Pruned and distilled safety classifier small enough to run on-device next to a 3B generator.',
    architecture: {
      layers: 16,
      hiddenSize: 2048,
      attentionHeads: 32,
      kvHeads: 8,
      vocabSize: 128_256,
      tiedEmbeddings: true,
    },
    languages: ['en'],
  },
  {
    id: 'code-llama-34b',
    repoId: 'codellama/CodeLlama-34b-Instruct-hf',
    name: 'Code Llama 34B Instruct',
    family: 'Code Llama',
    paramsB: 33.7,
    contextLength: 16_384,
    modality: 'text',
    license: 'Llama 2 Community License',
    gated: false,
    releasedAt: '2023-08-24',
    description:
      'Legacy code model kept here as a size reference point — it is the classic "does a 34B fit on my 24 GB card" case.',
    architecture: {
      layers: 48,
      hiddenSize: 8192,
      attentionHeads: 64,
      kvHeads: 8,
      vocabSize: 32_016,
    },
    languages: ['en'],
  },
];

export const MODEL_BY_ID = new Map(MODELS.map((m) => [m.id, m]));
export const DEFAULT_MODEL_ID = 'llama-3-1-8b';

export function getModel(id: string | null | undefined): LlamaModel {
  if (!id) return MODEL_BY_ID.get(DEFAULT_MODEL_ID)!;
  return MODEL_BY_ID.get(id) ?? MODEL_BY_ID.get(DEFAULT_MODEL_ID)!;
}

export const MODEL_FAMILIES = Array.from(new Set(MODELS.map((m) => m.family)));

/* -------------------------------------------------------------------------- */
/*  WebLLM playground catalog                                                  */
/* -------------------------------------------------------------------------- */

/**
 * A curated slice of WebLLM's prebuilt config. `vramMb` values are the
 * `vram_required_MB` figures WebLLM itself reports, so the "will this load"
 * check in the playground matches what the engine will actually try to do.
 */
export const WEBLLM_MODELS: WebLlmModelOption[] = [
  {
    id: 'Llama-3.2-1B-Instruct-q4f16_1-MLC',
    label: 'Llama 3.2 1B Instruct · q4f16',
    paramsB: 1.24,
    vramMb: 879,
    contextLength: 4096,
    recommended: true,
    family: 'Llama 3.2',
  },
  {
    id: 'Llama-3.2-1B-Instruct-q4f32_1-MLC',
    label: 'Llama 3.2 1B Instruct · q4f32',
    paramsB: 1.24,
    vramMb: 1129,
    contextLength: 4096,
    recommended: false,
    family: 'Llama 3.2',
  },
  {
    id: 'Llama-3.2-3B-Instruct-q4f16_1-MLC',
    label: 'Llama 3.2 3B Instruct · q4f16',
    paramsB: 3.21,
    vramMb: 2264,
    contextLength: 4096,
    recommended: true,
    family: 'Llama 3.2',
  },
  {
    id: 'Llama-3.2-3B-Instruct-q4f32_1-MLC',
    label: 'Llama 3.2 3B Instruct · q4f32',
    paramsB: 3.21,
    vramMb: 2952,
    contextLength: 4096,
    recommended: false,
    family: 'Llama 3.2',
  },
  {
    id: 'Llama-3.1-8B-Instruct-q4f16_1-MLC-1k',
    label: 'Llama 3.1 8B Instruct · q4f16 · 1k ctx',
    paramsB: 8.03,
    vramMb: 4598,
    contextLength: 1024,
    recommended: false,
    family: 'Llama 3.1',
  },
  {
    id: 'Llama-3.1-8B-Instruct-q4f16_1-MLC',
    label: 'Llama 3.1 8B Instruct · q4f16',
    paramsB: 8.03,
    vramMb: 5001,
    contextLength: 4096,
    recommended: true,
    family: 'Llama 3.1',
  },
  {
    id: 'Hermes-3-Llama-3.2-3B-q4f16_1-MLC',
    label: 'Hermes 3 (Llama 3.2 3B) · q4f16',
    paramsB: 3.21,
    vramMb: 2264,
    contextLength: 4096,
    recommended: false,
    family: 'Llama 3.2 fine-tune',
  },
  {
    id: 'DeepSeek-R1-Distill-Llama-8B-q4f16_1-MLC',
    label: 'DeepSeek-R1 Distill Llama 8B · q4f16',
    paramsB: 8.03,
    vramMb: 5001,
    contextLength: 4096,
    recommended: false,
    family: 'Llama 3.1 distill',
  },
  {
    id: 'TinyLlama-1.1B-Chat-v1.0-q4f16_1-MLC',
    label: 'TinyLlama 1.1B Chat · q4f16',
    paramsB: 1.1,
    vramMb: 697,
    contextLength: 2048,
    recommended: false,
    family: 'TinyLlama',
  },
  {
    id: 'SmolLM2-360M-Instruct-q4f16_1-MLC',
    label: 'SmolLM2 360M Instruct · q4f16 (fastest)',
    paramsB: 0.36,
    vramMb: 376,
    contextLength: 4096,
    recommended: false,
    family: 'SmolLM2',
  },
];

export const DEFAULT_WEBLLM_MODEL_ID = 'Llama-3.2-1B-Instruct-q4f16_1-MLC';

export function getWebLlmModel(id: string): WebLlmModelOption {
  return (
    WEBLLM_MODELS.find((m) => m.id === id) ??
    WEBLLM_MODELS.find((m) => m.id === DEFAULT_WEBLLM_MODEL_ID)!
  );
}

/**
 * GGUF repos the command generator points at. Meta's own repos are gated, so we
 * also surface the community mirrors people actually download from.
 */
export const GGUF_MIRRORS: Record<string, { repo: string; label: string }[]> = {
  'llama-3-1-8b': [
    { repo: 'bartowski/Meta-Llama-3.1-8B-Instruct-GGUF', label: 'bartowski (imatrix)' },
    { repo: 'lmstudio-community/Meta-Llama-3.1-8B-Instruct-GGUF', label: 'LM Studio Community' },
  ],
  'llama-3-3-70b': [
    { repo: 'bartowski/Llama-3.3-70B-Instruct-GGUF', label: 'bartowski (imatrix)' },
    { repo: 'unsloth/Llama-3.3-70B-Instruct-GGUF', label: 'Unsloth' },
  ],
  'llama-3-2-3b': [
    { repo: 'bartowski/Llama-3.2-3B-Instruct-GGUF', label: 'bartowski (imatrix)' },
  ],
  'llama-3-2-1b': [
    { repo: 'bartowski/Llama-3.2-1B-Instruct-GGUF', label: 'bartowski (imatrix)' },
  ],
  'llama-4-scout': [
    { repo: 'unsloth/Llama-4-Scout-17B-16E-Instruct-GGUF', label: 'Unsloth (dynamic quants)' },
  ],
  'llama-4-maverick': [
    { repo: 'unsloth/Llama-4-Maverick-17B-128E-Instruct-GGUF', label: 'Unsloth (dynamic quants)' },
  ],
};
