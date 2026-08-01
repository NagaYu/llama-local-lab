'use client';

import * as React from 'react';
import {
  AlertTriangle,
  Boxes,
  CheckCircle2,
  Download,
  ExternalLink,
  FileCode2,
  HardDrive,
  Layers,
  RotateCcw,
  Server,
  ShieldAlert,
  TerminalSquare,
  Wrench,
  Zap,
} from 'lucide-react';

import { CodeBlock, InlineCode } from '@/components/code-block';
import { CopyButton } from '@/components/copy-button';
import {
  Field,
  Footnote,
  NumberInput,
  SectionHeading,
  Stat,
  StatGrid,
} from '@/components/primitives';
import { useStudio } from '@/components/studio-provider';
import { Alert, AlertDescription, AlertTitle } from '@/components/ui/alert';
import { Badge } from '@/components/ui/badge';
import { Button } from '@/components/ui/button';
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card';
import { Input } from '@/components/ui/input';
import { Label } from '@/components/ui/label';
import {
  Select,
  SelectContent,
  SelectGroup,
  SelectItem,
  SelectLabel,
  SelectTrigger,
  SelectValue,
} from '@/components/ui/select';
import { Separator } from '@/components/ui/separator';
import { Switch } from '@/components/ui/switch';
import { InfoTip } from '@/components/ui/tooltip';
import {
  DEFAULT_MODEL_ID,
  GGUF_MIRRORS,
  MODELS,
  MODEL_FAMILIES,
  getModel,
} from '@/data/models';
import {
  DEFAULT_QUANT_ID,
  QUANT_FORMATS,
  TIER_META,
  getQuant,
  type QuantFormatWithEmbed,
} from '@/data/quants';
import { formatContext, formatGib, formatParams, formatPct } from '@/lib/format';
import { ggufSizeBytes, qualityLossPct } from '@/lib/quant';
import { STORAGE_KEYS, mergeDefaults, usePersistentState } from '@/lib/storage';
import type {
  CommandPlatform,
  CommandRecipe,
  CommandStep,
  GgufConfig,
  LlamaModel,
  QuantTier,
} from '@/lib/types';
import { clamp, cn, downloadFile } from '@/lib/utils';

/* -------------------------------------------------------------------------- */
/*  Local types                                                                */
/* -------------------------------------------------------------------------- */

interface GeneratedStep extends CommandStep {
  /** Steps the user toggled on rather than steps every recipe needs. */
  optional?: boolean;
  /**
   * Long-running or interactive. The composed script comments these out so a
   * downloaded `.sh` does not stall forever on `llama-server`.
   */
  blocking?: boolean;
  /** A companion block in another language — e.g. the shell that consumes a Modelfile. */
  extra?: { title: string; language: CommandStep['language']; code: string };
  /** When `code` is a file's contents, the path the composed script writes it to. */
  writeTo?: string;
}

interface GeneratedRecipe extends Omit<CommandRecipe, 'steps'> {
  steps: GeneratedStep[];
}

interface RecipeContext {
  model: LlamaModel;
  quant: QuantFormatWithEmbed;
  platform: CommandPlatform;
  useImatrix: boolean;
  threads: number;
  contextLength: number;
  gpuLayers: number;
  workDir: string;
  ollamaName: string;
}

/* -------------------------------------------------------------------------- */
/*  Constants                                                                  */
/* -------------------------------------------------------------------------- */

const GIB = 1024 ** 3;

/** Above this, the F16 intermediate stops being something you can casually keep. */
const BIG_F16_BYTES = 40 * GIB;

const LLAMA_CPP_REPO = 'https://github.com/ggml-org/llama.cpp';

const DEFAULT_GGUF: GgufConfig = {
  modelId: DEFAULT_MODEL_ID,
  quantId: DEFAULT_QUANT_ID,
  platform: 'linux',
  useImatrix: false,
  threads: 8,
  workDir: '~/models',
  // Empty means "derive from the model and quant", so the name keeps tracking the
  // selection until the user types their own.
  ollamaName: '',
  contextLength: 8192,
  gpuLayers: 99,
};

const REVIVE = { revive: mergeDefaults(DEFAULT_GGUF) };

const PLATFORMS: { id: CommandPlatform; label: string; note: string }[] = [
  { id: 'linux', label: 'Linux', note: 'CUDA · bash' },
  { id: 'macos', label: 'macOS', note: 'Metal · zsh' },
  { id: 'windows', label: 'Windows', note: 'CUDA · PowerShell' },
];

const PLATFORM_IDS = new Set<string>(PLATFORMS.map((p) => p.id));

const TIER_ORDER: QuantTier[] = ['lossless', 'recommended', 'aggressive', 'extreme'];

/**
 * Ollama library tags, so the shortcut card can offer `ollama pull` for models
 * that are actually in the registry. Only the quant types Ollama publishes are
 * usable as a tag suffix — the i-quants are not in the library at all.
 */
const OLLAMA_LIBRARY: Record<string, { name: string; quantized: boolean }> = {
  'llama-4-scout': { name: 'llama4:scout', quantized: false },
  'llama-4-maverick': { name: 'llama4:maverick', quantized: false },
  'llama-3-3-70b': { name: 'llama3.3:70b-instruct', quantized: true },
  'llama-3-1-405b': { name: 'llama3.1:405b-instruct', quantized: true },
  'llama-3-1-70b': { name: 'llama3.1:70b-instruct', quantized: true },
  'llama-3-1-8b': { name: 'llama3.1:8b-instruct', quantized: true },
  'llama-3-2-3b': { name: 'llama3.2:3b-instruct', quantized: true },
  'llama-3-2-1b': { name: 'llama3.2:1b-instruct', quantized: true },
  'llama-3-2-11b-vision': { name: 'llama3.2-vision:11b-instruct', quantized: true },
  'llama-3-2-90b-vision': { name: 'llama3.2-vision:90b-instruct', quantized: true },
  'llama-guard-3-8b': { name: 'llama-guard3:8b', quantized: false },
  'llama-guard-3-1b': { name: 'llama-guard3:1b', quantized: false },
  'code-llama-34b': { name: 'codellama:34b-instruct', quantized: true },
};

/** Quantization tags Ollama's library actually publishes. */
const OLLAMA_QUANT_TAGS = new Set([
  'Q2_K', 'Q3_K_S', 'Q3_K_M', 'Q3_K_L', 'Q4_0', 'Q4_K_S', 'Q4_K_M',
  'Q5_K_S', 'Q5_K_M', 'Q6_K', 'Q8_0', 'F16',
]);

const STEP_ICONS: Record<string, React.ComponentType<{ className?: string }>> = {
  prereq: Wrench,
  download: Download,
  convert: FileCode2,
  imatrix: Zap,
  quantize: Layers,
  verify: CheckCircle2,
  serve: Server,
  ollama: Boxes,
};

/* -------------------------------------------------------------------------- */
/*  Shell helpers                                                              */
/* -------------------------------------------------------------------------- */

function isWindows(platform: CommandPlatform) {
  return platform === 'windows';
}

function shellLanguage(platform: CommandPlatform): CommandStep['language'] {
  return isWindows(platform) ? 'powershell' : 'bash';
}

/** Join filesystem segments with the separator the target shell expects. */
function joinPath(platform: CommandPlatform, ...parts: string[]): string {
  const joined = parts.filter((p) => p.length > 0).join('/');
  return isWindows(platform) ? joined.replace(/\//g, '\\') : joined;
}

/**
 * A command spread over several lines. bash continues with a trailing backslash,
 * PowerShell with a trailing backtick — getting this wrong silently runs half
 * the command, so it is derived from the platform rather than hard-coded.
 */
function multiline(platform: CommandPlatform, head: string, args: string[]): string {
  if (args.length === 0) return head;
  const cont = isWindows(platform) ? '`' : '\\';
  return [head, ...args].join(` ${cont}\n  `);
}

/** `export NAME=value` on POSIX, `$env:NAME = "value"` in PowerShell. */
function exportVar(platform: CommandPlatform, name: string, value: string): string {
  return isWindows(platform) ? `$env:${name} = "${value}"` : `export ${name}=${value}`;
}

function makeDir(platform: CommandPlatform, path: string): string {
  return isWindows(platform)
    ? `New-Item -ItemType Directory -Force -Path ${path} | Out-Null`
    : `mkdir -p ${path}`;
}

function changeDir(platform: CommandPlatform, path: string): string {
  return isWindows(platform) ? `Set-Location ${path}` : `cd ${path}`;
}

function block(...lines: string[]): string {
  return lines.join('\n');
}

/* -------------------------------------------------------------------------- */
/*  Naming                                                                     */
/* -------------------------------------------------------------------------- */

/** `meta-llama/Llama-3.1-8B-Instruct` -> `Llama-3.1-8B-Instruct`. */
function repoBaseName(repoId: string): string {
  const parts = repoId.split('/');
  return parts[parts.length - 1];
}

function autoOllamaName(model: LlamaModel, quant: QuantFormatWithEmbed): string {
  return `${model.id}-${quant.id.toLowerCase()}`;
}

/**
 * Community GGUF repos name their files `<repo-without-GGUF>-<QUANT>.gguf`.
 * bartowski, lmstudio-community and Unsloth's static quants all follow it; the
 * dynamic-quant folders do not, which is why the card also offers a glob.
 */
function mirrorFileName(repo: string, quantId: string): string {
  const base = repoBaseName(repo).replace(/-GGUF$/i, '');
  return `${base}-${quantId}.gguf`;
}

/* -------------------------------------------------------------------------- */
/*  Ollama Modelfile                                                           */
/* -------------------------------------------------------------------------- */

interface ChatFormat {
  template: string;
  stops: string[];
}

/**
 * Ollama re-implements the chat template in Go, so it has to be written out by
 * hand. Llama 4 replaced the 3.x `<|start_header_id|>` pair with
 * `<|header_start|>` / `<|header_end|>` and `<|eot_id|>` with `<|eot|>`, and
 * Code Llama is still on the Llama 2 `[INST]` format — using the wrong one
 * produces a model that answers, badly, and never stops.
 */
function chatFormat(model: LlamaModel): ChatFormat {
  if (model.family === 'Llama 4') {
    return {
      template: [
        '<|begin_of_text|>{{ if .System }}<|header_start|>system<|header_end|>',
        '',
        '{{ .System }}<|eot|>{{ end }}<|header_start|>user<|header_end|>',
        '',
        '{{ .Prompt }}<|eot|><|header_start|>assistant<|header_end|>',
        '',
      ].join('\n'),
      stops: ['<|eot|>', '<|header_start|>', '<|header_end|>'],
    };
  }

  if (model.family === 'Code Llama') {
    return {
      template: [
        '[INST] {{ if .System }}<<SYS>>',
        '{{ .System }}',
        '<</SYS>>',
        '',
        '{{ end }}{{ .Prompt }} [/INST] ',
      ].join('\n'),
      stops: ['[INST]', '[/INST]', '<<SYS>>', '<</SYS>>'],
    };
  }

  return {
    template: [
      '<|begin_of_text|>{{ if .System }}<|start_header_id|>system<|end_header_id|>',
      '',
      '{{ .System }}<|eot_id|>{{ end }}<|start_header_id|>user<|end_header_id|>',
      '',
      '{{ .Prompt }}<|eot_id|><|start_header_id|>assistant<|end_header_id|>',
      '',
    ].join('\n'),
    stops: ['<|eot_id|>', '<|eom_id|>', '<|start_header_id|>', '<|end_header_id|>'],
  };
}

function buildModelfile(ctx: RecipeContext, ggufPath: string, modelfilePath: string): string {
  const { template, stops } = chatFormat(ctx.model);
  const isGuard = ctx.model.family === 'Llama Guard';

  const lines = [
    `# ${modelfilePath}`,
    `# ${ctx.model.name} · ${ctx.quant.id} · generated by Llama Local Lab`,
    '',
    `FROM ${ggufPath}`,
    '',
    `PARAMETER num_ctx ${ctx.contextLength}`,
    `PARAMETER num_gpu ${ctx.gpuLayers}`,
    `PARAMETER num_thread ${ctx.threads}`,
    'PARAMETER temperature 0.6',
    'PARAMETER top_p 0.9',
    'PARAMETER repeat_penalty 1.1',
    ...stops.map((s) => `PARAMETER stop "${s}"`),
    '',
    `TEMPLATE """${template}"""`,
  ];

  if (!isGuard) {
    lines.push(
      '',
      'SYSTEM """You are a helpful assistant. Answer concisely and say so when you are unsure."""',
    );
  } else {
    lines.push(
      '',
      '# Llama Guard takes its taxonomy in the user turn, not a SYSTEM block —',
      '# send the full safety prompt as the message and read the first token back.',
    );
  }

  return lines.join('\n');
}

/* -------------------------------------------------------------------------- */
/*  Recipe                                                                     */
/* -------------------------------------------------------------------------- */

function buildRecipe(ctx: RecipeContext): GeneratedRecipe {
  const { platform, model, quant } = ctx;
  const win = isWindows(platform);
  const lang = shellLanguage(platform);
  const py = win ? 'python' : 'python3';
  const bin = (name: string) =>
    win ? `.\\build\\bin\\Release\\${name}.exe` : `./build/bin/${name}`;

  const root = ctx.workDir.replace(/[\\/]+$/, '') || '~/models';
  const rootPath = joinPath(platform, root);
  const llamaDir = joinPath(platform, root, 'llama.cpp');
  const srcDir = joinPath(platform, root, repoBaseName(model.repoId));
  const ggufDir = joinPath(platform, root, 'gguf');
  const baseName = repoBaseName(model.repoId);
  const f16Path = joinPath(platform, ggufDir, `${baseName}-F16.gguf`);
  const quantPath = joinPath(platform, ggufDir, `${baseName}-${quant.id}.gguf`);
  const imatrixPath = joinPath(platform, ggufDir, `${baseName}-imatrix.dat`);
  const modelfilePath = joinPath(platform, ggufDir, 'Modelfile');

  // F16 is the conversion output, so selecting it means there is nothing to
  // quantize — the convert step already produced the final artifact.
  const quantizes = quant.id !== 'F16';
  const finalPath = quantizes ? quantPath : f16Path;
  const useImatrix = quantizes && ctx.useImatrix;

  const steps: GeneratedStep[] = [];

  /* --- 0. toolchain, clone, build --------------------------------------- */

  const buildFlag =
    platform === 'macos' ? '-DGGML_METAL=ON' : '-DGGML_CUDA=ON';
  const jobs =
    platform === 'linux'
      ? '$(nproc)'
      : platform === 'macos'
        ? '$(sysctl -n hw.ncpu)'
        : '$env:NUMBER_OF_PROCESSORS';

  const toolchain =
    platform === 'linux'
      ? block(
          '# Compiler, CMake, git-lfs and a Python that can run the converter.',
          'sudo apt-get update',
          'sudo apt-get install -y build-essential cmake git git-lfs python3 python3-pip python3-venv',
        )
      : platform === 'macos'
        ? block(
            '# Command line tools give you clang and Metal; the rest comes from Homebrew.',
            'xcode-select --install || true',
            'brew install cmake git git-lfs python@3.12',
          )
        : block(
            '# Run this in an elevated PowerShell, then reopen a',
            '# "Developer PowerShell for VS 2022" so cl.exe is on PATH.',
            'winget install --id Git.Git -e --source winget',
            'winget install --id Kitware.CMake -e --source winget',
            'winget install --id Python.Python.3.12 -e --source winget',
            'winget install --id Microsoft.VisualStudio.2022.BuildTools -e --source winget --override "--quiet --add Microsoft.VisualStudio.Workload.NativeDesktop --includeRecommended"',
            'winget install --id Nvidia.CUDA -e --source winget',
          );

  const venv =
    win
      ? block(
          'py -3 -m venv .venv',
          '.\\.venv\\Scripts\\Activate.ps1',
          'python -m pip install --upgrade pip',
          'python -m pip install -r requirements.txt "huggingface_hub[cli]>=0.34"',
        )
      : block(
          `${py} -m venv .venv`,
          'source .venv/bin/activate',
          `${py} -m pip install --upgrade pip`,
          `${py} -m pip install -r requirements.txt "huggingface_hub[cli]>=0.34"`,
        );

  steps.push({
    id: 'prereq',
    title: 'Install the toolchain and build llama.cpp',
    description:
      'llama.cpp is built with CMake now — the old `make` path is deprecated and the binaries moved into build/bin/. Build once; every later step reuses these binaries.',
    language: lang,
    code: block(
      toolchain,
      '',
      '# The project lives under the ggml-org organisation.',
      makeDir(platform, rootPath),
      changeDir(platform, rootPath),
      `git clone ${LLAMA_CPP_REPO}`,
      changeDir(platform, 'llama.cpp'),
      '',
      `# ${buildFlag} compiles the GPU backend. Drop it entirely for a CPU-only build.`,
      `cmake -B build ${buildFlag} -DCMAKE_BUILD_TYPE=Release`,
      `cmake --build build --config Release -j ${jobs}`,
      '',
      '# convert_hf_to_gguf.py is Python — keep its dependencies out of the system env.',
      venv,
    ),
    warning:
      platform === 'linux'
        ? 'GGML_CUDA=ON needs the NVIDIA CUDA Toolkit (nvcc) on PATH; add -DCMAKE_CUDA_ARCHITECTURES=native to compile only for the card you own. AMD builds use -DGGML_HIP=ON, portable GPU builds -DGGML_VULKAN=ON.'
        : platform === 'macos'
          ? 'Metal is compiled in by default on Apple Silicon — the flag is explicit, not required. Unified memory means -ngl 99 is almost always right, and there is no separate VRAM budget to plan around.'
          : 'The Visual Studio generator writes binaries to build\\bin\\Release\\. If you configure with -G Ninja they land in build\\bin\\ instead, so drop the Release\\ segment from every path below. Activate.ps1 needs Set-ExecutionPolicy -Scope Process RemoteSigned.',
  });

  /* --- 1. authenticate + download --------------------------------------- */

  const downloadHead = `hf download ${model.repoId}`;
  const downloadArgs = [
    `--local-dir ${srcDir}`,
    '--exclude "original/*" "*.pth" "*.gguf"',
  ];

  steps.push({
    id: 'download',
    title: model.gated ? 'Accept the licence and download the checkpoint' : 'Download the checkpoint',
    description: model.gated
      ? 'Meta repos are gated: your account has to be granted access on the model page before any token works. `hf auth login` replaced `huggingface-cli login` in huggingface_hub 0.34.'
      : 'A public repo, so no token is needed. The excludes skip the duplicate consolidated weights that would otherwise double the download.',
    language: lang,
    code: block(
      ...(model.gated
        ? [
            `# 1. Accept the licence at https://huggingface.co/${model.repoId} and wait for approval.`,
            '# 2. Paste a token with "read" scope from https://huggingface.co/settings/tokens.',
            'hf auth login',
            '',
          ]
        : []),
      '# Optional: hf_transfer saturates a fast link. Skip both lines on a slow one.',
      `${py} -m pip install hf_transfer`,
      exportVar(platform, 'HF_HUB_ENABLE_HF_TRANSFER', '1'),
      '',
      '# original/ holds Meta\'s consolidated .pth copy of the same weights.',
      multiline(platform, downloadHead, downloadArgs),
    ),
    warning: model.gated
      ? 'A 403 here almost always means the licence has not been accepted (or the request is still pending) — not a bad token. Check the model page while signed in.'
      : undefined,
  });

  /* --- 2. convert to F16 GGUF ------------------------------------------- */

  const convertHead = `${py} convert_hf_to_gguf.py ${srcDir}`;
  const convertArgs = [`--outfile ${f16Path}`, '--outtype f16'];

  steps.push({
    id: 'convert',
    title: quantizes ? 'Convert the checkpoint to an F16 GGUF' : 'Convert the checkpoint to GGUF',
    description: quantizes
      ? 'convert.py is gone; convert_hf_to_gguf.py reads config.json plus the safetensors shards and writes one F16 GGUF. This is a container change, not compression — the file is the same size as the weights.'
      : 'You picked F16, so the converter produces the final artifact directly and there is nothing left to quantize.',
    language: lang,
    code: block(
      makeDir(platform, ggufDir),
      '',
      `# Run from the llama.cpp checkout (${llamaDir}).`,
      multiline(platform, convertHead, convertArgs),
    ),
    warning:
      model.modality === 'multimodal'
        ? model.family === 'Llama 3.2'
          ? 'Llama 3.2 Vision uses cross-attention (mllama), which llama.cpp does not implement — the converter will reject it. Use a text-only sibling for GGUF, or run this checkpoint under vLLM / transformers instead.'
          : 'Multimodal checkpoints need a second artifact: pass --mmproj to emit the vision projector alongside the text model, and load it with --mmproj at serve time. Build from a recent master — projector support moves fast.'
        : undefined,
  });

  /* --- 3. importance matrix (optional) ---------------------------------- */

  if (useImatrix) {
    const fetchCorpus = win
      ? '& "$env:ProgramFiles\\Git\\bin\\bash.exe" scripts/get-wikitext-2.sh'
      : 'bash scripts/get-wikitext-2.sh';

    steps.push({
      id: 'imatrix',
      title: 'Build the importance matrix',
      description:
        'An imatrix records which weights actually move activations on a calibration corpus, so the quantizer can spend its bits where they matter. Required by every IQ format and a free quality win for the K-quants.',
      language: lang,
      optional: true,
      code: block(
        '# Any UTF-8 corpus works. llama.cpp ships a fetcher for wikitext-2; text that',
        '# looks like your own traffic works better. 200 chunks is plenty.',
        fetchCorpus,
        '',
        multiline(platform, bin('llama-imatrix'), [
          `-m ${f16Path}`,
          '-f wikitext-2-raw/wiki.train.raw',
          `-o ${imatrixPath}`,
          `-ngl ${ctx.gpuLayers}`,
          '--chunks 200',
        ]),
      ),
      warning: quant.needsImatrix
        ? `${quant.id} is an i-quant: llama-quantize will refuse to produce it without --imatrix, so this step is not optional here.`
        : 'This runs a full forward pass over the corpus at F16 — minutes on a small model, an hour or more on a 70B. It is the slowest step in the pipeline.',
    });
  }

  /* --- 4. quantize ------------------------------------------------------ */

  if (quantizes) {
    const quantizeArgs = [
      ...(useImatrix ? [`--imatrix ${imatrixPath}`] : []),
      f16Path,
      quantPath,
      quant.id,
      String(ctx.threads),
    ];

    steps.push({
      id: 'quantize',
      title: `Quantize to ${quant.id}`,
      description:
        'Positional arguments, in order: input GGUF, output GGUF, type, thread count. The type name is exactly the id llama.cpp uses — run llama-quantize with no arguments to see the full list.',
      language: lang,
      code: multiline(platform, bin('llama-quantize'), quantizeArgs),
      warning: `${quant.notes}`,
    });
  }

  /* --- 5. verify + smoke test ------------------------------------------- */

  const sizeCheck = win
    ? `Get-ChildItem ${finalPath} | Select-Object Name, @{ Name = 'GiB'; Expression = { [math]::Round($_.Length / 1GB, 2) } }`
    : `ls -lh ${finalPath}`;

  steps.push({
    id: 'verify',
    title: 'Check the file and smoke-test it',
    description:
      'The size should land within a percent or two of the estimate above. -no-cnv runs a single completion instead of dropping into the interactive chat loop, which is what you want in a script.',
    language: lang,
    code: block(
      sizeCheck,
      '',
      multiline(platform, bin('llama-cli'), [
        `-m ${finalPath}`,
        `-ngl ${ctx.gpuLayers}`,
        `-c ${ctx.contextLength}`,
        `-t ${ctx.threads}`,
        '-n 128',
        '-no-cnv',
        '-p "In two sentences, explain what quantization does to a language model."',
      ]),
    ),
    warning:
      'Watch the load log: llama.cpp prints how many layers it actually offloaded. If it is fewer than you asked for, the KV cache did not fit — lower -c or -ngl rather than letting it spill.',
  });

  /* --- 6. serve ---------------------------------------------------------- */

  const serveAlias = ctx.ollamaName || autoOllamaName(model, quant);
  const apiCall = win
    ? block(
        '# From a second PowerShell window:',
        '$body = @{',
        `  model      = '${serveAlias}'`,
        "  messages   = @(@{ role = 'user'; content = 'Ping' })",
        '  max_tokens = 64',
        '} | ConvertTo-Json -Depth 4',
        '',
        'Invoke-RestMethod -Uri http://localhost:8080/v1/chat/completions -Method Post -ContentType "application/json" -Body $body',
      )
    : block(
        '# From a second shell — the endpoint speaks the OpenAI schema.',
        multiline(platform, 'curl -s http://localhost:8080/v1/chat/completions', [
          '-H "Content-Type: application/json"',
          `-d '{"model":"${serveAlias}","messages":[{"role":"user","content":"Ping"}],"max_tokens":64}'`,
        ]),
      );

  steps.push({
    id: 'serve',
    title: 'Serve it over an OpenAI-compatible API',
    description:
      'llama-server exposes /v1/chat/completions, /v1/completions and /v1/embeddings, plus a built-in web UI on the same port. Any OpenAI SDK works against it by changing the base URL.',
    language: lang,
    blocking: true,
    code: block(
      ...(platform === 'macos'
        ? []
        : [
            '# Pin the server to one card when the box has several.',
            exportVar(platform, 'CUDA_VISIBLE_DEVICES', '0'),
            '',
          ]),
      multiline(platform, bin('llama-server'), [
        `-m ${finalPath}`,
        '--host 0.0.0.0',
        '--port 8080',
        `-ngl ${ctx.gpuLayers}`,
        `-c ${ctx.contextLength}`,
        `-t ${ctx.threads}`,
        `--alias ${serveAlias}`,
      ]),
      '',
      apiCall,
    ),
    warning:
      '--host 0.0.0.0 publishes the server on every interface with no authentication. On a shared network bind 127.0.0.1 instead, or pass --api-key <secret> and put a reverse proxy in front.',
  });

  /* --- 7. Ollama --------------------------------------------------------- */

  steps.push({
    id: 'ollama',
    title: 'Register the GGUF with Ollama',
    description:
      'A Modelfile points Ollama at a local GGUF and pins the sampling defaults and chat template. Ollama re-implements the template in Go, so it has to be spelled out — the one below matches this model family.',
    language: 'dockerfile',
    writeTo: modelfilePath,
    code: buildModelfile(ctx, finalPath, modelfilePath),
    extra: {
      title: 'Create and run',
      language: lang,
      code: block(
        changeDir(platform, ggufDir),
        `ollama create ${serveAlias} -f Modelfile`,
        `ollama run ${serveAlias} "Give me one sentence about llamas."`,
        'ollama list',
      ),
    },
    warning:
      'Ollama does not expand ~ in a FROM line — use an absolute path if the one above starts with a tilde. `ollama create` copies the GGUF into its own blob store, so budget the file size twice.',
  });

  return {
    id: `${model.id}-${quant.id}-${platform}`,
    title: `${model.name} → ${quant.id} GGUF`,
    summary: `${steps.length} steps from the HuggingFace checkpoint to a served ${quant.id} GGUF on ${
      PLATFORMS.find((p) => p.id === platform)?.label ?? platform
    }.`,
    steps,
  };
}

/* -------------------------------------------------------------------------- */
/*  Script composition                                                         */
/* -------------------------------------------------------------------------- */

/** Wrap file contents in a heredoc / here-string so the script can write them out. */
function writeFileCommand(platform: CommandPlatform, path: string, contents: string): string {
  if (isWindows(platform)) {
    return block("@'", contents, "'@ | Set-Content -Path " + path + ' -Encoding utf8');
  }
  return block(`cat > ${path} <<'LLAMA_LOCAL_LAB_EOF'`, contents, 'LLAMA_LOCAL_LAB_EOF');
}

function commentOut(code: string): string {
  return code
    .split('\n')
    .map((line) => (line.trim().length > 0 ? `# ${line}` : '#'))
    .join('\n');
}

function composeScript(recipe: GeneratedRecipe, ctx: RecipeContext, outputBytes: number): string {
  const win = isWindows(ctx.platform);
  const rule = `# ${'-'.repeat(74)}`;
  const lines: string[] = [];

  lines.push(
    win ? '#Requires -Version 5.1' : '#!/usr/bin/env bash',
    '#',
    '# Llama Local Lab — GGUF build script',
    `# Model    : ${ctx.model.repoId}`,
    `# Quant    : ${ctx.quant.id} (expected output ≈ ${formatGib(outputBytes)})`,
    `# Platform : ${ctx.platform}`,
    `# Imatrix  : ${ctx.useImatrix ? 'yes' : 'no'}`,
    '#',
    '# Read it before you run it: this downloads tens of gigabytes and writes',
    '# an F16 intermediate that is as large as the original checkpoint.',
    '',
    win ? "$ErrorActionPreference = 'Stop'" : 'set -euo pipefail',
  );

  recipe.steps.forEach((step, index) => {
    lines.push(
      '',
      rule,
      `# ${String(index).padStart(2, '0')}. ${step.title}${step.optional ? ' (optional)' : ''}`,
      `# ${step.description}`,
      ...(step.blocking ? ['# Commented out: this command blocks. Uncomment to run it.'] : []),
      rule,
      '',
    );

    const body =
      step.writeTo !== undefined
        ? writeFileCommand(ctx.platform, step.writeTo, step.code)
        : step.code;

    lines.push(step.blocking ? commentOut(body) : body);

    if (step.extra) {
      lines.push('');
      lines.push(step.blocking ? commentOut(step.extra.code) : step.extra.code);
    }
  });

  lines.push('');
  return lines.join('\n');
}

/* -------------------------------------------------------------------------- */
/*  Panel                                                                      */
/* -------------------------------------------------------------------------- */

export function GgufGeneratorPanel() {
  const { modelId, setModelId, quantId, setQuantId } = useStudio();
  const [config, setConfig, { reset: resetConfig }] = usePersistentState<GgufConfig>(
    STORAGE_KEYS.gguf,
    DEFAULT_GGUF,
    REVIVE,
  );

  const model = getModel(modelId);
  const quant = getQuant(quantId);
  const f16 = getQuant('F16');

  // localStorage is user-writable, so every persisted number is re-clamped here
  // rather than trusted into a shell command.
  const platform: CommandPlatform = PLATFORM_IDS.has(config.platform) ? config.platform : 'linux';
  const threads = clamp(Math.round(safeNumber(config.threads, 8)), 1, 256);
  const contextLength = clamp(
    Math.round(safeNumber(config.contextLength, 8192)),
    512,
    model.contextLength,
  );
  const gpuLayers = clamp(Math.round(safeNumber(config.gpuLayers, 99)), 0, 999);
  const workDir = config.workDir.trim() || '~/models';
  const derivedOllamaName = autoOllamaName(model, quant);
  const ollamaName = config.ollamaName.trim() || derivedOllamaName;

  const imatrixApplicable = quant.id !== 'F16';
  const imatrixRequired = quant.needsImatrix;
  const useImatrix = imatrixApplicable && (imatrixRequired || config.useImatrix);

  const ctx: RecipeContext = React.useMemo(
    () => ({
      model,
      quant,
      platform,
      useImatrix,
      threads,
      contextLength,
      gpuLayers,
      workDir,
      ollamaName,
    }),
    [model, quant, platform, useImatrix, threads, contextLength, gpuLayers, workDir, ollamaName],
  );

  const recipe = React.useMemo(() => buildRecipe(ctx), [ctx]);

  const outputBytes = ggufSizeBytes(model, quant);
  const f16Bytes = ggufSizeBytes(model, f16);
  // HF checkpoints ship as bf16 safetensors: two bytes per parameter, no metadata
  // worth counting at this scale.
  const checkpointBytes = model.paramsB * 1e9 * 2;
  const peakBytes = checkpointBytes + f16Bytes + (quant.id === 'F16' ? 0 : outputBytes);

  const script = React.useCallback(
    () => composeScript(recipe, ctx, outputBytes),
    [recipe, ctx, outputBytes],
  );

  const scriptName = `convert-${model.id}-${quant.id.toLowerCase()}.${
    platform === 'windows' ? 'ps1' : 'sh'
  }`;

  const mirrors = GGUF_MIRRORS[model.id] ?? [];
  const ollamaEntry = OLLAMA_LIBRARY[model.id];
  const ollamaTag = ollamaEntry
    ? ollamaEntry.quantized && OLLAMA_QUANT_TAGS.has(quant.id)
      ? `${ollamaEntry.name}-${quant.id === 'F16' ? 'fp16' : quant.id.toLowerCase()}`
      : ollamaEntry.name
    : null;

  const patch = React.useCallback(
    (next: Partial<GgufConfig>) => setConfig((prev) => ({ ...prev, ...next })),
    [setConfig],
  );

  const handleModelChange = (id: string) => {
    setModelId(id);
    const nextModel = getModel(id);
    patch({
      modelId: id,
      // A stored context longer than the new model supports would generate a
      // command llama.cpp rejects at load time.
      contextLength: clamp(contextLength, 512, nextModel.contextLength),
    });
  };

  const handleQuantChange = (id: string) => {
    setQuantId(id);
    patch({ quantId: id });
  };

  const handleDownload = () => downloadFile(scriptName, script(), 'text/x-shellscript');

  return (
    <div className="space-y-8">
      <SectionHeading
        eyebrow="03 · GGUF commands"
        title="From a HuggingFace checkpoint to a served GGUF"
        description="Every command below is generated from the selection on the right and is meant to be pasted verbatim. Flags are the current llama.cpp ones — CMake builds, convert_hf_to_gguf.py, and the llama-* binaries in build/bin."
        actions={
          <>
            <CopyButton value={script} label="Copy the whole script" showLabel variant="outline" />
            <Button type="button" onClick={handleDownload}>
              <Download />
              Download script
            </Button>
          </>
        }
      />

      {mirrors.length > 0 && (
        <Card className="border-meta-200 bg-meta-50/40">
          <CardHeader className="pb-3">
            <div className="flex flex-wrap items-start justify-between gap-3">
              <div className="space-y-1">
                <CardTitle className="flex items-center gap-2 text-meta-800">
                  <Zap className="h-4 w-4 text-meta-600" />
                  Skip the conversion
                </CardTitle>
                <p className="max-w-2xl text-sm text-meta-900/70">
                  Someone has already done all of this for {model.name}. Unless you are converting a
                  fine-tune of your own, pull the finished file and go straight to step 5.
                </p>
              </div>
              <Badge variant="meta" className="font-mono">
                {quant.id}
              </Badge>
            </div>
          </CardHeader>
          <CardContent className="space-y-4">
            {mirrors.map((mirror) => (
              <div key={mirror.repo} className="space-y-1.5">
                <div className="flex flex-wrap items-center gap-2">
                  <a
                    href={`https://huggingface.co/${mirror.repo}`}
                    target="_blank"
                    rel="noreferrer noopener"
                    className="inline-flex items-center gap-1 font-mono text-xs font-medium text-meta-700 hover:underline"
                  >
                    {mirror.repo}
                    <ExternalLink className="h-3 w-3" />
                  </a>
                  <Badge variant="outline" className="border-meta-200 bg-white text-[11px]">
                    {mirror.label}
                  </Badge>
                </div>
                <CodeBlock
                  language={shellLanguage(platform)}
                  code={multiline(platform, `hf download ${mirror.repo}`, [
                    mirrorFileName(mirror.repo, quant.id),
                    `--local-dir ${joinPath(platform, workDir, 'gguf')}`,
                  ])}
                />
              </div>
            ))}

            {ollamaTag && (
              <div className="space-y-1.5">
                <div className="text-xs font-medium text-meta-900/70">
                  Or let Ollama fetch and register it in one command:
                </div>
                <CodeBlock
                  language={shellLanguage(platform)}
                  code={`ollama pull ${ollamaTag}\nollama run ${ollamaTag}`}
                />
              </div>
            )}

            <Footnote className="text-meta-900/60">
              File names follow each repo&apos;s own convention. If the exact name 404s, download by
              pattern instead —{' '}
              <InlineCode>hf download {mirrors[0].repo} --include &quot;*{quant.id}*&quot;</InlineCode>{' '}
              — or check the repo&apos;s Files tab. Unsloth&apos;s dynamic quants use their own
              UD-prefixed names.
            </Footnote>
          </CardContent>
        </Card>
      )}

      {model.gated && (
        <Alert variant="warning">
          <ShieldAlert />
          <AlertTitle>{model.repoId} is a gated repository</AlertTitle>
          <AlertDescription className="space-y-1">
            <p>
              Downloading it requires accepting the {model.license} on the model page while signed
              in, and waiting for Meta to grant access. A read-scoped token alone is not enough —
              until the request is approved every download returns 403.
            </p>
            <p className="flex flex-wrap gap-x-4 gap-y-1 pt-1">
              <a
                href={`https://huggingface.co/${model.repoId}`}
                target="_blank"
                rel="noreferrer noopener"
                className="inline-flex items-center gap-1 font-medium underline underline-offset-2"
              >
                Model page
                <ExternalLink className="h-3 w-3" />
              </a>
              {model.licenseUrl && (
                <a
                  href={model.licenseUrl}
                  target="_blank"
                  rel="noreferrer noopener"
                  className="inline-flex items-center gap-1 font-medium underline underline-offset-2"
                >
                  Licence text
                  <ExternalLink className="h-3 w-3" />
                </a>
              )}
            </p>
          </AlertDescription>
        </Alert>
      )}

      {/* ---------------------------------------------------------------- */}
      {/*  Controls                                                         */}
      {/* ---------------------------------------------------------------- */}

      <Card>
        <CardHeader className="pb-4">
          <div className="flex flex-wrap items-start justify-between gap-3">
            <div className="space-y-1">
              <CardTitle className="flex items-center gap-2">
                <TerminalSquare className="h-4 w-4 text-meta-600" />
                Recipe inputs
              </CardTitle>
              <p className="text-sm text-muted-foreground">{recipe.summary}</p>
            </div>
            <Button type="button" variant="ghost" size="sm" onClick={resetConfig}>
              <RotateCcw />
              Reset options
            </Button>
          </div>
        </CardHeader>

        <CardContent className="space-y-5">
          <div className="grid grid-cols-1 gap-4 sm:grid-cols-2 lg:grid-cols-3">
            <Field label="Model" htmlFor="gguf-model" hint={formatParams(model.paramsB)}>
              <Select value={model.id} onValueChange={handleModelChange}>
                <SelectTrigger id="gguf-model" aria-label="Model to convert">
                  <SelectValue />
                </SelectTrigger>
                <SelectContent>
                  {MODEL_FAMILIES.map((family) => (
                    <SelectGroup key={family}>
                      <SelectLabel>{family}</SelectLabel>
                      {MODELS.filter((m) => m.family === family).map((m) => (
                        <SelectItem key={m.id} value={m.id}>
                          {m.name}
                        </SelectItem>
                      ))}
                    </SelectGroup>
                  ))}
                </SelectContent>
              </Select>
            </Field>

            <Field
              label="Quantization"
              htmlFor="gguf-quant"
              hint={`${quant.bpw.toFixed(2)} bpw`}
              help="The type name passed to llama-quantize verbatim. i-quants (IQ*) cannot be built without an importance matrix."
            >
              <Select value={quant.id} onValueChange={handleQuantChange}>
                <SelectTrigger id="gguf-quant" aria-label="Quantization format">
                  <SelectValue />
                </SelectTrigger>
                <SelectContent>
                  {TIER_ORDER.map((tier) => (
                    <SelectGroup key={tier}>
                      <SelectLabel>{TIER_META[tier].label}</SelectLabel>
                      {QUANT_FORMATS.filter((q) => q.tier === tier).map((q) => (
                        <SelectItem key={q.id} value={q.id}>
                          <span className="font-mono">{q.id}</span>
                          {q.needsImatrix && (
                            <span className="ml-2 text-xs text-muted-foreground">imatrix</span>
                          )}
                        </SelectItem>
                      ))}
                    </SelectGroup>
                  ))}
                </SelectContent>
              </Select>
            </Field>

            <Field
              label="Target platform"
              help="Changes the build flags (CUDA vs Metal), the binary paths, the line-continuation character and how environment variables are set."
            >
              <div
                role="group"
                aria-label="Target platform"
                className="inline-flex w-full rounded-md border bg-muted/40 p-0.5"
              >
                {PLATFORMS.map((p) => (
                  <button
                    key={p.id}
                    type="button"
                    aria-pressed={platform === p.id}
                    onClick={() => patch({ platform: p.id })}
                    className={cn(
                      'flex-1 rounded px-2 py-1.5 text-xs font-medium transition-colors',
                      'focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring',
                      platform === p.id
                        ? 'bg-background text-meta-700 shadow-sm'
                        : 'text-muted-foreground hover:text-foreground',
                    )}
                  >
                    {p.label}
                  </button>
                ))}
              </div>
            </Field>

            <Field
              label="Threads"
              htmlFor="gguf-threads"
              help="Passed to llama-quantize as its trailing positional argument and to llama-cli / llama-server as -t. Match your physical core count, not the hyperthread count."
            >
              <NumberInput
                id="gguf-threads"
                value={threads}
                min={1}
                max={256}
                onChange={(n) => patch({ threads: n })}
                suffix="-t"
              />
            </Field>

            <Field
              label="Context length"
              htmlFor="gguf-ctx"
              hint={`max ${formatContext(model.contextLength)}`}
              help="Sets -c at serve time and num_ctx in the Modelfile. The KV cache grows linearly with this number, so it is usually what decides whether the model still fits."
            >
              <NumberInput
                id="gguf-ctx"
                value={contextLength}
                min={512}
                max={model.contextLength}
                step={512}
                onChange={(n) => patch({ contextLength: n })}
                suffix="-c"
              />
            </Field>

            <Field
              label="GPU layers"
              htmlFor="gguf-ngl"
              hint={`${model.architecture.layers} in model`}
              help="How many transformer layers to offload to the GPU. Any number at or above the layer count means all of them; 0 keeps everything on the CPU."
            >
              <NumberInput
                id="gguf-ngl"
                value={gpuLayers}
                min={0}
                max={999}
                onChange={(n) => patch({ gpuLayers: n })}
                suffix="-ngl"
              />
            </Field>

            <Field
              label="Working directory"
              htmlFor="gguf-workdir"
              help="Everything is created under here: the llama.cpp checkout, the downloaded checkpoint and a gguf/ folder for the converted files."
            >
              <Input
                id="gguf-workdir"
                value={config.workDir}
                spellCheck={false}
                autoComplete="off"
                onChange={(e) => patch({ workDir: e.target.value })}
                className="font-mono text-xs"
                placeholder="~/models"
              />
            </Field>

            <Field
              label="Ollama model name"
              htmlFor="gguf-ollama"
              hint={config.ollamaName.trim() ? undefined : 'auto'}
              help="Used for `ollama create` and as the llama-server --alias, which is the model id clients send in their requests."
            >
              <Input
                id="gguf-ollama"
                value={config.ollamaName}
                spellCheck={false}
                autoComplete="off"
                onChange={(e) => patch({ ollamaName: e.target.value })}
                className="font-mono text-xs"
                placeholder={derivedOllamaName}
              />
            </Field>

            <div className="flex items-start gap-3 rounded-lg border bg-muted/30 p-3">
              <Switch
                id="gguf-imatrix"
                checked={useImatrix}
                disabled={imatrixRequired || !imatrixApplicable}
                onCheckedChange={(checked) => patch({ useImatrix: checked })}
                aria-label="Build an importance matrix before quantizing"
              />
              <div className="min-w-0 space-y-0.5">
                <Label htmlFor="gguf-imatrix" className="text-xs font-medium">
                  Importance matrix
                </Label>
                <p className="text-[11px] leading-relaxed text-muted-foreground">
                  {!imatrixApplicable
                    ? 'Not applicable: F16 is the conversion output, so nothing is quantized.'
                    : imatrixRequired
                      ? `Forced on — ${quant.id} is an i-quant and llama-quantize refuses to build one without --imatrix.`
                      : 'Adds a calibration pass. Costs one slow forward pass over a corpus, buys measurably lower perplexity at the same file size.'}
                </p>
              </div>
            </div>
          </div>

          <Separator />

          <StatGrid cols={4}>
            <Stat
              label={
                <InfoTip label="Output file">
                  ggufSizeBytes(): transformer weights at {quant.bpw.toFixed(2)} bpw plus the
                  embedding and output tensors at their own higher rate, plus GGUF metadata.
                  Typically within ~1% of published uploads.
                </InfoTip>
              }
              value={formatGib(outputBytes)}
              sub={<span className="font-mono">{quant.id}</span>}
              tone="accent"
            />
            <Stat
              label={
                <InfoTip label="F16 intermediate">
                  The converter&apos;s output, before quantization: roughly two bytes per parameter.
                  You can delete it once llama-quantize finishes.
                </InfoTip>
              }
              value={formatGib(f16Bytes)}
              sub="deletable after step 4"
            />
            <Stat
              label={
                <InfoTip label="Peak disk">
                  checkpoint + F16 GGUF{quant.id === 'F16' ? '' : ' + quantized GGUF'} — all three
                  exist at the same time unless you delete as you go.
                </InfoTip>
              }
              value={formatGib(peakBytes)}
              sub={`checkpoint ≈ ${formatGib(checkpointBytes)}`}
              tone={peakBytes > BIG_F16_BYTES ? 'warn' : 'default'}
            />
            <Stat
              label={
                <InfoTip label="Quality cost">
                  pplDelta ÷ 5.9066 × 100, from llama.cpp&apos;s Wikitext-2 perplexity table
                  measured on LLaMA-7B. A proxy for relative degradation, not a promise about your
                  workload.
                </InfoTip>
              }
              value={quant.id === 'F16' ? '—' : formatPct(qualityLossPct(quant))}
              sub="vs F16 perplexity"
              tone={qualityLossPct(quant) > 4 ? 'warn' : 'default'}
            />
          </StatGrid>

          {f16Bytes > BIG_F16_BYTES && (
            <Alert variant="warning">
              <HardDrive />
              <AlertTitle>Check your free space before you start</AlertTitle>
              <AlertDescription>
                The F16 intermediate for {model.name} is {formatGib(f16Bytes)} and it has to exist
                alongside the original checkpoint ({formatGib(checkpointBytes)}) while
                llama-quantize reads it — about {formatGib(peakBytes)} at peak. Convert onto a fast
                drive with room to spare; a full disk halfway through step 2 loses the whole
                conversion.
              </AlertDescription>
            </Alert>
          )}
        </CardContent>
      </Card>

      {/* ---------------------------------------------------------------- */}
      {/*  Steps                                                            */}
      {/* ---------------------------------------------------------------- */}

      <ol className="space-y-4">
        {recipe.steps.map((step, index) => {
          const Icon = STEP_ICONS[step.id] ?? TerminalSquare;
          return (
            <li key={step.id}>
              <Card>
                <CardHeader className="gap-3 pb-4">
                  <div className="flex items-start gap-3">
                    <span
                      aria-hidden
                      className="tabular mt-0.5 flex h-7 w-7 shrink-0 items-center justify-center rounded-lg border border-meta-200 bg-meta-50 font-mono text-xs font-semibold text-meta-700"
                    >
                      {String(index).padStart(2, '0')}
                    </span>
                    <div className="min-w-0 flex-1 space-y-1">
                      <div className="flex flex-wrap items-center gap-2">
                        <CardTitle className="flex items-center gap-2">
                          <Icon className="h-4 w-4 text-muted-foreground" />
                          {step.title}
                        </CardTitle>
                        {step.optional && (
                          <Badge variant="muted" className="text-[11px]">
                            optional
                          </Badge>
                        )}
                        {step.blocking && (
                          <Badge variant="outline" className="text-[11px]">
                            runs until stopped
                          </Badge>
                        )}
                      </div>
                      <p className="text-pretty text-sm leading-relaxed text-muted-foreground">
                        {step.description}
                      </p>
                    </div>
                  </div>
                </CardHeader>

                <CardContent className="space-y-3">
                  <CodeBlock
                    code={step.code}
                    language={step.language}
                    title={step.writeTo}
                    showLineNumbers={step.code.split('\n').length > 6}
                  />

                  {step.extra && (
                    <CodeBlock
                      code={step.extra.code}
                      language={step.extra.language}
                      title={step.extra.title}
                    />
                  )}

                  {step.warning && (
                    <div className="flex gap-2 rounded-lg border border-warning/25 bg-warning-soft px-3 py-2 text-xs leading-relaxed text-warning">
                      <AlertTriangle className="mt-0.5 h-3.5 w-3.5 shrink-0" />
                      <span className="text-pretty">{step.warning}</span>
                    </div>
                  )}
                </CardContent>
              </Card>
            </li>
          );
        })}
      </ol>

      <Footnote>
        Commands target llama.cpp as of 2025: the CMake build (<InlineCode>cmake -B build</InlineCode>
        ), <InlineCode>convert_hf_to_gguf.py</InlineCode> in place of the removed{' '}
        <InlineCode>convert.py</InlineCode>, and the <InlineCode>llama-*</InlineCode> binaries under{' '}
        <InlineCode>build/bin/</InlineCode>. Flag names do change — if one is rejected, run the
        binary with <InlineCode>--help</InlineCode>, which is always authoritative for the build you
        compiled. Sizes are estimates from the model architecture, not measurements of your file.
      </Footnote>
    </div>
  );
}

/** localStorage can hold anything; fall back rather than emit NaN into a command. */
function safeNumber(value: number, fallback: number): number {
  return typeof value === 'number' && Number.isFinite(value) ? value : fallback;
}
