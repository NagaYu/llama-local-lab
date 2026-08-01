'use client';

import * as React from 'react';
import {
  AlertTriangle,
  ArrowRight,
  Check,
  CheckCircle2,
  ChevronDown,
  Download,
  RotateCcw,
  X,
} from 'lucide-react';

import { DEFAULT_GPU_ID, GPUS, getGpu } from '@/data/gpus';
import { DEFAULT_MODEL_ID, MODELS, MODEL_FAMILIES, getModel } from '@/data/models';
import { BASELINE_PPL, KV_BYTES, QUANT_FORMATS, TIER_META, getQuant } from '@/data/quants';
import {
  GIB,
  bytesReadPerToken,
  estimateQuant,
  maxContextFor,
  qualityGrade,
  recommendQuant,
  splitParams,
} from '@/lib/quant';
import {
  formatBytes,
  formatContext,
  formatGib,
  formatNumber,
  formatPct,
  formatTps,
} from '@/lib/format';
import { toCsv } from '@/lib/csv';
import { cn, downloadFile } from '@/lib/utils';
import { STORAGE_KEYS, mergeDefaults, usePersistentState } from '@/lib/storage';
import type {
  GpuClass,
  KvCachePrecision,
  LlamaModel,
  QuantEstimate,
  QuantPlannerConfig,
} from '@/lib/types';

import { useStudio } from '@/components/studio-provider';
import {
  Field,
  Footnote,
  MeterBar,
  NumberInput,
  SectionHeading,
  Stat,
  StatGrid,
} from '@/components/primitives';
import { InlineCode } from '@/components/code-block';
import { Alert, AlertDescription, AlertTitle } from '@/components/ui/alert';
import { Badge } from '@/components/ui/badge';
import { Button } from '@/components/ui/button';
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card';
import { Label } from '@/components/ui/label';
import {
  Select,
  SelectContent,
  SelectGroup,
  SelectItem,
  SelectLabel,
  SelectSeparator,
  SelectTrigger,
  SelectValue,
} from '@/components/ui/select';
import { Slider } from '@/components/ui/slider';
import { Switch } from '@/components/ui/switch';
import {
  Table,
  TableBody,
  TableCell,
  TableHead,
  TableHeader,
  TableRow,
} from '@/components/ui/table';
import { InfoTip, Tooltip, TooltipContent, TooltipTrigger } from '@/components/ui/tooltip';

/* -------------------------------------------------------------------------- */
/*  Constants                                                                  */
/* -------------------------------------------------------------------------- */

/**
 * Power-of-two context steps. Anything finer is false precision: llama.cpp
 * rounds the KV allocation up to whole blocks anyway, and nobody deploys with
 * a 47,000-token window.
 */
const CONTEXT_STEPS = [1024, 2048, 4096, 8192, 16_384, 32_768, 65_536, 131_072];

/** Sentinel for the "type my own VRAM number" option in the GPU select. */
const CUSTOM_GPU = 'custom';

const GPU_CLASS_ORDER: { id: GpuClass; label: string }[] = [
  { id: 'datacenter', label: 'Datacenter' },
  { id: 'workstation', label: 'Workstation' },
  { id: 'consumer', label: 'Consumer' },
  { id: 'unified', label: 'Unified memory' },
];

/** How much smaller a quantized KV cache is than f16, straight from KV_BYTES. */
function kvSavingsPct(precision: KvCachePrecision): number {
  return Math.round((1 - KV_BYTES[precision] / KV_BYTES.f16) * 100);
}

const KV_OPTIONS: { id: KvCachePrecision; label: string; detail: string }[] = [
  { id: 'f16', label: 'f16', detail: 'Reference precision' },
  { id: 'q8_0', label: 'q8_0', detail: `${kvSavingsPct('q8_0')}% smaller cache` },
  { id: 'q4_0', label: 'q4_0', detail: `${kvSavingsPct('q4_0')}% smaller cache` },
];

function isKvPrecision(value: string): value is KvCachePrecision {
  return value === 'f16' || value === 'q8_0' || value === 'q4_0';
}

const GRADE_VARIANT: Record<
  ReturnType<typeof qualityGrade>['tone'],
  'success' | 'meta' | 'warning' | 'danger'
> = {
  good: 'success',
  ok: 'meta',
  warn: 'warning',
  bad: 'danger',
};

/**
 * The RTX 4090 default matches DEFAULT_GPU_ID so the planner opens on the most
 * common "will this run at home?" configuration rather than an empty form.
 */
const DEFAULT_CONFIG: QuantPlannerConfig = {
  modelId: DEFAULT_MODEL_ID,
  vramGb: 24,
  contextLength: 8192,
  kvPrecision: 'f16',
  reservedGb: 0.8,
  gpuId: DEFAULT_GPU_ID,
  onlyFitting: false,
};

/** Table column count, used by the full-width detail and empty rows. */
const COLUMN_COUNT = 10;

/* -------------------------------------------------------------------------- */
/*  Panel                                                                      */
/* -------------------------------------------------------------------------- */

export function QuantPlannerPanel() {
  const { modelId, setModelId, quantId, setQuantId, setGpuId, goTo } = useStudio();
  const [config, setConfig, { reset: resetConfig }] = usePersistentState<QuantPlannerConfig>(
    STORAGE_KEYS.planner,
    DEFAULT_CONFIG,
    { revive: mergeDefaults(DEFAULT_CONFIG) },
  );
  const [expanded, setExpanded] = React.useState<Record<string, boolean>>({});
  const fieldId = React.useId();

  const model = getModel(modelId);
  // `null` means the user is typing a raw VRAM number, so there is no bandwidth
  // figure and every throughput cell honestly reads as unknown.
  const gpu = getGpu(config.gpuId);

  /* --- derived configuration ---------------------------------------------- */

  const contextSteps = React.useMemo(() => {
    const steps = CONTEXT_STEPS.filter((s) => s <= model.contextLength);
    return steps.length > 0 ? steps : [model.contextLength];
  }, [model.contextLength]);

  // Snap the persisted value to the nearest available step instead of writing
  // it back in an effect — that keeps the first render deterministic for SSG.
  const contextIndex = React.useMemo(() => {
    let best = 0;
    for (let i = 1; i < contextSteps.length; i += 1) {
      const closer =
        Math.abs(contextSteps[i] - config.contextLength) <
        Math.abs(contextSteps[best] - config.contextLength);
      if (closer) best = i;
    }
    return best;
  }, [contextSteps, config.contextLength]);

  const contextLength = contextSteps[contextIndex];

  const effectiveConfig = React.useMemo<QuantPlannerConfig>(
    () => ({ ...config, modelId, contextLength }),
    [config, modelId, contextLength],
  );

  /* --- estimates ----------------------------------------------------------- */

  const rows = React.useMemo(
    () => QUANT_FORMATS.map((q) => estimateQuant(model, q, effectiveConfig, gpu)),
    [model, effectiveConfig, gpu],
  );

  const recommended = React.useMemo(
    () => recommendQuant(model, QUANT_FORMATS, effectiveConfig, gpu),
    [model, effectiveConfig, gpu],
  );

  const selectedFormat = getQuant(quantId);
  const selected = React.useMemo(
    () => rows.find((r) => r.format.id === selectedFormat.id) ?? rows[0],
    [rows, selectedFormat.id],
  );

  const smallest = React.useMemo(
    () => rows.reduce((a, b) => (b.totalRamBytes < a.totalRamBytes ? b : a), rows[0]),
    [rows],
  );

  const usableGb = Math.max(0, config.vramGb - config.reservedGb);
  const budgetBytes = usableGb * GIB;
  const headroomBytes = budgetBytes - selected.totalRamBytes;
  const selectedMaxContext = maxContextFor(model, selected.format, usableGb, config.kvPrecision);

  const visibleRows = config.onlyFitting ? rows.filter((r) => r.fits) : rows;

  const recommendReason = recommended
    ? recommended.format.id === 'F16'
      ? 'Full precision fits, so there is nothing to trade away.'
      : `Lowest measured quality loss among the formats that fit ${formatGib(budgetBytes)} of usable VRAM.`
    : null;

  /* --- handlers ------------------------------------------------------------ */

  const update = React.useCallback(
    <K extends keyof QuantPlannerConfig>(key: K, value: QuantPlannerConfig[K]) => {
      setConfig((prev) => ({ ...prev, [key]: value }));
    },
    [setConfig],
  );

  const onGpuChange = (value: string) => {
    if (value === CUSTOM_GPU) {
      setConfig((prev) => ({ ...prev, gpuId: null }));
      return;
    }
    const next = getGpu(value);
    if (!next) return;
    // The rest of the studio reasons about a real GPU, so keep it in step. The
    // preset seeds VRAM but does not lock it: two 24 GB cards are a valid
    // "RTX 4090 with 48 GB" setup, and the bandwidth model still applies.
    setGpuId(next.id);
    setConfig((prev) => ({ ...prev, gpuId: next.id, vramGb: next.vramGb }));
  };

  const toggleRow = (id: string) =>
    setExpanded((prev) => ({ ...prev, [id]: !prev[id] }));

  const exportCsv = () => {
    const header = [
      'quant',
      'tier',
      'bits_per_weight',
      'file_size_gib',
      'kv_cache_gib',
      'compute_buffers_gib',
      'total_vram_gib',
      'quality_loss_pct',
      'grade',
      'tokens_per_second',
      'gpu_layers',
      'fit',
    ];
    const body = rows.map((r) => [
      r.format.id,
      r.format.tier,
      r.format.bpw,
      (r.fileSizeBytes / GIB).toFixed(3),
      (r.kvCacheBytes / GIB).toFixed(3),
      (r.overheadBytes / GIB).toFixed(3),
      (r.totalRamBytes / GIB).toFixed(3),
      r.qualityLossPct.toFixed(3),
      qualityGrade(r.qualityLossPct).grade,
      r.tokensPerSecond === null ? '' : r.tokensPerSecond.toFixed(1),
      `${r.gpuLayers}/${model.architecture.layers}`,
      r.fits ? 'fits' : r.partialFit ? 'partial' : 'too-large',
    ]);
    const meta = [
      [`# model`, model.repoId],
      [`# gpu`, gpu ? gpu.name : 'custom'],
      [`# vram_gb`, config.vramGb],
      [`# reserved_gb`, config.reservedGb],
      [`# context_tokens`, contextLength],
      [`# kv_precision`, config.kvPrecision],
    ];
    downloadFile(
      `${model.id}-quant-plan.csv`,
      toCsv([...meta, header, ...body]),
      'text/csv',
    );
  };

  /* --- render -------------------------------------------------------------- */

  return (
    <div className="space-y-8">
      <SectionHeading
        eyebrow="02 · Quantization planner"
        title="Which quantization actually fits your GPU?"
        description="Every figure below is computed from the model's own architecture and llama.cpp's published quantization table — nothing is fetched and nothing is guessed. Pick hardware, pick a context window, and read off the format that survives the trade."
        actions={
          <>
            <Button variant="outline" size="sm" onClick={exportCsv}>
              <Download />
              Export CSV
            </Button>
            <Button
              variant="ghost"
              size="sm"
              onClick={resetConfig}
              aria-label="Reset planner inputs to defaults"
            >
              <RotateCcw />
              Reset
            </Button>
          </>
        }
      />

      {/* ---------------------------------------------------------------- */}
      {/*  Inputs                                                           */}
      {/* ---------------------------------------------------------------- */}
      <Card>
        <CardHeader className="pb-4">
          <CardTitle>Your setup</CardTitle>
          <CardDescription>
            The model and GPU chosen here follow you into the GGUF generator and the cost
            calculator.
          </CardDescription>
        </CardHeader>
        <CardContent className="space-y-6">
          <div className="grid gap-5 sm:grid-cols-2 lg:grid-cols-3">
            <Field label="Model" htmlFor={`${fieldId}-model`} hint={`${model.architecture.layers} layers`}>
              <Select value={modelId} onValueChange={setModelId}>
                <SelectTrigger id={`${fieldId}-model`} aria-label="Model">
                  <SelectValue placeholder="Select a model" />
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
              label="GPU preset"
              htmlFor={`${fieldId}-gpu`}
              hint={gpu ? `${formatNumber(gpu.bandwidthGbs)} GB/s` : 'no bandwidth data'}
              help="A preset fills in VRAM capacity and, more importantly, memory bandwidth — the number that sets single-stream decode speed. Choose Custom if you only want to size memory."
            >
              <Select value={config.gpuId ?? CUSTOM_GPU} onValueChange={onGpuChange}>
                <SelectTrigger id={`${fieldId}-gpu`} aria-label="GPU preset">
                  <SelectValue placeholder="Select a GPU" />
                </SelectTrigger>
                <SelectContent>
                  <SelectGroup>
                    <SelectItem value={CUSTOM_GPU}>Custom — enter VRAM manually</SelectItem>
                  </SelectGroup>
                  <SelectSeparator />
                  {GPU_CLASS_ORDER.map(({ id, label }) => {
                    const inClass = GPUS.filter((g) => g.class === id);
                    if (inClass.length === 0) return null;
                    return (
                      <SelectGroup key={id}>
                        <SelectLabel>{label}</SelectLabel>
                        {inClass.map((g) => (
                          <SelectItem key={g.id} value={g.id}>
                            {g.name} · {g.vramGb} GB
                          </SelectItem>
                        ))}
                      </SelectGroup>
                    );
                  })}
                </SelectContent>
              </Select>
            </Field>

            <Field
              label="VRAM budget"
              htmlFor={`${fieldId}-vram`}
              hint={
                gpu && config.vramGb !== gpu.vramGb
                  ? `${gpu.name} ships ${gpu.vramGb} GB`
                  : undefined
              }
            >
              <div className="flex items-center gap-3">
                <NumberInput
                  id={`${fieldId}-vram`}
                  value={config.vramGb}
                  onChange={(n) => update('vramGb', n)}
                  min={1}
                  max={2048}
                  step={1}
                  suffix="GB"
                  className="w-28 shrink-0"
                />
                <Slider
                  role="group"
                  aria-label="VRAM budget in gigabytes"
                  className="flex-1"
                  min={1}
                  max={192}
                  step={1}
                  value={[Math.min(192, Math.max(1, config.vramGb))]}
                  onValueChange={([v]) => update('vramGb', v)}
                />
              </div>
            </Field>

            <Field
              label="Context length"
              htmlFor={`${fieldId}-ctx`}
              hint={`${formatContext(contextLength)} · ${formatNumber(contextLength)} tokens`}
              help="The KV cache is sized for a full window. Planning for 128K when you only ever send 8K wastes VRAM you could have spent on a better quant."
            >
              <Slider
                id={`${fieldId}-ctx`}
                role="group"
                aria-label={`Context length, currently ${formatNumber(contextLength)} tokens`}
                min={0}
                max={contextSteps.length - 1}
                step={1}
                value={[contextIndex]}
                onValueChange={([i]) => update('contextLength', contextSteps[i])}
              />
              <div className="flex justify-between pt-1 font-mono text-[10px] text-muted-foreground">
                <span>{formatContext(contextSteps[0])}</span>
                <span>{formatContext(contextSteps[contextSteps.length - 1])}</span>
              </div>
            </Field>

            <Field
              label="KV cache precision"
              htmlFor={`${fieldId}-kv`}
              hint={`${KV_BYTES[config.kvPrecision]} B/element`}
              help={
                <>
                  Quantizing the KV cache is what makes long context affordable — q4_0 cuts cache
                  memory by {kvSavingsPct('q4_0')}% for a small, mostly-invisible quality cost that
                  grows with context length. llama.cpp needs flash attention (
                  <InlineCode>-fa</InlineCode>) enabled before it will quantize the V cache.
                </>
              }
            >
              <Select
                value={config.kvPrecision}
                onValueChange={(v) => {
                  if (isKvPrecision(v)) update('kvPrecision', v);
                }}
              >
                <SelectTrigger id={`${fieldId}-kv`} aria-label="KV cache precision">
                  <SelectValue />
                </SelectTrigger>
                <SelectContent>
                  {KV_OPTIONS.map((opt) => (
                    <SelectItem key={opt.id} value={opt.id}>
                      <span className="font-mono">{opt.label}</span>
                      <span className="ml-2 text-muted-foreground">{opt.detail}</span>
                    </SelectItem>
                  ))}
                </SelectContent>
              </Select>
            </Field>

            <Field
              label="Reserved VRAM"
              htmlFor={`${fieldId}-reserved`}
              hint={`budget ${formatGib(budgetBytes)}`}
              help="Held back for the desktop compositor, the browser and anything else already on the card. On a headless server you can take this to zero."
            >
              <NumberInput
                id={`${fieldId}-reserved`}
                value={config.reservedGb}
                onChange={(n) => update('reservedGb', n)}
                min={0}
                max={16}
                step={0.1}
                suffix="GB"
              />
            </Field>
          </div>

          <div className="flex items-center gap-3 border-t pt-4">
            <Switch
              id={`${fieldId}-only-fitting`}
              checked={config.onlyFitting}
              onCheckedChange={(v) => update('onlyFitting', v)}
            />
            <Label htmlFor={`${fieldId}-only-fitting`} className="text-sm font-normal">
              Only show quants that fit
            </Label>
            <span className="tabular ml-auto font-mono text-xs text-muted-foreground">
              {rows.filter((r) => r.fits).length}/{rows.length} fit
            </span>
          </div>
        </CardContent>
      </Card>

      {/* ---------------------------------------------------------------- */}
      {/*  Selected quant summary                                           */}
      {/* ---------------------------------------------------------------- */}
      <section className="space-y-4">
        <div className="flex flex-wrap items-center gap-2">
          <h3 className="text-sm font-semibold">Selected</h3>
          <Badge variant="meta" className="font-mono">
            {selected.format.id}
          </Badge>
          <span className="text-sm text-muted-foreground">
            {model.name} at {formatContext(contextLength)} context
            {gpu ? ` on ${gpu.name}` : ''}
          </span>
        </div>

        <StatGrid cols={5}>
          <Stat
            label="GGUF file size"
            value={formatGib(selected.fileSizeBytes)}
            sub={`${selected.format.bpw} bits/weight`}
          />
          <Stat
            label="Total VRAM needed"
            value={formatGib(selected.totalRamBytes)}
            sub="weights + KV cache + buffers"
          />
          <Stat
            label={headroomBytes >= 0 ? 'Headroom left' : 'Over budget'}
            value={formatGib(Math.abs(headroomBytes))}
            tone={headroomBytes >= 0 ? 'good' : 'bad'}
            sub={`against ${formatGib(budgetBytes)} usable`}
          />
          <Stat
            label="Estimated speed"
            value={formatTps(selected.tokensPerSecond)}
            tone="accent"
            sub={gpu ? `single stream on ${gpu.name}` : 'pick a GPU preset'}
          />
          <Stat
            label="Max context that fits"
            value={selectedMaxContext > 0 ? formatContext(selectedMaxContext) : '—'}
            sub={
              selectedMaxContext > 0
                ? `${formatNumber(selectedMaxContext)} tokens at ${config.kvPrecision}`
                : 'weights alone exceed the budget'
            }
          />
        </StatGrid>

        <Card>
          <CardContent className="space-y-4 p-5">
            <MeterBar
              capacity={budgetBytes}
              segments={[
                {
                  label: `Weights · ${formatGib(selected.fileSizeBytes)}`,
                  value: selected.fileSizeBytes,
                  className: 'bg-meta-500',
                },
                {
                  label: `KV cache · ${formatGib(selected.kvCacheBytes)}`,
                  value: selected.kvCacheBytes,
                  className: 'bg-meta-300',
                },
                {
                  label: `Compute buffers · ${formatGib(selected.overheadBytes)}`,
                  value: selected.overheadBytes,
                  className: 'bg-meta-100',
                },
              ]}
            />
            <Footnote>
              Capacity line at {formatGib(budgetBytes)} = {formatNumber(config.vramGb, 1)} GB VRAM −{' '}
              {formatNumber(config.reservedGb, 1)} GB reserved. KV cache is{' '}
              <InlineCode>2 × {model.architecture.layers} layers × {model.architecture.kvHeads} KV
              heads × {model.architecture.hiddenSize / model.architecture.attentionHeads} head dim ×
              context × {KV_BYTES[config.kvPrecision]} B</InlineCode>.
            </Footnote>

            {/* Compact call-to-action: state the verdict, then offer the next step. */}
            <div className="flex flex-wrap items-center justify-between gap-3 rounded-lg border bg-muted/30 p-3">
              <div className="flex min-w-0 items-start gap-2 text-sm">
                {selected.fits ? (
                  <CheckCircle2 className="mt-0.5 h-4 w-4 shrink-0 text-success" aria-hidden />
                ) : (
                  <AlertTriangle className="mt-0.5 h-4 w-4 shrink-0 text-warning" aria-hidden />
                )}
                <span className="text-pretty">
                  {selected.fits ? (
                    <>
                      <span className="font-mono">{selected.format.id}</span> fits
                      {gpu ? ` on the ${gpu.name}` : ' in your budget'} with{' '}
                      <strong className="tabular font-semibold">
                        {formatGib(headroomBytes)}
                      </strong>{' '}
                      to spare.
                    </>
                  ) : selected.partialFit ? (
                    <>
                      <span className="font-mono">{selected.format.id}</span> is{' '}
                      <strong className="tabular font-semibold">
                        {formatGib(-headroomBytes)}
                      </strong>{' '}
                      over budget — only {selected.gpuLayers} of {model.architecture.layers} layers
                      would run on the GPU.
                    </>
                  ) : (
                    <>
                      <span className="font-mono">{selected.format.id}</span> needs{' '}
                      <strong className="tabular font-semibold">
                        {formatGib(-headroomBytes)}
                      </strong>{' '}
                      more than this budget, with nothing left for a single layer.
                    </>
                  )}
                </span>
              </div>
              <div className="flex shrink-0 flex-wrap gap-2">
                {!selected.fits && recommended && (
                  <Button
                    variant="outline"
                    size="sm"
                    onClick={() => setQuantId(recommended.format.id)}
                  >
                    Switch to {recommended.format.id}
                  </Button>
                )}
                <Button
                  size="sm"
                  onClick={() => goTo('gguf', { modelId: model.id, quantId: selected.format.id })}
                >
                  Generate GGUF commands
                  <ArrowRight />
                </Button>
              </div>
            </div>
          </CardContent>
        </Card>

        {!recommended && (
          <Alert variant="warning">
            <AlertTriangle />
            <AlertTitle>Nothing fits this budget</AlertTitle>
            <AlertDescription>
              Even <span className="font-mono">{smallest.format.id}</span> needs{' '}
              {formatGib(smallest.totalRamBytes)} against {formatGib(budgetBytes)} of usable VRAM.
              Drop the context length, quantize the KV cache to q4_0, or add memory — at{' '}
              {formatContext(contextLength)} the cache alone is {formatGib(smallest.kvCacheBytes)}.
            </AlertDescription>
          </Alert>
        )}
      </section>

      {/* ---------------------------------------------------------------- */}
      {/*  The table                                                        */}
      {/* ---------------------------------------------------------------- */}
      <Card className="overflow-hidden">
        <CardHeader className="pb-3">
          <CardTitle>Every llama.cpp quantization format</CardTitle>
          <CardDescription>
            Ordered from lossless down to experimental. Click a row to select it; open a row for the
            format's notes and a full memory breakdown.
          </CardDescription>
        </CardHeader>
        <CardContent className="p-0">
          <Table className="min-w-[920px]">
            <TableHeader>
              <TableRow className="hover:bg-transparent">
                <TableHead className="pl-5">Quant</TableHead>
                <TableHead>Tier</TableHead>
                <TableHead className="text-right">
                  <InfoTip label="File size">
                    GGUF size on disk, with block weights and embedding tensors quantized at
                    different rates.
                  </InfoTip>
                </TableHead>
                <TableHead className="text-right">
                  <InfoTip label="KV cache">
                    Attention cache for a full {formatContext(contextLength)} window at{' '}
                    {config.kvPrecision}. Identical across quants — it does not depend on weight
                    precision.
                  </InfoTip>
                </TableHead>
                <TableHead className="text-right">
                  <InfoTip label="Total VRAM">
                    File size + KV cache + {formatGib(rows[0].overheadBytes)} of CUDA context and
                    compute buffers.
                  </InfoTip>
                </TableHead>
                <TableHead className="text-right">
                  <InfoTip label="Quality loss (Δ ppl)">
                    Wikitext-2 perplexity increase versus F16, measured on LLaMA-7B by llama.cpp and
                    used here as a proxy. It is not a benchmark score, and it does not capture how
                    quantization hits instruction following or JSON output first.
                  </InfoTip>
                </TableHead>
                <TableHead className="text-center">Grade</TableHead>
                <TableHead className="text-right">
                  <InfoTip label="Speed">
                    {gpu
                      ? `Single-stream decode: ${formatNumber(gpu.bandwidthGbs)} GB/s × 72% achieved bandwidth ÷ bytes read per token. Prefill and batched serving are faster.`
                      : 'Needs a GPU preset — throughput follows memory bandwidth, which a raw VRAM number does not give us.'}
                  </InfoTip>
                </TableHead>
                <TableHead>Fit</TableHead>
                <TableHead className="w-10 pr-5">
                  <span className="sr-only">Details</span>
                </TableHead>
              </TableRow>
            </TableHeader>
            <TableBody>
              {visibleRows.length === 0 && (
                <TableRow className="hover:bg-transparent">
                  <TableCell colSpan={COLUMN_COUNT} className="py-10 text-center">
                    <div className="text-sm font-medium">No format fits this budget</div>
                    <div className="mt-1 text-sm text-muted-foreground">
                      Turn the filter off to see how far off each one is.
                    </div>
                    <Button
                      variant="outline"
                      size="sm"
                      className="mt-3"
                      onClick={() => update('onlyFitting', false)}
                    >
                      Show all {rows.length} formats
                    </Button>
                  </TableCell>
                </TableRow>
              )}

              {visibleRows.map((est) => {
                const q = est.format;
                const isSelected = q.id === selected.format.id;
                const isRecommended = recommended?.format.id === q.id;
                const isOpen = Boolean(expanded[q.id]);
                const grade = qualityGrade(est.qualityLossPct);
                const overBytes = est.totalRamBytes - budgetBytes;

                return (
                  <React.Fragment key={q.id}>
                    <TableRow
                      data-state={isSelected ? 'selected' : undefined}
                      onClick={() => setQuantId(q.id)}
                      className={cn(
                        'cursor-pointer transition-opacity',
                        !est.fits && 'opacity-60 hover:opacity-100',
                        isRecommended && !isSelected && 'bg-meta-50/50',
                        isOpen && 'border-b-0',
                      )}
                    >
                      <TableCell
                        className={cn(
                          'border-l-2 pl-[18px]',
                          isSelected ? 'border-l-meta-500' : 'border-l-transparent',
                        )}
                      >
                        <button
                          type="button"
                          aria-pressed={isSelected}
                          onClick={(e) => {
                            e.stopPropagation();
                            setQuantId(q.id);
                          }}
                          className="rounded text-left focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring focus-visible:ring-offset-1"
                        >
                          <span
                            className={cn(
                              'font-mono text-sm',
                              isSelected ? 'font-semibold text-meta-700' : 'font-medium',
                            )}
                          >
                            {q.label}
                          </span>
                        </button>
                        <div className="mt-1 flex flex-wrap items-center gap-1.5">
                          {isRecommended && (
                            <Badge variant="meta" className="px-1.5 py-0 text-[10px]">
                              Recommended
                            </Badge>
                          )}
                          {q.needsImatrix && (
                            <Tooltip>
                              <TooltipTrigger asChild>
                                <Badge
                                  variant="outline"
                                  className="cursor-help px-1.5 py-0 text-[10px]"
                                >
                                  needs imatrix
                                </Badge>
                              </TooltipTrigger>
                              <TooltipContent>
                                i-quants need an importance matrix built from a calibration corpus
                                before <InlineCode>llama-quantize</InlineCode> will produce them.
                                The GGUF generator emits that step for you.
                              </TooltipContent>
                            </Tooltip>
                          )}
                        </div>
                        {isRecommended && recommendReason && (
                          <div className="mt-1 max-w-[15rem] text-pretty text-[11px] leading-snug text-meta-700">
                            {recommendReason}
                          </div>
                        )}
                      </TableCell>

                      <TableCell>
                        <span
                          className={cn(
                            'inline-flex items-center gap-1.5 whitespace-nowrap rounded-full border px-2 py-0.5 text-[11px] font-medium',
                            TIER_META[q.tier].className,
                          )}
                        >
                          <span
                            className={cn('h-1.5 w-1.5 rounded-full', TIER_META[q.tier].dot)}
                            aria-hidden
                          />
                          {TIER_META[q.tier].label}
                        </span>
                      </TableCell>

                      <TableCell className="tabular text-right font-mono">
                        {formatGib(est.fileSizeBytes)}
                      </TableCell>
                      <TableCell className="tabular text-right font-mono text-muted-foreground">
                        {formatGib(est.kvCacheBytes)}
                      </TableCell>
                      <TableCell className="tabular text-right font-mono font-medium">
                        {formatGib(est.totalRamBytes)}
                      </TableCell>

                      <TableCell className="text-right">
                        <Tooltip>
                          <TooltipTrigger asChild>
                            <span className="tabular cursor-help font-mono">
                              {formatPct(est.qualityLossPct, 2)}
                            </span>
                          </TooltipTrigger>
                          <TooltipContent>
                            Wikitext-2 perplexity rises by {q.pplDelta.toFixed(4)} from the{' '}
                            {formatNumber(5.9066, 4)} F16 baseline measured on LLaMA-7B by
                            llama.cpp. That is a proxy for quality, not a benchmark score, and it
                            was measured on a different model than the one selected.
                          </TooltipContent>
                        </Tooltip>
                        <div className="text-[10px] text-muted-foreground">
                          +{q.pplDelta.toFixed(4)} ppl
                        </div>
                      </TableCell>

                      <TableCell className="text-center">
                        <Badge
                          variant={GRADE_VARIANT[grade.tone]}
                          className="min-w-[2.25rem] justify-center px-1.5 font-mono"
                        >
                          {grade.grade}
                        </Badge>
                      </TableCell>

                      <TableCell className="tabular text-right font-mono">
                        {est.tokensPerSecond === null ? (
                          <span className="text-muted-foreground">—</span>
                        ) : (
                          <Tooltip>
                            <TooltipTrigger asChild>
                              <span className={cn('cursor-help', !est.fits && 'text-warning')}>
                                {formatTps(est.tokensPerSecond)}
                              </span>
                            </TooltipTrigger>
                            <TooltipContent>
                              {formatBytes(bytesReadPerToken(model, q))} of weights are read per
                              decoded token.
                              {est.fits
                                ? ' Single stream, fully resident on the GPU.'
                                : ' Already penalised for streaming the offloaded layers over PCIe every token.'}
                            </TooltipContent>
                          </Tooltip>
                        )}
                      </TableCell>

                      <TableCell>
                        <FitCell est={est} layers={model.architecture.layers} over={overBytes} />
                      </TableCell>

                      <TableCell className="pr-5 text-right">
                        <button
                          type="button"
                          aria-expanded={isOpen}
                          aria-label={`${isOpen ? 'Hide' : 'Show'} details for ${q.label}`}
                          onClick={(e) => {
                            e.stopPropagation();
                            toggleRow(q.id);
                          }}
                          className="inline-flex h-7 w-7 items-center justify-center rounded-md text-muted-foreground hover:bg-accent hover:text-foreground focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring"
                        >
                          <ChevronDown
                            className={cn('h-4 w-4 transition-transform', isOpen && 'rotate-180')}
                            aria-hidden
                          />
                        </button>
                      </TableCell>
                    </TableRow>

                    {isOpen && (
                      <TableRow className="hover:bg-transparent">
                        <TableCell colSpan={COLUMN_COUNT} className="bg-muted/25 px-5 py-4">
                          <div className="space-y-4">
                            <p className="max-w-3xl text-pretty text-sm leading-relaxed">
                              {q.notes}
                            </p>
                            <dl className="grid grid-cols-2 gap-x-6 gap-y-2 sm:grid-cols-3 lg:grid-cols-5">
                              <DetailItem label="Block bits/weight" value={`${q.bpw}`} />
                              <DetailItem
                                label={
                                  model.architecture.tiedEmbeddings
                                    ? 'Tied embedding bits'
                                    : 'Embedding bits'
                                }
                                value={`${
                                  model.architecture.tiedEmbeddings ? q.outputBpw : q.embedBpw
                                }`}
                              />
                              <DetailItem
                                label="Embedding share"
                                value={formatPct(embedSharePct(model), 1)}
                              />
                              <DetailItem
                                label="Layers on GPU"
                                value={`${est.gpuLayers}/${model.architecture.layers}`}
                              />
                              <DetailItem
                                label="Max context here"
                                value={
                                  maxContextFor(model, q, usableGb, config.kvPrecision) > 0
                                    ? formatContext(
                                        maxContextFor(model, q, usableGb, config.kvPrecision),
                                      )
                                    : '—'
                                }
                              />
                            </dl>
                            <div className="flex flex-wrap gap-2">
                              <Button
                                variant="outline"
                                size="sm"
                                onClick={() => setQuantId(q.id)}
                                disabled={isSelected}
                              >
                                {isSelected ? 'Selected' : 'Select this quant'}
                              </Button>
                              <Button
                                variant="ghost"
                                size="sm"
                                onClick={() =>
                                  goTo('gguf', { modelId: model.id, quantId: q.id })
                                }
                              >
                                Build {q.label} for {model.name}
                                <ArrowRight />
                              </Button>
                            </div>
                          </div>
                        </TableCell>
                      </TableRow>
                    )}
                  </React.Fragment>
                );
              })}
            </TableBody>
          </Table>
        </CardContent>
      </Card>

      <Footnote className="max-w-3xl">
        <strong className="font-semibold text-foreground">How these sizes are computed.</strong>{' '}
        Bits-per-weight is applied separately to the transformer blocks and to the{' '}
        <InlineCode>token_embd</InlineCode> / <InlineCode>output</InlineCode> tensors, which
        <InlineCode>llama-quantize</InlineCode> deliberately keeps at higher precision. On Llama 3+
        the 128k-token vocabulary makes those tensors large enough to matter — roughly 21% of Llama
        3.2 1B — so the flat <InlineCode>params × bits ÷ 8</InlineCode> formula is 10–25% wrong on
        small models. Splitting them lands these estimates within about 1% of published GGUF
        uploads. Throughput is a memory-bandwidth model at 72% achieved utilisation, calibrated
        against reported llama.cpp figures; treat it as an order-of-magnitude guide, not a
        benchmark. Nothing here is measured on your machine.
      </Footnote>
    </div>
  );
}

/* -------------------------------------------------------------------------- */
/*  Cells                                                                      */
/* -------------------------------------------------------------------------- */

/**
 * The fit verdict. The visible text stays short so the column scans; the full
 * explanation is duplicated into an `sr-only` span because a Radix tooltip on a
 * non-focusable cell is mouse-only.
 */
function FitCell({
  est,
  layers,
  over,
}: {
  est: QuantEstimate;
  layers: number;
  over: number;
}) {
  if (est.fits) {
    const detail = `Fits with ${formatGib(-over)} to spare.`;
    return (
      <Tooltip>
        <TooltipTrigger asChild>
          <span className="inline-flex cursor-help items-center gap-1.5 whitespace-nowrap text-sm font-medium text-success">
            <Check className="h-3.5 w-3.5" aria-hidden />
            Fits
            <span className="sr-only"> — {detail}</span>
          </span>
        </TooltipTrigger>
        <TooltipContent>
          All {layers} layers, the KV cache and the compute buffers sit inside the budget with{' '}
          {formatGib(-over)} left over.
        </TooltipContent>
      </Tooltip>
    );
  }

  if (est.partialFit) {
    return (
      <Tooltip>
        <TooltipTrigger asChild>
          <span className="inline-flex cursor-help items-center gap-1.5 whitespace-nowrap text-sm font-medium text-warning">
            <AlertTriangle className="h-3.5 w-3.5" aria-hidden />
            Partial — {est.gpuLayers}/{layers} layers
            <span className="sr-only">
              {' '}
              — {formatGib(over)} over budget; the remaining layers run on the CPU.
            </span>
          </span>
        </TooltipTrigger>
        <TooltipContent>
          {formatGib(over)} over budget. llama.cpp will keep {layers - est.gpuLayers} layers in
          system RAM and stream them across PCIe for every single token, so throughput collapses —
          the tok/s figure already reflects that penalty.
        </TooltipContent>
      </Tooltip>
    );
  }

  return (
    <Tooltip>
      <TooltipTrigger asChild>
        <span className="inline-flex cursor-help items-center gap-1.5 whitespace-nowrap text-sm font-medium text-danger">
          <X className="h-3.5 w-3.5" aria-hidden />
          Too large
          <span className="sr-only"> — {formatGib(over)} over budget.</span>
        </span>
      </TooltipTrigger>
      <TooltipContent>
        {formatGib(over)} over budget. The KV cache and compute buffers alone leave no room for a
        single transformer layer, so there is nothing to offload.
      </TooltipContent>
    </Tooltip>
  );
}

function DetailItem({ label, value }: { label: string; value: string }) {
  return (
    <div>
      <dt className="text-[11px] uppercase tracking-wide text-muted-foreground">{label}</dt>
      <dd className="tabular font-mono text-sm">{value}</dd>
    </div>
  );
}

/* -------------------------------------------------------------------------- */
/*  Local helpers                                                              */
/* -------------------------------------------------------------------------- */

/**
 * Share of the checkpoint's parameters living in the embedding/output tensors.
 * Surfaced per row because it is the whole reason two formats with the same
 * nominal bit width can differ by hundreds of megabytes on a small model.
 */
function embedSharePct(model: LlamaModel): number {
  const { embedParams, totalParams } = splitParams(model);
  return (embedParams / totalParams) * 100;
}
