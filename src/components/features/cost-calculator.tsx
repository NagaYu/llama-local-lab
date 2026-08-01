'use client';

import * as React from 'react';
import { AlertTriangle, ArrowUpRight, Download, Plus, RotateCcw, Table2, X } from 'lucide-react';

import { GPUS, PRICE_SNAPSHOT_DATE, PROVIDERS, getGpu } from '@/data/gpus';
import { MODELS, MODEL_FAMILIES, getModel } from '@/data/models';
import { QUANT_FORMATS, TIER_META, getQuant } from '@/data/quants';
import {
  API_REFERENCE_PRICES,
  DAYS_PER_MONTH,
  DEFAULT_COST_INPUTS,
  DEFAULT_SELF_HOST,
  HOURS_PER_MONTH,
  HOURS_PER_YEAR,
  apiMonthlyCost,
  computeCosts,
  resolvePrices,
  resolveThroughputOverride,
  selfHostHourly,
  throughputKey,
} from '@/lib/cost';
import { toCsv } from '@/lib/csv';
import { estimateThroughput } from '@/lib/quant';
import {
  formatCompact,
  formatContext,
  formatNumber,
  formatPct,
  formatTps,
  formatUsd,
} from '@/lib/format';
import { STORAGE_KEYS, mergeDefaults, usePersistentState } from '@/lib/storage';
import { cn, downloadFile } from '@/lib/utils';
import type { CostInputs, CostResult, GpuPriceTable, SelfHostAssumptions } from '@/lib/types';

import { CopyButton } from '@/components/copy-button';
import { Field, Footnote, NumberInput, SectionHeading, Stat, StatGrid } from '@/components/primitives';
import { useStudio } from '@/components/studio-provider';
import { Alert, AlertDescription, AlertTitle } from '@/components/ui/alert';
import { Badge } from '@/components/ui/badge';
import { Button } from '@/components/ui/button';
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card';
import {
  Dialog,
  DialogClose,
  DialogContent,
  DialogDescription,
  DialogFooter,
  DialogHeader,
  DialogTitle,
  DialogTrigger,
} from '@/components/ui/dialog';
import {
  Select,
  SelectContent,
  SelectGroup,
  SelectItem,
  SelectLabel,
  SelectTrigger,
  SelectValue,
} from '@/components/ui/select';
import { Slider } from '@/components/ui/slider';
import {
  Table,
  TableBody,
  TableCell,
  TableHead,
  TableHeader,
  TableRow,
} from '@/components/ui/table';
import { InfoTip } from '@/components/ui/tooltip';

/* -------------------------------------------------------------------------- */
/*  Static option lists                                                        */
/* -------------------------------------------------------------------------- */

/** Stable identity so `usePersistentState`'s reset always lands on the same object. */
const NO_OVERRIDES: GpuPriceTable = {};
const NO_TPS_OVERRIDES: Record<string, number> = {};

/** Self-hosted pricing is computed, not rented, so it has no editable cells. */
const CLOUD_PROVIDERS = PROVIDERS.filter((p) => p.kind !== 'self-hosted');

const MODEL_GROUPS = MODEL_FAMILIES.map((family) => ({
  family,
  models: MODELS.filter((m) => m.family === family),
}));

const QUANT_TIERS = ['lossless', 'recommended', 'aggressive', 'extreme'] as const;

const GPU_GROUPS = [
  { label: 'Datacenter', gpus: GPUS.filter((g) => g.class === 'datacenter') },
  { label: 'Workstation', gpus: GPUS.filter((g) => g.class === 'workstation') },
  { label: 'Consumer', gpus: GPUS.filter((g) => g.class === 'consumer') },
  { label: 'Unified memory', gpus: GPUS.filter((g) => g.class === 'unified') },
];

/**
 * Batch sizes worth reasoning about. A linear 1–128 slider spends 80% of its
 * travel in a range nobody deploys, so the control steps through the sizes
 * real serving stacks are actually configured with.
 */
const BATCH_STEPS = [1, 2, 4, 8, 12, 16, 24, 32, 48, 64, 96, 128];

/** Presets keep a 1:3 output:input ratio — roughly what RAG and chat traffic looks like. */
const WORKLOAD_PRESETS = [
  { id: 'side', label: 'Side project 100K/day', output: 100_000, input: 300_000 },
  { id: 'startup', label: 'Startup 5M/day', output: 5_000_000, input: 15_000_000 },
  { id: 'scale', label: 'Scale 100M/day', output: 100_000_000, input: 300_000_000 },
];

/* -------------------------------------------------------------------------- */
/*  Pure helpers                                                               */
/* -------------------------------------------------------------------------- */

function nearestBatchIndex(value: number): number {
  let best = 0;
  for (let i = 1; i < BATCH_STEPS.length; i += 1) {
    if (Math.abs(BATCH_STEPS[i] - value) < Math.abs(BATCH_STEPS[best] - value)) best = i;
  }
  return best;
}

/** `Infinity` is a legitimate result of the demand model; it is not a price. */
function finiteOrNull(n: number | null | undefined): number | null {
  return typeof n === 'number' && Number.isFinite(n) ? n : null;
}

function cheapestFinite(results: CostResult[]): CostResult | null {
  let best: CostResult | null = null;
  for (const r of results) {
    const monthly = finiteOrNull(r.monthlyUsd);
    if (monthly === null) continue;
    if (best === null || monthly < (best.monthlyUsd ?? Infinity)) best = r;
  }
  return best;
}

/** Rescale the workload while holding the input:output shape constant. */
function atVolume(inputs: CostInputs, outputPerDay: number): CostInputs {
  const ratio =
    inputs.outputTokensPerDay > 0 ? inputs.inputTokensPerDay / inputs.outputTokensPerDay : 3;
  return {
    ...inputs,
    outputTokensPerDay: outputPerDay,
    inputTokensPerDay: outputPerDay * ratio,
  };
}

const BREAK_EVEN_MIN = 1e3;
const BREAK_EVEN_MAX = 1e12;

/**
 * Output tokens/day where the cheapest self-serve option and a managed API cost
 * the same.
 *
 * Dividing the two per-token rates would be wrong: rented GPU-hours are linear
 * in volume, but self-hosting carries a fixed monthly floor (capex accrues on
 * all 730 hours whether or not you use them) and steps up whole GPUs at a time.
 * Bisecting the real cost function keeps those discontinuities honest.
 */
function breakEvenOutputPerDay(
  inputs: CostInputs,
  overrides: GpuPriceTable,
  price: { inputUsd: number; outputUsd: number },
  tpsOverrides: Record<string, number>,
): number | null {
  const delta = (outputPerDay: number): number | null => {
    const scaled = atVolume(inputs, outputPerDay);
    const best = cheapestFinite(computeCosts(scaled, overrides, tpsOverrides).results);
    const monthly = finiteOrNull(best?.monthlyUsd);
    if (monthly === null) return null;
    return monthly - apiMonthlyCost(price, scaled);
  };

  const low = delta(BREAK_EVEN_MIN);
  const high = delta(BREAK_EVEN_MAX);
  if (low === null || high === null) return null;
  if (low === 0) return BREAK_EVEN_MIN;
  if ((low > 0) === (high > 0)) return null;

  let lo = Math.log10(BREAK_EVEN_MIN);
  let hi = Math.log10(BREAK_EVEN_MAX);
  for (let i = 0; i < 44; i += 1) {
    const mid = (lo + hi) / 2;
    const d = delta(10 ** mid);
    if (d === null) return null;
    if ((d > 0) === (low > 0)) lo = mid;
    else hi = mid;
  }
  return 10 ** ((lo + hi) / 2);
}

/** Seed value offered when adding a price a provider does not currently list. */
function suggestedPrice(prices: GpuPriceTable, gpuId: string): number {
  const known = Object.values(prices)
    .map((row) => row[gpuId])
    .filter((v): v is number => typeof v === 'number' && Number.isFinite(v) && v > 0);
  if (known.length === 0) return 0.5;
  return Number((known.reduce((a, b) => a + b, 0) / known.length).toFixed(2));
}

/* -------------------------------------------------------------------------- */
/*  Editable price cell                                                        */
/* -------------------------------------------------------------------------- */

function PriceCell({
  providerName,
  gpuName,
  price,
  suggestion,
  edited,
  showSuffix = false,
  className,
  onChange,
}: {
  providerName: string;
  gpuName: string;
  price: number | null;
  suggestion: number;
  edited: boolean;
  showSuffix?: boolean;
  className?: string;
  onChange: (next: number | null) => void;
}) {
  if (price === null) {
    return (
      <div className={cn('flex items-center justify-end gap-1.5', className)}>
        <span className="text-xs text-muted-foreground">Not listed</span>
        <Button
          type="button"
          variant="ghost"
          size="icon-sm"
          aria-label={`Add an hourly price for ${gpuName} on ${providerName}`}
          onClick={() => onChange(suggestion)}
        >
          <Plus />
        </Button>
      </div>
    );
  }

  return (
    <div className={cn('flex items-center justify-end gap-1', className)}>
      <NumberInput
        value={price}
        min={0}
        step={0.01}
        suffix={showSuffix ? '$/hr' : undefined}
        className={showSuffix ? 'w-36' : 'w-24'}
        aria-label={`Hourly price for ${gpuName} on ${providerName}, US dollars`}
        onChange={(n) => onChange(n)}
      />
      <Button
        type="button"
        variant="ghost"
        size="icon-sm"
        aria-label={
          edited
            ? `Mark ${gpuName} as not offered by ${providerName}`
            : `Mark ${gpuName} as not offered by ${providerName}`
        }
        onClick={() => onChange(null)}
      >
        <X />
      </Button>
    </div>
  );
}

/* -------------------------------------------------------------------------- */
/*  Panel                                                                      */
/* -------------------------------------------------------------------------- */

export function CostCalculatorPanel() {
  const { modelId, setModelId, quantId, setQuantId, gpuId, setGpuId, goTo, hydrated } = useStudio();

  const [inputs, setInputs, { hydrated: inputsHydrated }] = usePersistentState<CostInputs>(
    STORAGE_KEYS.cost,
    DEFAULT_COST_INPUTS,
    { revive: mergeDefaults(DEFAULT_COST_INPUTS) },
  );
  const [overrides, setOverrides, { reset: resetOverrides }] = usePersistentState<GpuPriceTable>(
    STORAGE_KEYS.priceOverrides,
    NO_OVERRIDES,
  );
  // Measured tokens/sec, keyed by GPU + model + quant. Throughput is the least
  // trustworthy number here — it moves with the backend, driver and batching
  // strategy — so anyone who has benchmarked their own stack can type that
  // number in and have every cost below re-derive from it.
  const [tpsOverrides, setTpsOverrides, { reset: resetTpsOverrides }] = usePersistentState<
    Record<string, number>
  >(STORAGE_KEYS.throughputOverrides, NO_TPS_OVERRIDES);

  const [breakdownOpen, setBreakdownOpen] = React.useState(false);
  const [matrixOpen, setMatrixOpen] = React.useState(false);

  // The studio selection is shared across all six tools, so it wins over
  // whatever this panel last persisted — otherwise picking a model on the
  // Models tab would silently price a different one here.
  React.useEffect(() => {
    if (!hydrated || !inputsHydrated) return;
    setInputs((prev) =>
      prev.modelId === modelId && prev.quantId === quantId && prev.gpuId === gpuId
        ? prev
        : { ...prev, modelId, quantId, gpuId },
    );
  }, [hydrated, inputsHydrated, modelId, quantId, gpuId, setInputs]);

  const patch = React.useCallback(
    (next: Partial<CostInputs>) => setInputs((prev) => ({ ...prev, ...next })),
    [setInputs],
  );
  const patchSelfHost = React.useCallback(
    (next: Partial<SelfHostAssumptions>) =>
      setInputs((prev) => ({ ...prev, selfHost: { ...prev.selfHost, ...next } })),
    [setInputs],
  );

  const model = getModel(inputs.modelId);
  const quant = getQuant(inputs.quantId);
  const gpu = getGpu(inputs.gpuId);

  const { demand, results } = React.useMemo(
    () => computeCosts(inputs, overrides, tpsOverrides),
    [inputs, overrides, tpsOverrides],
  );

  const prices = React.useMemo(() => resolvePrices(overrides), [overrides]);

  // What the user measured for this exact selection, and what we would have
  // predicted — shown side by side so replacing the estimate is an informed act.
  const measuredTps = resolveThroughputOverride(inputs, tpsOverrides);
  const estimatedTps = React.useMemo(() => {
    const gpuForEstimate = getGpu(inputs.gpuId);
    if (!gpuForEstimate) return 0;
    return estimateThroughput(model, quant, gpuForEstimate, {
      contextLength: Math.min(8192, model.contextLength),
      kvPrecision: 'f16',
      batchSize: 1,
    }).decodeSingle;
  }, [inputs.gpuId, model, quant]);
  const selfCost = React.useMemo(
    () => selfHostHourly(inputs.gpuId, inputs.selfHost),
    [inputs.gpuId, inputs.selfHost],
  );

  const outputPerMonth = inputs.outputTokensPerDay * DAYS_PER_MONTH;

  /** Normalize non-finite results into explicit "unavailable" rows, then sort. */
  const rows = React.useMemo(() => {
    const normalized = results.map((r) => {
      const monthlyUsd = finiteOrNull(r.monthlyUsd);
      const unavailableReason =
        r.unavailableReason ??
        (monthlyUsd === null
          ? 'This GPU cannot finish the month’s tokens at any utilization.'
          : undefined);
      return {
        ...r,
        hourlyUsd: finiteOrNull(r.hourlyUsd),
        gpuHoursPerMonth: Number.isFinite(r.gpuHoursPerMonth) ? r.gpuHoursPerMonth : NaN,
        monthlyUsd,
        usdPerMillionOutput: finiteOrNull(r.usdPerMillionOutput),
        unavailableReason,
      };
    });
    return normalized.sort((a, b) => {
      if (a.monthlyUsd === null && b.monthlyUsd === null) {
        return a.provider.name.localeCompare(b.provider.name);
      }
      if (a.monthlyUsd === null) return 1;
      if (b.monthlyUsd === null) return -1;
      return a.monthlyUsd - b.monthlyUsd;
    });
  }, [results]);

  const cheapest = rows.find((r) => r.monthlyUsd !== null) ?? null;

  const apiRows = React.useMemo(
    () =>
      API_REFERENCE_PRICES.map((price) => {
        const monthly = apiMonthlyCost(price, inputs);
        return {
          price,
          monthly,
          perMillionOutput: outputPerMonth > 0 ? monthly / (outputPerMonth / 1_000_000) : null,
          breakEven: breakEvenOutputPerDay(inputs, overrides, price, tpsOverrides),
        };
      }),
    [inputs, overrides, outputPerMonth, tpsOverrides],
  );

  const cheapestApi = React.useMemo(
    () => apiRows.reduce<(typeof apiRows)[number] | null>((best, r) => (best === null || r.monthly < best.monthly ? r : best), null),
    [apiRows],
  );

  const overrideCount = React.useMemo(
    () => Object.values(overrides).reduce((n, row) => n + Object.keys(row).length, 0),
    [overrides],
  );

  // Mirrors the serving context cost.ts sizes the KV cache for: provisioning the
  // model's full 128k window per request is not what anyone actually deploys.
  const servingContext = Math.min(8192, model.contextLength);

  const setPrice = React.useCallback(
    (providerId: string, targetGpuId: string, value: number | null) => {
      setOverrides((prev) => ({
        ...prev,
        [providerId]: { ...(prev[providerId] ?? {}), [targetGpuId]: value },
      }));
    },
    [setOverrides],
  );

  const buildSummary = React.useCallback(() => {
    const lines = [
      `Llama Local Lab — inference cost estimate`,
      `Model: ${model.name} (${model.repoId})`,
      `Quant: ${quant.label} @ ${quant.bpw} bpw · GPU: ${gpu?.name ?? inputs.gpuId}`,
      `Workload: ${formatNumber(inputs.outputTokensPerDay)} output + ${formatNumber(inputs.inputTokensPerDay)} input tokens/day`,
      `Batch ${inputs.batchSize} · target utilization ${formatPct(inputs.targetUtilization * 100, 0)}`,
      '',
    ];
    if (demand) {
      lines.push(
        `Decode ${formatTps(demand.throughput.decodeSingle)} single / ${formatTps(demand.throughput.decodeBatched)} batched · prefill ${formatTps(demand.throughput.prefill)}`,
        `${formatNumber(demand.provisionedHours, 0)} GPU-hours/month across ${demand.gpusNeeded} GPU(s)`,
        '',
      );
    }
    for (const r of rows) {
      lines.push(
        r.unavailableReason
          ? `${r.provider.name}: ${r.unavailableReason}`
          : `${r.provider.name}: ${formatUsd(r.monthlyUsd, 0)}/mo (${formatUsd(r.hourlyUsd)}/GPU-hr, ${formatUsd(r.usdPerMillionOutput)} per 1M output tokens)`,
      );
    }
    lines.push('', `Prices from the ${PRICE_SNAPSHOT_DATE} snapshot; estimates, not quotes.`);
    return lines.join('\n');
  }, [demand, gpu, inputs, model, quant, rows]);

  const exportCsv = React.useCallback(() => {
    const header = [
      'provider',
      'kind',
      'usd_per_gpu_hour',
      'gpu_hours_per_month',
      'monthly_usd',
      'usd_per_million_output_tokens',
      'unavailable_reason',
    ];
    const body = rows.map((r) => [
      r.provider.name,
      r.provider.kind,
      r.hourlyUsd ?? '',
      Number.isFinite(r.gpuHoursPerMonth) ? Math.round(r.gpuHoursPerMonth) : '',
      r.monthlyUsd === null ? '' : r.monthlyUsd.toFixed(2),
      r.usdPerMillionOutput === null ? '' : r.usdPerMillionOutput.toFixed(4),
      r.unavailableReason ?? '',
    ]);
    downloadFile(
      `llama-cost-${inputs.modelId}-${inputs.quantId}-${inputs.gpuId}.csv`,
      toCsv([header, ...body]),
      'text/csv',
    );
  }, [inputs.gpuId, inputs.modelId, inputs.quantId, rows]);

  const batchIndex = nearestBatchIndex(inputs.batchSize);
  const utilizationPct = Math.round(inputs.targetUtilization * 100);

  return (
    <div className="space-y-8">
      <SectionHeading
        eyebrow="04 · Inference cost"
        title="What this workload actually costs to serve"
        description="Throughput first, then money. The calculator sizes the GPU-hours your token volume needs, then prices those hours against rented capacity and against owning the hardware. Every price is editable, because the ones shipped here went stale the day they were written."
        actions={
          <>
            <Button type="button" variant="outline" size="sm" onClick={exportCsv}>
              <Download />
              Export CSV
            </Button>
            <CopyButton value={buildSummary} label="Copy estimate" showLabel variant="outline" />
          </>
        }
      />

      {/* ------------------------------------------------------------------ */}
      {/*  Workload                                                          */}
      {/* ------------------------------------------------------------------ */}
      <Card>
        <CardHeader>
          <CardTitle>Workload</CardTitle>
          <CardDescription>
            What you are serving, and how hard you intend to push the card.
          </CardDescription>
        </CardHeader>
        <CardContent className="space-y-6">
          <div className="grid gap-4 sm:grid-cols-2 lg:grid-cols-3">
            <Field label="Model" htmlFor="cost-model">
              <Select
                value={inputs.modelId}
                onValueChange={(v) => {
                  setModelId(v);
                  patch({ modelId: v });
                }}
              >
                <SelectTrigger id="cost-model" aria-label="Model to price">
                  <SelectValue placeholder="Select a model" />
                </SelectTrigger>
                <SelectContent>
                  {MODEL_GROUPS.map((group) => (
                    <SelectGroup key={group.family}>
                      <SelectLabel>{group.family}</SelectLabel>
                      {group.models.map((m) => (
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
              hint={`${quant.bpw} bpw`}
              htmlFor="cost-quant"
              help="Fewer bits per weight means fewer bytes read per token, which is exactly proportional to decode speed on a memory-bound GPU."
            >
              <Select
                value={inputs.quantId}
                onValueChange={(v) => {
                  setQuantId(v);
                  patch({ quantId: v });
                }}
              >
                <SelectTrigger id="cost-quant" aria-label="Quantization format">
                  <SelectValue placeholder="Select a quant" />
                </SelectTrigger>
                <SelectContent>
                  {QUANT_TIERS.map((tier) => (
                    <SelectGroup key={tier}>
                      <SelectLabel>{TIER_META[tier].label}</SelectLabel>
                      {QUANT_FORMATS.filter((q) => q.tier === tier).map((q) => (
                        <SelectItem key={q.id} value={q.id} className="font-mono">
                          {q.label} · {q.bpw} bpw
                        </SelectItem>
                      ))}
                    </SelectGroup>
                  ))}
                </SelectContent>
              </Select>
            </Field>

            <Field
              label="GPU"
              hint={gpu ? `${gpu.vramGb} GB · ${formatNumber(gpu.bandwidthGbs)} GB/s` : undefined}
              htmlFor="cost-gpu"
            >
              <Select
                value={inputs.gpuId}
                onValueChange={(v) => {
                  setGpuId(v);
                  patch({ gpuId: v });
                }}
              >
                <SelectTrigger id="cost-gpu" aria-label="GPU to price">
                  <SelectValue placeholder="Select a GPU" />
                </SelectTrigger>
                <SelectContent>
                  {GPU_GROUPS.map((group) => (
                    <SelectGroup key={group.label}>
                      <SelectLabel>{group.label}</SelectLabel>
                      {group.gpus.map((g) => (
                        <SelectItem key={g.id} value={g.id}>
                          {g.name}
                        </SelectItem>
                      ))}
                    </SelectGroup>
                  ))}
                </SelectContent>
              </Select>
            </Field>
          </div>

          <div className="grid gap-4 sm:grid-cols-2">
            <Field
              label="Output tokens per day"
              hint={formatCompact(inputs.outputTokensPerDay)}
              htmlFor="cost-output"
              help="Generated tokens. These are the expensive ones: every single token requires reading the model's weights out of memory again."
            >
              <NumberInput
                id="cost-output"
                value={inputs.outputTokensPerDay}
                min={0}
                step={1000}
                suffix="tok"
                onChange={(n) => patch({ outputTokensPerDay: n })}
              />
            </Field>
            <Field
              label="Input tokens per day"
              hint={formatCompact(inputs.inputTokensPerDay)}
              htmlFor="cost-input"
              help="Prompt tokens. Prefill processes them in parallel at compute-bound speed, so they cost roughly an order of magnitude less per token than output."
            >
              <NumberInput
                id="cost-input"
                value={inputs.inputTokensPerDay}
                min={0}
                step={1000}
                suffix="tok"
                onChange={(n) => patch({ inputTokensPerDay: n })}
              />
            </Field>
          </div>

          <div className="flex flex-wrap items-center gap-2">
            <span className="text-xs font-medium text-muted-foreground">Presets</span>
            {WORKLOAD_PRESETS.map((preset) => {
              const active =
                inputs.outputTokensPerDay === preset.output &&
                inputs.inputTokensPerDay === preset.input;
              return (
                <Button
                  key={preset.id}
                  type="button"
                  variant={active ? 'default' : 'outline'}
                  size="sm"
                  onClick={() =>
                    patch({ outputTokensPerDay: preset.output, inputTokensPerDay: preset.input })
                  }
                >
                  {preset.label}
                </Button>
              );
            })}
            <span className="text-[11px] text-muted-foreground">
              Each preset assumes three prompt tokens per generated token.
            </span>
          </div>

          <div className="grid gap-6 sm:grid-cols-2">
            <div className="space-y-2" role="group" aria-labelledby="cost-batch-label">
              <div className="flex items-baseline justify-between gap-2">
                <span id="cost-batch-label" className="text-xs font-medium text-foreground">
                  <InfoTip label="Batch size">
                    Concurrent requests the server decodes together. One weight read serves the
                    whole batch, so aggregate throughput climbs with batch size while the memory
                    traffic per step stays roughly flat. This is where almost all of the cost
                    reduction in a serving stack comes from — far more than picking a cheaper
                    provider.
                  </InfoTip>
                </span>
                <span className="tabular font-mono text-[11px] text-muted-foreground">
                  {inputs.batchSize} concurrent
                </span>
              </div>
              <Slider
                aria-label="Batch size"
                min={0}
                max={BATCH_STEPS.length - 1}
                step={1}
                value={[batchIndex]}
                onValueChange={(v) => patch({ batchSize: BATCH_STEPS[v[0]] })}
              />
              <div className="flex justify-between font-mono text-[10px] text-muted-foreground">
                <span>1</span>
                <span>128</span>
              </div>
            </div>

            <div className="space-y-2" role="group" aria-labelledby="cost-util-label">
              <div className="flex items-baseline justify-between gap-2">
                <span id="cost-util-label" className="text-xs font-medium text-foreground">
                  <InfoTip label="Target utilization">
                    The fraction of provisioned hours the GPU is actually generating. You pay for
                    the hours you hold the instance, not the hours it is busy, so halving
                    utilization doubles the bill for identical output. It is the single biggest
                    lever on a cloud invoice and the one most teams never measure.
                  </InfoTip>
                </span>
                <span className="tabular font-mono text-[11px] text-muted-foreground">
                  {utilizationPct}%
                </span>
              </div>
              <Slider
                aria-label="Target utilization, percent"
                min={5}
                max={100}
                step={5}
                value={[utilizationPct]}
                onValueChange={(v) => patch({ targetUtilization: v[0] / 100 })}
              />
              <div className="flex justify-between font-mono text-[10px] text-muted-foreground">
                <span>5%</span>
                <span>100%</span>
              </div>
            </div>
          </div>
        </CardContent>
      </Card>

      {/* ------------------------------------------------------------------ */}
      {/*  OOM                                                               */}
      {/* ------------------------------------------------------------------ */}
      {demand?.throughput.oom && gpu && (
        <Alert variant="destructive">
          <AlertTriangle />
          <AlertTitle>
            {model.name} at {quant.label} does not fit on the {gpu.name}
          </AlertTitle>
          <AlertDescription className="space-y-3">
            <p>
              Weights, a {formatContext(servingContext)}-token KV cache for {inputs.batchSize}{' '}
              concurrent requests and llama.cpp&apos;s runtime buffers need{' '}
              <span className="tabular font-mono">
                {formatPct(demand.throughput.memoryUtilizationPct, 0)}
              </span>{' '}
              of this card&apos;s {gpu.vramGb} GB. The prices below still compute, but nothing here
              is servable until the configuration fits — drop a quant tier, shrink the batch, or
              move to a larger card.
            </p>
            <Button type="button" variant="outline" size="sm" onClick={() => goTo('quant')}>
              Open the quantization planner
            </Button>
          </AlertDescription>
        </Alert>
      )}

      {/* ------------------------------------------------------------------ */}
      {/*  Headline demand                                                   */}
      {/* ------------------------------------------------------------------ */}
      {demand && gpu && (
        <StatGrid cols={5}>
          <Stat
            label={
              <InfoTip label="Single-stream decode">
                {formatNumber(gpu.bandwidthGbs)} GB/s × 72% achieved bandwidth utilization ÷ the
                bytes {quant.label} reads per token. One request, no batching.
              </InfoTip>
            }
            value={formatTps(demand.throughput.decodeSingle)}
            sub={`${gpu.name} · ${quant.label}`}
            tone="accent"
          />
          <Stat
            label={
              <InfoTip label="Batched aggregate">
                Batching scales sublinearly — modelled as batch^0.85 — and is then capped by the
                card&apos;s compute ceiling at 25% MFU. This is the number that divides into your
                monthly token volume.
              </InfoTip>
            }
            value={formatTps(demand.throughput.decodeBatched)}
            sub={`across ${inputs.batchSize} concurrent requests`}
          />
          <Stat
            label={
              <InfoTip label="Prefill per stream">
                Prompt processing is compute bound: {formatNumber(gpu.fp16Tflops)} TFLOPS × 35% MFU
                ÷ 2 FLOPs per active parameter. Multiply by the batch size for the aggregate rate.
              </InfoTip>
            }
            value={formatTps(demand.throughput.prefill)}
            sub={`${formatNumber(demand.prefillHours, 0)} h/mo of prompt processing`}
          />
          <Stat
            label={
              <InfoTip label="GPU-hours / month">
                Busy hours ÷ target utilization. Busy hours are output tokens ÷ batched decode rate
                plus input tokens ÷ batched prefill rate, over {formatNumber(DAYS_PER_MONTH, 2)}{' '}
                days.
              </InfoTip>
            }
            value={
              Number.isFinite(demand.provisionedHours)
                ? formatNumber(demand.provisionedHours, 0)
                : '—'
            }
            sub={
              Number.isFinite(demand.busyHours)
                ? `${formatNumber(demand.busyHours, 0)} busy at ${utilizationPct}% utilization`
                : 'workload exceeds what this GPU can serve'
            }
          />
          <Stat
            label={
              <InfoTip label="GPUs required">
                ceil(provisioned hours ÷ {HOURS_PER_MONTH} hours per month). A month only has so
                many hours; past that you buy parallelism, not patience.
              </InfoTip>
            }
            value={Number.isFinite(demand.provisionedHours) ? formatNumber(demand.gpusNeeded) : '—'}
            sub={demand.gpusNeeded > 1 ? 'running in parallel' : 'a single card keeps up'}
            tone={demand.gpusNeeded > 1 ? 'warn' : 'default'}
          />
        </StatGrid>
      )}

      {/* ------------------------------------------------------------------ */}
      {/*  Provider comparison                                               */}
      {/* ------------------------------------------------------------------ */}
      <Card>
        <CardHeader>
          <CardTitle>Provider comparison</CardTitle>
          <CardDescription>
            The same {formatNumber(inputs.outputTokensPerDay)} output tokens per day, priced across
            rented and owned hardware. Sorted cheapest first.
          </CardDescription>
        </CardHeader>
        <CardContent className="space-y-3">
          <Table>
            <TableHeader>
              <TableRow>
                <TableHead>Provider</TableHead>
                <TableHead className="text-right">$ / GPU-hr</TableHead>
                <TableHead className="text-right">GPU-hours / mo</TableHead>
                <TableHead className="text-right">Monthly</TableHead>
                <TableHead className="text-right">$ / 1M output</TableHead>
              </TableRow>
            </TableHeader>
            <TableBody>
              {rows.map((row) => {
                const isCheapest = cheapest?.provider.id === row.provider.id;
                return (
                  <React.Fragment key={row.provider.id}>
                    <TableRow className={cn(isCheapest && 'bg-meta-50/60')}>
                      <TableCell className="align-top">
                        <div className="flex flex-wrap items-center gap-2">
                          <a
                            href={row.provider.url}
                            target="_blank"
                            rel="noreferrer noopener"
                            className="inline-flex items-center gap-1 font-medium hover:text-meta-600 hover:underline"
                          >
                            {row.provider.name}
                            <ArrowUpRight className="h-3.5 w-3.5 text-muted-foreground" />
                          </a>
                          {isCheapest && <Badge variant="success">Cheapest</Badge>}
                          {row.provider.kind === 'cloud-spot' && (
                            <Badge variant="muted">Interruptible</Badge>
                          )}
                        </div>
                        <div className="mt-0.5 max-w-sm text-pretty text-xs text-muted-foreground">
                          {row.provider.note}
                        </div>
                        {row.breakdown && (
                          <Button
                            type="button"
                            variant="ghost"
                            size="xs"
                            className="mt-1 -ml-2"
                            aria-expanded={breakdownOpen}
                            onClick={() => setBreakdownOpen((v) => !v)}
                          >
                            {breakdownOpen ? 'Hide breakdown' : 'Show breakdown'}
                          </Button>
                        )}
                      </TableCell>

                      {row.unavailableReason ? (
                        <TableCell
                          colSpan={4}
                          className="align-top text-right text-xs text-muted-foreground"
                        >
                          {row.unavailableReason}
                        </TableCell>
                      ) : (
                        <>
                          <TableCell className="tabular align-top text-right font-mono">
                            {formatUsd(row.hourlyUsd)}
                          </TableCell>
                          <TableCell className="tabular align-top text-right">
                            {Number.isFinite(row.gpuHoursPerMonth)
                              ? formatNumber(row.gpuHoursPerMonth, 0)
                              : '—'}
                          </TableCell>
                          <TableCell className="tabular align-top text-right font-semibold">
                            {formatUsd(
                              row.monthlyUsd,
                              row.monthlyUsd !== null && Math.abs(row.monthlyUsd) >= 1000 ? 0 : 2,
                            )}
                          </TableCell>
                          <TableCell className="tabular align-top text-right">
                            {formatUsd(row.usdPerMillionOutput)}
                          </TableCell>
                        </>
                      )}
                    </TableRow>

                    {breakdownOpen && row.breakdown && (
                      <TableRow className="bg-muted/30 hover:bg-muted/30">
                        <TableCell colSpan={5}>
                          <div className="grid gap-3 sm:grid-cols-3">
                            <div className="rounded-lg border bg-background p-3">
                              <div className="text-[11px] font-medium uppercase tracking-wide text-muted-foreground">
                                Hardware amortization
                              </div>
                              <div className="tabular mt-1 font-semibold">
                                {formatUsd(row.breakdown.capexUsd, 0)}
                              </div>
                              <p className="mt-1 text-xs text-muted-foreground">
                                Street price × 1.4 for the box around the GPU, spread over{' '}
                                {inputs.selfHost.amortizationYears} years, charged on all{' '}
                                {HOURS_PER_MONTH} hours a month × {demand?.gpusNeeded ?? 1} GPU.
                                Idle hardware still depreciates.
                              </p>
                            </div>
                            <div className="rounded-lg border bg-background p-3">
                              <div className="text-[11px] font-medium uppercase tracking-wide text-muted-foreground">
                                Electricity
                              </div>
                              <div className="tabular mt-1 font-semibold">
                                {formatUsd(row.breakdown.powerUsd, 2)}
                              </div>
                              <p className="mt-1 text-xs text-muted-foreground">
                                {gpu ? `${gpu.tdpWatts} W` : 'TDP'} ×{' '}
                                {inputs.selfHost.overheadFactor} overhead ×{' '}
                                {formatUsd(inputs.selfHost.electricityUsdPerKwh)}/kWh, billed only
                                for the{' '}
                                {demand && Number.isFinite(demand.busyHours)
                                  ? formatNumber(demand.busyHours, 0)
                                  : '—'}{' '}
                                busy hours.
                              </p>
                            </div>
                            <div className="rounded-lg border bg-background p-3">
                              <div className="text-[11px] font-medium uppercase tracking-wide text-muted-foreground">
                                Fixed overhead
                              </div>
                              <div className="tabular mt-1 font-semibold">
                                {formatUsd(row.breakdown.fixedUsd, 0)}
                              </div>
                              <p className="mt-1 text-xs text-muted-foreground">
                                Colo, bandwidth and the ops time nobody costs. Whatever you put in
                                the assumptions below, per GPU.
                              </p>
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

          <Footnote>
            Rented rows are the provider&rsquo;s hourly list price × provisioned GPU-hours; nothing
            here includes egress, storage, or the engineering time to keep a serving stack alive.
            The self-hosted row is the only one with a fixed floor, which is why it wins at high
            volume and loses badly at low volume.
          </Footnote>
        </CardContent>
      </Card>

      {/* ------------------------------------------------------------------ */}
      {/*  Prices + self-hosting assumptions                                 */}
      {/* ------------------------------------------------------------------ */}
      <div className="grid gap-6 lg:grid-cols-2">
        <Card>
          <CardHeader className="pb-4">
            <div className="flex flex-wrap items-start justify-between gap-3">
              <div className="space-y-1.5">
                <CardTitle>
                  GPU-hour prices
                  {overrideCount > 0 && (
                    <Badge variant="warning" className="ml-2 align-middle">
                      {overrideCount} edited
                    </Badge>
                  )}
                </CardTitle>
                <CardDescription>
                  Rates for the{' '}
                  <span className="font-mono">{gpu?.name ?? inputs.gpuId}</span>, from the{' '}
                  <span className="font-mono">{PRICE_SNAPSHOT_DATE}</span> snapshot.
                </CardDescription>
              </div>
              <div className="flex items-center gap-2">
                <Dialog open={matrixOpen} onOpenChange={setMatrixOpen}>
                  <DialogTrigger asChild>
                    <Button type="button" variant="outline" size="sm">
                      <Table2 />
                      Full matrix
                    </Button>
                  </DialogTrigger>
                  <DialogContent className="max-w-5xl">
                    <DialogHeader>
                      <DialogTitle>Provider × GPU price matrix</DialogTitle>
                      <DialogDescription>
                        USD per GPU-hour. Edits are stored locally and merged over the{' '}
                        <span className="font-mono">{PRICE_SNAPSHOT_DATE}</span> snapshot every time
                        a cost is computed. Clearing a cell marks that GPU as not offered.
                      </DialogDescription>
                    </DialogHeader>
                    <Table>
                      <TableHeader>
                        <TableRow>
                          <TableHead>GPU</TableHead>
                          {CLOUD_PROVIDERS.map((p) => (
                            <TableHead key={p.id} className="text-right">
                              {p.name}
                            </TableHead>
                          ))}
                        </TableRow>
                      </TableHeader>
                      <TableBody>
                        {GPUS.map((g) => (
                          <TableRow key={g.id}>
                            <TableCell>
                              <div className="font-mono text-xs">{g.id}</div>
                              <div className="text-[11px] text-muted-foreground">
                                {g.name} · {g.vramGb} GB
                              </div>
                            </TableCell>
                            {CLOUD_PROVIDERS.map((p) => (
                              <TableCell key={p.id} className="text-right">
                                <PriceCell
                                  providerName={p.name}
                                  gpuName={g.name}
                                  price={prices[p.id]?.[g.id] ?? null}
                                  suggestion={suggestedPrice(prices, g.id)}
                                  edited={overrides[p.id]?.[g.id] !== undefined}
                                  onChange={(next) => setPrice(p.id, g.id, next)}
                                />
                              </TableCell>
                            ))}
                          </TableRow>
                        ))}
                      </TableBody>
                    </Table>
                    <DialogFooter>
                      <Button
                        type="button"
                        variant="outline"
                        size="sm"
                        onClick={resetOverrides}
                        disabled={overrideCount === 0}
                      >
                        <RotateCcw />
                        Reset to snapshot
                      </Button>
                      <DialogClose asChild>
                        <Button type="button" size="sm">
                          Done
                        </Button>
                      </DialogClose>
                    </DialogFooter>
                  </DialogContent>
                </Dialog>
                <Button
                  type="button"
                  variant="ghost"
                  size="sm"
                  onClick={resetOverrides}
                  disabled={overrideCount === 0}
                >
                  <RotateCcw />
                  Reset
                </Button>
              </div>
            </div>
          </CardHeader>
          <CardContent className="space-y-3">
            <div className="space-y-2">
              {CLOUD_PROVIDERS.map((p) => {
                const edited = overrides[p.id]?.[inputs.gpuId] !== undefined;
                return (
                  <div
                    key={p.id}
                    className="flex flex-wrap items-center justify-between gap-3 rounded-lg border px-3 py-2"
                  >
                    <div className="min-w-0">
                      <div className="flex items-center gap-2 text-sm font-medium">
                        {p.name}
                        {edited && <Badge variant="warning">edited</Badge>}
                      </div>
                      <div className="font-mono text-[11px] text-muted-foreground">
                        {p.kind === 'cloud-spot' ? 'interruptible' : 'on-demand'}
                      </div>
                    </div>
                    <PriceCell
                      providerName={p.name}
                      gpuName={gpu?.name ?? inputs.gpuId}
                      price={prices[p.id]?.[inputs.gpuId] ?? null}
                      suggestion={suggestedPrice(prices, inputs.gpuId)}
                      edited={edited}
                      showSuffix
                      onChange={(next) => setPrice(p.id, inputs.gpuId, next)}
                    />
                  </div>
                );
              })}
            </div>
            <Footnote>
              GPU rental prices move constantly — spot markets reprice hourly, on-demand tiers
              reprice with supply, and every provider runs promotions. Treat the{' '}
              <span className="font-mono">{PRICE_SNAPSHOT_DATE}</span> numbers as starting points
              for arithmetic, not as quotes. Paste in the rate you were actually offered and the
              whole table re-derives.
            </Footnote>
          </CardContent>
        </Card>

        <Card>
          <CardHeader className="pb-4">
            <div className="flex flex-wrap items-start justify-between gap-3">
              <div className="space-y-1.5">
                <CardTitle>Measured throughput</CardTitle>
                <CardDescription>
                  Replace the estimate for this exact model, quant and GPU with a number you
                  benchmarked.
                </CardDescription>
              </div>
              <Button
                type="button"
                variant="ghost"
                size="sm"
                onClick={resetTpsOverrides}
                disabled={Object.keys(tpsOverrides).length === 0}
              >
                <RotateCcw />
                Use estimates
              </Button>
            </div>
          </CardHeader>
          <CardContent className="space-y-3">
            <Field
              label="Single-stream decode"
              hint={
                measuredTps === null
                  ? `estimated ${formatTps(estimatedTps)}`
                  : `estimate was ${formatTps(estimatedTps)}`
              }
              help="Run llama-bench, or read the tok/s off llama-server, and paste the single-stream decode rate. The batched and prefill figures rescale by the same ratio so the model stays internally consistent."
            >
              <NumberInput
                value={measuredTps ?? Math.round(estimatedTps)}
                min={0}
                step={1}
                suffix="tok/s"
                aria-label="Measured single-stream decode tokens per second"
                onChange={(next) =>
                  setTpsOverrides((prev) => ({
                    ...prev,
                    [throughputKey(inputs.gpuId, inputs.modelId, inputs.quantId)]: next,
                  }))
                }
              />
            </Field>
            {measuredTps !== null && (
              <Badge variant="meta">
                Costs below use your measured {formatTps(measuredTps)}
              </Badge>
            )}
            <Footnote>
              The bundled estimate is{' '}
              <span className="font-mono">bandwidth × utilization ÷ bytes-read-per-token</span>,
              which lands inside the commonly reported range for every GPU here — but a different
              llama.cpp build, driver or serving stack moves it by tens of percent. If you have
              measured yours, that number beats ours.
            </Footnote>
          </CardContent>
        </Card>

        <Card>
          <CardHeader className="pb-4">
            <div className="flex flex-wrap items-start justify-between gap-3">
              <div className="space-y-1.5">
                <CardTitle>Self-hosting assumptions</CardTitle>
                <CardDescription>
                  What owning the box costs per hour, before a single token is generated.
                </CardDescription>
              </div>
              <Button
                type="button"
                variant="ghost"
                size="sm"
                onClick={() => patch({ selfHost: DEFAULT_SELF_HOST })}
              >
                <RotateCcw />
                Defaults
              </Button>
            </div>
          </CardHeader>
          <CardContent className="space-y-4">
            <div className="grid gap-4 sm:grid-cols-2">
              <Field
                label="Amortization"
                htmlFor="cost-amort"
                help="How long you expect to keep the card in service before it is replaced or resold. Longer amortization lowers the hourly rate and raises the risk that the hardware is obsolete first."
              >
                <NumberInput
                  id="cost-amort"
                  value={inputs.selfHost.amortizationYears}
                  min={0.5}
                  max={10}
                  step={0.5}
                  suffix="yr"
                  onChange={(n) => patchSelfHost({ amortizationYears: n })}
                />
              </Field>
              <Field
                label="Electricity"
                htmlFor="cost-power"
                help="Your marginal rate per kilowatt-hour. US residential averages around $0.15; industrial colo contracts run well below that, and parts of Europe well above."
              >
                <NumberInput
                  id="cost-power"
                  value={inputs.selfHost.electricityUsdPerKwh}
                  min={0}
                  max={2}
                  step={0.01}
                  suffix="$/kWh"
                  onChange={(n) => patchSelfHost({ electricityUsdPerKwh: n })}
                />
              </Field>
              <Field
                label="Power overhead"
                htmlFor="cost-overhead"
                help="Multiplier on GPU TDP covering the CPU, RAM, fans, PSU losses and datacenter cooling. 1.45 is a reasonable single-node figure; a well-run facility with a low PUE lands nearer 1.2."
              >
                <NumberInput
                  id="cost-overhead"
                  value={inputs.selfHost.overheadFactor}
                  min={1}
                  max={3}
                  step={0.05}
                  suffix="×"
                  onChange={(n) => patchSelfHost({ overheadFactor: n })}
                />
              </Field>
              <Field
                label="Fixed monthly"
                htmlFor="cost-fixed"
                help="Rack space, bandwidth, monitoring, and the on-call rotation. Charged per GPU the workload needs."
              >
                <NumberInput
                  id="cost-fixed"
                  value={inputs.selfHost.fixedMonthlyUsd}
                  min={0}
                  step={5}
                  suffix="$/mo"
                  onChange={(n) => patchSelfHost({ fixedMonthlyUsd: n })}
                />
              </Field>
            </div>

            {selfCost ? (
              <div className="rounded-lg border bg-meta-50/50 p-3">
                <div className="flex flex-wrap items-baseline justify-between gap-2">
                  <span className="text-xs font-medium uppercase tracking-wide text-muted-foreground">
                    <InfoTip label="Derived cost of ownership">
                      Capex per hour = street price × 1.4 system multiplier ÷ (years ×{' '}
                      {formatNumber(HOURS_PER_YEAR)} hours). Power per hour = TDP ÷ 1000 × overhead
                      × price per kWh.
                    </InfoTip>
                  </span>
                  <span className="tabular font-mono text-lg font-semibold text-meta-700">
                    {formatUsd(selfCost.total)}/hr
                  </span>
                </div>
                <div className="mt-2 grid grid-cols-2 gap-2 text-xs text-muted-foreground">
                  <div>
                    Hardware{' '}
                    <span className="tabular font-mono text-foreground">
                      {formatUsd(selfCost.capexPerHour)}
                    </span>
                    /hr
                  </div>
                  <div>
                    Power{' '}
                    <span className="tabular font-mono text-foreground">
                      {formatUsd(selfCost.powerPerHour)}
                    </span>
                    /hr
                  </div>
                </div>
              </div>
            ) : (
              <div className="rounded-lg border border-dashed p-3 text-xs text-muted-foreground">
                No street price is bundled for the {gpu?.name ?? 'selected part'}, so ownership
                cannot be modelled — datacenter SKUs like H100 and MI300X are bought through OEM
                channels at prices nobody publishes. Pick a consumer or workstation card to compare
                buying against renting.
              </div>
            )}

            <Footnote>
              Utilization is not a separate assumption here: the calculator drives it from the
              target-utilization slider above, so the same idle-time penalty applies to owned and
              rented hardware. Hardware depreciation is charged on all {HOURS_PER_MONTH} hours a
              month; electricity only on the hours the GPU is busy.
            </Footnote>
          </CardContent>
        </Card>
      </div>

      {/* ------------------------------------------------------------------ */}
      {/*  Managed APIs                                                      */}
      {/* ------------------------------------------------------------------ */}
      <Card>
        <CardHeader>
          <CardTitle>Or just call an API</CardTitle>
          <CardDescription>
            Published list prices for hosted Llama endpoints, applied to the same{' '}
            {formatCompact(inputs.outputTokensPerDay)} output and{' '}
            {formatCompact(inputs.inputTokensPerDay)} input tokens per day. These are the providers&apos;
            own models, not the checkpoint selected above — a serverless Llama 4 Scout endpoint is
            not the same artifact as your GGUF.
          </CardDescription>
        </CardHeader>
        <CardContent className="space-y-3">
          <Table>
            <TableHeader>
              <TableRow>
                <TableHead>Endpoint</TableHead>
                <TableHead className="text-right">In / out per 1M</TableHead>
                <TableHead className="text-right">Monthly</TableHead>
                <TableHead className="text-right">$ / 1M output</TableHead>
                <TableHead className="text-right">
                  <InfoTip label="Break-even">
                    Output tokens per day at which the cheapest self-serve row above stops being
                    more expensive than this endpoint, holding the current input:output ratio.
                    Found by bisecting the real cost curve, so it accounts for the fixed monthly
                    floor of owning hardware.
                  </InfoTip>
                </TableHead>
              </TableRow>
            </TableHeader>
            <TableBody>
              {apiRows.map((row) => {
                const cheaperNow =
                  cheapest?.monthlyUsd !== null &&
                  cheapest?.monthlyUsd !== undefined &&
                  row.monthly < cheapest.monthlyUsd;
                return (
                  <TableRow key={row.price.name}>
                    <TableCell>
                      <div className="font-medium">{row.price.name}</div>
                      <div className="text-xs text-muted-foreground">{row.price.note}</div>
                    </TableCell>
                    <TableCell className="tabular text-right font-mono text-xs">
                      {formatUsd(row.price.inputUsd)} / {formatUsd(row.price.outputUsd)}
                    </TableCell>
                    <TableCell className="tabular text-right font-semibold">
                      {formatUsd(row.monthly, row.monthly >= 1000 ? 0 : 2)}
                      {cheaperNow && (
                        <Badge variant="success" className="ml-2">
                          cheaper now
                        </Badge>
                      )}
                    </TableCell>
                    <TableCell className="tabular text-right">
                      {formatUsd(row.perMillionOutput)}
                    </TableCell>
                    <TableCell className="tabular text-right">
                      {row.breakEven === null ? (
                        <span className="text-xs text-muted-foreground">no crossover</span>
                      ) : (
                        <span className="font-mono">{formatCompact(row.breakEven)}/day</span>
                      )}
                    </TableCell>
                  </TableRow>
                );
              })}
            </TableBody>
          </Table>

          <Footnote>
            {cheapest && cheapest.monthlyUsd !== null && cheapestApi ? (
              <>
                At this volume the cheapest self-serve option is{' '}
                <strong className="font-medium text-foreground">{cheapest.provider.name}</strong> at{' '}
                <span className="tabular font-mono">
                  {formatUsd(cheapest.monthlyUsd, cheapest.monthlyUsd >= 1000 ? 0 : 2)}
                </span>
                /month, against{' '}
                <span className="tabular font-mono">
                  {formatUsd(cheapestApi.monthly, cheapestApi.monthly >= 1000 ? 0 : 2)}
                </span>
                /month for {cheapestApi.price.name}.{' '}
                {cheapestApi.breakEven === null ? (
                  <>
                    The two curves never cross inside the range checked (
                    {formatCompact(BREAK_EVEN_MIN)}–{formatCompact(BREAK_EVEN_MAX)} output tokens
                    per day), so whichever is ahead today stays ahead at every volume worth
                    modelling.
                  </>
                ) : (
                  <>
                    They cross near{' '}
                    <span className="tabular font-mono">
                      {formatCompact(cheapestApi.breakEven)}
                    </span>{' '}
                    output tokens per day. You are at{' '}
                    <span className="tabular font-mono">
                      {formatCompact(inputs.outputTokensPerDay)}
                    </span>
                    , so the managed API is the{' '}
                    {inputs.outputTokensPerDay < cheapestApi.breakEven ? 'cheaper' : 'more expensive'}{' '}
                    option right now. Managed APIs almost always win below the crossover: they
                    amortize idle time across every customer, and you cannot.
                  </>
                )}{' '}
                None of this prices the work — fine-tunes you cannot deploy on a serverless
                endpoint, data that cannot leave your network, or the weeks it takes to run a
                serving stack well.
              </>
            ) : (
              <>
                No self-serve option can be priced for this configuration, so there is nothing to
                compare the managed endpoints against. Choose a GPU the model fits on.
              </>
            )}
          </Footnote>
        </CardContent>
      </Card>

      <Footnote>
        <strong className="font-medium text-foreground">How the throughput numbers are made.</strong>{' '}
        Decode is memory-bandwidth bound, not compute bound: generating one token requires reading
        every active weight out of VRAM, so tok/s ≈ (GPU bandwidth × achieved bandwidth utilization)
        ÷ bytes read per token, with utilization fixed at 72% — what a well-tuned llama.cpp or vLLM
        server reaches in practice. Prefill is the opposite: prompt tokens are processed in
        parallel, so it is compute bound at roughly 35% model-FLOPs utilization over 2 FLOPs per
        active parameter per token. Batching amortizes one weight read across concurrent requests
        and is modelled as batch^0.85, capped by the card&apos;s compute ceiling. Mixture-of-experts
        models read only their routed experts per token, which is why Llama 4 Scout decodes like a
        17B model despite weighing 109B. Every figure on this page is a model, not a measurement —
        benchmark the configuration you actually intend to ship before signing anything.
      </Footnote>
    </div>
  );
}
