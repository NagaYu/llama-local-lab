import { BASE_PRICES, getGpu, PROVIDERS } from '@/data/gpus';
import { getQuant } from '@/data/quants';
import { getModel } from '@/data/models';
import { estimateThroughput } from '@/lib/quant';
import type {
  CostInputs,
  CostResult,
  GpuPriceTable,
  SelfHostAssumptions,
  ThroughputEstimate,
} from '@/lib/types';

/** Average hours in a month (8766 / 12). Using 730 keeps monthly math honest. */
export const HOURS_PER_MONTH = 730;
export const HOURS_PER_YEAR = 8766;
export const DAYS_PER_MONTH = 30.44;

export const DEFAULT_SELF_HOST: SelfHostAssumptions = {
  amortizationYears: 3,
  electricityUsdPerKwh: 0.15,
  overheadFactor: 1.45,
  utilization: 0.6,
  fixedMonthlyUsd: 40,
};

/**
 * Capex is more than the GPU: chassis, CPU, RAM, PSU and networking typically
 * add 35–50% on a single-GPU box. 1.4 is the middle of what people report.
 */
const SYSTEM_CAPEX_MULTIPLIER = 1.4;

/**
 * Key for a measured-throughput override.
 *
 * Throughput is the least trustworthy number in the whole calculator — it moves
 * with the backend, the driver and the batching strategy. Anyone who has actually
 * benchmarked their stack should be able to type that number in and have every
 * cost downstream re-derive from it, rather than argue with our estimate.
 */
export function throughputKey(gpuId: string, modelId: string, quantId: string): string {
  return `${gpuId}|${modelId}|${quantId}`;
}

/** Decode tokens/sec override for the current selection, if the user set one. */
export function resolveThroughputOverride(
  inputs: CostInputs,
  overrides: Record<string, number> | undefined,
): number | null {
  if (!overrides) return null;
  const value = overrides[throughputKey(inputs.gpuId, inputs.modelId, inputs.quantId)];
  return typeof value === 'number' && Number.isFinite(value) && value > 0 ? value : null;
}

export interface WorkloadDemand {
  /** GPU-hours per month spent actually generating and prefilling. */
  busyHours: number;
  /** Hours you must pay for, once idle time at the target utilization is added. */
  provisionedHours: number;
  decodeHours: number;
  prefillHours: number;
  /** How many GPUs the workload needs to keep up inside one month. */
  gpusNeeded: number;
  throughput: ThroughputEstimate;
  /** True when even a fully saturated GPU cannot finish the month's tokens. */
  infeasible: boolean;
  /** True when the user replaced the estimate with a measured number. */
  usingOverride: boolean;
}

export function computeDemand(
  inputs: CostInputs,
  throughputOverrides?: Record<string, number>,
): WorkloadDemand | null {
  const model = getModel(inputs.modelId);
  const quant = getQuant(inputs.quantId);
  const gpu = getGpu(inputs.gpuId);
  if (!gpu) return null;

  // Size the KV cache for a typical serving context rather than the model max —
  // provisioning 128k of cache per request is not what anyone actually does.
  const servingContext = Math.min(8192, model.contextLength);
  const estimated = estimateThroughput(model, quant, gpu, {
    contextLength: servingContext,
    kvPrecision: 'f16',
    batchSize: inputs.batchSize,
  });

  // A measured single-stream number replaces the estimate, and the batched and
  // prefill figures are rescaled by the same ratio so the whole model stays
  // internally consistent rather than mixing measured and predicted rates.
  const override = resolveThroughputOverride(inputs, throughputOverrides);
  const ratio = override && estimated.decodeSingle > 0 ? override / estimated.decodeSingle : 1;
  const throughput: ThroughputEstimate = override
    ? {
        ...estimated,
        decodeSingle: override,
        decodeBatched: estimated.decodeBatched * ratio,
        prefill: estimated.prefill * ratio,
      }
    : estimated;

  const outputPerMonth = inputs.outputTokensPerDay * DAYS_PER_MONTH;
  const inputPerMonth = inputs.inputTokensPerDay * DAYS_PER_MONTH;

  const decodeHours = throughput.decodeBatched > 0 ? outputPerMonth / throughput.decodeBatched / 3600 : Infinity;
  const prefillHours = throughput.prefill > 0 ? inputPerMonth / (throughput.prefill * inputs.batchSize) / 3600 : Infinity;
  const busyHours = decodeHours + prefillHours;

  const utilization = Math.min(Math.max(inputs.targetUtilization, 0.01), 1);
  const provisionedHours = busyHours / utilization;
  const gpusNeeded = Math.max(1, Math.ceil(provisionedHours / HOURS_PER_MONTH));

  return {
    busyHours,
    provisionedHours,
    decodeHours,
    prefillHours,
    gpusNeeded,
    throughput,
    infeasible: !Number.isFinite(busyHours) || throughput.oom,
    usingOverride: override !== null,
  };
}

/** Hourly cost of owning the box, split into amortized hardware and power. */
export function selfHostHourly(
  gpuId: string,
  assumptions: SelfHostAssumptions,
): { capexPerHour: number; powerPerHour: number; total: number } | null {
  const gpu = getGpu(gpuId);
  if (!gpu || !gpu.msrpUsd) return null;

  const capexPerHour =
    (gpu.msrpUsd * SYSTEM_CAPEX_MULTIPLIER) /
    (assumptions.amortizationYears * HOURS_PER_YEAR);
  const powerPerHour =
    (gpu.tdpWatts / 1000) * assumptions.overheadFactor * assumptions.electricityUsdPerKwh;

  return { capexPerHour, powerPerHour, total: capexPerHour + powerPerHour };
}

/** Merge editable overrides on top of the snapshotted price table. */
export function resolvePrices(overrides: GpuPriceTable): GpuPriceTable {
  const merged: GpuPriceTable = {};
  for (const providerId of Object.keys(BASE_PRICES)) {
    merged[providerId] = { ...BASE_PRICES[providerId], ...(overrides[providerId] ?? {}) };
  }
  for (const providerId of Object.keys(overrides)) {
    if (!merged[providerId]) merged[providerId] = { ...overrides[providerId] };
  }
  return merged;
}

export function computeCosts(
  inputs: CostInputs,
  overrides: GpuPriceTable,
  throughputOverrides?: Record<string, number>,
): { demand: WorkloadDemand | null; results: CostResult[] } {
  const demand = computeDemand(inputs, throughputOverrides);
  const prices = resolvePrices(overrides);
  const gpu = getGpu(inputs.gpuId);

  const results: CostResult[] = PROVIDERS.map((provider) => {
    if (!demand || !gpu) {
      return {
        provider,
        hourlyUsd: null,
        gpuHoursPerMonth: 0,
        monthlyUsd: null,
        usdPerMillionOutput: null,
        unavailableReason: 'Select a GPU to price this workload.',
      };
    }

    const outputPerMonth = inputs.outputTokensPerDay * DAYS_PER_MONTH;
    const perMillion = (monthly: number) =>
      outputPerMonth > 0 ? monthly / (outputPerMonth / 1_000_000) : null;

    if (provider.kind === 'self-hosted') {
      const self = selfHostHourly(inputs.gpuId, inputs.selfHost);
      if (!self) {
        return {
          provider,
          hourlyUsd: null,
          gpuHoursPerMonth: demand.provisionedHours,
          monthlyUsd: null,
          usdPerMillionOutput: null,
          unavailableReason: 'No street price for this part — datacenter GPUs are rarely bought outright.',
        };
      }
      // You own the hardware around the clock, so capex accrues on all 730
      // hours per GPU. Power is only burned while the GPU is actually busy.
      const capexUsd = self.capexPerHour * HOURS_PER_MONTH * demand.gpusNeeded;
      const powerUsd = self.powerPerHour * demand.busyHours;
      const fixedUsd = inputs.selfHost.fixedMonthlyUsd * demand.gpusNeeded;
      const monthly = capexUsd + powerUsd + fixedUsd;
      return {
        provider,
        hourlyUsd: self.total,
        gpuHoursPerMonth: HOURS_PER_MONTH * demand.gpusNeeded,
        monthlyUsd: monthly,
        usdPerMillionOutput: perMillion(monthly),
        breakdown: { capexUsd, powerUsd, fixedUsd },
      };
    }

    const hourly = prices[provider.id]?.[inputs.gpuId] ?? null;
    if (hourly === null || hourly === undefined) {
      return {
        provider,
        hourlyUsd: null,
        gpuHoursPerMonth: demand.provisionedHours,
        monthlyUsd: null,
        usdPerMillionOutput: null,
        unavailableReason: `${gpu.name} is not listed on ${provider.name}.`,
      };
    }

    const monthly = hourly * demand.provisionedHours;
    return {
      provider,
      hourlyUsd: hourly,
      gpuHoursPerMonth: demand.provisionedHours,
      monthlyUsd: monthly,
      usdPerMillionOutput: perMillion(monthly),
    };
  });

  return { demand, results };
}

/**
 * Public API list prices per million tokens, for the "should I just call an API"
 * comparison. Blended at a 1:3 input:output ratio, which matches the calculator's
 * default workload shape.
 */
export const API_REFERENCE_PRICES: { name: string; inputUsd: number; outputUsd: number; note: string }[] = [
  { name: 'Together AI · Llama 3.3 70B Turbo', inputUsd: 0.88, outputUsd: 0.88, note: 'Serverless, FP8' },
  { name: 'Groq · Llama 3.3 70B Versatile', inputUsd: 0.59, outputUsd: 0.79, note: 'LPU, very high tok/s' },
  { name: 'Fireworks · Llama 3.1 8B', inputUsd: 0.2, outputUsd: 0.2, note: 'Serverless' },
  { name: 'Together AI · Llama 4 Scout', inputUsd: 0.18, outputUsd: 0.59, note: 'Serverless MoE' },
  { name: 'Together AI · Llama 4 Maverick', inputUsd: 0.27, outputUsd: 0.85, note: 'Serverless MoE' },
];

export function apiMonthlyCost(
  price: { inputUsd: number; outputUsd: number },
  inputs: CostInputs,
): number {
  const inTok = (inputs.inputTokensPerDay * DAYS_PER_MONTH) / 1_000_000;
  const outTok = (inputs.outputTokensPerDay * DAYS_PER_MONTH) / 1_000_000;
  return inTok * price.inputUsd + outTok * price.outputUsd;
}

export const DEFAULT_COST_INPUTS: CostInputs = {
  modelId: 'llama-3-1-8b',
  quantId: 'Q4_K_M',
  gpuId: 'rtx-4090',
  outputTokensPerDay: 5_000_000,
  inputTokensPerDay: 15_000_000,
  batchSize: 16,
  targetUtilization: 0.6,
  selfHost: DEFAULT_SELF_HOST,
};
