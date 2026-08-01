const GIB = 1024 ** 3;

/** Human-readable byte size in binary units (GiB), which is what GPUs report. */
export function formatBytes(bytes: number, digits = 2): string {
  if (!Number.isFinite(bytes) || bytes <= 0) return '—';
  const units = ['B', 'KiB', 'MiB', 'GiB', 'TiB'];
  let i = 0;
  let v = bytes;
  while (v >= 1024 && i < units.length - 1) {
    v /= 1024;
    i += 1;
  }
  return `${v.toFixed(i >= 3 ? digits : 0)} ${units[i]}`;
}

/** Byte size in GiB with a fixed unit, for table columns that must line up. */
export function formatGib(bytes: number, digits = 2): string {
  if (!Number.isFinite(bytes) || bytes <= 0) return '—';
  return `${(bytes / GIB).toFixed(digits)} GB`;
}

export function formatCompact(n: number): string {
  if (!Number.isFinite(n)) return '—';
  return new Intl.NumberFormat('en-US', { notation: 'compact', maximumFractionDigits: 1 }).format(n);
}

export function formatNumber(n: number, digits = 0): string {
  if (!Number.isFinite(n)) return '—';
  return new Intl.NumberFormat('en-US', {
    minimumFractionDigits: digits,
    maximumFractionDigits: digits,
  }).format(n);
}

export function formatUsd(n: number | null | undefined, digits = 2): string {
  if (n === null || n === undefined || !Number.isFinite(n)) return '—';
  const abs = Math.abs(n);
  const d = abs > 0 && abs < 0.01 ? 4 : digits;
  return new Intl.NumberFormat('en-US', {
    style: 'currency',
    currency: 'USD',
    minimumFractionDigits: d,
    maximumFractionDigits: d,
  }).format(n);
}

export function formatPct(n: number, digits = 1): string {
  if (!Number.isFinite(n)) return '—';
  return `${n.toFixed(digits)}%`;
}

/** Context lengths read better as 128K / 10M than as 131,072 / 10,485,760. */
export function formatContext(tokens: number): string {
  if (tokens >= 1_000_000) {
    const m = tokens / 1_048_576;
    return `${m % 1 === 0 ? m : m.toFixed(1)}M`;
  }
  if (tokens >= 1024) {
    const k = tokens / 1024;
    return `${k % 1 === 0 ? k : k.toFixed(1)}K`;
  }
  return String(tokens);
}

export function formatParams(paramsB: number): string {
  if (paramsB >= 1000) return `${(paramsB / 1000).toFixed(2)}T`;
  if (paramsB >= 10) return `${paramsB.toFixed(paramsB % 1 === 0 ? 0 : 1)}B`;
  return `${paramsB.toFixed(2)}B`;
}

export function formatDuration(ms: number): string {
  if (!Number.isFinite(ms)) return '—';
  if (ms < 1000) return `${Math.round(ms)} ms`;
  if (ms < 60_000) return `${(ms / 1000).toFixed(1)} s`;
  const min = Math.floor(ms / 60_000);
  const sec = Math.round((ms % 60_000) / 1000);
  return `${min}m ${sec}s`;
}

export function formatTps(tps: number | null): string {
  if (tps === null || !Number.isFinite(tps)) return '—';
  if (tps >= 1000) return `${formatNumber(tps, 0)} tok/s`;
  if (tps >= 10) return `${tps.toFixed(0)} tok/s`;
  return `${tps.toFixed(1)} tok/s`;
}

export function formatDate(iso: string | undefined): string {
  if (!iso) return '—';
  const d = new Date(iso);
  if (Number.isNaN(d.getTime())) return '—';
  return d.toLocaleDateString('en-US', { year: 'numeric', month: 'short', day: 'numeric' });
}

/** Rough token estimate for the eval suite: ~4 characters per token for Llama 3. */
export function estimateTokens(text: string): number {
  return Math.max(1, Math.round(text.length / 4));
}
