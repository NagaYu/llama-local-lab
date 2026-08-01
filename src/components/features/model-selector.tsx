'use client';

import * as React from 'react';
import {
  Check,
  CloudOff,
  Download,
  ExternalLink,
  Heart,
  Image as ImageIcon,
  Languages,
  Layers,
  Lock,
  Receipt,
  RefreshCw,
  Scale,
  Search,
  SearchX,
  Sparkles,
  TerminalSquare,
  Type,
  X,
} from 'lucide-react';

import { GGUF_MIRRORS, MODELS, MODEL_FAMILIES, getModel } from '@/data/models';
import { fetchHubStats, ggufUrl, hubUrl, invalidateHubCache } from '@/lib/hf';
import {
  formatBytes,
  formatCompact,
  formatContext,
  formatDate,
  formatNumber,
  formatParams,
  formatPct,
} from '@/lib/format';
import { cn } from '@/lib/utils';
import type { HubStats, LlamaModel, ModelFamily, ModelModality } from '@/lib/types';
import { useStudio, type StudioTab } from '@/components/studio-provider';
import { EmptyState, Field, Footnote, SectionHeading } from '@/components/primitives';
import { Badge } from '@/components/ui/badge';
import { Button, buttonVariants } from '@/components/ui/button';
import { Card, CardContent, CardFooter, CardHeader } from '@/components/ui/card';
import { Input } from '@/components/ui/input';
import { Separator } from '@/components/ui/separator';
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from '@/components/ui/select';
import {
  Accordion,
  AccordionContent,
  AccordionItem,
  AccordionTrigger,
} from '@/components/ui/accordion';
import { InfoTip, Tooltip, TooltipContent, TooltipTrigger } from '@/components/ui/tooltip';

/* -------------------------------------------------------------------------- */
/*  Local helpers                                                              */
/* -------------------------------------------------------------------------- */

const MONTHS = [
  'Jan',
  'Feb',
  'Mar',
  'Apr',
  'May',
  'Jun',
  'Jul',
  'Aug',
  'Sep',
  'Oct',
  'Nov',
  'Dec',
];

/**
 * Format a bundled `YYYY-MM-DD` release date without constructing a `Date`.
 *
 * `new Date('2024-12-06')` is parsed as UTC midnight and then rendered in the
 * viewer's timezone, so a page prerendered in UTC would print a different day
 * for anyone west of it — a hydration mismatch on a statically exported page.
 * `formatDate` is still the right helper for Hub timestamps, which only ever
 * render after the fetch resolves in an effect.
 */
function formatReleaseDay(iso: string): string {
  const parts = /^(\d{4})-(\d{2})-(\d{2})/.exec(iso);
  if (!parts) return iso;
  const month = MONTHS[Number(parts[2]) - 1] ?? parts[2];
  return `${month} ${Number(parts[3])}, ${parts[1]}`;
}

type SortKey = 'params-desc' | 'params-asc' | 'context-desc' | 'released-desc' | 'downloads-desc';

const SORT_OPTIONS: { id: SortKey; label: string }[] = [
  { id: 'params-desc', label: 'Parameters — large first' },
  { id: 'params-asc', label: 'Parameters — small first' },
  { id: 'context-desc', label: 'Context length' },
  { id: 'released-desc', label: 'Release date — newest' },
  { id: 'downloads-desc', label: 'Downloads — most' },
];

function isSortKey(value: string): value is SortKey {
  return SORT_OPTIONS.some((option) => option.id === value);
}

type ModalityFilter = 'all' | ModelModality;

const MODALITY_OPTIONS: { id: ModalityFilter; label: string }[] = [
  { id: 'all', label: 'All modalities' },
  { id: 'text', label: 'Text only' },
  { id: 'multimodal', label: 'Multimodal' },
];

function isModalityFilter(value: string): value is ModalityFilter {
  return MODALITY_OPTIONS.some((option) => option.id === value);
}

/** Family chip counts are computed from the bundled catalog, not the filtered view. */
const FAMILY_COUNTS: { family: ModelFamily; count: number }[] = MODEL_FAMILIES.map((family) => ({
  family,
  count: MODELS.filter((model) => model.family === family).length,
}));

function matchesQuery(model: LlamaModel, terms: string[]): boolean {
  if (terms.length === 0) return true;
  const haystack =
    `${model.name} ${model.repoId} ${model.family} ${model.description} ${model.id}`.toLowerCase();
  return terms.every((term) => haystack.includes(term));
}

/* -------------------------------------------------------------------------- */
/*  HuggingFace stats loading                                                  */
/* -------------------------------------------------------------------------- */

interface HubState {
  status: 'loading' | 'ready' | 'error';
  entries: Record<string, HubStats>;
  fetchedAt?: string;
  message?: string;
}

function pickFetchedAt(entries: Record<string, HubStats>): string | undefined {
  for (const stats of Object.values(entries)) {
    if (stats.fetchedAt) return stats.fetchedAt;
  }
  return undefined;
}

/** Shimmering placeholder for a stat that is still in flight. */
function StatSkeleton({ className }: { className?: string }) {
  return (
    <span
      aria-hidden="true"
      className={cn(
        'relative inline-block h-3 w-14 overflow-hidden rounded bg-muted align-middle',
        className,
      )}
    >
      <span className="absolute inset-0 -translate-x-full animate-shimmer bg-gradient-to-r from-transparent via-background/90 to-transparent" />
    </span>
  );
}

/* -------------------------------------------------------------------------- */
/*  Card sub-parts                                                             */
/* -------------------------------------------------------------------------- */

function Metric({
  label,
  value,
  help,
}: {
  label: string;
  value: React.ReactNode;
  help?: React.ReactNode;
}) {
  return (
    <div className="min-w-0">
      <div className="text-[11px] font-medium uppercase tracking-wide text-muted-foreground">
        {help ? <InfoTip label={label}>{help}</InfoTip> : label}
      </div>
      <div className="tabular mt-0.5 truncate text-sm font-medium">{value}</div>
    </div>
  );
}

function ArchItem({
  label,
  value,
  help,
}: {
  label: string;
  value: React.ReactNode;
  help?: React.ReactNode;
}) {
  return (
    <div className="min-w-0">
      <dt className="text-[11px] text-muted-foreground">
        {help ? <InfoTip label={label}>{help}</InfoTip> : label}
      </dt>
      <dd className="tabular font-mono text-xs font-medium">{value}</dd>
    </div>
  );
}

function HubStatsRow({ stats, loading }: { stats: HubStats | undefined; loading: boolean }) {
  if (!stats && loading) {
    return (
      <div className="flex items-center gap-4 text-xs text-muted-foreground">
        <span className="sr-only">Loading HuggingFace statistics</span>
        <span className="inline-flex items-center gap-1.5">
          <Download className="h-3.5 w-3.5" aria-hidden="true" />
          <StatSkeleton />
        </span>
        <span className="inline-flex items-center gap-1.5">
          <Heart className="h-3.5 w-3.5" aria-hidden="true" />
          <StatSkeleton className="w-10" />
        </span>
      </div>
    );
  }

  if (!stats) {
    return (
      <div className="flex items-center gap-1.5 text-xs text-muted-foreground">
        <CloudOff className="h-3.5 w-3.5" aria-hidden="true" />
        <span>Live Hub stats unavailable — specs below are bundled and unaffected.</span>
      </div>
    );
  }

  return (
    <div className="flex flex-wrap items-center gap-x-4 gap-y-1 text-xs text-muted-foreground">
      <span className="inline-flex items-center gap-1.5">
        <Download className="h-3.5 w-3.5" aria-hidden="true" />
        <span className="tabular font-medium text-foreground">
          <InfoTip
            label={stats.downloads === undefined ? '—' : formatCompact(stats.downloads)}
          >
            HuggingFace reports downloads over a rolling 30-day window, not since release.
            {stats.downloads !== undefined && ` Exact count: ${formatNumber(stats.downloads)}.`}
          </InfoTip>
        </span>
        downloads
      </span>
      <span className="inline-flex items-center gap-1.5">
        <Heart className="h-3.5 w-3.5" aria-hidden="true" />
        <span className="tabular font-medium text-foreground">
          {stats.likes === undefined ? '—' : formatCompact(stats.likes)}
        </span>
        likes
      </span>
      <span className="text-[11px]">
        from HuggingFace
        {stats.lastModified ? ` · repo updated ${formatDate(stats.lastModified)}` : ''}
      </span>
    </div>
  );
}

/* -------------------------------------------------------------------------- */
/*  Model card                                                                 */
/* -------------------------------------------------------------------------- */

function ModelCard({
  model,
  stats,
  statsLoading,
  selected,
  onSelect,
  onGoTo,
}: {
  model: LlamaModel;
  stats: HubStats | undefined;
  statsLoading: boolean;
  selected: boolean;
  onSelect: (id: string) => void;
  onGoTo: (tab: StudioTab, modelId: string) => void;
}) {
  const arch = model.architecture;
  const moe = arch.moe;
  const mirrors = GGUF_MIRRORS[model.id] ?? [];
  const headDim = arch.hiddenSize / arch.attentionHeads;
  const gqaRatio = arch.attentionHeads / arch.kvHeads;
  // f16 KV cache for 1024 tokens: 2 tensors (K and V) × layers × kvHeads × headDim × 2 bytes.
  const kvBytesPerKTokens = 2 * arch.layers * arch.kvHeads * headDim * 2 * 1024;
  const activeShare = moe ? (moe.activeParamsB / model.paramsB) * 100 : 0;

  const handleCardClick = (event: React.MouseEvent<HTMLDivElement>) => {
    // Real controls (links, buttons, the accordion trigger) handle their own clicks.
    if (event.target instanceof Element && event.target.closest('a,button')) return;
    onSelect(model.id);
  };

  return (
    <Card
      onClick={handleCardClick}
      className={cn(
        'flex h-full flex-col overflow-hidden shadow-none transition-colors',
        selected
          ? 'border-meta-300 bg-meta-50/60 ring-2 ring-meta-500'
          : 'hover:border-meta-200 hover:bg-meta-50/30',
      )}
    >
      <CardHeader className="gap-3 p-5 pb-3">
        <div className="flex items-start justify-between gap-2">
          <button
            type="button"
            onClick={() => onSelect(model.id)}
            aria-pressed={selected}
            aria-label={`Select ${model.name} as the active model`}
            className="group min-w-0 grow rounded-md text-left focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring focus-visible:ring-offset-2 ring-offset-background"
          >
            <span className="flex flex-wrap items-center gap-2">
              <span className="text-base font-semibold leading-tight tracking-tight">
                {model.name}
              </span>
              {selected && (
                <Badge variant="meta" className="px-1.5 py-0 text-[10px]">
                  <Check className="h-3 w-3" aria-hidden="true" />
                  Active
                </Badge>
              )}
            </span>
            <span className="mt-1 block truncate font-mono text-[11px] text-muted-foreground transition-colors group-hover:text-meta-600">
              {model.repoId}
            </span>
          </button>

          <Tooltip>
            <TooltipTrigger asChild>
              <a
                href={hubUrl(model.repoId)}
                target="_blank"
                rel="noreferrer noopener"
                aria-label={`Open ${model.repoId} on HuggingFace`}
                className={cn(
                  buttonVariants({ variant: 'ghost', size: 'icon-sm' }),
                  'shrink-0 text-muted-foreground hover:text-meta-600',
                )}
              >
                <ExternalLink className="h-4 w-4" aria-hidden="true" />
              </a>
            </TooltipTrigger>
            <TooltipContent>Open the repository on HuggingFace</TooltipContent>
          </Tooltip>
        </div>

        <div className="flex flex-wrap items-center gap-1.5">
          <Badge variant="meta">{model.family}</Badge>
          <Badge variant="secondary">
            {model.modality === 'multimodal' ? (
              <ImageIcon className="h-3 w-3" aria-hidden="true" />
            ) : (
              <Type className="h-3 w-3" aria-hidden="true" />
            )}
            {model.modality === 'multimodal' ? 'Multimodal' : 'Text'}
          </Badge>
          {moe && (
            <Badge variant="outline">
              <Sparkles className="h-3 w-3" aria-hidden="true" />
              MoE
            </Badge>
          )}
          {model.gated && (
            <Tooltip>
              <TooltipTrigger className="rounded-full focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring focus-visible:ring-offset-1 ring-offset-background">
                <Badge variant="warning">
                  <Lock className="h-3 w-3" aria-hidden="true" />
                  Gated
                </Badge>
              </TooltipTrigger>
              <TooltipContent>
                Meta gates this repository. Sign in to HuggingFace, accept the community license
                on the model page, and wait for access to be granted before `huggingface-cli
                download` or `from_pretrained` will work. Community GGUF mirrors are usually
                ungated.
              </TooltipContent>
            </Tooltip>
          )}
          {model.languages && model.languages.length > 0 && (
            <span className="inline-flex items-center gap-1 text-[11px] text-muted-foreground">
              <Languages className="h-3 w-3" aria-hidden="true" />
              <InfoTip label={`${model.languages.length} languages`}>
                Officially supported by Meta for this checkpoint: {model.languages.join(', ')}.
                The tokenizer covers far more, but quality is only claimed for these.
              </InfoTip>
            </span>
          )}
        </div>
      </CardHeader>

      <CardContent className="flex grow flex-col gap-4 p-5 pt-0">
        <div className="grid grid-cols-2 gap-x-4 gap-y-3 rounded-lg border bg-background p-3">
          <Metric
            label="Parameters"
            value={formatParams(model.paramsB)}
            help={
              moe
                ? `Total weights that must be resident in memory. Only ${formatParams(
                    moe.activeParamsB,
                  )} are used per token.`
                : 'Total parameter count as published by Meta. Drives file size at every quantization level.'
            }
          />
          <Metric
            label="Context"
            value={`${formatContext(model.contextLength)} tokens`}
            help={`Maximum position count: ${formatNumber(
              model.contextLength,
            )} tokens. Serving that full window needs a KV cache far larger than the weights — the quantization planner sizes it for you.`}
          />
          <Metric label="Released" value={formatReleaseDay(model.releasedAt)} />
          <Metric
            label="License"
            value={
              model.licenseUrl ? (
                <a
                  href={model.licenseUrl}
                  target="_blank"
                  rel="noreferrer noopener"
                  className="inline-flex items-center gap-1 text-meta-600 underline decoration-meta-300 underline-offset-2 hover:decoration-meta-600"
                >
                  <Scale className="h-3 w-3 shrink-0" aria-hidden="true" />
                  <span className="truncate">{model.license}</span>
                </a>
              ) : (
                <span className="inline-flex items-center gap-1">
                  <Scale className="h-3 w-3 shrink-0 text-muted-foreground" aria-hidden="true" />
                  <span className="truncate">{model.license}</span>
                </span>
              )
            }
          />
        </div>

        {moe && (
          <div className="rounded-lg border border-meta-200 bg-meta-50 p-3">
            <div className="flex items-center gap-1.5 text-[11px] font-semibold uppercase tracking-wide text-meta-700">
              <Sparkles className="h-3.5 w-3.5" aria-hidden="true" />
              Mixture of experts
            </div>
            <div className="mt-2 grid grid-cols-3 gap-2">
              <div>
                <div className="text-[11px] text-muted-foreground">Experts</div>
                <div className="tabular font-mono text-xs font-medium">
                  {formatNumber(moe.experts)}
                </div>
              </div>
              <div>
                <div className="text-[11px] text-muted-foreground">Active / token</div>
                <div className="tabular font-mono text-xs font-medium">
                  {formatNumber(moe.activeExperts)}
                </div>
              </div>
              <div>
                <div className="text-[11px] text-muted-foreground">Active params</div>
                <div className="tabular font-mono text-xs font-medium">
                  {formatParams(moe.activeParamsB)}
                </div>
              </div>
            </div>
            <p className="mt-2 text-[11px] leading-relaxed text-muted-foreground">
              <InfoTip label={`${formatPct(activeShare, 0)} of weights run per token`}>
                activeParamsB ÷ total paramsB = {formatParams(moe.activeParamsB)} ÷{' '}
                {formatParams(model.paramsB)}.
              </InfoTip>{' '}
              Memory must hold every expert; decode speed tracks the active slice.
            </p>
          </div>
        )}

        <p className="text-pretty text-sm leading-relaxed text-muted-foreground">
          {model.description}
        </p>

        <HubStatsRow stats={stats} loading={statsLoading} />

        {mirrors.length > 0 && (
          <div className="space-y-1.5">
            <div className="text-[11px] font-semibold uppercase tracking-wide text-muted-foreground">
              Community GGUF mirrors
            </div>
            <div className="space-y-1">
              {mirrors.map((mirror) => (
                <a
                  key={mirror.repo}
                  href={ggufUrl(mirror.repo)}
                  target="_blank"
                  rel="noreferrer noopener"
                  aria-label={`Browse GGUF files in ${mirror.repo} on HuggingFace`}
                  className="flex items-center gap-2 rounded-md border bg-background px-2 py-1.5 transition-colors hover:border-meta-300 hover:bg-meta-50"
                >
                  <span className="min-w-0 grow truncate font-mono text-[11px]">{mirror.repo}</span>
                  <span className="shrink-0 text-[10px] text-muted-foreground">{mirror.label}</span>
                  <ExternalLink
                    className="h-3 w-3 shrink-0 text-muted-foreground"
                    aria-hidden="true"
                  />
                </a>
              ))}
            </div>
            <Footnote className="text-[11px]">
              Community conversions, not published by Meta. Check the quant list and the file
              checksums before you run one.
            </Footnote>
          </div>
        )}

        <Accordion type="single" collapsible className="mt-auto border-t">
          <AccordionItem value="architecture" className="border-b-0">
            <AccordionTrigger className="py-3 text-xs font-medium text-muted-foreground hover:text-foreground hover:no-underline">
              Architecture
            </AccordionTrigger>
            <AccordionContent className="pb-2">
              <dl className="grid grid-cols-2 gap-x-4 gap-y-2.5 sm:grid-cols-3">
                <ArchItem
                  label="Layers"
                  value={formatNumber(arch.layers)}
                  help="Transformer blocks. Every offloaded layer is one -ngl step in llama.cpp, and layer count scales the KV cache linearly."
                />
                <ArchItem label="Hidden size" value={formatNumber(arch.hiddenSize)} />
                <ArchItem label="Attention heads" value={formatNumber(arch.attentionHeads)} />
                <ArchItem
                  label="KV heads"
                  value={
                    <span className="inline-flex items-center gap-1.5">
                      {formatNumber(arch.kvHeads)}
                      {gqaRatio > 1 && (
                        <Badge variant="muted" className="px-1.5 py-0 font-sans text-[10px]">
                          GQA {gqaRatio}:1
                        </Badge>
                      )}
                    </span>
                  }
                  help={
                    gqaRatio > 1
                      ? `Grouped-query attention: ${arch.attentionHeads} query heads share ${arch.kvHeads} key/value heads, so the KV cache is ${gqaRatio}× smaller than full multi-head attention would need.`
                      : 'Multi-head attention — one key/value head per query head.'
                  }
                />
                <ArchItem
                  label="Head dim"
                  value={formatNumber(headDim)}
                  help={`hiddenSize ÷ attentionHeads = ${formatNumber(
                    arch.hiddenSize,
                  )} ÷ ${formatNumber(arch.attentionHeads)}.`}
                />
                <ArchItem
                  label="Vocabulary"
                  value={formatCompact(arch.vocabSize)}
                  help={`${formatNumber(arch.vocabSize)} tokens.${
                    arch.tiedEmbeddings
                      ? ' Input and output embeddings are tied, so the table is stored once.'
                      : ' Input and output embeddings are separate matrices.'
                  }`}
                />
                <ArchItem
                  label="KV cache / 1K tok"
                  value={formatBytes(kvBytesPerKTokens)}
                  help={`f16 cache: 2 (K and V) × ${arch.layers} layers × ${arch.kvHeads} KV heads × ${formatNumber(
                    headDim,
                  )} head dim × 2 bytes × 1024 tokens. Quantizing the cache to q8_0 roughly halves it.`}
                />
                <ArchItem
                  label="Tied embeddings"
                  value={arch.tiedEmbeddings ? 'yes' : 'no'}
                  help="When lm_head reuses the token embedding table, the embedding contribution to file size is counted once instead of twice — it is why small Llamas are smaller than a naive bits-per-weight estimate suggests."
                />
              </dl>
            </AccordionContent>
          </AccordionItem>
        </Accordion>
      </CardContent>

      <CardFooter className="flex flex-wrap gap-2 p-5 pt-0">
        <Button
          variant="outline"
          size="xs"
          onClick={() => onGoTo('quant', model.id)}
          aria-label={`Plan quantization for ${model.name}`}
        >
          <Layers className="h-3.5 w-3.5" aria-hidden="true" />
          Plan quantization
        </Button>
        <Button
          variant="outline"
          size="xs"
          onClick={() => onGoTo('gguf', model.id)}
          aria-label={`Generate commands for ${model.name}`}
        >
          <TerminalSquare className="h-3.5 w-3.5" aria-hidden="true" />
          Generate commands
        </Button>
        <Button
          variant="outline"
          size="xs"
          onClick={() => onGoTo('cost', model.id)}
          aria-label={`Estimate cost for ${model.name}`}
        >
          <Receipt className="h-3.5 w-3.5" aria-hidden="true" />
          Estimate cost
        </Button>
      </CardFooter>
    </Card>
  );
}

/* -------------------------------------------------------------------------- */
/*  Panel                                                                      */
/* -------------------------------------------------------------------------- */

export function ModelSelectorPanel() {
  const { modelId, setModelId, goTo } = useStudio();

  const [query, setQuery] = React.useState('');
  const [families, setFamilies] = React.useState<ModelFamily[]>([]);
  const [modality, setModality] = React.useState<ModalityFilter>('all');
  const [sort, setSort] = React.useState<SortKey>('params-desc');

  const [hub, setHub] = React.useState<HubState>({ status: 'loading', entries: {} });
  // Bumping the token re-runs the effect, whose cleanup aborts the in-flight request.
  const [reloadToken, setReloadToken] = React.useState(0);

  React.useEffect(() => {
    const controller = new AbortController();
    let cancelled = false;

    setHub((prev) => ({ ...prev, status: 'loading', message: undefined }));

    fetchHubStats(controller.signal)
      .then((entries) => {
        if (cancelled) return;
        if (Object.keys(entries).length === 0) {
          // Every request failed (offline, rate limited, blocked). Keep whatever
          // we already had rather than blanking the catalog.
          setHub((prev) => ({
            status: 'error',
            entries: prev.entries,
            fetchedAt: prev.fetchedAt,
            message: 'no repository answered',
          }));
          return;
        }
        setHub({ status: 'ready', entries, fetchedAt: pickFetchedAt(entries) });
      })
      .catch((error: unknown) => {
        if (cancelled) return;
        setHub((prev) => ({
          status: 'error',
          entries: prev.entries,
          fetchedAt: prev.fetchedAt,
          message: error instanceof Error ? error.message : 'unknown error',
        }));
      });

    return () => {
      cancelled = true;
      controller.abort();
    };
  }, [reloadToken]);

  const refresh = React.useCallback(() => {
    invalidateHubCache();
    setReloadToken((token) => token + 1);
  }, []);

  const toggleFamily = (family: ModelFamily) => {
    setFamilies((prev) =>
      prev.includes(family) ? prev.filter((f) => f !== family) : [...prev, family],
    );
  };

  const filtersActive = query.trim() !== '' || families.length > 0 || modality !== 'all';

  const clearFilters = () => {
    setQuery('');
    setFamilies([]);
    setModality('all');
  };

  const visible = React.useMemo(() => {
    const terms = query.trim().toLowerCase().split(/\s+/).filter(Boolean);
    const entries = hub.entries;

    const filtered = MODELS.filter(
      (model) =>
        matchesQuery(model, terms) &&
        (families.length === 0 || families.includes(model.family)) &&
        (modality === 'all' || model.modality === modality),
    );

    const byParamsDesc = (a: LlamaModel, b: LlamaModel) => b.paramsB - a.paramsB;

    return [...filtered].sort((a, b) => {
      switch (sort) {
        case 'params-asc':
          return a.paramsB - b.paramsB;
        case 'context-desc':
          return b.contextLength - a.contextLength || byParamsDesc(a, b);
        case 'released-desc':
          // ISO dates sort correctly as strings — no Date construction needed.
          return b.releasedAt.localeCompare(a.releasedAt) || byParamsDesc(a, b);
        case 'downloads-desc':
          return (
            (entries[b.id]?.downloads ?? -1) - (entries[a.id]?.downloads ?? -1) ||
            byParamsDesc(a, b)
          );
        case 'params-desc':
        default:
          return byParamsDesc(a, b);
      }
    });
  }, [query, families, modality, sort, hub.entries]);

  const coverage = Object.keys(hub.entries).length;
  const statsLoading = hub.status === 'loading';
  const activeModel = getModel(modelId);

  return (
    <div className="space-y-6">
      <SectionHeading
        eyebrow="01 · Catalog"
        title="Llama model catalog"
        description="Every Llama checkpoint the studio can plan for, with the architecture numbers the rest of the tools depend on. Pick one and it stays selected across the quantization planner, the command generator and the cost model."
        actions={
          <div className="flex items-center gap-2">
            {hub.status === 'ready' && (
              <Badge variant="meta" className="font-normal">
                Live Hub stats
                {hub.fetchedAt ? ` · ${formatDate(hub.fetchedAt)}` : ''}
              </Badge>
            )}
            {hub.status === 'loading' && (
              <Badge variant="muted" className="font-normal">
                <RefreshCw className="h-3 w-3 animate-spin" aria-hidden="true" />
                Fetching Hub stats
              </Badge>
            )}
            {hub.status === 'error' && (
              <Badge variant="muted" className="font-normal">
                <CloudOff className="h-3 w-3" aria-hidden="true" />
                Bundled data only
              </Badge>
            )}
            <Button
              variant="outline"
              size="sm"
              onClick={refresh}
              disabled={hub.status === 'loading'}
              aria-label="Refresh HuggingFace statistics"
            >
              <RefreshCw
                className={cn('h-4 w-4', hub.status === 'loading' && 'animate-spin')}
                aria-hidden="true"
              />
              Refresh
            </Button>
          </div>
        }
      />

      <Card className="shadow-none">
        <CardContent className="space-y-4 p-4 sm:p-5">
          <div className="grid gap-4 sm:grid-cols-2 lg:grid-cols-[minmax(0,1fr)_13rem_15rem]">
            <Field
              label="Search"
              htmlFor="model-search"
              hint={filtersActive ? `${visible.length} match${visible.length === 1 ? '' : 'es'}` : undefined}
            >
              <div className="relative">
                <Search
                  className="pointer-events-none absolute left-3 top-1/2 h-4 w-4 -translate-y-1/2 text-muted-foreground"
                  aria-hidden="true"
                />
                <Input
                  id="model-search"
                  type="text"
                  autoComplete="off"
                  spellCheck={false}
                  value={query}
                  onChange={(event) => setQuery(event.target.value)}
                  placeholder="Name, repo id, family or description…"
                  className="pl-9 pr-9"
                />
                {query !== '' && (
                  <button
                    type="button"
                    onClick={() => setQuery('')}
                    aria-label="Clear the search box"
                    className="absolute right-2 top-1/2 -translate-y-1/2 rounded p-1 text-muted-foreground transition-colors hover:text-foreground focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring"
                  >
                    <X className="h-3.5 w-3.5" aria-hidden="true" />
                  </button>
                )}
              </div>
            </Field>

            <Field label="Modality" htmlFor="model-modality">
              <Select
                value={modality}
                onValueChange={(value) => {
                  if (isModalityFilter(value)) setModality(value);
                }}
              >
                <SelectTrigger id="model-modality" aria-label="Filter by modality">
                  <SelectValue />
                </SelectTrigger>
                <SelectContent>
                  {MODALITY_OPTIONS.map((option) => (
                    <SelectItem key={option.id} value={option.id}>
                      {option.label}
                    </SelectItem>
                  ))}
                </SelectContent>
              </Select>
            </Field>

            <Field
              label="Sort by"
              htmlFor="model-sort"
              help="Downloads come from the HuggingFace API. When that fetch fails, the download sort falls back to parameter count so the order stays deterministic."
            >
              <Select
                value={sort}
                onValueChange={(value) => {
                  if (isSortKey(value)) setSort(value);
                }}
              >
                <SelectTrigger id="model-sort" aria-label="Sort the catalog">
                  <SelectValue />
                </SelectTrigger>
                <SelectContent>
                  {SORT_OPTIONS.map((option) => (
                    <SelectItem key={option.id} value={option.id}>
                      {option.label}
                    </SelectItem>
                  ))}
                </SelectContent>
              </Select>
            </Field>
          </div>

          <Separator />

          <div className="flex flex-wrap items-center gap-2">
            <span className="text-xs font-medium text-foreground">Family</span>
            <div role="group" aria-label="Filter by model family" className="flex flex-wrap gap-1.5">
              <button
                type="button"
                onClick={() => setFamilies([])}
                aria-pressed={families.length === 0}
                className={cn(
                  'rounded-full border px-2.5 py-1 text-xs transition-colors focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring focus-visible:ring-offset-1 ring-offset-background',
                  families.length === 0
                    ? 'border-meta-300 bg-meta-50 font-medium text-meta-700'
                    : 'border-input text-muted-foreground hover:bg-muted',
                )}
              >
                All families
              </button>
              {FAMILY_COUNTS.map(({ family, count }) => {
                const active = families.includes(family);
                return (
                  <button
                    key={family}
                    type="button"
                    onClick={() => toggleFamily(family)}
                    aria-pressed={active}
                    className={cn(
                      'inline-flex items-center gap-1.5 rounded-full border px-2.5 py-1 text-xs transition-colors focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring focus-visible:ring-offset-1 ring-offset-background',
                      active
                        ? 'border-meta-300 bg-meta-50 font-medium text-meta-700'
                        : 'border-input text-muted-foreground hover:bg-muted',
                    )}
                  >
                    {family}
                    <span className="tabular font-mono text-[10px] opacity-70">{count}</span>
                  </button>
                );
              })}
            </div>
            {filtersActive && (
              <Button variant="ghost" size="xs" onClick={clearFilters} className="ml-auto">
                <X className="h-3.5 w-3.5" aria-hidden="true" />
                Clear filters
              </Button>
            )}
          </div>
        </CardContent>
      </Card>

      {hub.status === 'error' && (
        <div className="flex flex-wrap items-center gap-x-2 gap-y-1 rounded-lg border border-dashed px-3 py-2 text-xs leading-relaxed text-muted-foreground">
          <CloudOff className="h-3.5 w-3.5 shrink-0" aria-hidden="true" />
          <span className="min-w-0">
            Live HuggingFace statistics are unavailable
            {hub.message ? ` (${hub.message})` : ''}. Download and like counts are hidden; every
            architecture, context and license figure below is bundled with the app and is still
            accurate.
          </span>
          <Button variant="ghost" size="xs" onClick={refresh}>
            Try again
          </Button>
        </div>
      )}

      {hub.status === 'ready' && coverage < MODELS.length && (
        <div className="flex flex-wrap items-center gap-2 rounded-lg border border-dashed px-3 py-2 text-xs text-muted-foreground">
          <CloudOff className="h-3.5 w-3.5 shrink-0" aria-hidden="true" />
          <span>
            HuggingFace answered for {coverage} of {MODELS.length} repositories. The rest show
            bundled specs only.
          </span>
        </div>
      )}

      <div className="flex flex-wrap items-baseline justify-between gap-x-4 gap-y-1">
        <p className="text-sm text-muted-foreground">
          Showing <span className="tabular font-medium text-foreground">{visible.length}</span> of{' '}
          <span className="tabular">{MODELS.length}</span> checkpoints
        </p>
        <p className="text-xs text-muted-foreground">
          Active model <span className="font-mono text-foreground">{activeModel.repoId}</span>
        </p>
      </div>

      {visible.length === 0 ? (
        <EmptyState
          icon={<SearchX className="h-8 w-8" aria-hidden="true" />}
          title="No checkpoints match these filters"
          description="Try a shorter search term, or widen the family and modality filters. The bundled catalog holds every Llama checkpoint the studio can model."
          action={
            <Button variant="outline" size="sm" onClick={clearFilters}>
              <X className="h-4 w-4" aria-hidden="true" />
              Clear filters
            </Button>
          }
        />
      ) : (
        <div className="grid grid-cols-1 gap-4 md:grid-cols-2 xl:grid-cols-3">
          {visible.map((model) => (
            <ModelCard
              key={model.id}
              model={model}
              stats={hub.entries[model.id]}
              statsLoading={statsLoading}
              selected={model.id === modelId}
              onSelect={setModelId}
              onGoTo={(tab, id) => goTo(tab, { modelId: id })}
            />
          ))}
        </div>
      )}

      <Footnote>
        Architecture figures (layers, hidden size, head counts, vocabulary) come from each
        repository&rsquo;s <span className="font-mono">config.json</span> and Meta&rsquo;s model
        cards, bundled with this app so the catalog works with no network at all. Download and like
        counts are fetched from the public HuggingFace API in your browser — no API key, no
        proxy — and cached in <span className="font-mono">localStorage</span> for six hours;
        HuggingFace counts downloads over a rolling 30-day window. Context lengths are the maximum
        the checkpoint supports, not what your hardware can serve; the quantization planner sizes
        the KV cache for the window you actually intend to run.
      </Footnote>
    </div>
  );
}
