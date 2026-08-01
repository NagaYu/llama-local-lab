'use client';

import * as React from 'react';
import {
  AlertTriangle,
  Check,
  ClipboardCheck,
  Download,
  Eye,
  FileJson,
  Info,
  Loader2,
  Play,
  Plus,
  RotateCcw,
  Square,
  Trash2,
  Upload,
  X,
} from 'lucide-react';

import { CodeBlock, InlineCode } from '@/components/code-block';
import { CopyButton } from '@/components/copy-button';
import {
  EmptyState,
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
import {
  Card,
  CardContent,
  CardDescription,
  CardHeader,
  CardTitle,
} from '@/components/ui/card';
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
import { Progress } from '@/components/ui/progress';
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from '@/components/ui/select';
import { Separator } from '@/components/ui/separator';
import {
  Table,
  TableBody,
  TableCell,
  TableHead,
  TableHeader,
  TableRow,
} from '@/components/ui/table';
import { Textarea } from '@/components/ui/textarea';
import { InfoTip } from '@/components/ui/tooltip';
import { evalRunToCsv, evalRunToJsonl, graderToScoringFn, parseQaPairs } from '@/lib/csv';
import { generate, resetChat, useEngine } from '@/lib/engine';
import {
  estimateTokens,
  formatDuration,
  formatNumber,
  formatPct,
  formatTps,
} from '@/lib/format';
import { STORAGE_KEYS, useMounted, usePersistentState } from '@/lib/storage';
import type {
  EvalCase,
  EvalCaseResult,
  EvalRun,
  EvalSummary,
  GenerationStats,
  GraderKind,
  SamplingParams,
} from '@/lib/types';
import { cn, downloadFile, nextId } from '@/lib/utils';

/* -------------------------------------------------------------------------- */
/*  Constants                                                                  */
/* -------------------------------------------------------------------------- */

/**
 * Seed cases.
 *
 * Ids are literal strings rather than `nextId()` output because they end up in
 * `id`/`htmlFor` attributes: a randomly generated id would differ between the
 * prerendered HTML and the client's first render and blow up hydration.
 *
 * The questions are deliberately chosen to line up with the demo engine's
 * canned replies, so the entire flow — run, grade, summarize, export — is
 * demonstrable on a machine that has never downloaded a model.
 */
const SEED_CASES: EvalCase[] = [
  { id: 'seed-capital-france', question: 'What is the capital of France?', expected: 'Paris' },
  { id: 'seed-arithmetic', question: 'What is 2 + 2?', expected: '4' },
  {
    id: 'seed-largest-planet',
    question: 'Which is the largest planet in our solar system?',
    expected: 'Jupiter',
  },
  {
    id: 'seed-romeo-juliet',
    question: 'Who wrote Romeo and Juliet?',
    expected: 'William Shakespeare',
  },
  {
    id: 'seed-speed-of-light',
    question: 'What is the speed of light in a vacuum, in meters per second?',
    expected: '299,792,458 meters per second',
  },
];

const DEFAULT_SYSTEM_PROMPT =
  'You are being evaluated. Answer with the shortest correct answer and nothing else. Do not explain your reasoning.';

/** Token-F1 cut-off. 0.6 is the conventional operating point for SQuAD-style F1. */
const FUZZY_PASS_THRESHOLD = 0.6;

const MAX_HISTORY = 10;

const EMPTY_SUMMARY: EvalSummary = {
  total: 0,
  passed: 0,
  accuracy: 0,
  meanScore: 0,
  meanLatencyMs: 0,
  meanTokensPerSecond: 0,
};

const GRADERS: { id: GraderKind; label: string; blurb: string }[] = [
  {
    id: 'exact-match',
    label: 'Exact match',
    blurb:
      'Case-insensitive string equality after trimming, collapsing runs of whitespace and dropping trailing punctuation. Binary 1 or 0. Unforgiving, but the only grader with no judgement calls in it.',
  },
  {
    id: 'contains',
    label: 'Contains',
    blurb:
      'Passes when the normalized expected answer appears anywhere inside the normalized model output. Binary 1 or 0. Tolerates preambles like "The answer is…", and rewards a model that buries the right answer in a wall of text.',
  },
  {
    id: 'fuzzy',
    label: 'Fuzzy (token F1)',
    blurb:
      'SQuAD-style token-level F1 over whitespace-split tokens, with articles removed and edge punctuation stripped. Continuous 0–1; a case passes at 0.60 or above.',
  },
  {
    id: 'regex',
    label: 'Regex',
    blurb:
      'The expected column is treated as a case-insensitive JavaScript regular expression and tested against the raw model output. An invalid pattern is recorded on the row as an error instead of aborting the run.',
  },
  {
    id: 'llm-judge',
    label: 'LLM as judge',
    blurb:
      'A second generation against the same local model, asked for a single CORRECT / INCORRECT verdict plus a one-line reason. Roughly doubles the run time.',
  },
];

const JUDGE_SYSTEM_PROMPT =
  'You are a strict grader. Reply with exactly one word on the first line: CORRECT or INCORRECT. On the second line give a reason of at most fifteen words. Output nothing else.';

function buildJudgePrompt(question: string, expected: string, actual: string): string {
  return [
    `Question: ${question}`,
    `Reference answer: ${expected}`,
    `Candidate answer: ${actual}`,
    '',
    'Is the candidate answer factually equivalent to the reference answer? Ignore wording, formatting and extra detail.',
  ].join('\n');
}

/** Greedy decoding for the judge — a grader that changes its mind is not a grader. */
const JUDGE_SAMPLING: SamplingParams = {
  temperature: 0,
  topP: 1,
  maxTokens: 96,
  repetitionPenalty: 1,
  presencePenalty: 0,
  frequencyPenalty: 0,
  seed: null,
};

/* -------------------------------------------------------------------------- */
/*  Graders                                                                    */
/* -------------------------------------------------------------------------- */

interface GradeOutcome {
  score: number;
  passed: boolean;
  rationale?: string;
  error?: string;
}

const ARTICLES = new Set(['a', 'an', 'the']);

/**
 * Lowercase, collapse whitespace, drop wrapping quotes and trailing sentence
 * punctuation. "Paris." and " paris " must compare equal, or exact-match is
 * useless against any instruction-tuned model.
 */
function normalize(text: string): string {
  return text
    .trim()
    .toLowerCase()
    .replace(/\s+/g, ' ')
    .replace(/^["'([]+/, '')
    .replace(/[.,;:!?'")\]]+$/, '')
    .trim();
}

/**
 * SQuAD-style tokens: normalized, punctuation stripped from each token's edges,
 * articles removed. Internal punctuation is kept so `299,792,458` stays one
 * token instead of fragmenting into three numbers that all match by accident.
 */
function tokensOf(text: string): string[] {
  return normalize(text)
    .split(' ')
    .map((token) => token.replace(/^[^\p{L}\p{N}]+|[^\p{L}\p{N}]+$/gu, ''))
    .filter((token) => token.length > 0 && !ARTICLES.has(token));
}

/** Token-level F1 with multiset (not set) intersection, as SQuAD defines it. */
function tokenF1(expected: string, actual: string): number {
  const gold = tokensOf(expected);
  const pred = tokensOf(actual);

  // Two empty token sets agree vacuously. One empty set means there is no
  // overlap to measure at all and precision/recall are undefined, which SQuAD
  // scores as 0 rather than as a divide-by-zero.
  if (gold.length === 0 && pred.length === 0) return 1;
  if (gold.length === 0 || pred.length === 0) return 0;

  const remaining = new Map<string, number>();
  for (const token of gold) remaining.set(token, (remaining.get(token) ?? 0) + 1);

  let common = 0;
  for (const token of pred) {
    const left = remaining.get(token) ?? 0;
    if (left > 0) {
      remaining.set(token, left - 1);
      common += 1;
    }
  }
  if (common === 0) return 0;

  const precision = common / pred.length;
  const recall = common / gold.length;
  return (2 * precision * recall) / (precision + recall);
}

/** The four graders that need no second model call. */
function gradeDeterministic(
  grader: Exclude<GraderKind, 'llm-judge'>,
  expected: string,
  actual: string,
): GradeOutcome {
  if (grader === 'exact-match') {
    const hit = normalize(expected) === normalize(actual);
    return { score: hit ? 1 : 0, passed: hit };
  }

  if (grader === 'contains') {
    const hit = normalize(actual).includes(normalize(expected));
    return { score: hit ? 1 : 0, passed: hit };
  }

  if (grader === 'fuzzy') {
    const score = tokenF1(expected, actual);
    return { score, passed: score >= FUZZY_PASS_THRESHOLD };
  }

  // regex — a user-authored pattern is untrusted input, so a syntax error is a
  // property of the row, not a reason to lose the rest of the run.
  try {
    const hit = new RegExp(expected, 'i').test(actual);
    return { score: hit ? 1 : 0, passed: hit };
  } catch (err) {
    return {
      score: 0,
      passed: false,
      error: `Invalid regular expression: ${errorMessage(err)}`,
    };
  }
}

/** Pull a verdict and a short reason out of the judge's free-form reply. */
function parseVerdict(raw: string): GradeOutcome {
  const text = raw.trim();
  // \b prevents CORRECT from matching inside INCORRECT.
  const match = text.match(/\b(in)?correct\b/i);

  if (!match) {
    return {
      score: 0,
      passed: false,
      rationale: text.slice(0, 240) || undefined,
      error: 'The judge did not emit a CORRECT / INCORRECT verdict.',
    };
  }

  const rationale = text
    .slice((match.index ?? 0) + match[0].length)
    .replace(/^[\s.:,;\-–—]+/, '')
    .replace(/\s+/g, ' ')
    .trim()
    .slice(0, 240);

  const correct = !match[1];
  return { score: correct ? 1 : 0, passed: correct, rationale: rationale || undefined };
}

async function runJudge(args: {
  question: string;
  expected: string;
  actual: string;
  signal: AbortSignal;
}): Promise<GradeOutcome> {
  // The judge must not see the answer turn's KV cache, or it grades its own
  // continuation instead of the pair it was handed.
  await resetChat();

  let text = '';
  for await (const delta of generate({
    messages: [
      { role: 'system', content: JUDGE_SYSTEM_PROMPT },
      { role: 'user', content: buildJudgePrompt(args.question, args.expected, args.actual) },
    ],
    sampling: JUDGE_SAMPLING,
    signal: args.signal,
  })) {
    text = delta.text;
  }

  return parseVerdict(text);
}

/* -------------------------------------------------------------------------- */
/*  Helpers                                                                    */
/* -------------------------------------------------------------------------- */

type CaseStatus = 'pending' | 'running' | 'passed' | 'failed';

function errorMessage(err: unknown): string {
  return err instanceof Error ? err.message : String(err);
}

/** `accuracy` and `meanScore` are fractions in 0–1; the UI multiplies for display. */
function summarize(results: EvalCaseResult[]): EvalSummary {
  const total = results.length;
  if (total === 0) return { ...EMPTY_SUMMARY };

  const passed = results.filter((r) => r.passed).length;
  const sum = (pick: (r: EvalCaseResult) => number) => results.reduce((n, r) => n + pick(r), 0);
  // Throughput averages only over cases that actually decoded tokens, so a case
  // that errored before generation does not drag tok/s toward zero.
  const timed = results.filter((r) => r.tokensPerSecond > 0);

  return {
    total,
    passed,
    accuracy: passed / total,
    meanScore: sum((r) => r.score) / total,
    meanLatencyMs: sum((r) => r.latencyMs) / total,
    meanTokensPerSecond: timed.length
      ? timed.reduce((n, r) => n + r.tokensPerSecond, 0) / timed.length
      : 0,
  };
}

function graderLabel(grader: GraderKind): string {
  return GRADERS.find((g) => g.id === grader)?.label ?? grader;
}

function fileSlug(text: string): string {
  return (
    text
      .replace(/[^A-Za-z0-9._-]+/g, '-')
      .replace(/^-+|-+$/g, '')
      .slice(0, 60) || 'model'
  );
}

function formatTimestamp(iso: string): string {
  const d = new Date(iso);
  if (Number.isNaN(d.getTime())) return '—';
  return d.toLocaleString('en-US', {
    month: 'short',
    day: 'numeric',
    hour: '2-digit',
    minute: '2-digit',
  });
}

function reviveCases(raw: unknown): EvalCase[] {
  if (!Array.isArray(raw)) return SEED_CASES;
  return raw
    .filter((row): row is Record<string, unknown> => !!row && typeof row === 'object')
    .map((row, i) => ({
      id: typeof row.id === 'string' && row.id ? row.id : `case-restored-${i}`,
      question: typeof row.question === 'string' ? row.question : '',
      expected: typeof row.expected === 'string' ? row.expected : '',
    }));
}

function isEvalRun(value: unknown): value is EvalRun {
  if (!value || typeof value !== 'object') return false;
  // Shape check on a persisted blob: every field the exporters touch must exist,
  // otherwise a stale localStorage entry crashes the CSV writer.
  const run = value as Partial<EvalRun>;
  return (
    typeof run.id === 'string' &&
    typeof run.startedAt === 'string' &&
    typeof run.modelId === 'string' &&
    Array.isArray(run.results) &&
    !!run.summary &&
    !!run.sampling
  );
}

function reviveRuns(raw: unknown): EvalRun[] {
  if (!Array.isArray(raw)) return [];
  return raw.filter(isEvalRun).slice(0, MAX_HISTORY);
}

/* -------------------------------------------------------------------------- */
/*  Panel                                                                      */
/* -------------------------------------------------------------------------- */

export function EvalSuitePanel() {
  const { goTo } = useStudio();
  const engine = useEngine();
  const mounted = useMounted();

  const [cases, setCases, casesMeta] = usePersistentState<EvalCase[]>(
    STORAGE_KEYS.evalCases,
    SEED_CASES,
    { revive: reviveCases },
  );
  const [history, setHistory] = usePersistentState<EvalRun[]>(STORAGE_KEYS.evalRuns, [], {
    revive: reviveRuns,
  });

  const [grader, setGrader] = React.useState<GraderKind>('exact-match');
  const [systemPrompt, setSystemPrompt] = React.useState(DEFAULT_SYSTEM_PROMPT);
  const [temperature, setTemperature] = React.useState(0);
  const [maxTokens, setMaxTokens] = React.useState(160);

  const [run, setRun] = React.useState<EvalRun | null>(null);
  const [running, setRunning] = React.useState(false);
  const [progress, setProgress] = React.useState({ done: 0, total: 0 });
  const [caseStatus, setCaseStatus] = React.useState<Record<string, CaseStatus>>({});
  const [detail, setDetail] = React.useState<EvalCaseResult | null>(null);

  const abortRef = React.useRef<AbortController | null>(null);

  // Radix unmounts an inactive tab panel; without this, switching tabs mid-run
  // would leave a generation streaming into a dead component.
  React.useEffect(
    () => () => {
      abortRef.current?.abort();
    },
    [],
  );

  const runnable = React.useMemo(
    () => cases.filter((c) => c.question.trim().length > 0),
    [cases],
  );
  const missingExpected = runnable.filter((c) => c.expected.trim().length === 0).length;

  const estimatedPromptTokens = React.useMemo(
    () =>
      runnable.reduce(
        (n, c) => n + estimateTokens(systemPrompt) + estimateTokens(c.question),
        0,
      ),
    [runnable, systemPrompt],
  );

  const activeGrader = GRADERS.find((g) => g.id === grader) ?? GRADERS[0];
  const canRun = engine.ready && runnable.length > 0 && !running;

  /* ---------------------------------------------------------------------- */
  /*  Case editing                                                          */
  /* ---------------------------------------------------------------------- */

  const updateCase = (id: string, patch: Partial<EvalCase>) => {
    setCases((prev) => prev.map((c) => (c.id === id ? { ...c, ...patch } : c)));
  };

  const addCase = () => {
    setCases((prev) => [...prev, { id: nextId('case'), question: '', expected: '' }]);
  };

  const removeCase = (id: string) => {
    setCases((prev) => prev.filter((c) => c.id !== id));
  };

  const importPairs = (pairs: { question: string; expected: string }[], mode: 'replace' | 'append') => {
    const imported: EvalCase[] = pairs.map((p) => ({
      id: nextId('case'),
      question: p.question,
      expected: p.expected,
    }));
    setCases((prev) => (mode === 'replace' ? imported : [...prev, ...imported]));
    setCaseStatus({});
  };

  /* ---------------------------------------------------------------------- */
  /*  Runner                                                                */
  /* ---------------------------------------------------------------------- */

  const stop = () => {
    abortRef.current?.abort();
  };

  const start = async () => {
    if (!engine.ready || runnable.length === 0 || running) return;

    const controller = new AbortController();
    abortRef.current = controller;

    const sampling: SamplingParams = {
      temperature,
      topP: temperature === 0 ? 1 : 0.9,
      maxTokens,
      repetitionPenalty: 1.1,
      presencePenalty: 0,
      frequencyPenalty: 0,
      seed: null,
    };

    const draft: EvalRun = {
      id: nextId('run'),
      startedAt: new Date().toISOString(),
      modelId: engine.modelId ?? 'unknown',
      backend: engine.backend ?? 'mock',
      grader,
      sampling,
      systemPrompt,
      results: [],
      summary: { ...EMPTY_SUMMARY },
    };

    setRunning(true);
    setDetail(null);
    setRun(draft);
    setProgress({ done: 0, total: runnable.length });
    setCaseStatus(Object.fromEntries(runnable.map((c) => [c.id, 'pending' as CaseStatus])));

    const results: EvalCaseResult[] = [];

    for (const evalCase of runnable) {
      if (controller.signal.aborted) break;
      setCaseStatus((prev) => ({ ...prev, [evalCase.id]: 'running' }));

      let result: EvalCaseResult;

      try {
        // A fresh conversation per case: without this the previous answer stays
        // in the KV cache and case n+1 can be answered by case n's context.
        await resetChat();

        const startedAt = performance.now();
        let actual = '';
        let stats: GenerationStats | undefined;

        for await (const delta of generate({
          messages: [
            ...(systemPrompt.trim()
              ? [{ role: 'system' as const, content: systemPrompt }]
              : []),
            { role: 'user' as const, content: evalCase.question },
          ],
          sampling,
          signal: controller.signal,
        })) {
          actual = delta.text;
          if (delta.stats) stats = delta.stats;
        }

        if (controller.signal.aborted) {
          // Stopped mid-case: keep everything already collected, and leave this
          // row pending rather than scoring a truncated answer as a failure.
          setCaseStatus((prev) => ({ ...prev, [evalCase.id]: 'pending' }));
          break;
        }

        const latencyMs = stats?.totalMs ?? performance.now() - startedAt;
        const outcome = await gradeOne(grader, evalCase, actual, controller.signal);

        result = {
          caseId: evalCase.id,
          question: evalCase.question,
          expected: evalCase.expected,
          actual,
          score: outcome.score,
          passed: outcome.passed,
          grader,
          rationale: outcome.rationale,
          latencyMs,
          tokensPerSecond: stats?.tokensPerSecond ?? 0,
          completionTokens: stats?.completionTokens ?? 0,
          error: outcome.error,
        };
      } catch (err) {
        if (controller.signal.aborted) break;
        // One bad case must not kill the run: record it and carry on.
        result = {
          caseId: evalCase.id,
          question: evalCase.question,
          expected: evalCase.expected,
          actual: '',
          score: 0,
          passed: false,
          grader,
          latencyMs: 0,
          tokensPerSecond: 0,
          completionTokens: 0,
          error: errorMessage(err),
        };
      }

      results.push(result);
      setCaseStatus((prev) => ({ ...prev, [evalCase.id]: result.passed ? 'passed' : 'failed' }));
      setProgress({ done: results.length, total: runnable.length });
      setRun((prev) =>
        prev && prev.id === draft.id
          ? { ...prev, results: [...results], summary: summarize(results) }
          : prev,
      );
    }

    const finished: EvalRun = {
      ...draft,
      finishedAt: new Date().toISOString(),
      results,
      summary: summarize(results),
    };

    setRun(finished);
    setRunning(false);
    abortRef.current = null;

    if (results.length > 0) {
      setHistory((prev) => [finished, ...prev.filter((r) => r.id !== finished.id)].slice(0, MAX_HISTORY));
    }
  };

  /* ---------------------------------------------------------------------- */
  /*  Export                                                                */
  /* ---------------------------------------------------------------------- */

  const exportName = (extension: string) =>
    run
      ? `eval-${fileSlug(run.modelId)}-${run.startedAt.replace(/[:.]/g, '-')}.${extension}`
      : `eval.${extension}`;

  const viewHistoric = (entry: EvalRun) => {
    setRun(entry);
    setDetail(null);
    setCaseStatus({});
  };

  const deleteHistoric = (id: string) => {
    setHistory((prev) => prev.filter((r) => r.id !== id));
    setRun((prev) => (prev?.id === id ? null : prev));
  };

  /* ---------------------------------------------------------------------- */
  /*  Render                                                                */
  /* ---------------------------------------------------------------------- */

  const progressPct = progress.total > 0 ? (progress.done / progress.total) * 100 : 0;

  return (
    <div className="space-y-8">
      <SectionHeading
        eyebrow="06 — Evaluation"
        title="Evaluation mini-suite"
        description="Paste a handful of question/answer pairs, run them against the model loaded in this tab, score them with a grader you can read the source of, and export in the column layout Llama Stack's scoring API expects."
        actions={
          engine.ready && engine.modelId ? (
            <Badge variant="success" className="font-mono">
              <Check className="h-3 w-3" />
              {engine.modelId}
            </Badge>
          ) : (
            <Badge variant="muted">No model loaded</Badge>
          )
        }
      />

      {!engine.ready && (
        <Alert variant="info">
          <Info />
          <AlertTitle>The evaluation runs against the model in your browser</AlertTitle>
          <AlertDescription className="space-y-2">
            <p>
              Nothing here calls an API. Load a model — or the zero-download demo engine — in the
              playground first, and it stays resident when you come back to this tab.
            </p>
            <Button variant="outline" size="sm" onClick={() => goTo('playground')}>
              Go to the playground
            </Button>
          </AlertDescription>
        </Alert>
      )}

      <div className="grid gap-6 lg:grid-cols-3">
        {/* ---------------------------------------------------------------- */}
        {/*  Case editor                                                     */}
        {/* ---------------------------------------------------------------- */}
        <Card className="lg:col-span-2">
          <CardHeader className="flex-row flex-wrap items-start justify-between gap-3 space-y-0">
            <div className="min-w-0 space-y-1.5">
              <CardTitle className="flex items-center gap-2">
                Cases
                <Badge variant="meta" className="tabular font-mono">
                  {cases.length}
                </Badge>
              </CardTitle>
              <CardDescription>
                One question and one reference answer per row. Rows with an empty question are
                skipped when the suite runs.
              </CardDescription>
            </div>
            <div className="flex flex-wrap items-center gap-2">
              <ImportDialog onImport={importPairs} disabled={running} />
              <Button variant="outline" size="sm" onClick={casesMeta.reset} disabled={running}>
                <RotateCcw />
                Reset to examples
              </Button>
              <Button size="sm" onClick={addCase} disabled={running}>
                <Plus />
                Add case
              </Button>
            </div>
          </CardHeader>

          <CardContent className="space-y-3">
            {cases.length === 0 ? (
              <EmptyState
                icon={<ClipboardCheck className="h-6 w-6" />}
                title="No cases"
                description="Add a case by hand, import a batch of Q&A pairs, or restore the five bundled examples."
                action={
                  <Button size="sm" variant="outline" onClick={casesMeta.reset}>
                    <RotateCcw />
                    Restore examples
                  </Button>
                }
              />
            ) : (
              <ul className="space-y-3">
                {cases.map((evalCase, index) => (
                  <li key={evalCase.id} className="rounded-lg border bg-card p-3">
                    <div className="mb-2 flex items-center justify-between gap-2">
                      <div className="flex items-center gap-2">
                        <span className="tabular font-mono text-xs text-muted-foreground">
                          {String(index + 1).padStart(2, '0')}
                        </span>
                        <CaseStatusBadge status={caseStatus[evalCase.id] ?? 'pending'} />
                      </div>
                      <Button
                        variant="ghost"
                        size="icon-sm"
                        aria-label={`Delete case ${index + 1}`}
                        onClick={() => removeCase(evalCase.id)}
                        disabled={running}
                        className="text-muted-foreground hover:text-danger"
                      >
                        <Trash2 />
                      </Button>
                    </div>

                    <div className="grid gap-3 sm:grid-cols-2">
                      <Field label="Question" htmlFor={`${evalCase.id}-question`}>
                        <Textarea
                          id={`${evalCase.id}-question`}
                          rows={2}
                          value={evalCase.question}
                          onChange={(e) => updateCase(evalCase.id, { question: e.target.value })}
                          placeholder="What is the capital of France?"
                          className="min-h-[64px]"
                          disabled={running}
                        />
                      </Field>
                      <Field
                        label={grader === 'regex' ? 'Expected (regex)' : 'Expected answer'}
                        htmlFor={`${evalCase.id}-expected`}
                      >
                        <Textarea
                          id={`${evalCase.id}-expected`}
                          rows={2}
                          value={evalCase.expected}
                          onChange={(e) => updateCase(evalCase.id, { expected: e.target.value })}
                          placeholder={grader === 'regex' ? '^\\s*Paris\\b' : 'Paris'}
                          className={cn('min-h-[64px]', grader === 'regex' && 'font-mono text-xs')}
                          disabled={running}
                        />
                      </Field>
                    </div>
                  </li>
                ))}
              </ul>
            )}

            {missingExpected > 0 && (
              <Footnote>
                {missingExpected} {missingExpected === 1 ? 'case has' : 'cases have'} no reference
                answer. Those rows will run and be recorded with a grading error, because there is
                nothing to score against.
              </Footnote>
            )}
          </CardContent>
        </Card>

        {/* ---------------------------------------------------------------- */}
        {/*  Run configuration                                               */}
        {/* ---------------------------------------------------------------- */}
        <Card>
          <CardHeader>
            <CardTitle>Run configuration</CardTitle>
            <CardDescription>
              Scoring and sampling for the whole suite. Both are recorded in the export.
            </CardDescription>
          </CardHeader>

          <CardContent className="space-y-4">
            <Field
              label="Grader"
              htmlFor="eval-grader"
              help="How a generated answer is turned into a score. Every grader in this list is implemented client-side; nothing is sent anywhere."
            >
              <Select value={grader} onValueChange={(v) => setGrader(v as GraderKind)}>
                <SelectTrigger id="eval-grader" aria-label="Grader">
                  <SelectValue />
                </SelectTrigger>
                <SelectContent>
                  {GRADERS.map((g) => (
                    <SelectItem key={g.id} value={g.id}>
                      {g.label}
                    </SelectItem>
                  ))}
                </SelectContent>
              </Select>
            </Field>

            <p className="text-pretty text-xs leading-relaxed text-muted-foreground">
              {activeGrader.blurb}
            </p>

            <div className="rounded-lg border bg-muted/40 p-2.5 text-xs">
              <span className="text-muted-foreground">Exports as </span>
              <InlineCode>{graderToScoringFn(grader)}</InlineCode>
            </div>

            {grader === 'llm-judge' && (
              <div className="space-y-3">
                <Alert variant="warning">
                  <AlertTriangle />
                  <AlertTitle>A small local model judging itself is a weak signal</AlertTitle>
                  <AlertDescription>
                    The judge and the candidate are the same weights, so its errors are correlated
                    with the answers it is grading. Treat this as a smoke test that the harness
                    works end to end, not as a number worth putting on a leaderboard.
                  </AlertDescription>
                </Alert>
                <CodeBlock
                  code={`${JUDGE_SYSTEM_PROMPT}\n\n${buildJudgePrompt(
                    '{question}',
                    '{expected answer}',
                    '{model output}',
                  )}`}
                  language="text"
                  title="judge prompt"
                  maxHeight={190}
                />
                {engine.backend === 'mock' && (
                  <Footnote>
                    The demo engine replies with canned text and will not produce a verdict token,
                    so every row here records a judge error. Use exact match, contains or fuzzy to
                    tour the suite without a real model.
                  </Footnote>
                )}
              </div>
            )}

            <Separator />

            <Field
              label="System prompt"
              htmlFor="eval-system"
              help="Prepended to every case. Kept short and directive because most graders here compare strings, and a chatty model fails exact match on content it got right."
            >
              <Textarea
                id="eval-system"
                rows={3}
                value={systemPrompt}
                onChange={(e) => setSystemPrompt(e.target.value)}
                disabled={running}
                className="text-xs"
              />
            </Field>

            <div className="grid grid-cols-2 gap-3">
              <Field
                label="Temperature"
                htmlFor="eval-temperature"
                help="0 means greedy decoding: the model always takes its top token, so the same suite run twice gives the same answers. Anything above 0 makes a regression impossible to distinguish from sampling noise."
              >
                <NumberInput
                  id="eval-temperature"
                  value={temperature}
                  onChange={setTemperature}
                  min={0}
                  max={2}
                  step={0.05}
                  disabled={running}
                />
              </Field>
              <Field
                label="Max tokens"
                htmlFor="eval-max-tokens"
                help="Upper bound on the answer length per case. Short answers keep exact match honest and keep the run fast."
                hint="cap"
              >
                <NumberInput
                  id="eval-max-tokens"
                  value={maxTokens}
                  onChange={setMaxTokens}
                  min={8}
                  max={2048}
                  step={8}
                  disabled={running}
                />
              </Field>
            </div>

            {temperature > 0 && (
              <Footnote>
                Temperature is above 0, so two runs of this suite can disagree on the same model.
                Set it back to 0 before comparing quants.
              </Footnote>
            )}

            <Separator />

            <div className="space-y-3">
              {running ? (
                <Button variant="destructive" className="w-full" onClick={stop}>
                  <Square />
                  Stop run
                </Button>
              ) : (
                <Button className="w-full" onClick={start} disabled={!canRun}>
                  <Play />
                  Run {runnable.length} {runnable.length === 1 ? 'case' : 'cases'}
                </Button>
              )}

              {!engine.ready && (
                <p className="text-xs leading-relaxed text-muted-foreground">
                  Disabled because no engine is loaded. The suite generates with the model running
                  in this browser tab, so there is nothing to run against yet.
                </p>
              )}
              {engine.ready && runnable.length === 0 && (
                <p className="text-xs leading-relaxed text-muted-foreground">
                  Disabled because every case has an empty question.
                </p>
              )}

              {running && (
                <div className="space-y-1.5">
                  <Progress value={progressPct} aria-label="Evaluation progress" />
                  <div className="flex items-center justify-between text-xs text-muted-foreground">
                    <span className="flex items-center gap-1.5">
                      <Loader2 className="h-3 w-3 animate-spin" />
                      Case {Math.min(progress.done + 1, progress.total)} of {progress.total}
                    </span>
                    <span className="tabular font-mono">{formatPct(progressPct, 0)}</span>
                  </div>
                </div>
              )}

              {!running && runnable.length > 0 && (
                <Footnote>
                  Cases run sequentially with the chat state reset between them, so one answer
                  cannot condition the next. Roughly{' '}
                  <span className="tabular font-mono">
                    {formatNumber(estimatedPromptTokens)}
                  </span>{' '}
                  prompt tokens across the suite, estimated at four characters per token
                  {grader === 'llm-judge' ? ', before the judge pass' : ''}.
                </Footnote>
              )}
            </div>
          </CardContent>
        </Card>
      </div>

      {/* ------------------------------------------------------------------ */}
      {/*  Results                                                           */}
      {/* ------------------------------------------------------------------ */}
      <ResultsSection
        run={run}
        running={running}
        onOpenDetail={setDetail}
        exportName={exportName}
      />

      {/* ------------------------------------------------------------------ */}
      {/*  History                                                           */}
      {/* ------------------------------------------------------------------ */}
      <Card>
        <CardHeader>
          <CardTitle>Run history</CardTitle>
          <CardDescription>
            The last {MAX_HISTORY} runs, kept in this browser's localStorage. Nothing is uploaded.
          </CardDescription>
        </CardHeader>
        <CardContent>
          {history.length === 0 ? (
            <Footnote>
              No saved runs yet. Every completed or stopped run with at least one scored case is
              saved here automatically.
            </Footnote>
          ) : (
            <ul className="divide-y">
              {history.map((entry) => (
                <li
                  key={entry.id}
                  className={cn(
                    'flex flex-wrap items-center justify-between gap-3 py-2.5',
                    run?.id === entry.id && 'bg-meta-50/60',
                  )}
                >
                  <div className="flex min-w-0 flex-wrap items-center gap-x-3 gap-y-1">
                    <span className="tabular w-32 font-mono text-xs text-muted-foreground">
                      {mounted ? formatTimestamp(entry.startedAt) : '—'}
                    </span>
                    <span className="truncate font-mono text-xs">{entry.modelId}</span>
                    <Badge variant="outline">{graderLabel(entry.grader)}</Badge>
                    <Badge variant={entry.summary.accuracy >= 0.8 ? 'success' : 'muted'}>
                      <span className="tabular">
                        {formatPct(entry.summary.accuracy * 100, 0)} · {entry.summary.passed}/
                        {entry.summary.total}
                      </span>
                    </Badge>
                  </div>
                  <div className="flex items-center gap-1">
                    <Button
                      variant="ghost"
                      size="sm"
                      onClick={() => viewHistoric(entry)}
                      disabled={running}
                    >
                      <Eye />
                      View
                    </Button>
                    <Button
                      variant="ghost"
                      size="icon-sm"
                      aria-label={`Delete run from ${entry.startedAt}`}
                      onClick={() => deleteHistoric(entry.id)}
                      disabled={running}
                      className="text-muted-foreground hover:text-danger"
                    >
                      <Trash2 />
                    </Button>
                  </div>
                </li>
              ))}
            </ul>
          )}
        </CardContent>
      </Card>

      <DetailDialog result={detail} onClose={() => setDetail(null)} />
    </div>
  );
}

/* -------------------------------------------------------------------------- */
/*  Grading entry point                                                        */
/* -------------------------------------------------------------------------- */

async function gradeOne(
  grader: GraderKind,
  evalCase: EvalCase,
  actual: string,
  signal: AbortSignal,
): Promise<GradeOutcome> {
  if (!evalCase.expected.trim()) {
    return {
      score: 0,
      passed: false,
      error: 'No reference answer, so there is nothing to score this against.',
    };
  }

  if (grader === 'llm-judge') {
    return runJudge({
      question: evalCase.question,
      expected: evalCase.expected,
      actual,
      signal,
    });
  }

  return gradeDeterministic(grader, evalCase.expected, actual);
}

/* -------------------------------------------------------------------------- */
/*  Sub-components                                                             */
/* -------------------------------------------------------------------------- */

function CaseStatusBadge({ status }: { status: CaseStatus }) {
  if (status === 'running') {
    return (
      <Badge variant="meta">
        <Loader2 className="h-3 w-3 animate-spin" />
        Running
      </Badge>
    );
  }
  if (status === 'passed') {
    return (
      <Badge variant="success">
        <Check className="h-3 w-3" />
        Passed
      </Badge>
    );
  }
  if (status === 'failed') {
    return (
      <Badge variant="danger">
        <X className="h-3 w-3" />
        Failed
      </Badge>
    );
  }
  return <Badge variant="muted">Pending</Badge>;
}

function ResultsSection({
  run,
  running,
  onOpenDetail,
  exportName,
}: {
  run: EvalRun | null;
  running: boolean;
  onOpenDetail: (result: EvalCaseResult) => void;
  exportName: (extension: string) => string;
}) {
  if (!run || run.results.length === 0) {
    return (
      <Card>
        <CardContent className="pt-5">
          <EmptyState
            icon={<ClipboardCheck className="h-7 w-7" />}
            title={running ? 'Running…' : 'No results yet'}
            description={
              running
                ? 'The first case is generating. Rows appear here as each one is scored.'
                : 'Run the suite to populate the summary, the per-case table and the Llama Stack export.'
            }
          />
        </CardContent>
      </Card>
    );
  }

  const { summary } = run;
  const csv = () => evalRunToCsv(run);

  return (
    <Card>
      <CardHeader className="flex-row flex-wrap items-start justify-between gap-3 space-y-0">
        <div className="min-w-0 space-y-1.5">
          <CardTitle className="flex flex-wrap items-center gap-2">
            Results
            <Badge variant="outline">{graderLabel(run.grader)}</Badge>
            <Badge variant="meta" className="font-mono">
              {run.modelId}
            </Badge>
            {running && (
              <Badge variant="warning">
                <Loader2 className="h-3 w-3 animate-spin" />
                Live
              </Badge>
            )}
          </CardTitle>
          <CardDescription>
            {run.results.length} scored {run.results.length === 1 ? 'case' : 'cases'} · backend{' '}
            <span className="font-mono">{run.backend}</span> · temperature{' '}
            <span className="tabular font-mono">{run.sampling.temperature}</span>
          </CardDescription>
        </div>
        <div className="flex flex-wrap items-center gap-2">
          <Button
            variant="outline"
            size="sm"
            onClick={() => downloadFile(exportName('csv'), csv(), 'text/csv')}
          >
            <Download />
            Export CSV
          </Button>
          <Button
            variant="outline"
            size="sm"
            onClick={() =>
              downloadFile(exportName('jsonl'), evalRunToJsonl(run), 'application/x-ndjson')
            }
          >
            <FileJson />
            Export JSONL
          </Button>
          <CopyButton value={csv} label="Copy CSV" showLabel variant="ghost" />
        </div>
      </CardHeader>

      <CardContent className="space-y-5">
        <StatGrid cols={5}>
          <Stat
            label={
              <InfoTip label="Accuracy">
                passed ÷ total, where a case passes when its score clears the grader&apos;s
                threshold (1.0 for the binary graders, 0.60 for token F1).
              </InfoTip>
            }
            value={formatPct(summary.accuracy * 100, 1)}
            sub={`${summary.passed} of ${summary.total} cases`}
            tone={summary.accuracy >= 0.8 ? 'good' : summary.accuracy >= 0.5 ? 'warn' : 'bad'}
          />
          <Stat
            label={
              <InfoTip label="Mean score">
                Arithmetic mean of the per-case score. Identical to accuracy for the binary
                graders; only token F1 puts anything between 0 and 1 in this column.
              </InfoTip>
            }
            value={summary.meanScore.toFixed(3)}
          />
          <Stat
            label={
              <InfoTip label="Mean latency">
                Mean wall-clock time of the answer generation, including prefill. The judge pass is
                not counted.
              </InfoTip>
            }
            value={formatDuration(summary.meanLatencyMs)}
          />
          <Stat
            label={
              <InfoTip label="Mean tok/s">
                Mean decode throughput reported by the engine, averaged over the cases that
                actually produced tokens.
              </InfoTip>
            }
            value={formatTps(summary.meanTokensPerSecond)}
          />
          <Stat label="Cases" value={formatNumber(summary.total)} sub="scored this run" />
        </StatGrid>

        <Table>
          <TableHeader>
            <TableRow>
              <TableHead className="w-10 text-right">#</TableHead>
              <TableHead>Question</TableHead>
              <TableHead>Expected</TableHead>
              <TableHead>Actual</TableHead>
              <TableHead className="text-right">Score</TableHead>
              <TableHead>Result</TableHead>
              <TableHead className="text-right">Latency</TableHead>
              <TableHead className="text-right">tok/s</TableHead>
              <TableHead className="w-10">
                <span className="sr-only">Details</span>
              </TableHead>
            </TableRow>
          </TableHeader>
          <TableBody>
            {run.results.map((result, index) => (
              <TableRow key={`${result.caseId}-${index}`}>
                <TableCell className="tabular text-right font-mono text-xs text-muted-foreground">
                  {index + 1}
                </TableCell>
                <TableCell>
                  <span className="block max-w-[16rem] truncate" title={result.question}>
                    {result.question}
                  </span>
                </TableCell>
                <TableCell>
                  <span
                    className="block max-w-[12rem] truncate font-mono text-xs"
                    title={result.expected}
                  >
                    {result.expected || '—'}
                  </span>
                </TableCell>
                <TableCell>
                  <span
                    className="block max-w-[18rem] truncate text-muted-foreground"
                    title={result.actual}
                  >
                    {result.actual.replace(/\s+/g, ' ').trim() || '—'}
                  </span>
                </TableCell>
                <TableCell className="tabular text-right font-mono">
                  {result.score.toFixed(2)}
                </TableCell>
                <TableCell>
                  {result.error ? (
                    <Badge variant="warning">
                      <AlertTriangle className="h-3 w-3" />
                      Error
                    </Badge>
                  ) : result.passed ? (
                    <Badge variant="success">
                      <Check className="h-3 w-3" />
                      Pass
                    </Badge>
                  ) : (
                    <Badge variant="danger">
                      <X className="h-3 w-3" />
                      Fail
                    </Badge>
                  )}
                </TableCell>
                <TableCell className="tabular text-right font-mono text-xs">
                  {formatDuration(result.latencyMs)}
                </TableCell>
                <TableCell className="tabular text-right font-mono text-xs">
                  {result.tokensPerSecond > 0 ? result.tokensPerSecond.toFixed(1) : '—'}
                </TableCell>
                <TableCell>
                  <Button
                    variant="ghost"
                    size="icon-sm"
                    aria-label={`Show the full output for case ${index + 1}`}
                    onClick={() => onOpenDetail(result)}
                  >
                    <Eye />
                  </Button>
                </TableCell>
              </TableRow>
            ))}
          </TableBody>
        </Table>

        <Footnote>
          The CSV columns match Llama Stack&apos;s scoring API — <InlineCode>input_query</InlineCode>
          , <InlineCode>expected_answer</InlineCode>, <InlineCode>generated_answer</InlineCode>,{' '}
          <InlineCode>score</InlineCode> and <InlineCode>scoring_function</InlineCode> — so the file
          can be replayed server-side without reshaping. This run&apos;s{' '}
          <span className="font-mono">{graderLabel(run.grader)}</span> grader is exported as{' '}
          <InlineCode>{graderToScoringFn(run.grader)}</InlineCode>, one of the scoring functions
          Llama Stack registers by default. The JSONL export carries the same rows with a
          reconstructed <InlineCode>chat_completion_input</InlineCode>, which is the dataset shape{' '}
          <InlineCode>/eval/run</InlineCode> takes.
        </Footnote>
      </CardContent>
    </Card>
  );
}

function DetailDialog({
  result,
  onClose,
}: {
  result: EvalCaseResult | null;
  onClose: () => void;
}) {
  return (
    <Dialog
      open={result !== null}
      onOpenChange={(open) => {
        if (!open) onClose();
      }}
    >
      <DialogContent className="max-w-2xl">
        {result && (
          <>
            <DialogHeader>
              <DialogTitle>Case detail</DialogTitle>
              <DialogDescription>
                Graded with {graderLabel(result.grader)} · score{' '}
                <span className="tabular font-mono">{result.score.toFixed(3)}</span> ·{' '}
                {result.passed ? 'passed' : 'failed'}
              </DialogDescription>
            </DialogHeader>

            <div className="space-y-4 text-sm">
              <DetailBlock label="Question">{result.question}</DetailBlock>
              <DetailBlock label="Expected" mono>
                {result.expected || '—'}
              </DetailBlock>
              <DetailBlock label="Model output" mono scroll>
                {result.actual || '—'}
              </DetailBlock>

              {result.rationale && (
                <DetailBlock label="Judge rationale">{result.rationale}</DetailBlock>
              )}

              {result.error && (
                <Alert variant="destructive">
                  <AlertTriangle />
                  <AlertTitle>Grading error</AlertTitle>
                  <AlertDescription>{result.error}</AlertDescription>
                </Alert>
              )}

              <div className="grid grid-cols-3 gap-3 border-t pt-3 text-xs">
                <div>
                  <div className="text-muted-foreground">Latency</div>
                  <div className="tabular font-mono">{formatDuration(result.latencyMs)}</div>
                </div>
                <div>
                  <div className="text-muted-foreground">Throughput</div>
                  <div className="tabular font-mono">{formatTps(result.tokensPerSecond)}</div>
                </div>
                <div>
                  <div className="text-muted-foreground">Completion tokens</div>
                  <div className="tabular font-mono">{formatNumber(result.completionTokens)}</div>
                </div>
              </div>
            </div>

            <DialogFooter>
              <CopyButton
                value={result.actual}
                label="Copy output"
                showLabel
                variant="outline"
                size="sm"
              />
              <DialogClose asChild>
                <Button size="sm">Close</Button>
              </DialogClose>
            </DialogFooter>
          </>
        )}
      </DialogContent>
    </Dialog>
  );
}

function DetailBlock({
  label,
  children,
  mono,
  scroll,
}: {
  label: string;
  children: React.ReactNode;
  mono?: boolean;
  scroll?: boolean;
}) {
  return (
    <div className="space-y-1">
      <div className="text-[11px] font-medium uppercase tracking-wide text-muted-foreground">
        {label}
      </div>
      <div
        className={cn(
          'whitespace-pre-wrap break-words rounded-lg border bg-muted/40 px-3 py-2 leading-relaxed',
          mono && 'font-mono text-xs',
          scroll && 'max-h-56 overflow-y-auto scrollbar-thin',
        )}
      >
        {children}
      </div>
    </div>
  );
}

function ImportDialog({
  onImport,
  disabled,
}: {
  onImport: (pairs: { question: string; expected: string }[], mode: 'replace' | 'append') => void;
  disabled?: boolean;
}) {
  const [open, setOpen] = React.useState(false);
  const [text, setText] = React.useState('');

  const pairs = React.useMemo(() => parseQaPairs(text), [text]);

  const apply = (mode: 'replace' | 'append') => {
    if (pairs.length === 0) return;
    onImport(pairs, mode);
    setText('');
    setOpen(false);
  };

  return (
    <Dialog
      open={open}
      onOpenChange={(next) => {
        setOpen(next);
        if (!next) setText('');
      }}
    >
      <DialogTrigger asChild>
        <Button variant="outline" size="sm" disabled={disabled}>
          <Upload />
          Import
        </Button>
      </DialogTrigger>
      <DialogContent className="max-w-2xl">
        <DialogHeader>
          <DialogTitle>Import Q&amp;A pairs</DialogTitle>
          <DialogDescription>
            Paste CSV, TSV, <InlineCode>Q:</InlineCode> / <InlineCode>A:</InlineCode> blocks, or a
            JSON array of objects. The first column is the question, the second is the reference
            answer; a leading <InlineCode>question,answer</InlineCode> header row is skipped.
          </DialogDescription>
        </DialogHeader>

        <Field label="Pasted pairs" htmlFor="eval-import">
          <Textarea
            id="eval-import"
            value={text}
            onChange={(e) => setText(e.target.value)}
            rows={10}
            className="font-mono text-xs"
            placeholder={
              'What is the capital of France?,Paris\nWhat is 2 + 2?,4\n\nor\n\nQ: Who wrote Romeo and Juliet?\nA: William Shakespeare'
            }
          />
        </Field>

        {pairs.length > 0 ? (
          <div className="space-y-2 rounded-lg border bg-muted/40 p-3">
            <div className="text-xs font-medium">
              <span className="tabular font-mono">{pairs.length}</span>{' '}
              {pairs.length === 1 ? 'pair' : 'pairs'} detected
            </div>
            <ul className="space-y-1 text-xs text-muted-foreground">
              {pairs.slice(0, 3).map((pair, i) => (
                <li key={i} className="truncate">
                  <span className="text-foreground">{pair.question}</span>
                  {' → '}
                  <span className="font-mono">{pair.expected || '(no answer)'}</span>
                </li>
              ))}
              {pairs.length > 3 && <li>…and {pairs.length - 3} more</li>}
            </ul>
          </div>
        ) : (
          <Footnote>
            {text.trim()
              ? 'Nothing parsed yet — check that each line has a question and an answer separated by a comma or a tab.'
              : 'A live count appears here as soon as the text parses.'}
          </Footnote>
        )}

        <DialogFooter>
          <Button variant="outline" size="sm" onClick={() => apply('append')} disabled={pairs.length === 0}>
            <Plus />
            Append {pairs.length || ''}
          </Button>
          <Button size="sm" onClick={() => apply('replace')} disabled={pairs.length === 0}>
            Replace all with {pairs.length || ''}
          </Button>
        </DialogFooter>
      </DialogContent>
    </Dialog>
  );
}
