import type { EvalRun } from '@/lib/types';

function escapeCell(value: unknown): string {
  const s = value === null || value === undefined ? '' : String(value);
  if (/[",\n\r]/.test(s)) {
    return `"${s.replace(/"/g, '""')}"`;
  }
  return s;
}

export function toCsv(rows: (string | number | boolean | null | undefined)[][]): string {
  return rows.map((row) => row.map(escapeCell).join(',')).join('\r\n');
}

/**
 * Export an evaluation run in the column layout Llama Stack's `/eval` scoring
 * API expects, so results can be diffed against a server-side run without
 * reshaping. Extra columns are ignored by the scorer.
 */
export function evalRunToCsv(run: EvalRun): string {
  const header = [
    'row_index',
    'input_query',
    'expected_answer',
    'generated_answer',
    'score',
    'passed',
    'scoring_function',
    'judge_rationale',
    'latency_ms',
    'tokens_per_second',
    'completion_tokens',
    'model_id',
    'backend',
    'temperature',
    'top_p',
    'max_tokens',
    'system_prompt',
    'run_id',
    'started_at',
    'error',
  ];

  const rows = run.results.map((r, i) => [
    i,
    r.question,
    r.expected,
    r.actual,
    r.score.toFixed(4),
    r.passed ? 'true' : 'false',
    graderToScoringFn(r.grader),
    r.rationale ?? '',
    Math.round(r.latencyMs),
    r.tokensPerSecond.toFixed(2),
    r.completionTokens,
    run.modelId,
    run.backend,
    run.sampling.temperature,
    run.sampling.topP,
    run.sampling.maxTokens,
    run.systemPrompt,
    run.id,
    run.startedAt,
    r.error ?? '',
  ]);

  return toCsv([header, ...rows]);
}

/** Map our grader names onto Llama Stack's registered scoring function ids. */
export function graderToScoringFn(grader: string): string {
  switch (grader) {
    case 'exact-match':
      return 'basic::equality';
    case 'contains':
      return 'basic::subset_of';
    case 'regex':
      return 'basic::regex_parser_multiple_choice_answer';
    case 'fuzzy':
      return 'basic::bleu';
    case 'llm-judge':
      return 'llm-as-judge::base';
    default:
      return grader;
  }
}

/** JSONL in Llama Stack's eval dataset shape, for re-running server-side. */
export function evalRunToJsonl(run: EvalRun): string {
  return run.results
    .map((r) =>
      JSON.stringify({
        input_query: r.question,
        expected_answer: r.expected,
        generated_answer: r.actual,
        chat_completion_input: JSON.stringify([
          ...(run.systemPrompt ? [{ role: 'system', content: run.systemPrompt }] : []),
          { role: 'user', content: r.question },
        ]),
        scoring_function: graderToScoringFn(r.grader),
        score: r.score,
      }),
    )
    .join('\n');
}

/** Parse pasted Q&A pairs. Accepts CSV, TSV, `Q: / A:` blocks and JSON arrays. */
export function parseQaPairs(text: string): { question: string; expected: string }[] {
  const trimmed = text.trim();
  if (!trimmed) return [];

  // JSON array of objects.
  if (trimmed.startsWith('[')) {
    try {
      const parsed = JSON.parse(trimmed) as Record<string, string>[];
      return parsed
        .map((row) => ({
          question: row.question ?? row.input ?? row.input_query ?? row.q ?? '',
          expected: row.expected ?? row.answer ?? row.expected_answer ?? row.a ?? '',
        }))
        .filter((r) => r.question);
    } catch {
      /* fall through to the line-based parsers */
    }
  }

  // `Q: ... / A: ...` blocks.
  if (/^\s*Q\s*:/im.test(trimmed)) {
    const pairs: { question: string; expected: string }[] = [];
    const blocks = trimmed.split(/\n(?=\s*Q\s*:)/i);
    for (const block of blocks) {
      const q = block.match(/Q\s*:\s*([\s\S]*?)(?=\n\s*A\s*:|$)/i)?.[1]?.trim();
      const a = block.match(/A\s*:\s*([\s\S]*)$/i)?.[1]?.trim();
      if (q) pairs.push({ question: q, expected: a ?? '' });
    }
    if (pairs.length) return pairs;
  }

  // Delimited: prefer tab, then a comma that is not inside quotes.
  const lines = trimmed.split(/\r?\n/).filter((l) => l.trim());
  const delimiter = lines[0]?.includes('\t') ? '\t' : ',';
  const rows = lines.map((line) => splitDelimited(line, delimiter));

  const start =
    rows[0] && /^(question|input|prompt|q)$/i.test(rows[0][0]?.trim() ?? '') ? 1 : 0;

  return rows
    .slice(start)
    .filter((cols) => cols[0]?.trim())
    .map((cols) => ({ question: cols[0].trim(), expected: (cols[1] ?? '').trim() }));
}

function splitDelimited(line: string, delimiter: string): string[] {
  const out: string[] = [];
  let cur = '';
  let inQuotes = false;
  for (let i = 0; i < line.length; i += 1) {
    const ch = line[i];
    if (ch === '"') {
      if (inQuotes && line[i + 1] === '"') {
        cur += '"';
        i += 1;
      } else {
        inQuotes = !inQuotes;
      }
    } else if (ch === delimiter && !inQuotes) {
      out.push(cur);
      cur = '';
    } else {
      cur += ch;
    }
  }
  out.push(cur);
  return out;
}
