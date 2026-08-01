'use client';

import * as React from 'react';
import { CopyButton } from '@/components/copy-button';
import { cn } from '@/lib/utils';

type Language = 'bash' | 'powershell' | 'python' | 'json' | 'dockerfile' | 'text';

interface CodeBlockProps {
  code: string;
  language?: Language;
  /** Filename or short caption shown in the block's title bar. */
  title?: string;
  className?: string;
  /** Show 1-based line numbers in a gutter. */
  showLineNumbers?: boolean;
  /** Cap the visible height and scroll inside the block. */
  maxHeight?: number;
}

/**
 * A syntax-highlighted code block with a copy button.
 *
 * The highlighter is deliberately small and hand-rolled: a full grammar-based
 * tokenizer would add hundreds of kilobytes to a static bundle for four
 * languages of shell-flavored output. It tokenizes in one pass so nothing can
 * be highlighted twice, and it escapes before it decorates.
 */
export function CodeBlock({
  code,
  language = 'bash',
  title,
  className,
  showLineNumbers = false,
  maxHeight,
}: CodeBlockProps) {
  const html = React.useMemo(() => highlight(code, language), [code, language]);
  const lines = React.useMemo(() => code.split('\n').length, [code]);

  return (
    <div
      className={cn(
        'group relative overflow-hidden rounded-lg border bg-[#0B1220] text-[13px] shadow-sm',
        className,
      )}
    >
      <div className="flex items-center justify-between gap-2 border-b border-white/10 bg-white/[0.03] px-3 py-1.5">
        <div className="flex min-w-0 items-center gap-2">
          <span className="font-mono text-[11px] uppercase tracking-wider text-slate-500">
            {language}
          </span>
          {title && (
            <span className="truncate font-mono text-[11px] text-slate-400">{title}</span>
          )}
        </div>
        <CopyButton
          value={code}
          className="h-6 w-6 text-slate-400 hover:bg-white/10 hover:text-slate-100"
        />
      </div>
      <div
        className="overflow-auto scrollbar-thin"
        style={maxHeight ? { maxHeight } : undefined}
      >
        <pre className="flex min-w-full font-mono leading-relaxed">
          {showLineNumbers && (
            <code
              aria-hidden
              className="select-none border-r border-white/5 px-3 py-3 text-right text-slate-600"
            >
              {Array.from({ length: lines }, (_, i) => (
                <span key={i} className="block">
                  {i + 1}
                </span>
              ))}
            </code>
          )}
          <code
            className="block flex-1 whitespace-pre px-4 py-3 text-slate-200"
            dangerouslySetInnerHTML={{ __html: html }}
          />
        </pre>
      </div>
    </div>
  );
}

/* -------------------------------------------------------------------------- */
/*  Minimal highlighter                                                        */
/* -------------------------------------------------------------------------- */

const ESCAPES: Record<string, string> = {
  '&': '&amp;',
  '<': '&lt;',
  '>': '&gt;',
  '"': '&quot;',
  "'": '&#39;',
};

function escapeHtml(s: string): string {
  return s.replace(/[&<>"']/g, (c) => ESCAPES[c]);
}

const SHELL_KEYWORDS = [
  'if', 'then', 'else', 'elif', 'fi', 'for', 'while', 'do', 'done', 'case', 'esac',
  'function', 'return', 'export', 'source', 'cd', 'set', 'local', 'in',
];

const PY_KEYWORDS = [
  'import', 'from', 'as', 'def', 'class', 'return', 'if', 'elif', 'else', 'for',
  'while', 'with', 'try', 'except', 'finally', 'raise', 'pass', 'None', 'True',
  'False', 'and', 'or', 'not', 'in', 'is', 'lambda', 'yield', 'await', 'async',
];

const COLORS = {
  comment: 'text-slate-500 italic',
  string: 'text-emerald-300',
  keyword: 'text-violet-300',
  flag: 'text-sky-300',
  number: 'text-amber-300',
  command: 'text-meta-300 font-medium',
  punctuation: 'text-slate-400',
  key: 'text-sky-300',
} as const;

function wrap(kind: keyof typeof COLORS, text: string) {
  return `<span class="${COLORS[kind]}">${escapeHtml(text)}</span>`;
}

/**
 * One regex with named alternatives, scanned left to right. Because every match
 * consumes its span, a URL inside a string can never be re-tokenized as a flag.
 */
function highlight(code: string, language: Language): string {
  if (language === 'text') return escapeHtml(code);

  if (language === 'json') return highlightJson(code);

  const isPython = language === 'python';
  const keywords = isPython ? PY_KEYWORDS : SHELL_KEYWORDS;
  const commentMarker = language === 'powershell' ? '#' : '#';

  const pattern = new RegExp(
    [
      `(?<comment>${escapeRegex(commentMarker)}[^\\n]*)`,
      `(?<string>"(?:[^"\\\\\\n]|\\\\.)*"|'(?:[^'\\\\\\n]|\\\\.)*')`,
      `(?<flag>(?<=\\s)--?[A-Za-z][\\w-]*)`,
      `(?<number>\\b\\d+(?:\\.\\d+)?\\b)`,
      `(?<word>[A-Za-z_][\\w.-]*)`,
    ].join('|'),
    'g',
  );

  let out = '';
  let last = 0;
  let atLineStart = true;

  for (const match of code.matchAll(pattern)) {
    const index = match.index ?? 0;
    const plain = code.slice(last, index);
    out += escapeHtml(plain);
    if (plain.includes('\n')) atLineStart = /\n\s*$/.test(plain);
    else if (plain.trim()) atLineStart = false;

    const g = match.groups!;
    if (g.comment) out += wrap('comment', g.comment);
    else if (g.string) out += wrap('string', g.string);
    else if (g.flag) out += wrap('flag', g.flag);
    else if (g.number) out += wrap('number', g.number);
    else if (g.word) {
      if (keywords.includes(g.word)) out += wrap('keyword', g.word);
      else if (atLineStart && !isPython) out += wrap('command', g.word);
      else out += escapeHtml(g.word);
      atLineStart = false;
    }

    last = index + match[0].length;
  }

  out += escapeHtml(code.slice(last));
  return out;
}

function highlightJson(code: string): string {
  const pattern =
    /(?<key>"(?:[^"\\]|\\.)*")(?=\s*:)|(?<string>"(?:[^"\\]|\\.)*")|(?<number>-?\b\d+(?:\.\d+)?(?:[eE][+-]?\d+)?\b)|(?<keyword>\btrue\b|\bfalse\b|\bnull\b)|(?<punctuation>[{}[\],:])/g;

  let out = '';
  let last = 0;
  for (const match of code.matchAll(pattern)) {
    const index = match.index ?? 0;
    out += escapeHtml(code.slice(last, index));
    const g = match.groups!;
    if (g.key) out += wrap('key', g.key);
    else if (g.string) out += wrap('string', g.string);
    else if (g.number) out += wrap('number', g.number);
    else if (g.keyword) out += wrap('keyword', g.keyword);
    else if (g.punctuation) out += wrap('punctuation', g.punctuation);
    last = index + match[0].length;
  }
  out += escapeHtml(code.slice(last));
  return out;
}

function escapeRegex(s: string) {
  return s.replace(/[.*+?^${}()|[\]\\]/g, '\\$&');
}

/** Inline code for use inside prose. */
export function InlineCode({ children, className }: { children: React.ReactNode; className?: string }) {
  return (
    <code
      className={cn(
        'rounded border bg-muted px-1.5 py-0.5 font-mono text-[0.85em] text-foreground',
        className,
      )}
    >
      {children}
    </code>
  );
}
