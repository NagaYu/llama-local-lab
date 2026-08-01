'use client';

import * as React from 'react';
import { cn } from '@/lib/utils';
import { Label } from '@/components/ui/label';
import { InfoTip } from '@/components/ui/tooltip';

/**
 * Layout and typography atoms shared by every feature panel, so the six tools
 * read as one product rather than six.
 */

export function SectionHeading({
  eyebrow,
  title,
  description,
  actions,
  className,
}: {
  eyebrow?: string;
  title: React.ReactNode;
  description?: React.ReactNode;
  actions?: React.ReactNode;
  className?: string;
}) {
  return (
    <div className={cn('flex flex-wrap items-end justify-between gap-4', className)}>
      <div className="min-w-0 max-w-3xl space-y-1.5">
        {eyebrow && (
          <div className="font-mono text-[11px] font-semibold uppercase tracking-[0.14em] text-meta-600">
            {eyebrow}
          </div>
        )}
        <h2 className="text-balance text-xl font-semibold tracking-tight sm:text-2xl">{title}</h2>
        {description && (
          <p className="text-pretty text-sm leading-relaxed text-muted-foreground">{description}</p>
        )}
      </div>
      {actions && <div className="flex shrink-0 flex-wrap items-center gap-2">{actions}</div>}
    </div>
  );
}

/** A labelled form row with optional help text and an inline unit suffix. */
export function Field({
  label,
  hint,
  help,
  htmlFor,
  children,
  className,
}: {
  label: React.ReactNode;
  hint?: React.ReactNode;
  help?: React.ReactNode;
  htmlFor?: string;
  children: React.ReactNode;
  className?: string;
}) {
  return (
    <div className={cn('space-y-1.5', className)}>
      <div className="flex items-baseline justify-between gap-2">
        <Label htmlFor={htmlFor} className="text-xs font-medium text-foreground">
          {help ? <InfoTip label={label}>{help}</InfoTip> : label}
        </Label>
        {hint && <span className="tabular font-mono text-[11px] text-muted-foreground">{hint}</span>}
      </div>
      {children}
    </div>
  );
}

/** A single headline metric. */
export function Stat({
  label,
  value,
  sub,
  tone = 'default',
  className,
}: {
  label: React.ReactNode;
  value: React.ReactNode;
  sub?: React.ReactNode;
  tone?: 'default' | 'good' | 'warn' | 'bad' | 'accent';
  className?: string;
}) {
  const toneClass = {
    default: 'text-foreground',
    accent: 'text-meta-600',
    good: 'text-success',
    warn: 'text-warning',
    bad: 'text-danger',
  }[tone];

  return (
    <div className={cn('rounded-lg border bg-card p-3.5', className)}>
      <div className="text-[11px] font-medium uppercase tracking-wide text-muted-foreground">
        {label}
      </div>
      <div className={cn('tabular mt-1 text-xl font-semibold tracking-tight', toneClass)}>
        {value}
      </div>
      {sub && <div className="mt-0.5 text-xs text-muted-foreground">{sub}</div>}
    </div>
  );
}

export function StatGrid({
  children,
  className,
  cols = 4,
}: {
  children: React.ReactNode;
  className?: string;
  cols?: 2 | 3 | 4 | 5;
}) {
  const colClass = {
    2: 'sm:grid-cols-2',
    3: 'sm:grid-cols-2 lg:grid-cols-3',
    4: 'sm:grid-cols-2 lg:grid-cols-4',
    5: 'sm:grid-cols-2 lg:grid-cols-5',
  }[cols];
  return <div className={cn('grid grid-cols-1 gap-3', colClass, className)}>{children}</div>;
}

/** Horizontal capacity bar used for VRAM and memory-budget readouts. */
export function MeterBar({
  segments,
  capacity,
  className,
}: {
  segments: { label: string; value: number; className: string }[];
  capacity: number;
  className?: string;
}) {
  const total = segments.reduce((n, s) => n + s.value, 0);
  const scale = Math.max(capacity, total);

  return (
    <div className={cn('space-y-1.5', className)}>
      <div className="relative flex h-3 w-full overflow-hidden rounded-full bg-muted">
        {segments.map((s) => (
          <div
            key={s.label}
            className={cn('h-full', s.className)}
            style={{ width: `${Math.max(0, (s.value / scale) * 100)}%` }}
            title={`${s.label}`}
          />
        ))}
        {capacity < total && (
          <div
            className="absolute top-0 h-full w-0.5 bg-danger"
            style={{ left: `${(capacity / scale) * 100}%` }}
            title="VRAM limit"
          />
        )}
      </div>
      <div className="flex flex-wrap gap-x-4 gap-y-1">
        {segments.map((s) => (
          <span key={s.label} className="flex items-center gap-1.5 text-[11px] text-muted-foreground">
            <span className={cn('h-2 w-2 rounded-sm', s.className)} />
            {s.label}
          </span>
        ))}
      </div>
    </div>
  );
}

/** Small caption for provenance notes under tables. */
export function Footnote({ children, className }: { children: React.ReactNode; className?: string }) {
  return (
    <p className={cn('text-pretty text-xs leading-relaxed text-muted-foreground', className)}>
      {children}
    </p>
  );
}

/** Empty state with an icon slot. */
export function EmptyState({
  icon,
  title,
  description,
  action,
  className,
}: {
  icon?: React.ReactNode;
  title: React.ReactNode;
  description?: React.ReactNode;
  action?: React.ReactNode;
  className?: string;
}) {
  return (
    <div
      className={cn(
        'flex flex-col items-center justify-center gap-2 rounded-lg border border-dashed px-6 py-12 text-center',
        className,
      )}
    >
      {icon && <div className="text-muted-foreground/60">{icon}</div>}
      <div className="font-medium">{title}</div>
      {description && (
        <div className="max-w-md text-pretty text-sm text-muted-foreground">{description}</div>
      )}
      {action && <div className="mt-2">{action}</div>}
    </div>
  );
}

/** A numeric input that keeps text state so the user can clear and retype. */
export function NumberInput({
  value,
  onChange,
  min,
  max,
  step = 1,
  suffix,
  className,
  id,
  ...rest
}: {
  value: number;
  onChange: (n: number) => void;
  min?: number;
  max?: number;
  step?: number;
  suffix?: React.ReactNode;
  className?: string;
  id?: string;
} & Omit<React.InputHTMLAttributes<HTMLInputElement>, 'value' | 'onChange' | 'min' | 'max' | 'step'>) {
  const [text, setText] = React.useState(String(value));
  const [focused, setFocused] = React.useState(false);

  React.useEffect(() => {
    if (!focused) setText(String(value));
  }, [value, focused]);

  const commit = (raw: string) => {
    const n = Number(raw);
    if (raw.trim() === '' || Number.isNaN(n)) {
      setText(String(value));
      return;
    }
    let next = n;
    if (min !== undefined) next = Math.max(min, next);
    if (max !== undefined) next = Math.min(max, next);
    onChange(next);
    setText(String(next));
  };

  return (
    <div className={cn('relative', className)}>
      <input
        id={id}
        type="number"
        inputMode="decimal"
        value={text}
        min={min}
        max={max}
        step={step}
        onFocus={() => setFocused(true)}
        onChange={(e) => {
          setText(e.target.value);
          const n = Number(e.target.value);
          if (e.target.value.trim() !== '' && !Number.isNaN(n)) onChange(n);
        }}
        onBlur={(e) => {
          setFocused(false);
          commit(e.target.value);
        }}
        className={cn(
          'tabular flex h-9 w-full rounded-md border border-input bg-background px-3 py-1 text-sm shadow-sm',
          'focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring focus-visible:ring-offset-1 ring-offset-background',
          '[appearance:textfield] [&::-webkit-inner-spin-button]:appearance-none [&::-webkit-outer-spin-button]:appearance-none',
          suffix && 'pr-12',
        )}
        {...rest}
      />
      {suffix && (
        <span className="pointer-events-none absolute right-3 top-1/2 -translate-y-1/2 font-mono text-[11px] text-muted-foreground">
          {suffix}
        </span>
      )}
    </div>
  );
}
