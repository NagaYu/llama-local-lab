'use client';

import { useCallback, useEffect, useRef, useState } from 'react';

const PREFIX = 'llama-local-lab:';

/**
 * localStorage-backed state.
 *
 * The first render always returns `initial` so server-rendered markup and the
 * client's first paint agree; the stored value is applied in an effect right
 * after. Without that split, static export produces a hydration mismatch on
 * every persisted control in the app.
 */
export function usePersistentState<T>(
  key: string,
  initial: T,
  options?: { revive?: (raw: unknown) => T },
): [T, (value: T | ((prev: T) => T)) => void, { hydrated: boolean; reset: () => void }] {
  const [value, setValue] = useState<T>(initial);
  const [hydrated, setHydrated] = useState(false);
  const initialRef = useRef(initial);
  const reviveRef = useRef(options?.revive);
  reviveRef.current = options?.revive;

  useEffect(() => {
    try {
      const raw = window.localStorage.getItem(PREFIX + key);
      if (raw !== null) {
        const parsed = JSON.parse(raw) as unknown;
        setValue(reviveRef.current ? reviveRef.current(parsed) : (parsed as T));
      }
    } catch {
      // Corrupt or unreadable entry — fall back to the default silently.
    }
    setHydrated(true);
  }, [key]);

  const update = useCallback(
    (next: T | ((prev: T) => T)) => {
      setValue((prev) => {
        const resolved =
          typeof next === 'function' ? (next as (p: T) => T)(prev) : next;
        try {
          window.localStorage.setItem(PREFIX + key, JSON.stringify(resolved));
        } catch {
          // Quota exceeded or private mode — keep working in memory.
        }
        return resolved;
      });
    },
    [key],
  );

  const reset = useCallback(() => {
    try {
      window.localStorage.removeItem(PREFIX + key);
    } catch {
      /* ignore */
    }
    setValue(initialRef.current);
  }, [key]);

  return [value, update, { hydrated, reset }];
}

/**
 * Merge a persisted object over defaults so adding a new field to a config type
 * does not break users who already have state in localStorage.
 */
export function mergeDefaults<T extends object>(defaults: T) {
  return (raw: unknown): T => {
    if (!raw || typeof raw !== 'object') return defaults;
    const merged = { ...defaults } as Record<string, unknown>;
    for (const [k, v] of Object.entries(raw as Record<string, unknown>)) {
      if (v === undefined) continue;
      const base = (defaults as Record<string, unknown>)[k];
      if (base && typeof base === 'object' && !Array.isArray(base) && typeof v === 'object' && v !== null && !Array.isArray(v)) {
        merged[k] = { ...(base as object), ...(v as object) };
      } else {
        merged[k] = v;
      }
    }
    return merged as T;
  };
}

/** True once the component has mounted on the client. */
export function useMounted() {
  const [mounted, setMounted] = useState(false);
  useEffect(() => setMounted(true), []);
  return mounted;
}

/** Clear every key this app owns. Used by the "reset studio" control. */
export function clearAllState() {
  try {
    const keys: string[] = [];
    for (let i = 0; i < window.localStorage.length; i += 1) {
      const k = window.localStorage.key(i);
      if (k?.startsWith(PREFIX)) keys.push(k);
    }
    keys.forEach((k) => window.localStorage.removeItem(k));
  } catch {
    /* ignore */
  }
}

export const STORAGE_KEYS = {
  planner: 'planner',
  cost: 'cost',
  gguf: 'gguf',
  playground: 'playground',
  evalCases: 'eval-cases',
  evalRuns: 'eval-runs',
  priceOverrides: 'price-overrides',
  throughputOverrides: 'throughput-overrides',
  hfCache: 'hf-cache',
  activeTab: 'active-tab',
} as const;
