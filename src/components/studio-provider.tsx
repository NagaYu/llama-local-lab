'use client';

import * as React from 'react';
import { DEFAULT_MODEL_ID } from '@/data/models';
import { DEFAULT_QUANT_ID } from '@/data/quants';
import { DEFAULT_GPU_ID } from '@/data/gpus';
import { usePersistentState, STORAGE_KEYS } from '@/lib/storage';

export type StudioTab = 'models' | 'quant' | 'gguf' | 'cost' | 'playground' | 'eval';

interface StudioContextValue {
  /** The model every panel is currently reasoning about. */
  modelId: string;
  setModelId: (id: string) => void;
  quantId: string;
  setQuantId: (id: string) => void;
  gpuId: string;
  setGpuId: (id: string) => void;
  tab: StudioTab;
  /** Switch tabs, optionally carrying a model/quant selection across. */
  goTo: (tab: StudioTab, opts?: { modelId?: string; quantId?: string; gpuId?: string }) => void;
  hydrated: boolean;
}

const StudioContext = React.createContext<StudioContextValue | null>(null);

const TAB_IDS: StudioTab[] = ['models', 'quant', 'gguf', 'cost', 'playground', 'eval'];

function isTab(value: unknown): value is StudioTab {
  return typeof value === 'string' && (TAB_IDS as string[]).includes(value);
}

/**
 * Cross-panel selection state.
 *
 * The six tools are one workflow: pick a model, plan a quant, generate the
 * commands, price it, then actually try it. Threading the selection through a
 * context means "Plan quantization" on a model card lands on the planner with
 * that model already chosen, instead of making the user re-pick it five times.
 */
export function StudioProvider({ children }: { children: React.ReactNode }) {
  const [modelId, setModelId] = usePersistentState<string>('selected-model', DEFAULT_MODEL_ID);
  const [quantId, setQuantId] = usePersistentState<string>('selected-quant', DEFAULT_QUANT_ID);
  const [gpuId, setGpuId] = usePersistentState<string>('selected-gpu', DEFAULT_GPU_ID);
  const [tab, setTab, { hydrated }] = usePersistentState<StudioTab>(
    STORAGE_KEYS.activeTab,
    'models',
    { revive: (raw) => (isTab(raw) ? raw : 'models') },
  );

  const goTo = React.useCallback(
    (next: StudioTab, opts?: { modelId?: string; quantId?: string; gpuId?: string }) => {
      if (opts?.modelId) setModelId(opts.modelId);
      if (opts?.quantId) setQuantId(opts.quantId);
      if (opts?.gpuId) setGpuId(opts.gpuId);
      setTab(next);
      // The tab strip is sticky, so scroll the panel back into view rather than
      // dropping the user halfway down a long table.
      if (typeof window !== 'undefined') {
        window.requestAnimationFrame(() => {
          document.getElementById('studio')?.scrollIntoView({ behavior: 'smooth', block: 'start' });
        });
      }
    },
    [setModelId, setQuantId, setGpuId, setTab],
  );

  const value = React.useMemo(
    () => ({ modelId, setModelId, quantId, setQuantId, gpuId, setGpuId, tab, goTo, hydrated }),
    [modelId, setModelId, quantId, setQuantId, gpuId, setGpuId, tab, goTo, hydrated],
  );

  return <StudioContext.Provider value={value}>{children}</StudioContext.Provider>;
}

export function useStudio(): StudioContextValue {
  const ctx = React.useContext(StudioContext);
  if (!ctx) throw new Error('useStudio must be used inside <StudioProvider>.');
  return ctx;
}
