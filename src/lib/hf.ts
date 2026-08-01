'use client';

import { MODELS } from '@/data/models';
import type { HubStats } from '@/lib/types';

const HF_API = 'https://huggingface.co/api/models';
const CACHE_KEY = 'llama-local-lab:hf-cache';
const CACHE_TTL_MS = 6 * 60 * 60 * 1000; // 6 hours

interface CacheEnvelope {
  fetchedAt: number;
  entries: Record<string, HubStats>;
}

function readCache(): CacheEnvelope | null {
  try {
    const raw = window.localStorage.getItem(CACHE_KEY);
    if (!raw) return null;
    const parsed = JSON.parse(raw) as CacheEnvelope;
    if (!parsed?.entries || typeof parsed.fetchedAt !== 'number') return null;
    return parsed;
  } catch {
    return null;
  }
}

function writeCache(envelope: CacheEnvelope) {
  try {
    window.localStorage.setItem(CACHE_KEY, JSON.stringify(envelope));
  } catch {
    /* quota — the app works fine without the cache */
  }
}

interface HfApiModel {
  id?: string;
  downloads?: number;
  likes?: number;
  lastModified?: string;
  pipeline_tag?: string;
  tags?: string[];
  gated?: boolean | string;
}

/**
 * Fetch live download/like counts for the bundled catalog straight from the
 * browser. This is the only network call the studio makes, it needs no API key,
 * and every failure mode degrades to the bundled numbers rather than an error.
 */
export async function fetchHubStats(signal?: AbortSignal): Promise<Record<string, HubStats>> {
  const cached = readCache();
  if (cached && Date.now() - cached.fetchedAt < CACHE_TTL_MS) {
    return cached.entries;
  }

  const entries: Record<string, HubStats> = {};
  const fetchedAt = new Date().toISOString();

  // One request per repo, run concurrently. The HF search endpoint would be a
  // single call, but it does not reliably return gated Meta repos by exact id.
  const results = await Promise.allSettled(
    MODELS.map(async (model) => {
      const url = `${HF_API}/${encodeURI(model.repoId)}`;
      const res = await fetch(url, { signal, headers: { Accept: 'application/json' } });
      if (!res.ok) throw new Error(`${res.status}`);
      const json = (await res.json()) as HfApiModel;
      return { id: model.id, json };
    }),
  );

  for (const result of results) {
    if (result.status !== 'fulfilled') continue;
    const { id, json } = result.value;
    entries[id] = {
      downloads: json.downloads,
      likes: json.likes,
      lastModified: json.lastModified,
      pipelineTag: json.pipeline_tag,
      tags: json.tags?.slice(0, 12),
      source: 'huggingface',
      fetchedAt,
    };
  }

  if (Object.keys(entries).length > 0) {
    writeCache({ fetchedAt: Date.now(), entries });
  }

  return entries;
}

export function hubUrl(repoId: string) {
  return `https://huggingface.co/${repoId}`;
}

export function ggufUrl(repoId: string) {
  return `https://huggingface.co/${repoId}/tree/main`;
}

/** Drop the cached response so the next load refetches. */
export function invalidateHubCache() {
  try {
    window.localStorage.removeItem(CACHE_KEY);
  } catch {
    /* ignore */
  }
}
