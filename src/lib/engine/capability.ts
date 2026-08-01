'use client';

import type { GpuCapability } from './types';

/**
 * Probe what this browser can actually run.
 *
 * `navigator.gpu` existing is not enough — Chrome exposes it on machines where
 * `requestAdapter()` returns null (no compatible device, or the GPU is
 * blocklisted), and Safari 17 exposes it behind a flag that fails at device
 * creation. We request an adapter for real so the UI never promises inference
 * it cannot deliver.
 */
export async function detectCapability(): Promise<GpuCapability> {
  const base: GpuCapability = {
    webgpu: false,
    wasm: typeof WebAssembly !== 'undefined',
    crossOriginIsolated:
      typeof globalThis !== 'undefined' && Boolean((globalThis as { crossOriginIsolated?: boolean }).crossOriginIsolated),
    hardwareConcurrency:
      typeof navigator !== 'undefined' ? navigator.hardwareConcurrency || 4 : 4,
  };

  const gpu = (navigator as Navigator & { gpu?: GPU }).gpu;
  if (!gpu) {
    return {
      ...base,
      reason:
        'This browser does not expose WebGPU. Chrome 113+, Edge 113+, or Safari 18+ on macOS/iOS 18 are required.',
    };
  }

  try {
    const adapter = await gpu.requestAdapter({ powerPreference: 'high-performance' });
    if (!adapter) {
      return {
        ...base,
        reason:
          'WebGPU is present but no compatible adapter was returned. The GPU may be blocklisted or disabled — check chrome://gpu.',
      };
    }

    // `requestAdapterInfo` is not in every browser and is being replaced by
    // `adapter.info`; try both and do not fail if neither exists.
    let description: string | undefined;
    const info = (adapter as GPUAdapter & { info?: GPUAdapterInfo }).info;
    if (info) {
      description = [info.vendor, info.architecture, info.description]
        .filter(Boolean)
        .join(' · ');
    } else if ('requestAdapterInfo' in adapter) {
      try {
        const legacy = await (
          adapter as GPUAdapter & { requestAdapterInfo: () => Promise<GPUAdapterInfo> }
        ).requestAdapterInfo();
        description = [legacy.vendor, legacy.architecture, legacy.description]
          .filter(Boolean)
          .join(' · ');
      } catch {
        /* optional */
      }
    }

    return {
      ...base,
      webgpu: true,
      adapter: description && description.trim() ? description : 'WebGPU adapter',
      maxBufferSize: adapter.limits?.maxBufferSize,
      maxStorageBufferBindingSize: adapter.limits?.maxStorageBufferBindingSize,
    };
  } catch (err) {
    return {
      ...base,
      reason: `WebGPU adapter request failed: ${err instanceof Error ? err.message : String(err)}`,
    };
  }
}

/**
 * WebLLM allocates the largest weight shard as a single buffer, so a model
 * whose VRAM requirement exceeds the adapter's `maxBufferSize` will fail at
 * load time no matter how much memory the GPU has.
 */
export function canFitModel(capability: GpuCapability, vramMb: number): { ok: boolean; note?: string } {
  if (!capability.webgpu) return { ok: false, note: capability.reason };
  const limit = capability.maxStorageBufferBindingSize ?? capability.maxBufferSize;
  if (!limit) return { ok: true };
  const requiredBytes = vramMb * 1024 * 1024;
  // WebLLM shards weights, so the hard limit applies per-shard rather than to
  // the whole model. Warn well before the model total crosses ~4× the limit.
  if (requiredBytes > limit * 8) {
    return {
      ok: false,
      note: `Needs ~${Math.round(vramMb)} MB but this adapter caps buffers at ${Math.round(limit / 1024 / 1024)} MB.`,
    };
  }
  return { ok: true };
}
