'use client';

import { ArrowRight, Cpu, Gauge, Terminal } from 'lucide-react';
import { Button } from '@/components/ui/button';
import { useStudio } from '@/components/studio-provider';

const PROOF_POINTS = [
  {
    icon: Gauge,
    title: 'Will it fit?',
    body: 'Size every GGUF quant against your VRAM, with the KV cache and compute buffers counted — not just the weights.',
  },
  {
    icon: Cpu,
    title: 'What does it cost?',
    body: 'Throughput derived from memory bandwidth, priced against RunPod, Vast.ai, Lambda and owning the hardware.',
  },
  {
    icon: Terminal,
    title: 'How do I ship it?',
    body: 'Exact convert, quantize and serve commands for llama.cpp and Ollama. Then evaluate the result in this tab.',
  },
];

export function Hero() {
  const { goTo } = useStudio();

  return (
    <section className="grid-backdrop relative overflow-hidden border-b">
      <div className="pointer-events-none absolute inset-0 bg-gradient-to-b from-background/40 via-background/85 to-background" />
      <div className="container relative py-16 sm:py-20">
        <div className="max-w-3xl">
          <div className="inline-flex items-center gap-2 rounded-full border border-meta-200 bg-meta-50 px-3 py-1 text-xs font-medium text-meta-700">
            <span className="relative flex h-1.5 w-1.5">
              <span className="absolute inline-flex h-full w-full animate-ping rounded-full bg-meta-500 opacity-60" />
              <span className="relative inline-flex h-1.5 w-1.5 rounded-full bg-meta-500" />
            </span>
            Runs entirely in your browser
          </div>

          <h1 className="mt-5 text-balance text-4xl font-semibold leading-[1.08] tracking-tight sm:text-5xl">
            The three questions everyone asks before running{' '}
            <span className="text-meta-500">Llama</span> locally.
          </h1>

          <p className="mt-5 max-w-2xl text-pretty text-base leading-relaxed text-muted-foreground sm:text-lg">
            Which quantization fits my GPU, how much quality does it cost me, and what will
            inference actually run per month? Llama Local Lab answers all three from published
            architecture specs — then lets you load a real model into this tab and check the answer
            yourself.
          </p>

          <div className="mt-7 flex flex-wrap items-center gap-3">
            <Button size="lg" onClick={() => goTo('quant')}>
              Plan a quantization
              <ArrowRight />
            </Button>
            <Button size="lg" variant="outline" onClick={() => goTo('playground')}>
              Run Llama in this tab
            </Button>
          </div>

          <p className="mt-4 font-mono text-xs text-muted-foreground">
            No account. No API key. No request ever leaves your machine.
          </p>
        </div>

        <div className="mt-14 grid gap-px overflow-hidden rounded-xl border bg-border sm:grid-cols-3">
          {PROOF_POINTS.map(({ icon: Icon, title, body }) => (
            <div key={title} className="bg-background p-5">
              <Icon className="h-5 w-5 text-meta-500" />
              <div className="mt-3 text-sm font-semibold">{title}</div>
              <p className="mt-1 text-pretty text-sm leading-relaxed text-muted-foreground">
                {body}
              </p>
            </div>
          ))}
        </div>
      </div>
    </section>
  );
}
