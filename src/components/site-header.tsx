'use client';

import * as React from 'react';
import { Github, RotateCcw, ShieldCheck } from 'lucide-react';
import { Badge } from '@/components/ui/badge';
import { Button } from '@/components/ui/button';
import {
  Dialog,
  DialogContent,
  DialogDescription,
  DialogFooter,
  DialogHeader,
  DialogTitle,
  DialogTrigger,
  DialogClose,
} from '@/components/ui/dialog';
import { clearAllState } from '@/lib/storage';

/** Meta's infinity mark, drawn inline so the app ships with zero image assets. */
function LlamaMark({ className }: { className?: string }) {
  return (
    <svg viewBox="0 0 36 24" className={className} aria-hidden focusable="false">
      <path
        d="M6.4 3C2.9 3 .8 6.4.8 11.3c0 4.9 2.1 8.2 5.4 8.2 2.6 0 4.4-1.7 6.6-5.2l1.9-3.1c.6-1 1.2-2 1.8-2.9.7 1 1.4 2.1 2 3.2l1.7 2.8c2.2 3.6 4 5.2 6.6 5.2 3.4 0 5.4-3.4 5.4-8.4C32.2 6.3 30.1 3 26.6 3c-2.5 0-4.5 1.7-6.7 5.2l-1.5 2.4-1.4-2.3C14.8 4.7 12.8 3 10.3 3H6.4Zm3.5 3.4c1.2 0 2.4 1 4 3.6l.8 1.3-1.4 2.3c-1.7 2.8-2.7 3.6-3.9 3.6-1.6 0-2.7-1.9-2.7-5.3 0-3.6 1.2-5.5 3.2-5.5Zm16.4 0c2 0 3.2 1.9 3.2 5.4 0 3.5-1.1 5.4-2.7 5.4-1.2 0-2.2-.8-3.9-3.6l-1.5-2.4.9-1.4c1.6-2.5 2.8-3.4 4-3.4Z"
        fill="currentColor"
      />
    </svg>
  );
}

export function SiteHeader() {
  const [resetOpen, setResetOpen] = React.useState(false);

  const onReset = () => {
    clearAllState();
    setResetOpen(false);
    window.location.reload();
  };

  return (
    <header className="sticky top-0 z-40 w-full border-b bg-background/85 backdrop-blur supports-[backdrop-filter]:bg-background/70">
      <div className="container flex h-14 items-center justify-between gap-4">
        <a href="#studio" className="flex min-w-0 items-center gap-2.5">
          <LlamaMark className="h-5 w-auto shrink-0 text-meta-500" />
          <span className="truncate text-[15px] font-semibold tracking-tight">
            Llama Local Lab
          </span>
          <Badge variant="muted" className="hidden shrink-0 font-mono text-[10px] sm:inline-flex">
            community build
          </Badge>
        </a>

        <div className="flex items-center gap-1.5">
          <span className="hidden items-center gap-1.5 rounded-full border border-success/20 bg-success-soft px-2.5 py-1 text-[11px] font-medium text-success md:inline-flex">
            <ShieldCheck className="h-3.5 w-3.5" />
            No server · no API key
          </span>

          <Dialog open={resetOpen} onOpenChange={setResetOpen}>
            <DialogTrigger asChild>
              <Button variant="ghost" size="icon-sm" aria-label="Reset saved state">
                <RotateCcw />
              </Button>
            </DialogTrigger>
            <DialogContent>
              <DialogHeader>
                <DialogTitle>Reset the studio?</DialogTitle>
                <DialogDescription>
                  This clears every setting this app has saved in your browser: planner inputs,
                  price overrides, playground config, evaluation cases and past runs. Downloaded
                  model weights stay in your browser cache and are not touched.
                </DialogDescription>
              </DialogHeader>
              <DialogFooter>
                <DialogClose asChild>
                  <Button variant="outline">Cancel</Button>
                </DialogClose>
                <Button variant="destructive" onClick={onReset}>
                  Clear saved state
                </Button>
              </DialogFooter>
            </DialogContent>
          </Dialog>

          <Button variant="ghost" size="icon-sm" asChild aria-label="llama.cpp on GitHub">
            <a href="https://github.com/ggml-org/llama.cpp" target="_blank" rel="noreferrer noopener">
              <Github />
            </a>
          </Button>
        </div>
      </div>
    </header>
  );
}

export function SiteFooter() {
  return (
    <footer className="mt-20 border-t bg-muted/30">
      <div className="container flex flex-col gap-6 py-10 md:flex-row md:items-start md:justify-between">
        <div className="max-w-lg space-y-2">
          <div className="flex items-center gap-2">
            <LlamaMark className="h-4 w-auto text-meta-500" />
            <span className="text-sm font-semibold">Llama Local Lab</span>
          </div>
          <p className="text-pretty text-xs leading-relaxed text-muted-foreground">
            An independent, community-built tool. Not affiliated with, endorsed by, or sponsored by
            Meta. Llama and the Llama mark are properties of Meta Platforms, Inc. Model weights are
            licensed under the relevant Llama Community License — read it before you deploy.
          </p>
        </div>
        <div className="grid grid-cols-2 gap-x-10 gap-y-2 text-xs sm:grid-cols-3">
          <FooterLink href="https://www.llama.com/">Llama by Meta</FooterLink>
          <FooterLink href="https://github.com/ggml-org/llama.cpp">llama.cpp</FooterLink>
          <FooterLink href="https://github.com/mlc-ai/web-llm">WebLLM</FooterLink>
          <FooterLink href="https://github.com/ngxson/wllama">wllama</FooterLink>
          <FooterLink href="https://huggingface.co/meta-llama">HF · meta-llama</FooterLink>
          <FooterLink href="https://github.com/meta-llama/llama-stack">Llama Stack</FooterLink>
        </div>
      </div>
      <div className="container pb-8">
        <p className="text-[11px] text-muted-foreground">
          Every number in this studio is an estimate from published specifications, not a
          measurement of your machine. Verify before you commit budget.
        </p>
      </div>
    </footer>
  );
}

function FooterLink({ href, children }: { href: string; children: React.ReactNode }) {
  return (
    <a
      href={href}
      target="_blank"
      rel="noreferrer noopener"
      className="text-muted-foreground transition-colors hover:text-meta-600"
    >
      {children}
    </a>
  );
}
