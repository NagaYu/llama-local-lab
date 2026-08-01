'use client';

import * as React from 'react';
import {
  AlertTriangle,
  ArrowDown,
  Bot,
  Braces,
  Check,
  Cpu,
  Eraser,
  HardDriveDownload,
  Info,
  Loader2,
  Lock,
  MessageSquare,
  RotateCcw,
  Send,
  Sparkles,
  Square,
  Trash2,
  Wrench,
  Zap,
} from 'lucide-react';

import { CodeBlock } from '@/components/code-block';
import { CopyButton } from '@/components/copy-button';
import { EmptyState, Field, Footnote, NumberInput, SectionHeading } from '@/components/primitives';
import { Alert, AlertDescription, AlertTitle } from '@/components/ui/alert';
import { Badge } from '@/components/ui/badge';
import { Button } from '@/components/ui/button';
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card';
import { Label } from '@/components/ui/label';
import { Progress } from '@/components/ui/progress';
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from '@/components/ui/select';
import { Separator } from '@/components/ui/separator';
import { Slider } from '@/components/ui/slider';
import { Switch } from '@/components/ui/switch';
import { Textarea } from '@/components/ui/textarea';
import { InfoTip } from '@/components/ui/tooltip';
import { DEFAULT_WEBLLM_MODEL_ID, WEBLLM_MODELS, getWebLlmModel } from '@/data/models';
import {
  DEFAULT_TOOLS_JSON,
  WLLAMA_MODELS,
  buildToolSystemPrompt,
  canFitModel,
  evictModel,
  extractToolCall,
  generate,
  isModelCached,
  parseTools,
  resetChat,
  toolNames,
  useCapability,
  useEngine,
} from '@/lib/engine';
import {
  estimateTokens,
  formatContext,
  formatDuration,
  formatNumber,
  formatTps,
} from '@/lib/format';
import { STORAGE_KEYS, mergeDefaults, usePersistentState } from '@/lib/storage';
import type {
  ChatMessage,
  EngineBackend,
  GenerationStats,
  PlaygroundConfig,
  SamplingParams,
} from '@/lib/types';
import { clamp, cn, nextId } from '@/lib/utils';

/* -------------------------------------------------------------------------- */
/*  Constants                                                                  */
/* -------------------------------------------------------------------------- */

const BACKEND_LABEL: Record<EngineBackend, string> = {
  webllm: 'WebGPU',
  wllama: 'WASM',
  mock: 'Demo',
};

const BACKENDS: {
  id: EngineBackend;
  name: string;
  tagline: string;
  detail: string;
  icon: React.ComponentType<{ className?: string }>;
}[] = [
  {
    id: 'webllm',
    name: 'WebGPU',
    tagline: 'Fast — recommended',
    detail:
      'MLC WebLLM compiles the model to WebGPU compute shaders. The browser caches the weights, so the second load is near-instant and works offline.',
    icon: Zap,
  },
  {
    id: 'wllama',
    name: 'WebAssembly',
    tagline: 'Works without WebGPU',
    detail:
      'llama.cpp compiled to WASM, running real GGUF files on the CPU. Meaningfully slower than WebGPU, but it needs no GPU at all.',
    icon: Cpu,
  },
  {
    id: 'mock',
    name: 'Demo engine',
    tagline: 'Instant, no download',
    detail:
      'A scripted stand-in with no weights. Zero bytes downloaded, so every control on this page stays demonstrable offline.',
    icon: Sparkles,
  },
];

const DEFAULT_SYSTEM_PROMPT =
  'You are a helpful assistant running entirely inside the user’s browser. Answer accurately and keep responses tight.';

const SYSTEM_PRESETS: { id: string; label: string; prompt: string }[] = [
  { id: 'default', label: 'Default assistant', prompt: DEFAULT_SYSTEM_PROMPT },
  {
    id: 'terse',
    label: 'Terse',
    prompt:
      'Answer in exactly one line. No preamble, no restating the question, no markdown, no closing pleasantries.',
  },
  {
    id: 'json',
    label: 'JSON only',
    prompt:
      'Reply with a single valid JSON object and nothing else. No prose, no explanation, no markdown code fences.',
  },
  {
    id: 'guard',
    label: 'Llama Guard style',
    prompt: [
      'You are a content-safety classifier in the style of Llama Guard 3.',
      '',
      'Hazard categories: S1 Violent Crimes, S2 Non-Violent Crimes, S3 Sex-Related Crimes,',
      'S4 Child Sexual Exploitation, S5 Defamation, S6 Specialized Advice, S7 Privacy,',
      'S8 Intellectual Property, S9 Indiscriminate Weapons, S10 Hate, S11 Suicide & Self-Harm,',
      'S12 Sexual Content, S13 Elections.',
      '',
      'Classify the last user message. Reply with "safe", or with "unsafe" on the first line',
      'followed by the violated category codes, comma separated, on the second line.',
      'Output nothing else.',
    ].join('\n'),
  },
];

const DEFAULT_SAMPLING: SamplingParams = {
  temperature: 0.7,
  topP: 0.95,
  maxTokens: 512,
  repetitionPenalty: 1.1,
  presencePenalty: 0,
  frequencyPenalty: 0,
  seed: null,
};

const DEFAULT_CONFIG: PlaygroundConfig = {
  modelId: DEFAULT_WEBLLM_MODEL_ID,
  systemPrompt: DEFAULT_SYSTEM_PROMPT,
  sampling: DEFAULT_SAMPLING,
  functionCallingEnabled: false,
  toolsJson: DEFAULT_TOOLS_JSON,
};

/**
 * Which sampling knobs each backend actually forwards to its sampler.
 *
 * WebLLM takes the full OpenAI-shaped set. wllama maps onto llama.cpp's sampler,
 * which has no presence/frequency terms. The demo engine only honours the token
 * budget. Saying so beside the slider beats letting the user believe a control
 * is doing something it is not.
 */
const IGNORED_SAMPLING: Record<EngineBackend, ReadonlySet<keyof SamplingParams>> = {
  webllm: new Set(),
  wllama: new Set<keyof SamplingParams>(['presencePenalty', 'frequencyPenalty']),
  mock: new Set<keyof SamplingParams>([
    'temperature',
    'topP',
    'repetitionPenalty',
    'presencePenalty',
    'frequencyPenalty',
    'seed',
  ]),
};

const STARTER_PROMPTS = [
  'What is quantization, in one paragraph?',
  'What is GGUF and how do I produce one?',
  'What is the capital of France?',
];

/** Local widening of ChatMessage: failures render as a bubble, not as model output. */
interface ChatEntry extends ChatMessage {
  error?: string;
}

type CacheState = 'unknown' | 'checking' | 'cached' | 'cold';

/* -------------------------------------------------------------------------- */
/*  Panel                                                                      */
/* -------------------------------------------------------------------------- */

export function PlaygroundPanel() {
  const [config, setConfig] = usePersistentState<PlaygroundConfig>(
    STORAGE_KEYS.playground,
    DEFAULT_CONFIG,
    { revive: mergeDefaults(DEFAULT_CONFIG) },
  );

  const { capability, checking } = useCapability();
  const engine = useEngine();
  const status = engine.status;

  // Backend starts on the demo engine so the first paint is identical on the
  // server and the client; the capability probe upgrades it in an effect.
  const [backend, setBackend] = React.useState<EngineBackend>('mock');
  const backendChosen = React.useRef(false);
  const [wllamaModelId, setWllamaModelId] = React.useState<string>(WLLAMA_MODELS[0].id);

  const [messages, setMessages] = React.useState<ChatEntry[]>([]);
  const [input, setInput] = React.useState('');
  const [streamText, setStreamText] = React.useState('');
  const [generating, setGenerating] = React.useState(false);
  const [live, setLive] = React.useState<{ tokens: number; tps: number }>({ tokens: 0, tps: 0 });

  const [cacheState, setCacheState] = React.useState<CacheState>('unknown');
  const [evicting, setEvicting] = React.useState(false);
  const [evicted, setEvicted] = React.useState(false);
  const [cancelRequested, setCancelRequested] = React.useState(false);
  const [readyInfo, setReadyInfo] = React.useState<{
    modelId: string;
    backend: EngineBackend;
    loadMs: number;
  } | null>(null);

  const [pinnedToBottom, setPinnedToBottom] = React.useState(true);
  const listRef = React.useRef<HTMLDivElement | null>(null);
  const abortRef = React.useRef<AbortController | null>(null);
  const composerRef = React.useRef<HTMLTextAreaElement | null>(null);

  /* ---------------------------------------------------------------------- */
  /*  Capability, status and cache effects                                   */
  /* ---------------------------------------------------------------------- */

  React.useEffect(() => {
    if (!capability || backendChosen.current) return;
    setBackend(capability.webgpu ? 'webllm' : 'mock');
  }, [capability]);

  // `generate()` flips the shared status to `generating` and back to `ready`
  // with loadMs: 0, so the real load time is only trustworthy on the transition
  // out of `loading`. Latch it instead of reading it straight from the status.
  React.useEffect(() => {
    if (status.phase === 'ready' && status.loadMs > 0) {
      setReadyInfo({ modelId: status.modelId, backend: status.backend, loadMs: status.loadMs });
      setEvicted(false);
    } else if (
      status.phase === 'idle' ||
      status.phase === 'error' ||
      status.phase === 'loading' ||
      status.phase === 'unsupported'
    ) {
      setReadyInfo(null);
    }
  }, [status]);

  React.useEffect(() => {
    if (status.phase !== 'loading') setCancelRequested(false);
  }, [status.phase]);

  const webllmModel = getWebLlmModel(config.modelId);
  const wllamaModel = WLLAMA_MODELS.find((m) => m.id === wllamaModelId) ?? WLLAMA_MODELS[0];

  const loaded = React.useMemo(() => {
    if (readyInfo) return { ...readyInfo, loadMs: readyInfo.loadMs as number | null };
    if (engine.ready && engine.modelId && engine.backend) {
      // The engine was loaded on another tab of the studio: no load time to show.
      return { modelId: engine.modelId, backend: engine.backend, loadMs: null };
    }
    return null;
  }, [readyInfo, engine.ready, engine.modelId, engine.backend]);

  const loadedModelId = loaded?.modelId ?? null;

  React.useEffect(() => {
    if (backend !== 'webllm') {
      setCacheState('unknown');
      return;
    }
    let cancelled = false;
    setCacheState('checking');
    isModelCached(webllmModel.id)
      .then((hit) => {
        if (!cancelled) setCacheState(hit ? 'cached' : 'cold');
      })
      .catch(() => {
        if (!cancelled) setCacheState('unknown');
      });
    return () => {
      cancelled = true;
    };
    // `loadedModelId` re-runs the probe after a successful load so the label
    // flips from "download" to "cached" without a page refresh.
  }, [backend, webllmModel.id, loadedModelId]);

  React.useEffect(
    () => () => {
      abortRef.current?.abort();
    },
    [],
  );

  /* ---------------------------------------------------------------------- */
  /*  Derived values                                                         */
  /* ---------------------------------------------------------------------- */

  const fitById = React.useMemo(() => {
    const map = new Map<string, { ok: boolean; note?: string }>();
    if (!capability) return map;
    for (const m of WEBLLM_MODELS) map.set(m.id, canFitModel(capability, m.vramMb));
    return map;
  }, [capability]);

  const parsedTools = React.useMemo(() => parseTools(config.toolsJson), [config.toolsJson]);
  const toolsActive =
    config.functionCallingEnabled && parsedTools.error === null && parsedTools.tools.length > 0;
  const activeToolNames = React.useMemo(
    () => (toolsActive ? toolNames(parsedTools.tools) : []),
    [toolsActive, parsedTools],
  );
  const toolPreamble = React.useMemo(
    () => (toolsActive ? buildToolSystemPrompt(parsedTools.tools) : ''),
    [toolsActive, parsedTools],
  );

  const systemContent = React.useMemo(() => {
    const base = config.systemPrompt.trim();
    if (!toolPreamble) return base;
    return base ? `${base}\n\n${toolPreamble}` : toolPreamble;
  }, [config.systemPrompt, toolPreamble]);

  const selectedModelId =
    backend === 'webllm' ? webllmModel.id : backend === 'wllama' ? wllamaModel.id : 'demo-engine';

  const alreadyLoaded = loaded?.backend === backend && loaded?.modelId === selectedModelId;
  const effectiveBackend = loaded?.backend ?? backend;
  const ignored = IGNORED_SAMPLING[effectiveBackend];

  const setSampling = React.useCallback(
    (patch: Partial<SamplingParams>) => {
      setConfig((prev) => ({ ...prev, sampling: { ...prev.sampling, ...patch } }));
    },
    [setConfig],
  );

  /* ---------------------------------------------------------------------- */
  /*  Scrolling                                                              */
  /* ---------------------------------------------------------------------- */

  const handleScroll = React.useCallback(() => {
    const el = listRef.current;
    if (!el) return;
    // 48px of slack: a user parked "at the bottom" should not be un-pinned by
    // the half-line of jitter that a streaming token adds.
    setPinnedToBottom(el.scrollHeight - el.scrollTop - el.clientHeight < 48);
  }, []);

  const scrollToBottom = React.useCallback(() => {
    const el = listRef.current;
    if (!el) return;
    el.scrollTop = el.scrollHeight;
    setPinnedToBottom(true);
  }, []);

  React.useEffect(() => {
    const el = listRef.current;
    if (el && pinnedToBottom) el.scrollTop = el.scrollHeight;
  }, [messages, streamText, pinnedToBottom]);

  /* ---------------------------------------------------------------------- */
  /*  Generation                                                             */
  /* ---------------------------------------------------------------------- */

  async function runGeneration(prompt: string) {
    const userEntry: ChatEntry = { id: nextId('msg'), role: 'user', content: prompt };
    const history = [...messages, userEntry];

    setMessages(history);
    setPinnedToBottom(true);
    setStreamText('');
    setLive({ tokens: 0, tps: 0 });
    setGenerating(true);

    const controller = new AbortController();
    abortRef.current = controller;

    const payload = [
      ...(systemContent ? [{ role: 'system' as const, content: systemContent }] : []),
      // Failed turns carry no usable model output, so they stay out of context.
      ...history
        .filter((m) => !m.error)
        .map((m) => ({ role: m.role, content: m.content })),
    ];

    const startedAt = performance.now();
    let firstDeltaAt: number | null = null;
    let text = '';
    let stats: GenerationStats | undefined;

    try {
      for await (const chunk of generate({
        messages: payload,
        sampling: config.sampling,
        tools: toolsActive ? parsedTools.tools : undefined,
        signal: controller.signal,
      })) {
        if (chunk.delta && firstDeltaAt === null) firstDeltaAt = performance.now();
        text = chunk.text;
        setStreamText(chunk.text);

        // Live rate is measured from the first token so prefill latency does
        // not drag the readout down for the whole response.
        const elapsed = performance.now() - (firstDeltaAt ?? startedAt);
        const tokens = estimateTokens(chunk.text);
        setLive({ tokens, tps: elapsed > 0 ? (tokens / elapsed) * 1000 : 0 });

        if (chunk.done && chunk.stats) stats = chunk.stats;
      }

      const entry: ChatEntry = { id: nextId('msg'), role: 'assistant', content: text, stats };
      if (activeToolNames.length > 0) {
        const call = extractToolCall(text, activeToolNames);
        if (call) entry.toolCall = call;
      }
      if (!text.trim() && !entry.toolCall) {
        entry.error = controller.signal.aborted
          ? 'Stopped before the model produced any output.'
          : 'The model returned an empty response. Try raising max tokens or lowering the repetition penalty.';
      }
      setMessages((prev) => [...prev, entry]);
    } catch (err) {
      const message = err instanceof Error ? err.message : String(err);
      setMessages((prev) => [
        ...prev,
        { id: nextId('msg'), role: 'assistant', content: text, error: message },
      ]);
    } finally {
      setGenerating(false);
      setStreamText('');
      abortRef.current = null;
    }
  }

  function handleSend() {
    const prompt = input.trim();
    if (!prompt || !engine.ready || generating) return;
    setInput('');
    void runGeneration(prompt);
  }

  function handleStop() {
    abortRef.current?.abort();
  }

  function handleClear() {
    abortRef.current?.abort();
    setMessages([]);
    setStreamText('');
    setLive({ tokens: 0, tps: 0 });
    setPinnedToBottom(true);
    void resetChat();
  }

  async function handleEvict() {
    if (!loaded || loaded.backend !== 'webllm') return;
    setEvicting(true);
    try {
      await evictModel(loaded.modelId);
      setEvicted(true);
      if (loaded.modelId === webllmModel.id) setCacheState('cold');
    } catch {
      // Nothing cached under that id — the disk is already reclaimed.
      setEvicted(true);
    } finally {
      setEvicting(false);
    }
  }

  function handleModelChange(id: string) {
    if (backend === 'webllm') setConfig((prev) => ({ ...prev, modelId: id }));
    else if (backend === 'wllama') setWllamaModelId(id);
  }

  function handleFormatJson() {
    try {
      const formatted = JSON.stringify(JSON.parse(config.toolsJson), null, 2);
      setConfig((prev) => ({ ...prev, toolsJson: formatted }));
    } catch {
      // The button is disabled while the JSON is invalid; this is belt and braces.
    }
  }

  /** Seed the composer from a starter chip and put the caret in it. */
  function applyStarterPrompt(prompt: string) {
    setInput(prompt);
    composerRef.current?.focus();
  }

  const buildTranscript = React.useCallback((): string => {
    const s = config.sampling;
    const lines: string[] = [
      '# Llama Local Lab — playground transcript',
      '',
      `- Exported: ${new Date().toISOString()}`,
      `- Backend: ${BACKEND_LABEL[effectiveBackend]} (\`${effectiveBackend}\`)`,
      `- Model: \`${loaded?.modelId ?? selectedModelId}\``,
      `- Sampling: temperature ${s.temperature}, top_p ${s.topP}, max_tokens ${s.maxTokens}, ` +
        `repetition_penalty ${s.repetitionPenalty}, presence_penalty ${s.presencePenalty}, ` +
        `frequency_penalty ${s.frequencyPenalty}, seed ${s.seed === null ? 'random' : s.seed}`,
      '',
      '## System prompt',
      '',
      '```text',
      systemContent || '(empty)',
      '```',
      '',
      '## Conversation',
    ];

    for (const m of messages) {
      lines.push('');
      if (m.role === 'user') {
        lines.push('### User', '', m.content);
        continue;
      }
      if (m.error) {
        lines.push('### Assistant — error', '', `> ${m.error}`);
        if (m.content.trim()) lines.push('', 'Partial output:', '', m.content);
        continue;
      }
      if (m.toolCall) {
        lines.push(
          `### Assistant — tool call \`${m.toolCall.name}\``,
          '',
          '```json',
          m.toolCall.arguments,
          '```',
        );
      } else {
        lines.push('### Assistant', '', m.content);
      }
      if (m.stats) {
        lines.push(
          '',
          `_${formatTps(m.stats.tokensPerSecond)} · TTFT ${formatDuration(m.stats.ttftMs)} · ` +
            `${m.stats.completionTokens} completion tokens · ${BACKEND_LABEL[m.stats.backend]}_`,
        );
      }
    }

    return lines.join('\n');
  }, [config.sampling, effectiveBackend, loaded, selectedModelId, systemContent, messages]);

  /* ---------------------------------------------------------------------- */
  /*  Render                                                                 */
  /* ---------------------------------------------------------------------- */

  const statusBadge = (() => {
    switch (status.phase) {
      case 'loading':
        return <Badge variant="meta">Loading weights</Badge>;
      case 'generating':
        return <Badge variant="meta">Generating</Badge>;
      case 'ready':
        return <Badge variant="success">Engine ready</Badge>;
      case 'error':
        return <Badge variant="danger">Load failed</Badge>;
      default:
        return <Badge variant="muted">No engine loaded</Badge>;
    }
  })();

  return (
    <div className="space-y-6">
      <SectionHeading
        eyebrow="05 · Playground"
        title="Run Llama in this tab"
        description="Weights are compiled to WebGPU shaders (or run through llama.cpp in WebAssembly) inside your browser. There is no server, no API key and no request leaves this page once the model is downloaded."
        actions={statusBadge}
      />

      {/* -------------------------------------------------------------- */}
      {/*  Backend + model bar                                            */}
      {/* -------------------------------------------------------------- */}
      <Card>
        <CardHeader className="pb-4">
          <CardTitle>Engine</CardTitle>
          <CardDescription>
            Pick where inference runs, choose a checkpoint, then load it. The demo engine is always
            one click away and downloads nothing.
          </CardDescription>
        </CardHeader>
        <CardContent className="space-y-4">
          <div role="radiogroup" aria-label="Inference backend" className="grid gap-2 sm:grid-cols-3">
            {BACKENDS.map((b) => {
              const unavailable =
                (b.id === 'webllm' && capability !== null && !capability.webgpu) ||
                (b.id === 'wllama' && capability !== null && !capability.wasm);
              const selected = backend === b.id;
              const Icon = b.icon;
              return (
                <button
                  key={b.id}
                  type="button"
                  role="radio"
                  aria-checked={selected}
                  disabled={unavailable || engine.busy}
                  onClick={() => {
                    backendChosen.current = true;
                    setBackend(b.id);
                  }}
                  className={cn(
                    'rounded-lg border p-3 text-left transition-colors',
                    'focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring focus-visible:ring-offset-1',
                    selected
                      ? 'border-meta-500 bg-meta-50 ring-1 ring-meta-500'
                      : 'hover:bg-muted/50',
                    (unavailable || engine.busy) && 'cursor-not-allowed opacity-50',
                  )}
                >
                  <span className="flex items-center gap-2">
                    <Icon className={cn('h-4 w-4', selected ? 'text-meta-600' : 'text-muted-foreground')} />
                    <span className="text-sm font-medium">{b.name}</span>
                    {unavailable && (
                      <Badge variant="muted" className="ml-auto px-1.5 py-0 text-[10px]">
                        Unavailable
                      </Badge>
                    )}
                  </span>
                  <span className="mt-1 block text-[11px] font-medium text-meta-700">{b.tagline}</span>
                  <span className="mt-1 block text-[11px] leading-relaxed text-muted-foreground">
                    {b.detail}
                  </span>
                </button>
              );
            })}
          </div>

          {checking && (
            <p className="flex items-center gap-2 font-mono text-[11px] text-muted-foreground">
              <Loader2 className="h-3 w-3 animate-spin" />
              Probing this browser for a WebGPU adapter…
            </p>
          )}

          {capability?.webgpu && (
            <p className="truncate font-mono text-[11px] text-muted-foreground">
              adapter: {capability.adapter} · {capability.hardwareConcurrency} logical cores
              {capability.maxStorageBufferBindingSize
                ? ` · max storage buffer ${formatNumber(
                    capability.maxStorageBufferBindingSize / 1024 / 1024,
                  )} MB`
                : ''}
            </p>
          )}

          {capability && !capability.webgpu && (
            <Alert variant="info">
              <Info />
              <AlertTitle>WebGPU is not available here</AlertTitle>
              <AlertDescription className="space-y-1.5">
                <p>{capability.reason ?? 'The browser did not report a reason.'}</p>
                <p>
                  Use <strong>WebAssembly</strong> to run a real GGUF file on the CPU
                  {capability.wasm ? '' : ' (also unavailable in this browser)'}, or the{' '}
                  <strong>Demo engine</strong> for an instant, download-free tour of the studio.
                </p>
              </AlertDescription>
            </Alert>
          )}

          <Separator />

          <div className="grid gap-3 sm:grid-cols-[minmax(0,1fr)_auto] sm:items-end">
            <Field
              label="Model"
              htmlFor="pg-model"
              hint={
                backend === 'webllm'
                  ? `${formatContext(webllmModel.contextLength)} ctx`
                  : backend === 'wllama'
                    ? `${formatContext(wllamaModel.contextLength)} ctx`
                    : 'scripted'
              }
            >
              <Select
                value={selectedModelId}
                onValueChange={handleModelChange}
                disabled={engine.busy || backend === 'mock'}
              >
                <SelectTrigger id="pg-model" aria-label="Model">
                  <SelectValue placeholder="Choose a model" />
                </SelectTrigger>
                <SelectContent>
                  {backend === 'webllm' &&
                    WEBLLM_MODELS.map((m) => {
                      const fit = fitById.get(m.id);
                      const blocked = fit ? !fit.ok : false;
                      return (
                        <SelectItem key={m.id} value={m.id} disabled={blocked} textValue={m.label}>
                          <span className="flex flex-wrap items-center gap-2">
                            <span className="font-mono text-xs">{m.label}</span>
                            <span className="tabular font-mono text-[11px] text-muted-foreground">
                              {formatNumber(m.vramMb)} MB
                            </span>
                            {m.recommended && (
                              <Badge variant="meta" className="px-1.5 py-0 text-[10px]">
                                Recommended
                              </Badge>
                            )}
                            {blocked && fit?.note && (
                              <span className="text-[10px] text-danger">{fit.note}</span>
                            )}
                          </span>
                        </SelectItem>
                      );
                    })}

                  {backend === 'wllama' &&
                    WLLAMA_MODELS.map((m) => (
                      <SelectItem key={m.id} value={m.id} textValue={m.label}>
                        <span className="flex flex-wrap items-center gap-2">
                          <span className="font-mono text-xs">{m.label}</span>
                          <span className="tabular font-mono text-[11px] text-muted-foreground">
                            {formatNumber(m.sizeMb)} MB
                          </span>
                        </span>
                      </SelectItem>
                    ))}

                  {backend === 'mock' && (
                    <SelectItem value="demo-engine" textValue="Demo engine">
                      <span className="flex items-center gap-2">
                        <span className="font-mono text-xs">Demo engine</span>
                        <Badge variant="muted" className="px-1.5 py-0 text-[10px]">
                          0 MB
                        </Badge>
                      </span>
                    </SelectItem>
                  )}
                </SelectContent>
              </Select>
            </Field>

            <Button
              onClick={() => void engine.load(backend, selectedModelId)}
              disabled={engine.busy || alreadyLoaded}
              className="w-full sm:w-auto"
            >
              {status.phase === 'loading' ? (
                <Loader2 className="animate-spin" />
              ) : (
                <HardDriveDownload />
              )}
              {alreadyLoaded
                ? 'Loaded'
                : backend === 'mock'
                  ? 'Start demo engine'
                  : cacheState === 'cached'
                    ? 'Load model'
                    : 'Download & load'}
            </Button>
          </div>

          <p className="text-xs text-muted-foreground">
            {backend === 'webllm' && cacheState === 'checking' && 'Checking the browser cache…'}
            {backend === 'webllm' && cacheState === 'cached' && (
              <span className="font-medium text-success">
                Cached — loads instantly, nothing to download.
              </span>
            )}
            {backend === 'webllm' && (cacheState === 'cold' || cacheState === 'unknown') && (
              <>
                <InfoTip label={`~${formatNumber(webllmModel.vramMb)} MB download`}>
                  WebLLM publishes <span className="font-mono">vram_required_MB</span> ={' '}
                  {formatNumber(webllmModel.vramMb)} for this build. The download is the weight
                  shards, which come to roughly that figure; the browser keeps them in Cache Storage
                  afterwards.
                </InfoTip>{' '}
                on first load, then cached by the browser.
              </>
            )}
            {backend === 'wllama' && (
              <>
                {formatNumber(wllamaModel.sizeMb)} MB GGUF pulled from{' '}
                <span className="font-mono">{wllamaModel.repo}</span> on load. Same artifact the
                quantization planner sizes.
              </>
            )}
            {backend === 'mock' && (
              <>Nothing is downloaded. The demo engine streams a scripted reply at a plausible rate.</>
            )}
          </p>

          {status.phase === 'loading' && (
            <div className="space-y-2 rounded-lg border bg-muted/30 p-3">
              <div className="flex items-center justify-between gap-3">
                <span className="flex items-center gap-2 text-sm font-medium">
                  <Loader2 className="h-4 w-4 animate-spin text-meta-600" />
                  Loading weights
                </span>
                <span className="tabular font-mono text-xs text-muted-foreground">
                  {formatNumber(clamp(status.progress, 0, 1) * 100)}%
                </span>
              </div>
              <Progress
                value={clamp(status.progress, 0, 1) * 100}
                aria-label="Model load progress"
              />
              <div className="flex items-start justify-between gap-3">
                <p className="min-w-0 break-words font-mono text-[11px] leading-relaxed text-muted-foreground">
                  {status.text}
                </p>
                <Button
                  variant="outline"
                  size="sm"
                  onClick={engine.cancelLoad}
                  disabled={cancelRequested}
                  className="shrink-0"
                >
                  {cancelRequested ? 'Cancelling…' : 'Cancel'}
                </Button>
              </div>
              {cancelRequested && (
                <Footnote>
                  The in-flight fetch finishes its current shard before the load aborts, so this can
                  take a few seconds.
                </Footnote>
              )}
            </div>
          )}

          {status.phase === 'error' && (
            <Alert variant="destructive">
              <AlertTriangle />
              <AlertTitle>Could not load the model</AlertTitle>
              <AlertDescription className="space-y-1.5">
                <p className="break-words">{status.message}</p>
                <p>
                  The rest of this page still works — switch to the demo engine to keep exploring.
                </p>
              </AlertDescription>
            </Alert>
          )}

          {loaded && status.phase !== 'loading' && (
            <div className="flex flex-wrap items-center gap-x-4 gap-y-2 rounded-lg border border-success/25 bg-success-soft px-3 py-2.5">
              <span className="flex items-center gap-2 text-sm font-medium text-success">
                <Check className="h-4 w-4" />
                Loaded
              </span>
              <code className="min-w-0 break-all font-mono text-xs">{loaded.modelId}</code>
              <Badge variant="meta">{BACKEND_LABEL[loaded.backend]}</Badge>
              {loaded.loadMs !== null && (
                <span className="tabular font-mono text-xs text-muted-foreground">
                  load {formatDuration(loaded.loadMs)}
                </span>
              )}
              <div className="ml-auto flex flex-wrap items-center gap-2">
                <Button
                  variant="outline"
                  size="sm"
                  onClick={() => void engine.unload()}
                  disabled={generating}
                >
                  Unload
                </Button>
                {loaded.backend === 'webllm' && (
                  <Button
                    variant="ghost"
                    size="sm"
                    onClick={() => void handleEvict()}
                    disabled={evicting}
                  >
                    <Trash2 />
                    {evicting ? 'Evicting…' : 'Evict weights'}
                  </Button>
                )}
              </div>
              {evicted && (
                <p className="w-full text-xs text-muted-foreground">
                  Cached weights deleted from browser storage. The copy already in memory keeps
                  working until you press Unload.
                </p>
              )}
            </div>
          )}
        </CardContent>
      </Card>

      {/* -------------------------------------------------------------- */}
      {/*  Chat + settings                                                */}
      {/* -------------------------------------------------------------- */}
      <div className="grid gap-6 lg:grid-cols-3">
        {/* Chat */}
        <section className="order-2 min-w-0 space-y-3 lg:order-1 lg:col-span-2">
          <Card className="overflow-hidden">
            <CardHeader className="flex-row items-start justify-between gap-3 space-y-0 border-b py-3.5">
              <div className="min-w-0">
                <CardTitle className="text-sm">Conversation</CardTitle>
                <CardDescription className="text-xs">
                  {messages.length === 0
                    ? 'Nothing sent yet.'
                    : `${messages.filter((m) => m.role === 'user').length} prompt${
                        messages.filter((m) => m.role === 'user').length === 1 ? '' : 's'
                      } this session`}
                </CardDescription>
              </div>
              <div className="flex shrink-0 items-center gap-1.5">
                <CopyButton
                  value={buildTranscript}
                  label="Copy transcript"
                  showLabel
                  variant="outline"
                  aria-label="Copy the conversation transcript as markdown"
                  disabled={messages.length === 0}
                />
                <Button
                  variant="ghost"
                  size="sm"
                  onClick={handleClear}
                  disabled={messages.length === 0 && !generating}
                >
                  <Eraser />
                  Clear
                </Button>
              </div>
            </CardHeader>

            <CardContent className="p-0">
              <div className="relative">
                <div
                  ref={listRef}
                  onScroll={handleScroll}
                  className="max-h-[62vh] min-h-[20rem] space-y-5 overflow-y-auto scrollbar-thin p-4 sm:p-5"
                >
                  {messages.length === 0 && !generating ? (
                    <EmptyState
                      icon={<MessageSquare className="h-6 w-6" />}
                      title="No messages yet"
                      description={
                        engine.ready
                          ? 'Type below, or start with one of these.'
                          : 'Load an engine above — the demo engine takes about a second and downloads nothing.'
                      }
                      action={
                        <div className="flex flex-wrap justify-center gap-1.5">
                          {STARTER_PROMPTS.map((p) => (
                            <Button
                              key={p}
                              variant="outline"
                              size="xs"
                              onClick={() => applyStarterPrompt(p)}
                            >
                              {p}
                            </Button>
                          ))}
                        </div>
                      }
                      className="border-0"
                    />
                  ) : (
                    messages.map((m) => <MessageBubble key={m.id} entry={m} />)
                  )}

                  {generating && (
                    <AssistantShell>
                      <div className="whitespace-pre-wrap break-words text-sm leading-relaxed">
                        {streamText}
                        <span
                          aria-hidden
                          className="ml-0.5 inline-block h-4 w-[2px] translate-y-0.5 animate-pulse bg-meta-500"
                        />
                      </div>
                      <span className="sr-only" aria-live="polite">
                        Generating a response.
                      </span>
                    </AssistantShell>
                  )}
                </div>

                {!pinnedToBottom && (
                  <Button
                    size="sm"
                    variant="outline"
                    onClick={scrollToBottom}
                    className="absolute bottom-3 left-1/2 -translate-x-1/2 bg-background shadow-sm"
                  >
                    <ArrowDown />
                    Jump to latest
                  </Button>
                )}
              </div>

              {generating && (
                <div className="flex flex-wrap items-center gap-x-3 gap-y-1 border-t bg-meta-50 px-4 py-1.5 text-[11px] text-meta-800">
                  <span className="relative flex h-2 w-2" aria-hidden>
                    <span className="absolute inline-flex h-full w-full animate-ping rounded-full bg-meta-400 opacity-75" />
                    <span className="relative inline-flex h-2 w-2 rounded-full bg-meta-500" />
                  </span>
                  <span className="tabular font-mono font-semibold">{formatTps(live.tps)}</span>
                  <span className="tabular font-mono text-meta-700">
                    ≈{formatNumber(live.tokens)} tokens
                  </span>
                  <InfoTip label="live estimate">
                    While streaming there is no token count yet, so this is characters ÷ 4 divided by
                    the time since the first token. The exact figures replace it the moment the turn
                    finishes.
                  </InfoTip>
                </div>
              )}

              <div className="space-y-2 border-t p-4">
                <Label htmlFor="pg-composer" className="sr-only">
                  Message
                </Label>
                <Textarea
                  id="pg-composer"
                  ref={composerRef}
                  value={input}
                  onChange={(e) => setInput(e.target.value)}
                  onKeyDown={(e) => {
                    if (e.key === 'Enter' && !e.shiftKey && !e.nativeEvent.isComposing) {
                      e.preventDefault();
                      handleSend();
                    }
                  }}
                  disabled={!engine.ready}
                  rows={3}
                  placeholder={
                    engine.ready
                      ? 'Ask the model something…'
                      : 'Load an engine above to start generating.'
                  }
                  className="min-h-[78px] resize-y"
                />
                <div className="flex flex-wrap items-center justify-between gap-2">
                  <p className="text-[11px] text-muted-foreground">
                    <kbd className="font-mono">Enter</kbd> sends ·{' '}
                    <kbd className="font-mono">Shift</kbd>+<kbd className="font-mono">Enter</kbd> for
                    a newline
                    {input.trim() ? ` · ≈${formatNumber(estimateTokens(input))} prompt tokens` : ''}
                  </p>
                  <div className="flex items-center gap-2">
                    {generating && (
                      <Button variant="outline" size="sm" onClick={handleStop}>
                        <Square />
                        Stop
                      </Button>
                    )}
                    <Button
                      size="sm"
                      onClick={handleSend}
                      disabled={!engine.ready || generating || input.trim().length === 0}
                    >
                      <Send />
                      Send
                    </Button>
                  </div>
                </div>
              </div>
            </CardContent>
          </Card>

          <Footnote>
            Per-turn figures come straight from the engine. <strong>tok/s</strong> is completion
            tokens ÷ (total time − time to first token); on WebGPU it is WebLLM&rsquo;s own{' '}
            <span className="font-mono">decode_tokens_per_s</span>. <strong>TTFT</strong> is the wall
            time from submitting the prompt to the first streamed token — it is dominated by prefill,
            so it grows with conversation length. Token counts are exact on WebGPU (reported in the
            usage block) and estimated at 4 characters per token on the WASM and demo backends, which
            do not expose a tokenizer count.
          </Footnote>
        </section>

        {/* Settings sidebar */}
        <aside className="order-1 min-w-0 space-y-4 lg:order-2">
          {/* System prompt */}
          <Card>
            <CardHeader className="pb-3">
              <CardTitle className="text-sm">System prompt</CardTitle>
              <CardDescription className="text-xs">
                Sent as the first message on every turn. It is not shown as a bubble in the
                transcript.
              </CardDescription>
            </CardHeader>
            <CardContent className="space-y-3">
              <div className="flex flex-wrap gap-1.5">
                {SYSTEM_PRESETS.map((p) => (
                  <Button
                    key={p.id}
                    variant={config.systemPrompt === p.prompt ? 'default' : 'outline'}
                    size="xs"
                    onClick={() => setConfig((prev) => ({ ...prev, systemPrompt: p.prompt }))}
                    aria-pressed={config.systemPrompt === p.prompt}
                  >
                    {p.label}
                  </Button>
                ))}
              </div>
              <Field
                label="Prompt text"
                htmlFor="pg-system"
                hint={`≈${formatNumber(estimateTokens(config.systemPrompt))} tok`}
              >
                <Textarea
                  id="pg-system"
                  value={config.systemPrompt}
                  onChange={(e) =>
                    setConfig((prev) => ({ ...prev, systemPrompt: e.target.value }))
                  }
                  rows={6}
                  spellCheck={false}
                  className="min-h-[120px] font-mono text-[11px] leading-relaxed"
                />
              </Field>
              <Footnote>
                The token count is the same 4-characters-per-token estimate the rest of the studio
                uses. The system prompt is re-sent with every turn, so it is charged against the
                context window each time.
              </Footnote>
            </CardContent>
          </Card>

          {/* Sampling */}
          <Card>
            <CardHeader className="pb-3">
              <CardTitle className="text-sm">Sampling</CardTitle>
              <CardDescription className="text-xs">
                Applied to the <span className="font-mono">{BACKEND_LABEL[effectiveBackend]}</span>{' '}
                backend on the next turn.
              </CardDescription>
            </CardHeader>
            <CardContent className="space-y-4">
              <SliderRow
                label="Temperature"
                help="Scales the logits before sampling. 0 is greedy and deterministic; above ~1.2 Llama 3 starts to lose coherence. 0.6–0.8 is the usual instruct range."
                value={config.sampling.temperature}
                display={config.sampling.temperature.toFixed(2)}
                min={0}
                max={2}
                step={0.05}
                onChange={(v) => setSampling({ temperature: v })}
                ignoredBy={ignored.has('temperature') ? BACKEND_LABEL[effectiveBackend] : null}
              />
              <SliderRow
                label="Top-p"
                help="Nucleus sampling: only the smallest set of tokens whose probabilities sum to p can be chosen. 1.0 disables the filter."
                value={config.sampling.topP}
                display={config.sampling.topP.toFixed(2)}
                min={0}
                max={1}
                step={0.01}
                onChange={(v) => setSampling({ topP: v })}
                ignoredBy={ignored.has('topP') ? BACKEND_LABEL[effectiveBackend] : null}
              />
              <SliderRow
                label="Max tokens"
                help="Hard ceiling on the completion length. The model still stops early at an end-of-turn token; this only bounds the worst case."
                value={config.sampling.maxTokens}
                display={formatNumber(config.sampling.maxTokens)}
                min={16}
                max={2048}
                step={16}
                onChange={(v) => setSampling({ maxTokens: Math.round(v) })}
                ignoredBy={ignored.has('maxTokens') ? BACKEND_LABEL[effectiveBackend] : null}
              />
              <SliderRow
                label="Repetition penalty"
                help="Divides the logit of tokens already present in the context. 1.0 is off; 1.05–1.15 is enough to stop small quantized models looping."
                value={config.sampling.repetitionPenalty}
                display={config.sampling.repetitionPenalty.toFixed(2)}
                min={1}
                max={1.5}
                step={0.01}
                onChange={(v) => setSampling({ repetitionPenalty: v })}
                ignoredBy={
                  ignored.has('repetitionPenalty') ? BACKEND_LABEL[effectiveBackend] : null
                }
              />
              <SliderRow
                label="Presence penalty"
                help="A flat penalty applied once to any token that has already appeared, pushing the model toward new topics. Negative values encourage repetition."
                value={config.sampling.presencePenalty}
                display={config.sampling.presencePenalty.toFixed(1)}
                min={-2}
                max={2}
                step={0.1}
                onChange={(v) => setSampling({ presencePenalty: v })}
                ignoredBy={ignored.has('presencePenalty') ? BACKEND_LABEL[effectiveBackend] : null}
              />
              <SliderRow
                label="Frequency penalty"
                help="Scales with how often a token has already appeared, so it suppresses runaway repetition more aggressively than the presence penalty."
                value={config.sampling.frequencyPenalty}
                display={config.sampling.frequencyPenalty.toFixed(1)}
                min={-2}
                max={2}
                step={0.1}
                onChange={(v) => setSampling({ frequencyPenalty: v })}
                ignoredBy={ignored.has('frequencyPenalty') ? BACKEND_LABEL[effectiveBackend] : null}
              />

              <Separator />

              <div className="flex items-center justify-between gap-3">
                <Label htmlFor="pg-seed-toggle" className="text-xs font-medium">
                  <InfoTip label="Fixed seed">
                    Pins the RNG so the same prompt and sampling settings reproduce the same
                    completion. Leave it off for a fresh sample each time.
                  </InfoTip>
                </Label>
                <Switch
                  id="pg-seed-toggle"
                  checked={config.sampling.seed !== null}
                  onCheckedChange={(on) => setSampling({ seed: on ? 42 : null })}
                  aria-label="Use a fixed sampling seed"
                />
              </div>
              {config.sampling.seed !== null && (
                <NumberInput
                  id="pg-seed"
                  aria-label="Seed value"
                  value={config.sampling.seed}
                  onChange={(n) => setSampling({ seed: Math.max(0, Math.round(n)) })}
                  min={0}
                  max={2147483647}
                  step={1}
                />
              )}
              {ignored.has('seed') && config.sampling.seed !== null && (
                <p className="text-[10px] text-muted-foreground">
                  Not applied by the {BACKEND_LABEL[effectiveBackend]} backend.
                </p>
              )}

              <Button
                variant="outline"
                size="xs"
                onClick={() => setSampling(DEFAULT_SAMPLING)}
                className="w-full"
              >
                <RotateCcw />
                Reset to defaults
              </Button>
            </CardContent>
          </Card>

          {/* Function calling */}
          <Card>
            <CardHeader className="flex-row items-start justify-between gap-3 space-y-0 pb-3">
              <div className="min-w-0">
                <CardTitle className="flex items-center gap-2 text-sm">
                  <Wrench className="h-3.5 w-3.5 text-meta-600" />
                  Function calling
                </CardTitle>
                <CardDescription className="text-xs">
                  Prompt-level tool use, the way Meta documents it for Llama 3.1+.
                </CardDescription>
              </div>
              <Switch
                checked={config.functionCallingEnabled}
                onCheckedChange={(on) =>
                  setConfig((prev) => ({ ...prev, functionCallingEnabled: on }))
                }
                aria-label="Enable function calling"
                className="mt-0.5 shrink-0"
              />
            </CardHeader>
            {config.functionCallingEnabled && (
              <CardContent className="space-y-2.5">
                <div className="flex flex-wrap items-center justify-between gap-2">
                  <Label htmlFor="pg-tools" className="text-xs font-medium">
                    Tool definitions (JSON)
                  </Label>
                  <div className="flex items-center gap-1">
                    <Button
                      variant="ghost"
                      size="xs"
                      onClick={handleFormatJson}
                      disabled={parsedTools.error !== null}
                    >
                      <Braces />
                      Format JSON
                    </Button>
                    <Button
                      variant="ghost"
                      size="xs"
                      onClick={() => setConfig((prev) => ({ ...prev, toolsJson: DEFAULT_TOOLS_JSON }))}
                    >
                      <RotateCcw />
                      Reset
                    </Button>
                  </div>
                </div>

                <Textarea
                  id="pg-tools"
                  value={config.toolsJson}
                  onChange={(e) => setConfig((prev) => ({ ...prev, toolsJson: e.target.value }))}
                  rows={14}
                  spellCheck={false}
                  aria-invalid={parsedTools.error !== null}
                  className={cn(
                    'min-h-[220px] font-mono text-[11px] leading-relaxed',
                    parsedTools.error !== null && 'border-danger focus-visible:ring-danger',
                  )}
                />

                {parsedTools.error !== null ? (
                  <p className="flex items-start gap-1.5 rounded-md border border-danger/25 bg-danger-soft px-2.5 py-1.5 text-xs text-danger">
                    <AlertTriangle className="mt-0.5 h-3.5 w-3.5 shrink-0" />
                    <span className="min-w-0 break-words">{parsedTools.error}</span>
                  </p>
                ) : parsedTools.tools.length === 0 ? (
                  <p className="text-xs text-muted-foreground">
                    No functions defined — nothing will be appended to the system prompt.
                  </p>
                ) : (
                  <p className="flex flex-wrap items-center gap-1.5 text-xs text-success">
                    <Check className="h-3.5 w-3.5 shrink-0" />
                    {parsedTools.tools.length} function
                    {parsedTools.tools.length === 1 ? '' : 's'} parsed:
                    <span className="font-mono">{toolNames(parsedTools.tools).join(', ')}</span>
                  </p>
                )}

                {toolPreamble && (
                  <details className="rounded-lg border bg-muted/30 px-3 py-2">
                    <summary className="cursor-pointer text-xs font-medium">
                      Preview the text appended to the system prompt
                    </summary>
                    <div className="mt-2">
                      <CodeBlock
                        code={toolPreamble}
                        language="text"
                        title="tool preamble"
                        maxHeight={220}
                      />
                    </div>
                  </details>
                )}

                <Footnote>
                  WebLLM&rsquo;s OpenAI-style <span className="font-mono">tools</span> parameter only
                  works for the few models on its function-calling list, none of which are Llama 3.2.
                  So the definitions above are rendered into the system prompt instead, and each
                  reply is scanned for a JSON object naming one of the declared functions — bare, in
                  a fenced block, or behind Llama 3.1&rsquo;s{' '}
                  <span className="font-mono">&lt;|python_tag|&gt;</span> prefix. That is prompt-level
                  tool use, and it is what you would ship if you served these weights yourself.
                </Footnote>
              </CardContent>
            )}
          </Card>

          {/* Privacy */}
          <Card className="border-meta-200 bg-meta-50/50">
            <CardContent className="flex gap-3 p-4">
              <Lock className="mt-0.5 h-4 w-4 shrink-0 text-meta-600" />
              <div className="space-y-1 text-xs leading-relaxed text-meta-900/80">
                <p className="font-semibold text-meta-900">Everything runs on this machine.</p>
                <p>
                  Prompts, responses and settings never leave the tab — there is no backend and no
                  API key. The only network request this page makes is the one-time model download
                  from a public CDN, after which the weights are cached by your browser and
                  inference works offline. Settings are stored in{' '}
                  <span className="font-mono">localStorage</span>; use Evict weights to reclaim the
                  disk.
                </p>
              </div>
            </CardContent>
          </Card>
        </aside>
      </div>
    </div>
  );
}

/* -------------------------------------------------------------------------- */
/*  Sub-components                                                             */
/* -------------------------------------------------------------------------- */

/**
 * Radix's slider puts `role="slider"` on the thumb, and the shared primitive
 * does not forward an aria-label that far, so the group wrapper carries the name.
 */
function SliderRow({
  label,
  help,
  value,
  display,
  min,
  max,
  step,
  onChange,
  ignoredBy,
}: {
  label: string;
  help: React.ReactNode;
  value: number;
  display: string;
  min: number;
  max: number;
  step: number;
  onChange: (value: number) => void;
  ignoredBy: string | null;
}) {
  return (
    <Field label={label} help={help} hint={display}>
      <div role="group" aria-label={label} className="py-1">
        <Slider
          value={[value]}
          min={min}
          max={max}
          step={step}
          onValueChange={(next) => onChange(next[0])}
        />
      </div>
      {ignoredBy && (
        <p className="text-[10px] text-muted-foreground">
          Not applied by the {ignoredBy} backend.
        </p>
      )}
    </Field>
  );
}

function AssistantShell({ children }: { children: React.ReactNode }) {
  return (
    <div className="flex gap-3">
      <div className="mt-0.5 flex h-7 w-7 shrink-0 items-center justify-center rounded-md border bg-muted">
        <Bot className="h-3.5 w-3.5 text-meta-600" />
      </div>
      <div className="min-w-0 flex-1 space-y-2">{children}</div>
    </div>
  );
}

function MessageBubble({ entry }: { entry: ChatEntry }) {
  if (entry.role === 'user') {
    return (
      <div className="flex justify-end">
        <div className="max-w-[85%] whitespace-pre-wrap break-words rounded-lg rounded-br-sm border border-meta-200 bg-meta-50 px-3.5 py-2.5 text-sm leading-relaxed text-meta-900">
          {entry.content}
        </div>
      </div>
    );
  }

  return (
    <AssistantShell>
      {entry.error ? (
        <div className="rounded-lg border border-danger/25 bg-danger-soft px-3 py-2.5">
          <p className="flex items-start gap-1.5 text-sm font-medium text-danger">
            <AlertTriangle className="mt-0.5 h-3.5 w-3.5 shrink-0" />
            <span className="min-w-0 break-words">{entry.error}</span>
          </p>
          {entry.content.trim() && (
            <p className="mt-2 whitespace-pre-wrap break-words border-t border-danger/15 pt-2 text-sm leading-relaxed text-foreground">
              {entry.content}
            </p>
          )}
        </div>
      ) : entry.toolCall ? (
        <div className="space-y-2 rounded-lg border border-meta-200 bg-meta-50/60 p-3">
          <div className="flex flex-wrap items-center gap-2">
            <Wrench className="h-3.5 w-3.5 text-meta-600" />
            <span className="font-mono text-xs font-semibold text-meta-800">
              {entry.toolCall.name}
            </span>
            <Badge variant="meta" className="px-1.5 py-0 text-[10px]">
              tool call
            </Badge>
          </div>
          <CodeBlock
            code={entry.toolCall.arguments}
            language="json"
            title="arguments"
            maxHeight={260}
          />
          <details className="text-[11px] text-muted-foreground">
            <summary className="cursor-pointer">Raw model output</summary>
            <pre className="mt-1.5 overflow-x-auto whitespace-pre-wrap break-words rounded-md border bg-background p-2 font-mono">
              {entry.content}
            </pre>
          </details>
        </div>
      ) : (
        <div className="whitespace-pre-wrap break-words text-sm leading-relaxed">
          {entry.content}
        </div>
      )}

      {entry.stats && <StatsRow stats={entry.stats} />}
    </AssistantShell>
  );
}

function StatsRow({ stats }: { stats: GenerationStats }) {
  return (
    <div className="flex flex-wrap items-center gap-x-3 gap-y-1 text-[11px] text-muted-foreground">
      <Badge variant="muted" className="px-1.5 py-0 font-mono text-[10px]">
        {BACKEND_LABEL[stats.backend]}
      </Badge>
      <span className="tabular font-medium text-foreground">
        <InfoTip label={formatTps(stats.tokensPerSecond)}>
          Decode rate: completion tokens ÷ (total time − time to first token). On WebGPU this is
          WebLLM&rsquo;s own <span className="font-mono">decode_tokens_per_s</span>; the other
          backends compute it from wall-clock timing.
        </InfoTip>
      </span>
      <span className="tabular">TTFT {formatDuration(stats.ttftMs)}</span>
      <span className="tabular">
        {formatNumber(stats.completionTokens)} tok
        {stats.backend !== 'webllm' ? ' (est.)' : ''}
      </span>
      <span className="tabular">total {formatDuration(stats.totalMs)}</span>
    </div>
  );
}
