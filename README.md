# Llama Local Lab

**Zero-Cost Llama Evaluation Studio** — answer the three questions everyone asks before
running Llama locally, entirely in the browser.

> Which quantization fits my GPU? How much quality do I lose? What does inference actually
> cost per month?

No server. No database. No API routes. No API key. Nothing you type or generate leaves
your machine. The whole thing is a static export you can drop on Cloudflare Pages or
GitHub Pages for free.

---

## What it does

| # | Tool | What you get |
|---|------|--------------|
| 01 | **Model selector** | The Llama catalog — Llama 4 Scout & Maverick, Llama 3.3 70B, the 3.1 and 3.2 families, Llama Guard — with parameters, context length, license and gating status. Live download counts are fetched client-side from the HuggingFace API; the bundled specs work offline. |
| 02 | **Quantization planner** | Every GGUF quant from F16 down to IQ1_M, sized against *your* VRAM with the KV cache and compute buffers counted. Filter to what fits, see the quality cost, get a recommendation. |
| 03 | **GGUF command generator** | The exact `convert_hf_to_gguf.py` → `llama-quantize` → `ollama create` commands for your model, quant and platform. Copy any block, copy the whole script, or download it. |
| 04 | **Cost calculator** | Throughput derived from GPU memory bandwidth, priced against RunPod, Vast.ai, Lambda and owning the hardware. Every price is editable and persisted. |
| 05 | **In-browser playground** | Load Llama 3.2 1B (or 3B, or 3.1 8B) via WebGPU and chat with it in the tab. System prompt, sampling controls, function-calling JSON editor, live tokens/sec. |
| 06 | **Evaluation mini-suite** | Paste Q&A pairs, run them against the local model, score with exact-match / contains / token-F1 / regex / LLM-as-judge, export CSV or JSONL in Llama Stack's eval shape. |

---

## Quick start

```bash
npm install
npm run dev
```

Open <http://localhost:3000>.

To produce the static site:

```bash
npm run build
```

The output lands in `out/`. Serve it with any static file server:

```bash
npx serve@latest out
```

---

## Deploying

### Cloudflare Pages

| Setting | Value |
|---|---|
| Build command | `npm run build` |
| Output directory | `out` |
| Node version | `20` |

Nothing else to configure. `public/_headers` is picked up automatically.

### GitHub Pages

`.github/workflows/deploy.yml` is included and handles both cases:

- **Project site** (`user.github.io/llama-local-lab`) — the workflow sets `BASE_PATH`
  to `/llama-local-lab` from the repository name.
- **User/org site** (`user.github.io`) — no base path.

Enable Pages with **Source: GitHub Actions** in the repository settings and push to `main`.

To build for a project site locally:

```bash
BASE_PATH=/llama-local-lab npm run build
```

### Anywhere else

`out/` is plain HTML, CSS and JS. Netlify, S3 + CloudFront, Vercel, `python -m http.server` —
all fine. The only requirement is HTTPS (or `localhost`), because WebGPU and the Cache
Storage API used for model weights need a secure context.

---

## Browser support for local inference

The first four tools are pure calculation and work in any browser. The playground and the
evaluation suite need one of three backends, and the app detects which are available:

| Backend | Requirement | Speed | Notes |
|---|---|---|---|
| **WebGPU** (WebLLM) | Chrome/Edge 113+, Safari 18+, Firefox 141+ | Fast | Weights are compiled to WebGPU shaders and cached by the browser. Second load is instant and works offline. |
| **WebAssembly** (wllama) | Any browser with WASM SIMD | Slow | Runs real GGUF files via llama.cpp. Falls back to the single-threaded build when `SharedArrayBuffer` is unavailable. |
| **Demo engine** | None | Instant | A scripted stand-in with no download, so every screen is demonstrable offline. Always labelled as a demo. |

The demo engine is why this repo needs no fixtures: the playground streams, the tokens/sec
readout moves, the evaluation suite runs end to end and the CSV export produces a real
file, with zero bytes downloaded.

---

## How the numbers are calculated

Everything is derived from published architecture specifications and hardware datasheets.
None of it is measured on your machine — the playground is there so you can check the
estimates against reality.

### File size

A flat `params × bits ÷ 8` is wrong for Llama 3 and later, often by 10–25% on small models.
Llama 3's 128k-token vocabulary makes `token_embd` and `output` a large share of the
parameters (about 21% of Llama 3.2 1B), and `llama-quantize` deliberately keeps those
tensors at higher precision than the transformer blocks.

So the planner splits them:

```
size = non_embed_params × bpw ÷ 8
     + embed_params     × embed_bpw ÷ 8
     + metadata
```

Tied embeddings (Llama 3.2 1B and 3B) count one table, and that table stays at *output*
precision — llama.cpp does not drop a shared `lm_head` to 2 bits just because the blocks
went there. Untied models count two tables and average a cheap lookup against a Q6_K
output head.

The per-format values in [`src/data/quants.ts`](src/data/quants.ts) were back-solved
against published GGUF uploads. Across 36 model/quant pairs spanning Llama 3.1 8B,
3.3 70B, 3.2 3B and 3.2 1B, mean absolute error is **0.63%** and the worst case is 2.6% —
for example Llama 3.1 8B at Q4_K_M predicts 4.95 GB against a real file of 4.92 GB.

### KV cache

```
kv_bytes = 2 × layers × kv_heads × head_dim × context × bytes_per_element
```

The leading 2 covers the separate K and V tensors. Grouped-query attention pins `kv_heads`
at 8 for every modern Llama, which is the only reason 128k context is affordable. The
planner lets you quantize the cache to `q8_0` or `q4_0`, which is the usual way to buy back
context length.

### Throughput

Single-stream decode is memory-bandwidth bound — every generated token requires reading
the weights that participate in the forward pass:

```
tokens/sec ≈ (bandwidth_GB/s × utilization) ÷ bytes_read_per_token
```

Utilization is 0.72, which is what a well-tuned llama.cpp or vLLM deployment achieves on
CUDA, scaled by a per-GPU efficiency factor: Metal and ROCm reach materially less of peak
bandwidth than CUDA does, and treating them as equal over-predicted Apple Silicon by about
50%. The embedding table is a lookup so only one row is touched, but the output head is a
full matmul against the vocabulary and does count. For mixture-of-experts models only the
routed experts are read, which is why Llama 4 Scout decodes like a 17B model despite
weighing 109B.

Prefill is compute bound instead, at roughly 35% MFU against the GPU's dense FP16 TFLOPS.
Batched decode amortizes the weight read across concurrent requests and scales
sublinearly until it hits the compute ceiling.

Spot checks against commonly reported figures — all eleven land inside the reported range:

| Model | Quant | GPU | Predicted | Commonly reported |
|---|---|---|---|---|
| Llama 3.1 8B | Q4_K_M | RTX 4090 | 158 tok/s | 130–160 |
| Llama 3.1 8B | Q4_K_M | RTX 3090 | 129 tok/s | 110–135 |
| Llama 3.1 8B | Q8_0 | RTX 4090 | 93 tok/s | 75–90 |
| Llama 3.1 8B | Q4_K_M | A100 80GB | 304 tok/s | 280–330 |
| Llama 3.1 8B | Q4_K_M | L4 | 47 tok/s | 40–50 |
| Llama 3.1 8B | Q4_K_M | RX 7900 XTX | 117 tok/s | 100–130 |
| Llama 3.3 70B | Q4_K_M | A100 80GB | 33 tok/s | 25–35 |
| Llama 3.3 70B | Q4_K_M | H100 SXM | 57 tok/s | 45–60 |
| Llama 3.2 1B | Q4_K_M | M2 Pro | 116 tok/s | 100–140 |
| Llama 3.2 1B | Q4_K_M | M4 Max | 316 tok/s | 270–330 |
| Llama 4 Scout | Q4_K_M | H100 SXM | 223 tok/s | 170–230 |

### Quality loss

The "quality loss" column is the Wikitext-2 **perplexity increase versus F16**, measured on
LLaMA-7B by llama.cpp and expressed as a percentage of the F16 baseline of 5.9066. It is a
proxy, not a benchmark score — a model can hold its perplexity and still lose the ability
to emit valid JSON. Treat it as a ranking signal and verify anything below Q4 on your own
task, which is what tool 06 is for.

### Cost

```
gpu_hours_busy       = tokens ÷ throughput ÷ 3600
gpu_hours_provisioned = gpu_hours_busy ÷ target_utilization
monthly_cloud        = gpu_hours_provisioned × $/hr
monthly_self_hosted  = amortized_capex × 730h + power × busy_hours + fixed
```

Self-hosting applies a 1.4× multiplier to the GPU's street price for the rest of the box,
amortizes over a configurable term, and charges electricity only while the GPU is busy —
because you own the hardware around the clock whether or not it is working.

GPU rental prices move constantly. The bundled table is a snapshot, every cell is editable
in the UI, and your overrides persist in `localStorage`.

---

## Architecture

```
src/
  app/                        Next.js App Router — one static page
  components/
    ui/                       shadcn/ui primitives (Radix + Tailwind)
    features/                 the six tools, one file each
    code-block.tsx            syntax highlighting + copy button
    primitives.tsx            SectionHeading, Field, Stat, MeterBar, NumberInput …
    studio-provider.tsx       cross-panel model/quant/GPU selection
  data/                       models, quantization formats, GPUs, provider pricing
  lib/
    quant.ts                  size, KV cache and throughput math
    cost.ts                   workload demand and provider pricing
    engine/                   WebGPU / WASM / demo inference backends
    hf.ts                     client-side HuggingFace stats, cached 6h
    csv.ts                    Llama Stack eval export + Q&A parsing
    storage.ts                SSR-safe localStorage hooks
```

**All state is in `localStorage`**, namespaced under `llama-local-lab:`. The header has a
reset control that clears it. Persisted state is merged over defaults on read, so adding a
field to a config type never breaks an existing user.

**Hydration safety.** `usePersistentState` returns the default on first render and applies
the stored value in an effect. Without that split, a static export produces a hydration
mismatch on every persisted control.

---

## Verification

```bash
npm run typecheck   # tsc --noEmit, strict
npm run lint        # next lint
npm run build       # static export to out/
```

---

## Caveats worth stating plainly

- **The estimates are estimates.** They come from specification sheets, not from your
  machine. A different llama.cpp build, driver, or batching strategy will move throughput
  by tens of percent.
- **Perplexity is not quality.** See above.
- **Prices are a snapshot** and are editable for exactly that reason.
- **The demo engine is not a model.** It is scripted, labelled everywhere, and exists so
  the UI is usable before you commit to a download.
- **Llama weights are licensed.** Meta's repositories are gated: you must accept the
  relevant Llama Community License on HuggingFace before you can download them. Read the
  acceptable use policy before you deploy.

---

## Credits

Built on [llama.cpp](https://github.com/ggml-org/llama.cpp),
[WebLLM](https://github.com/mlc-ai/web-llm) and
[wllama](https://github.com/ngxson/wllama). Quantization reference numbers come from
llama.cpp's published perplexity tables.

This is an independent, community-built tool. It is not affiliated with, endorsed by, or
sponsored by Meta. Llama and the Llama mark are properties of Meta Platforms, Inc.

## License

MIT for the application code. Model weights carry their own licenses.
