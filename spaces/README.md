---
title: Llama Local Lab
emoji: 🦙
colorFrom: blue
colorTo: indigo
sdk: static
app_file: index.html
pinned: false
license: mit
short_description: Which Llama quant fits your GPU, and what it costs
tags:
  - llama
  - quantization
  - gguf
  - llama-cpp
  - webgpu
  - webllm
  - inference
  - cost-estimation
  - static
---

# Llama Local Lab

**The three questions everyone asks before running Llama locally** — answered in your
browser, with no server, no API key and no request leaving this page.

Which quantization fits my GPU? How much quality do I lose? What does inference cost per month?

## Six tools

| # | Tool | What you get |
|---|------|--------------|
| 01 | **Model catalog** | Llama 4 Scout & Maverick, 3.3 70B, the 3.1/3.2 families and Llama Guard, with the architecture numbers the other tools depend on. Download counts come live from this Hub. |
| 02 | **Quantization planner** | Every GGUF format from F16 to IQ1_M, sized against *your* VRAM with the KV cache and compute buffers counted — not just the weights. |
| 03 | **GGUF commands** | The exact `convert_hf_to_gguf.py` → `llama-quantize` → `ollama create` sequence for your model, quant and platform. Or the one-line `hf download` if a community GGUF already exists. |
| 04 | **Cost calculator** | Throughput from GPU memory bandwidth, priced against RunPod, Vast.ai, Lambda and owning the hardware. Every price editable. |
| 05 | **Playground** | Load Llama 3.2 1B via WebGPU and chat with it in this tab. Sampling controls, function-calling JSON editor, live tokens/sec. |
| 06 | **Evaluation** | Paste Q&A pairs, run them locally, score with exact-match / token-F1 / regex / LLM-as-judge, export in Llama Stack's scoring format. |

Nothing to install. A **demo engine** runs with zero downloads, so every screen works before
you commit to fetching weights.

## Why the sizing numbers are different

Most calculators use `params × bits ÷ 8`. On Llama 3+ that is wrong by 10–25% for small
models: the 128k-token vocabulary makes `token_embd` and `output` about 21% of Llama 3.2 1B,
and `llama-quantize` deliberately keeps those tensors at higher precision than the blocks.

This tool sizes them separately, and handles the tied-embedding case (Llama 3.2 1B/3B) where
the shared table stays at *output* precision instead of following the blocks down to 2 bits.

Calibrated against 36 published GGUF files spanning Llama 3.1 8B, 3.3 70B, 3.2 3B and 3.2 1B:

- **Mean absolute error 0.63%**, worst case 2.6%
- Llama 3.1 8B at Q4_K_M predicts 4.96 GB against a real file of 4.92 GB

Throughput is derived from memory bandwidth rather than a lookup table
(`bandwidth × utilization ÷ bytes-read-per-token`), with a per-architecture efficiency
factor because Metal and ROCm reach materially less of peak than CUDA. All 11 spot-checks
land inside the commonly reported range.

Every estimate comes from published specifications, not from your machine — which is exactly
why tool 05 lets you load a real model and check the answer yourself.

## Links

- Source: [github.com/NagaYu/llama-local-lab](https://github.com/NagaYu/llama-local-lab)
- Built on [llama.cpp](https://github.com/ggml-org/llama.cpp), [WebLLM](https://github.com/mlc-ai/web-llm) and [wllama](https://github.com/ngxson/wllama)

An independent, community-built tool. Not affiliated with, endorsed by, or sponsored by Meta.
Llama and the Llama mark are properties of Meta Platforms, Inc. Model weights carry their own
licenses — the MIT license here covers the application code only.
