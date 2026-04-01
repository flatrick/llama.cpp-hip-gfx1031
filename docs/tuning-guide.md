# llama-server Tuning Guide

A comprehensive reference for every parameter used in this project's launch scripts,
how each affects VRAM usage, inference speed, and output quality, and how to find the
right balance for your hardware and workload.

All examples reference the RX 6700 XT (12 GB VRAM) and Qwen3.5-9B GGUF models used
in this repository, but the principles apply to any GPU and model.

---

## Table of Contents

- [How VRAM Is Used](#how-vram-is-used)
- [Parameter Reference](#parameter-reference)
  - [--ctx-size](#--ctx-size)
  - [--cache-type-k / --cache-type-v](#--cache-type-k----cache-type-v)
  - [--batch-size](#--batch-size)
  - [--ubatch-size](#--ubatch-size)
  - [--flash-attn](#--flash-attn)
  - [--n-gpu-layers](#--n-gpu-layers)
  - [--parallel](#--parallel)
  - [--defrag-thold](#--defrag-thold)
  - [-cram](#-cram)
  - [--no-warmup](#--no-warmup)
  - [--swa-full](#--swa-full)
  - [--jinja](#--jinja)
  - [--no-mmproj](#--no-mmproj)
- [Sampling Parameters](#sampling-parameters)
  - [--temp](#--temp)
  - [--top-k](#--top-k)
  - [--top-p](#--top-p)
  - [--presence-penalty](#--presence-penalty)
- [KV Cache Deep Dive](#kv-cache-deep-dive)
  - [What the KV Cache Is](#what-the-kv-cache-is)
  - [How Quantization Types Compare](#how-quantization-types-compare)
  - [Asymmetric K/V Quantization](#asymmetric-kv-quantization)
  - [Quality Impact by Task Type](#quality-impact-by-task-type)
- [VRAM Budget Breakdown](#vram-budget-breakdown)
  - [The Formula](#the-formula)
  - [Worked Example](#worked-example)
  - [What-If Scenarios](#what-if-scenarios)
- [Performance Characteristics](#performance-characteristics)
  - [Prefill vs Generation](#prefill-vs-generation)
  - [How Parameters Affect Each Phase](#how-parameters-affect-each-phase)
- [Practical Tuning Strategies](#practical-tuning-strategies)
  - [Strategy 1: Maximum Context Length](#strategy-1-maximum-context-length)
  - [Strategy 2: Maximum Generation Speed](#strategy-2-maximum-generation-speed)
  - [Strategy 3: Balanced (Recommended Starting Point)](#strategy-3-balanced-recommended-starting-point)
- [Troubleshooting](#troubleshooting)

---

## How VRAM Is Used

When llama-server starts and processes requests, GPU memory is consumed by four
major components:

```
Total VRAM = Model Weights + KV Cache + Compute Buffer + Runtime Overhead
```

| Component | When Allocated | Scales With | Typical Size (9B model) |
|-----------|---------------|-------------|------------------------|
| **Model weights** | At startup (fixed) | Model size & quantization | 5.3-5.8 GB |
| **KV cache** | At startup (pre-allocated) | ctx-size x cache-type x model architecture | 0.5-4.0 GB |
| **Compute buffer** | During inference | batch-size, ubatch-size | 0.2-1.8 GB |
| **Runtime overhead** | At startup (fixed) | Backend (ROCm vs Vulkan) | ~0.15 GB per-process |

The KV cache and compute buffer are the two components you can tune. Model weights
are fixed by your choice of model and quantization. Runtime overhead is fixed by
your GPU driver.

---

## Parameter Reference

### --ctx-size

**What it does:** Sets the maximum number of tokens the server can hold in a single
conversation. This is the total context window -- prompt tokens + generated tokens
must fit within this limit. If a request exceeds it, the server returns HTTP 400.

**How it affects VRAM:** The KV cache is **pre-allocated** at startup based on
ctx-size. Doubling ctx-size roughly doubles the KV cache VRAM. This is the single
largest tuning lever for VRAM.

**How it affects quality:** Larger context lets the model "see" more of the
conversation history. With too-small context, the model loses access to earlier
messages and instructions, which can cause:
- Forgetting instructions from a system prompt
- Losing track of variable definitions in code
- Repeating itself or contradicting earlier statements
- Failing to follow multi-step instructions that span many messages

**How it affects speed:** No direct impact on tokens-per-second during generation.
However, very large prompts that fill a large context will have longer prefill times
simply because there's more to process.

**Typical values:**
| ctx-size | Approx. tokens | Use case |
|----------|---------------|----------|
| 8,192 | ~6K usable | Short conversations, quick Q&A |
| 32,768 | ~24K usable | Code review, moderate documents |
| 49,152 | ~37K usable | Large codebases, long documents |
| 65,536 | ~49K usable | Maximum for this GPU/model combo |

> "Usable" tokens = ctx-size minus tokens reserved for generation (max_tokens)
> and any system prompt overhead.

**On this hardware (RX 6700 XT + Qwen3.5-9B):**
- 65,536 with ROCm + non-f16 KV cache: VRAM can grow to ~11.50 GB at full context (headless only)
- 65,536 with ROCm + f16/f16 KV cache: VRAM stays flat at ~8.13 GB (desktop-safe; `f16` for both K and V is the only sane ROCm choice on this machine — see below)
- The previously-documented 49,152 + ROCm + q8_0 desktop compromise is superseded by 65,536 + ROCm + f16/f16
- 262,144 with Vulkan + q8_0: VRAM flat at 10.31 GB (full native context, 1.7 GB headroom)

---

### --cache-type-k / --cache-type-v

**What they do:** Control the numerical precision of the KV (Key/Value) cache
stored in VRAM. The K cache stores attention keys, the V cache stores attention
values. These can be set independently (asymmetric quantization).

**Available types and their memory cost per element:**

| Type | Bytes/element | Relative to f16 | Notes |
|------|--------------|-----------------|-------|
| `f16` | 2.000 | 100% (baseline) | Full half-precision, no quality loss |
| `f32` | 4.000 | 200% | Overkill; wastes VRAM with no benefit |
| `q8_0` | 1.000 | 50% | Excellent quality, half the VRAM of f16 |
| `q5_1` | 0.688 | 34% | Good quality, noticeable VRAM savings |
| `q5_0` | 0.625 | 31% | Slightly less precise than q5_1 |
| `q4_1` | 0.563 | 28% | Moderate quality trade-off |
| `q4_0` | 0.500 | 25% | Quarter of f16; aggressive but usable |

**How they affect VRAM:** On paper, lower-precision KV types reduce the KV cache
footprint. In practice, on this ROCm setup, any choice other than `f16` for both
K and V causes the backend to keep dequantized working copies around, so the
expected VRAM savings disappear. For ROCm on this machine, changing K/V away from
`f16` does **not** reduce real memory usage.

**How they affect speed:** Counter-intuitive results on ROCm:

- **f16 vs non-f16 on ROCm:** `f16` is faster despite being larger on paper. On
  this hardware, if either KV cache is not `f16`, the ROCm HIP backend keeps a
  dequantized copy of both K and V for attention computation, and those buffers
  accumulate in the GGML HIP caching pool. `f16`/`f16` requires no such duplicate
  buffers: the GPU reads and computes in native half-precision. At 62k context,
  this shows up as **34 tok/s (`f16`) vs 24.5 tok/s (`q8_0`)**.
- **q8_0 vs q4_0:** q8_0 is ~2 tok/s faster than q4_0. The GPU memory controller
  handles byte-aligned reads more efficiently than 4-bit packed reads. The smaller
  cache size of q4_0 doesn't compensate because generation is bandwidth-bound, not
  capacity-bound.
- **Vulkan** uses native shader operations; q8_0 dequantization is handled
  efficiently in SPIR-V without accumulating intermediate buffers.

**How they affect quality:** See the [KV Cache Deep Dive](#kv-cache-deep-dive)
section for a detailed breakdown by task type.

**Recommendations:**
| Priority | K cache | V cache | Backend | Rationale |
|----------|---------|---------|---------|-----------|
| **ROCm** | f16 | f16 | ROCm | Only reasonable choice on this hardware — if K or V is not `f16`, ROCm keeps dequantized K/V copies and real VRAM usage goes up instead of down |
| Vulkan — recommended | q8_0 | q8_0 | Vulkan | No HIP pool issue; q8_0 is fast and saves VRAM vs f16 |
| Vulkan — maximum context | q4_0 | q4_0 | Vulkan | Smallest KV cache, most room for ctx-size |
| Vulkan — experimental | q5_1 | q4_0 | Vulkan | Middle ground; test with your workload |

> **ROCm + non-f16 KV cache warning:** On this machine, if `--cache-type-k` or
> `--cache-type-v` is anything other than `f16`, the GGML HIP backend keeps
> dequantized copies of both K and V for attention computation. Those buffers
> accumulate in the HIP caching pool (high-water mark is never released). At 66k
> context with `q8_0`, this adds ~3.4 GB of growth on top of what llama.cpp
> reports. `f16`/`f16` is the only configuration here that gives flat VRAM under ROCm.

---

### --batch-size

**What it does:** Controls how many tokens are processed in a single batch during
**prefill** (the phase where the model reads your entire prompt before generating).
This is a processing throughput setting, not a context size setting.

**How it affects VRAM:** Larger batch sizes require a larger compute buffer in VRAM.
This buffer is allocated during inference and contributes to **peak** VRAM usage
(the spike you see during prompt processing). The buffer scales roughly linearly:

| batch-size | Est. compute buffer |
|-----------|-------------------|
| 256 | ~0.45 GB |
| 512 | ~0.90 GB |
| 1024 | ~1.80 GB |
| 2048 | ~3.60 GB |

**How it affects speed:** Directly controls prefill speed. Larger batches mean
fewer GPU dispatch cycles to process the prompt, so prefill completes faster.
**Has zero effect on generation speed** (tokens/second during output).

**How it affects quality:** None. This is purely a performance parameter.

**The nuance:** batch-size does **not** change the OOM boundary for context size.
A prompt that fits in the KV cache will still fit regardless of batch-size. However,
a large batch-size can cause a transient VRAM spike during prefill that pushes total
usage over the GPU limit, even if the steady-state usage (model + KV cache) fits
fine. This is why you sometimes see "peak VRAM" much higher than "post VRAM" in
stress test results.

**On this hardware:**
- 1024 gives good prefill speed without excessive VRAM spikes
- 512 is safer if you're running close to VRAM limits
- 2048+ is not recommended on 12 GB cards with large context sizes

---

### --ubatch-size

**What it does:** Controls the **micro-batch** size -- the number of tokens
processed in each individual GPU kernel dispatch within a batch. The batch is split
into ubatch-sized chunks, each dispatched separately to the GPU.

**How it affects VRAM:** Smaller ubatch means less working memory per GPU dispatch.
With flash attention enabled, this directly controls the size of the attention tile
being computed at once. Think of it as: batch-size controls how many tokens per
round, ubatch-size controls how many per GPU call within that round.

**How it affects speed:** Larger ubatch = fewer kernel launches = less dispatch
overhead = faster prefill. But the effect is smaller than batch-size because the
bottleneck is usually memory bandwidth, not dispatch count. Beyond ~512, returns
diminish on consumer GPUs.

**How it affects quality:** None. Purely a performance/memory parameter.

**Relationship with batch-size:** ubatch-size must be <= batch-size. The total
batch is divided into ceil(batch-size / ubatch-size) micro-batches. For example,
batch-size=1024 with ubatch-size=256 means 4 GPU dispatches per batch.

**On this hardware:**
- 256 is a good default -- keeps flash attention tile memory low while maintaining
  reasonable dispatch efficiency
- 128 can help if you're seeing VRAM spikes during prefill of very long prompts
- 512 gives slightly faster prefill but higher peak VRAM

---

### --flash-attn

**What it does:** Enables Flash Attention, an optimized attention algorithm that
computes attention in tiles rather than materializing the full N x N attention
matrix in memory. This is a fundamental algorithmic optimization, not just a
performance tweak.

**How it affects VRAM:** Dramatically reduces VRAM for long contexts. Without
flash attention, the attention matrix scales as O(n^2) with sequence length.
A 65K context would need an enormous temporary matrix. With flash attention,
memory scales as O(n) because only small tiles are held at any time.

**How it affects speed:** Generally faster for long contexts due to better memory
access patterns (fewer reads from slow global GPU memory). Can be slightly slower
for very short contexts due to tiling overhead, but this is negligible.

**How it affects quality:** None. Flash attention computes mathematically identical
results to standard attention (within floating-point rounding). It is a pure
optimization.

**Values:** `on` or `off`. Default is `off`.

**Recommendation:** Always use `on` for context sizes above ~4K. At 65K context,
this is not optional -- you will OOM without it on a 12 GB card.

---

### --n-gpu-layers

**What it does:** Controls how many of the model's transformer layers are offloaded
to the GPU. Setting to `-1` means "all layers on GPU" (full offload).

**How it affects VRAM:** Each layer on the GPU consumes VRAM for its weights. With
a 9B model at Q4 quantization, each of the 32 layers is roughly 165 MB. Partial
offload (e.g., 24 layers) would save ~1.3 GB but at a severe speed penalty.

**How it affects speed:** Massive impact. GPU inference is 5-10x faster than CPU.
Every layer left on CPU creates a bottleneck because data must transfer between
CPU RAM and GPU VRAM for each layer, each token.

**How it affects quality:** None.

**Recommendation:** Always use `-1` (all layers on GPU) unless the model physically
doesn't fit. If it doesn't fit, you need a smaller model or more aggressive weight
quantization, not partial offload -- the speed penalty of partial offload makes it
impractical for interactive use.

---

### --parallel

**What it does:** Sets the number of concurrent request slots the server maintains.
Each slot has its own independent KV cache, allowing multiple users (or multiple
requests) to be processed simultaneously.

**How it affects VRAM:** KV cache is allocated **per slot**. `--parallel 2` doubles
the KV cache VRAM. `--parallel 4` quadruples it. On a 12 GB card running a 9B
model with 65K context, even `--parallel 2` would OOM.

**How it affects speed:** Multiple slots enable concurrent request processing, but
on a single consumer GPU, the slots share compute resources. Each individual request
may be slower due to contention. Useful for serving multiple users; not useful for
single-user local inference.

**How it affects quality:** None.

**Recommendation:** Use `1` for local/single-user setups. Only increase if you
genuinely need concurrent request handling AND have the VRAM budget.

---

### --defrag-thold (DEPRECATED)

> **As of llama.cpp build 8495+, this flag is deprecated.** KV cache defragmentation
> is now automatic and always active. You can safely remove this from launch scripts.

**What it did:** Set the fragmentation threshold that triggered automatic KV cache
defragmentation. When the ratio of wasted (fragmented) KV cache space exceeded
this value, the server would pause briefly to compact the cache.

**Background (still relevant to understanding KV cache behavior):** During
multi-turn conversations, especially agentic workflows where context is added and
removed frequently, the KV cache can develop "holes" -- freed token slots
interspersed with active ones. These holes waste VRAM and can prevent new
allocations even when total free space is sufficient. Defragmentation compacts
the cache to reclaim this space. This now happens automatically without needing
any configuration.

---

### -cram

**What it does:** Enables a CPU RAM-based prompt cache of the specified size (in MB).
The server stores processed prompt embeddings in system RAM, so if a subsequent
request shares a prefix with a previous one (e.g., same system prompt), the cached
portion can be reused without re-processing on the GPU.

**How it affects VRAM:** None. The cache lives entirely in CPU RAM.

**How it affects system RAM:** Directly. `-cram 2048` reserves up to 2 GB of system
RAM for the prompt cache.

**How it affects speed:** Dramatically faster for repeated prefixes. In multi-turn
conversations where the system prompt and conversation history are sent with every
request, only the new tokens need prefill processing. A 2000-token system prompt
that takes 3 seconds to process on first request will be nearly instant on subsequent
requests.

**How it affects quality:** None. The cached embeddings are mathematically identical
to freshly computed ones.

**When it helps most:**
- Multi-turn chat (system prompt reused every message)
- Agentic workflows (tool-calling loops with shared context)
- Applications that prepend large instruction sets

**On this hardware:** 2048 MB (2 GB) is a good value with 32 GB system RAM. The
cache hit rate depends on how consistent your prompt prefixes are.

---

### --no-warmup

**What it does:** Skips the initial forward pass that llama-server normally runs at
startup to pre-allocate compute buffers. Without this flag, the server processes a
dummy prompt at startup to ensure all GPU memory is allocated upfront, which means
VRAM usage at idle reflects the true maximum.

**How it affects VRAM:** Defers compute buffer allocation until the first real
request. This means VRAM at idle is lower (model weights + KV cache only), but the
first request will see a higher VRAM spike as buffers are allocated on-demand.

**How it affects speed:** Faster server startup (no dummy forward pass). First
request may be slightly slower due to on-demand allocation.

**Trade-off:** Without warmup, you might not discover that your configuration
exceeds VRAM until the first real request arrives. With warmup, an OOM at startup
tells you immediately that you need to adjust settings. For development/testing,
you may want to **omit** this flag to catch OOM early. For production, it saves
startup time.

---

### --swa-full

**What it does:** Forces the server to use the full Sliding Window Attention (SWA)
context range rather than the model's configured window size. This is relevant only
for models that use SWA (a technique where each layer only attends to a fixed-size
window of recent tokens rather than the entire context).

**On Qwen3.5-9B:** This model has `n_swa = 0`, meaning it does not use sliding
window attention. The flag has **no effect** and can be safely omitted or left in
without impact.

**When it matters:** For models that do use SWA (e.g., Mistral, some Phi variants),
this flag determines whether the model can attend to the full context or is limited
to its native window size.

---

### --jinja

**What it does:** Enables Jinja2 template rendering for the model's chat template.
The chat template is metadata embedded in the GGUF file that defines how
messages (system/user/assistant roles) are formatted into the raw token stream
the model expects.

**Why it matters for Qwen3.5:** Qwen3.5 uses a chat template with special
thinking tokens (`<think>` / `</think>`) that control the model's chain-of-thought
reasoning. The Jinja template handles these tokens correctly. Without `--jinja`,
the server falls back to a generic template that may not handle thinking tokens,
leading to malformed output or missing reasoning.

**How it affects quality:** Critical for correct behavior. Without it, the model
may not engage its thinking mode, or thinking tokens may leak into visible output.

**How it affects speed/VRAM:** Negligible. Template rendering is a trivial string
operation.

---

### --no-mmproj

**What it does:** Disables loading of multimodal projection weights. These weights
are used by vision-language models to process image inputs. For text-only models
like Qwen3.5-9B (non-VL variant), there are no multimodal weights to load, but
this flag makes it explicit and prevents the server from searching for them.

**How it affects VRAM:** Prevents accidental loading of projection weights if a
multimodal model file happens to be present.

**How it affects quality:** None for text-only models.

---

## Sampling Parameters

These parameters control how the model selects the next token during generation.
They affect output quality, creativity, and determinism but have **zero impact on
VRAM or speed**.

### --temp

**What it does:** Controls randomness in token selection. Temperature scales the
logits (raw prediction scores) before converting to probabilities.

| Value | Behavior |
|-------|----------|
| 0.0 | Greedy/deterministic -- always picks the highest-probability token |
| 0.1-0.3 | Very focused, nearly deterministic |
| 0.5-0.7 | Balanced creativity and coherence |
| 0.8-1.0 | More creative/varied, occasionally surprising |
| 1.5+ | Very random, often incoherent |

**For coding tasks:** 0.0-0.3 is typical (you want deterministic, correct code).

**For creative writing:** 0.7-1.0 allows more varied expression.

**For general assistant use:** 0.7 (the value used in this project) is a solid default.

---

### --top-k

**What it does:** After computing probabilities for all tokens in the vocabulary,
top-k filtering keeps only the K most probable tokens and redistributes probability
among them. All other tokens are eliminated from consideration.

| Value | Behavior |
|-------|----------|
| 1 | Equivalent to greedy (only top token survives) |
| 10-20 | Tight focus on most likely continuations |
| 40 | Moderate diversity (common default) |
| 100+ | Very permissive, rarely filters anything |
| 0 | Disabled (no top-k filtering) |

**This project uses 20**, which is on the tighter side. This works well with
Qwen3.5's strong confidence calibration and keeps outputs focused.

**Interaction with top-p:** Top-k is applied first, then top-p further filters
the remaining candidates. They work together, not in competition.

---

### --top-p

**What it does:** Also called "nucleus sampling." After top-k filtering, top-p
keeps the smallest set of tokens whose cumulative probability exceeds the threshold
P. This adapts dynamically: when the model is confident (one token has 90%
probability), only 1-2 tokens survive. When uncertain, many tokens survive.

| Value | Behavior |
|-------|----------|
| 0.1 | Extremely restrictive (only the very top probability mass) |
| 0.5 | Moderate restriction |
| 0.8 | Mild restriction (this project's value) |
| 0.95 | Very permissive (common default) |
| 1.0 | Disabled (no top-p filtering) |

**This project uses 0.8**, slightly tighter than the common 0.95 default. Combined
with top-k=20, this gives focused but not fully deterministic outputs.

---

### --presence-penalty

**What it does:** Applies a fixed penalty to tokens that have already appeared in
the generated text, regardless of how many times they appeared. This discourages
repetition.

| Value | Behavior |
|-------|----------|
| 0.0 | No penalty (model may repeat freely) |
| 0.5 | Mild anti-repetition |
| 1.0-1.5 | Moderate anti-repetition |
| 2.0 | Strong anti-repetition (may cause incoherent topic-hopping) |

**This project uses 1.5**, which is moderately aggressive. This helps prevent the
model from getting stuck in repetitive loops during long generation, which is a
common failure mode for local models with extended contexts.

**Distinction from frequency_penalty:** Presence penalty is binary (appeared or
not). Frequency penalty scales with how many times a token appeared. Presence
penalty is generally preferred as it discourages repetition without penalizing
legitimately-repeated tokens (like common words).

---

## KV Cache Deep Dive

### What the KV Cache Is

During attention, each transformer layer computes Key and Value vectors for every
token in the context. These vectors are stored in the KV cache so they don't need
to be recomputed when generating the next token. Without the cache, generating
token N would require reprocessing all N-1 previous tokens -- making generation
O(n^2) instead of O(n).

For Qwen3.5-9B specifically:
- **8 KV attention layers** (not 32 — this is a hybrid model; the remaining 24
  layers use recurrent state instead of attention)
- 4 KV heads per layer (Grouped Query Attention / GQA)
- 256-dimensional head vectors
- Total per token: 2 (K+V) x 8 layers x 4 heads x 256 dim = 16,384 elements

At q8_0 (1 byte each): 16,384 bytes = 16 KB per token
At q4_0 (0.5 bytes each): 8,192 bytes = 8 KB per token

For a 53K context (desktop-safe configuration):
- q8_0: 16 KB x 53,248 = **~832 MiB** (confirmed: server reports 952 MiB with alignment)
- q4_0: 8 KB x 53,248 = **~416 MiB**

> **Important — hybrid architecture:** Because Qwen3.5-9B only uses 8 attention
> layers for its KV cache, the KV cache is a much smaller fraction of total VRAM
> than you'd expect from a 9B model. The dominant VRAM consumer is model weights
> (~5.7 GB). Under ROCm with any non-`f16` KV setting, a secondary VRAM consumer
> emerges: dequantized K/V copies accumulate in the GGML HIP pool as context fills
> (up to ~3.4 GB extra at 66k with `q8_0`). With `f16`/`f16` this pool growth
> does not happen. The model also allocates a separate **recurrent state (RS) buffer**
> (~50 MiB) for the non-attention layers.

### How Quantization Types Compare

Quantization replaces full-precision floating point values with lower-bit
representations, trading precision for memory savings. Each type uses a block-based
scheme where values are grouped, scaled, and rounded:

| Type | Bits | Scheme | Quality | Speed — Vulkan | Speed — ROCm |
|------|------|--------|---------|----------------|--------------|
| f16 | 16 | Native half-float | Perfect | Baseline | **Fastest** (no dequant) |
| q8_0 | 8 | Block-scaled 8-bit integers | Near-perfect | **Fastest** (aligned reads) | Slower (dequant pool grows) |
| q5_1 | 5.5 | 5-bit + 2 scale values per block | Very good | Moderate | Avoid |
| q5_0 | 5 | 5-bit + 1 scale value per block | Good | Moderate | Avoid |
| q4_1 | 4.5 | 4-bit + 2 scale values per block | Good | Moderate | Avoid |
| q4_0 | 4 | 4-bit + 1 scale value per block | Acceptable | Slower than q8_0 | Avoid |

Speed is backend-dependent because the two backends handle dequantization differently:

- **Vulkan:** q8_0 KV tensors are dequantized efficiently in SPIR-V shaders without
  accumulating intermediate allocations. q8_0 is the fastest choice because the GPU
  memory controller handles 8-bit-aligned reads better than 4-bit packed values.
- **ROCm:** On this hardware, if either cache is not `f16`, ROCm keeps dequantized
  copies of both K and V before attention computation. These buffers accumulate in
  the GGML HIP caching pool and are never freed (pool high-water mark grows with
  context fill). At 66k context with `q8_0`, this pool adds ~3.4 GB beyond what
  llama.cpp reports. **`f16` for both K and V avoids this entirely** and gives flat
  VRAM and faster generation than `q8_0` on ROCm.

> **WARNING — ROCm + non-`f16` KV cache:** On this machine, using anything other
> than `f16` for either K or V causes progressive VRAM growth as context fills,
> because ROCm keeps dequantized K/V copies in the HIP pool. At 66k context,
> ROCm `q8_0` peaks at 11.50 GB vs 8.13 GB for ROCm `f16`/`f16` — headless-only
> vs desktop-safe. **Use `f16` for both K and V caches under ROCm.**

> **WARNING — CPU load spike with sub-byte cache types:** On this hardware
> (RX 6700 XT, ROCm via Docker, `-cram` enabled), using any sub-byte V-cache
> type (q4_0, q4_1, q5_0, q5_1) causes CPU usage to jump from ~200% to ~600%.
> The dequantization appears to hit the CPU path (likely through the `-cram`
> prompt cache), not just the GPU. This applies regardless of backend.

### Asymmetric K/V Quantization

Keys and Values play different roles in attention and have different sensitivity to
quantization:

**Keys** determine *which tokens to attend to*. They're used in the dot product
that computes attention weights. Quantization errors in keys shift these weights,
potentially causing the model to focus on wrong parts of the context. Keys are
more sensitive to quantization.

**Values** determine *what information to extract* from attended tokens. They're
weighted-summed according to the attention weights. Quantization errors in values
add noise to the extracted information, but the averaging effect of the weighted
sum provides natural error correction. Values are more tolerant of quantization.

This means you can often use more aggressive quantization for V than K without
noticeable quality loss:

| Configuration | K VRAM | V VRAM | Total KV | Quality |
|--------------|--------|--------|----------|---------|
| q8_0 / q8_0 | 466 MiB | 466 MiB | 932 MiB | Best |
| q8_0 / q4_0 | 466 MiB | 233 MiB | 699 MiB | Very good |
| q8_0 / q5_1 | 466 MiB | 320 MiB | 786 MiB | Very good |
| q4_0 / q4_0 | 233 MiB | 233 MiB | 466 MiB | Acceptable |

(Values shown for 53K context with Qwen3.5-9B hybrid architecture — 8 KV layers.
Pure-attention models with 32 layers would use 4x these values.)

### Quality Impact by Task Type

| Task | K sensitivity | V sensitivity | Safe minimum |
|------|--------------|--------------|--------------|
| Short Q&A (<4K tokens) | Low | Low | q4_0 / q4_0 |
| Code generation | Medium | Low | q8_0 / q4_0 |
| Long document summarization | Medium | Medium | q8_0 / q5_1 |
| Multi-step reasoning | High | Medium | q8_0 / q8_0 |
| Precise fact retrieval from deep context | High | High | q8_0 / q8_0 |
| Creative writing | Low | Low | q4_0 / q4_0 |

The general pattern: tasks that require the model to precisely locate and extract
specific details from far back in the context are most affected. Tasks where the
model primarily uses recent context or generates freely are least affected.

---

## VRAM Budget Breakdown

### The Formula

For pure-attention models:
```
Total VRAM = Weights + KV_Cache + Compute + Overhead

Where:
  Weights       = model file size on GPU (from GGUF / server log)
  KV_Cache      = 2 x kv_layers x kv_heads x head_dim x ctx_size x bytes_per_element
  Compute       = compute buffer (from server log, typically 200-300 MiB)
  Overhead      = ROCm/CUDA runtime (see below)
```

> **Note on VRAM measurements:** Always use `stress_test.py` with per-process
> VRAM tracking rather than system-wide tools. System-wide measurements include
> the Wayland compositor and other GPU clients, which can add 3-4 GB on a desktop
> and make the runtime look far more expensive than it actually is. The per-process
> overhead above what llama.cpp reports is ~150 MB for both ROCm and Vulkan.

### Actual Measured Breakdown (Qwen3.5-9B UD-Q5_K_XL, ctx-size 66048)

From the server log (`llama_context` / `sched_reserve` output):

```
Model weights (ROCm0)   :  5,753.94 MiB   (fixed)
Model weights (CPU)     :    666.88 MiB   (CPU-mapped, not on GPU)
KV cache (q8_0/q8_0)   :  ~1,096.00 MiB  (K: 548, V: 548, 8 attn layers)
Recurrent state (RS)    :     50.25 MiB   (hybrid model, 32 recurrent layers)
Compute buffer (ROCm0)  :    246.50 MiB   (batch processing scratch)
Output buffer           :      0.95 MiB
────────────────────────────────────────
Reported GPU total      :  ~7,147  MiB   (~6.98 GB)
ROCm per-process (baseline)  :  7.13 GB   (~150 MiB overhead above reported)
Vulkan per-process (baseline):  6.99 GB   (~10 MiB overhead above reported)
```

Under ROCm on this hardware, VRAM grows as context fills whenever either KV cache
type is not `f16`, because the GGML HIP pool accumulates dequantized K/V buffers
(high-water mark never released). ROCm with `f16`/`f16` KV cache has flat VRAM —
identical behavior to Vulkan.
Vulkan always pre-commits all allocations at startup, so per-process VRAM is flat
regardless of KV cache type.

At 62,444 tokens filled: **ROCm q8_0: 11.50 GB | ROCm f16: 8.13 GB | Vulkan q8_0: 6.99 GB**.

### Measured Configurations (RX 6700 XT, 12 GB, UD-Q5_K_XL)

Per-process measurements from `stress_test.py` (not system-wide — see note above).

**ROCm backend (Docker container), f16/f16 KV cache (recommended):**

| ctx-size | Baseline VRAM | Peak VRAM (full ctx) | Headroom | Verdict |
|----------|--------------|---------------------|----------|---------|
| 66,048 | 8.07 GB | **8.13 GB (flat)** | ~3.9 GB | Desktop-safe, 42→34 tok/s |

**ROCm backend (Docker container), q8_0/q8_0 KV cache (not recommended):**

| ctx-size | Baseline VRAM | Peak VRAM (full ctx) | Headroom | Verdict |
|----------|--------------|---------------------|----------|---------|
| 40,960 | ~6.7 GB | ~9.31 GB | ~2.7 GB | Comfortable |
| 53,248 | ~6.9 GB | 10.39 GB | ~1.6 GB | Desktop-safe (barely) |
| 57,344 | ~7.0 GB | ~10.89 GB | ~1.1 GB | Tight for desktop |
| 66,048 | 7.13 GB | 11.50 GB | ~0.5 GB | Headless only |

**Vulkan backend (local or Docker — identical per-process VRAM):**

| ctx-size | VRAM at startup (flat) | Headroom | Verdict |
|----------|------------------------|----------|---------|
| 66,048 | **6.99 GB** | ~5.0 GB | Desktop-safe, ~47→38 tok/s |
| 262,144 | **10.31 GB** | ~1.7 GB | Full native context, ~47→18 tok/s |

Vulkan pre-commits all allocations at startup; VRAM does not grow as context
fills. ROCm f16 achieves the same flat behavior because no dequantization pool
accumulates.

> These are real measurements. Use `stress_test.py` to verify after any
> configuration change. The harness auto-detects local server processes and
> uses per-process VRAM tracking whether running in Docker or natively.

---

## Performance Characteristics

### Prefill vs Generation

LLM inference has two distinct phases with completely different performance
profiles:

**Prefill** (prompt processing):
- Processes all input tokens at once in batches
- Compute-bound: GPU is doing matrix multiplications at full speed
- Speed measured in "tokens processed per second" (typically 500-2000+ tok/s)
- Controlled by: batch-size, ubatch-size, flash-attn
- VRAM impact: transient compute buffer spike

**Generation** (token output):
- Produces one token at a time, sequentially
- Memory-bandwidth-bound: each token requires reading all model weights from VRAM
- Speed measured in "tokens generated per second" (ROCm f16: 34-42 tok/s; Vulkan q8_0: 38-48 tok/s for 9B)
- Controlled by: cache-type, model size, backend — but cache-type interaction is backend-dependent
- VRAM impact: stable for Vulkan and ROCm f16; grows under ROCm with quantized KV types
- **Backend matters significantly:** Vulkan is ~10-40% faster at generation than ROCm f16 on RDNA2.
  ROCm with q8_0 is much worse than ROCm f16 at large context due to dequantization overhead.

This is why batch-size affects prefill speed but not generation speed, and why KV cache type
has outsized impact on ROCm (dequantization pool accumulation) but not on Vulkan.

### How Parameters Affect Each Phase

| Parameter | Prefill speed | Generation speed | Peak VRAM | Steady VRAM |
|-----------|:------------:|:----------------:|:---------:|:-----------:|
| ctx-size ↑ | -- | -- | ↑↑ | ↑↑ |
| batch-size ↑ | ↑↑ | -- | ↑ | -- |
| ubatch-size ↑ | ↑ | -- | ↑ | -- |
| cache f16→q8 (ROCm) | -- | ↓↓ (~10 tok/s at large ctx) | ↑↑ (pool growth) | ↑ |
| cache q8→q4 (Vulkan) | -- | ↓ (~2 tok/s) | ↓↓ | ↓↓ |
| flash-attn on | ↑ | -- | ↓↓ | -- |
| parallel ↑ | -- | -- | ↑↑ | ↑↑ |

Legend: ↑ increases, ↓ decreases, -- no effect, ↑↑/↓↓ significant effect

---

## Practical Tuning Strategies

### Strategy 1: Desktop Use — Vulkan (Recommended)

**Goal:** Maximum context with best generation speed, safe for desktop with GUI apps.

```
--ctx-size 66048     # or up to ~256000 for near-full native context
--cache-type-k q8_0
--cache-type-v q8_0
--batch-size 1024
--ubatch-size 256
--flash-attn on
-cram 2048
```

**Measured (Vulkan, 66k):** 6.99 GB per-process flat regardless of context fill,
~5 GB headroom at 66k. Generation: **38-48 tok/s**. Cold prefill of 62k tokens: ~162s.

**Full context (Vulkan, 262k):** VRAM rises to **10.31 GB** at startup (KV cache
pre-committed for the full 262k window) and stays flat from there — 1.7 GB headroom.
Generation: **18 tok/s at 248k tokens**, 47 tok/s at 4k tokens. Cold-start prefill
of ~248k tokens takes ~28 min (~147 tok/s), so use `QUICK=1` for stress testing.
All phases pass. See `docs/stress-test-results.md` for the full breakdown.

**Best for:** Interactive coding with OpenCode, multi-turn agentic workflows on a
desktop. The preferred backend for the 9B model on this hardware.

### Strategy 2: Desktop Use — ROCm

**Goal:** Maximum context with ROCm while keeping desktop usable.

```
--ctx-size 66048
--cache-type-k f16
--cache-type-v f16
--batch-size 1024
--ubatch-size 256
--flash-attn on
-cram 2048
```

**Measured (ROCm f16):** 8.07 GB baseline → 8.13 GB flat throughout all context
fills, ~3.9 GB headroom. Generation: **34–42 tok/s** (vs 24.5–40 tok/s with q8_0).
Cold prefill of 62k tokens: ~115s (~543 tok/s). All stress test phases passed.

**Why `f16`/`f16`, not anything else:** On this hardware, if either cache type is
not `f16`, ROCm keeps dequantized copies of both K and V internally. Those buffers
accumulate in the GGML HIP caching pool and are never freed. With `q8_0` at 66k
context, this pushes usage to ~11.50 GB (headless-only). With `f16`/`f16`,
no duplicate K/V buffers are needed and VRAM stays flat at 8.13 GB.

**Best for:** ROCm-only setups that need desktop use. `f16`/`f16` gives more
context, faster generation, and flat VRAM. On this machine there is no real VRAM
win from using other K/V types under ROCm, because the backend gives it back as
dequantized copies.

### Strategy 3: Headless — ROCm (maximum context)

**Goal:** Squeeze more context out of ROCm if the 3.9 GB f16 headroom is not enough.

```
--ctx-size 66048          # or higher if VRAM allows
--cache-type-k f16
--cache-type-v f16
--batch-size 1024
--ubatch-size 256
--flash-attn on
-cram 2048
```

**Measured (ROCm f16, 66k):** 8.13 GB flat, ~3.9 GB headroom — already comfortable
for desktop. For headless use the primary lever to increase context is to simply
raise `--ctx-size`: each 10k additional tokens adds roughly 160 MiB of KV cache
(8 attention layers × q8_0 effective density at f16).

**Best for:** Headless servers or SSH-only machines where you want to push context
beyond 66k without Vulkan.

### Strategy 4: Maximum Generation Speed

**Goal:** Fastest possible token output for quick interactive chat.

```
--ctx-size 32768
--cache-type-k q8_0
--cache-type-v q8_0
--batch-size 1024
--ubatch-size 256
--flash-attn on
-cram 2048
```

**Estimated (ROCm):** ~9.5 GB at full context, ~2.7 GB headroom. Generation: ~30 tok/s.
**Estimated (Vulkan):** ~6.99 GB flat. Generation: ~42-44 tok/s.

**Best for:** Quick Q&A, short coding questions, rapid iteration where long context
is not needed. Vulkan gives noticeably faster responses here too.

---

## Troubleshooting

### OOM (Out of Memory) during startup

The KV cache is pre-allocated at startup. If you see OOM immediately:
1. Reduce `--ctx-size` (biggest impact)
2. Switch to `q4_0` cache types
3. Check for other processes using VRAM (`rocm-smi` or check
   `/sys/class/drm/card*/device/mem_info_vram_used`)

### OOM during inference (first request)

If using `--no-warmup`, the compute buffer isn't allocated until the first request.
The first large prompt may OOM even though startup succeeded:
1. Remove `--no-warmup` to catch this at startup
2. Reduce `--batch-size`
3. Reduce `--ctx-size`

### VRAM creeping up during long sessions

Likely KV cache fragmentation (now handled automatically by llama.cpp):
1. Run `stress_test.py` to verify no memory leaks
2. Check if multiple processes are accumulating VRAM
3. Restart the server to reset all allocations

### Generation is slow (~5 tok/s instead of ~25 tok/s)

The model is running on CPU, not GPU:
1. Check `--n-gpu-layers -1` is set
2. Verify `HSA_OVERRIDE_GFX_VERSION=10.3.0` is set (for gfx1031 GPUs)
3. Check container has `--device /dev/kfd --device /dev/dri`
4. Run `rocm-smi` to verify GPU is detected

### Server returns HTTP 400 on large prompts

The prompt exceeds ctx-size. Either:
1. Increase `--ctx-size` (if VRAM allows)
2. Truncate the prompt
3. Use a summarization step to compact earlier context

### q4_0 is slower than q8_0

This is expected on AMD GPUs. The sub-byte unpacking overhead outweighs bandwidth
savings. If you need both maximum context and maximum speed, consider a smaller
model rather than aggressive cache quantization.
