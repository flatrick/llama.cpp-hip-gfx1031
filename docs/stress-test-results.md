# Stress Test Results

Verified configurations on the RX 6700 XT (12 GB VRAM, gfx1031).

All results produced by the `python stress_test.py` entrypoint, which now wraps
the reusable `stress_harness/` package and runs five phases: ramp, sustained
load, cold-start leak detection, defrag stress, and boundary
(ctx_size+1 must return HTTP 400).

---

## Qwen3.5-2B UD-Q5_K_XL — 200K context

**Script:** `Qwen3.5-2B-UD-Q5_K_XL.200k.b8495.sh`
**Date:** 2026-03-31
**Result:** ALL PHASES PASSED

### Configuration

```
--ctx-size 209715
--cache-type-k q8_0
--cache-type-v q8_0
--batch-size 1024
--ubatch-size 256
--flash-attn on
--parallel 1
-cram 2048
```

### Memory breakdown (from llama-server log)

```
                          total   free    self   model   context   compute    unaccounted
ROCm0 (RX 6700 XT)      12272 =  860 + (3031 =  1388 +    1326 +     317) +        8380
Host                                      606 =   397 +       0 +     209
```

- **Model weights (GPU):** 1,388 MiB
- **KV cache:** 1,326 MiB
- **Compute buffer:** 317 MiB
- **Host (CPU):** 606 MiB (397 model + 209 compute)
- **Free VRAM:** 860 MiB
- **Unaccounted (ROCm overhead):** 8,380 MiB

### Performance

| Context fill | Prefill speed | Gen tok/s | VRAM |
|-------------|--------------|-----------|------|
| 4,000 tokens | 3,820t / 3.8s (~1005 t/s) | 105.6 | 3.18 GB |
| 16,000 tokens | 8,261t / 6.1s (~1354 t/s) | 90.9 | 3.53 GB |
| 52,480 tokens | 20,834t / 15.5s (~1344 t/s) | 68.1 | 4.95 GB |
| 104,960 tokens | 52,838t / 46.7s (~1131 t/s) | 50.6 | 7.12 GB |
| 146,944 tokens | 42,170t / 51.8s (~814 t/s) | 42.0 | 8.64 GB |
| 199,168 tokens | 6,356t / 17.4s (~365 t/s) | 34.7 | 11.06 GB |

### Sustained load (20 rounds at ~199K tokens)

- Generation speed: 34.7 tok/s (stable, ±0.2)
- VRAM: 11.06 GB (constant)

### Cold-start (8 rounds, fresh KV each time)

- Full prefill: ~198,900 tokens in ~190s (~1047 t/s)
- No VRAM leak detected

### Defrag stress (10 fill→evict cycles)

- Fill speed: ~190s per cycle (stable)
- Evict generation: ~112.6 tok/s
- No peak VRAM drift

### Verdict

Stable but tight. 860 MiB free VRAM means no room to increase context further.
The 11.06 GB peak exceeds the 11.0 GB warning threshold but runs reliably
through all stress phases. Suitable for headless use; on a desktop the
compositor will compete for the remaining VRAM.

Generation speed degrades gracefully from 105 tok/s (short context) to 35 tok/s
(full 200K context). Prefill throughput also drops at very high context fills
due to KV cache pressure.

---

## Qwen3.5-0.8B UD-Q5_K_XL — 230K context

**Script:** `Qwen3.5-0.8B-UD-Q5_K_XL.230k.b8495.sh`
**Date:** 2026-03-31
**Result:** ALL PHASES PASSED

The 0.8B model cannot fit the full 262K native context (OOM), but runs
stably at 230,686 tokens — about 20K more than the 2B model can manage.

### Configuration

```
--ctx-size 230686
--cache-type-k q8_0
--cache-type-v q8_0
--batch-size 1024
--ubatch-size 256
--flash-attn on
--parallel 1
-cram 2048
```

### Memory breakdown (from llama-server log)

```
                          total   free    self   model   context   compute    unaccounted
ROCm0 (RX 6700 XT)      12272 =  640 + (2371 =   568 +    1456 +     346) +        9260
Host                                      426 =   198 +       0 +     227
```

- **Model weights (GPU):** 568 MiB (vs 1,388 MiB for 2B — 820 MiB saved)
- **KV cache:** 1,456 MiB (vs 1,326 MiB for 2B — larger due to higher ctx-size)
- **Compute buffer:** 346 MiB
- **Host (CPU):** 426 MiB (198 model + 227 compute)
- **Free VRAM:** 640 MiB
- **Unaccounted (ROCm overhead):** 9,260 MiB

### Performance

| Context fill | Prefill speed | Gen tok/s | VRAM |
|-------------|--------------|-----------|------|
| 4,000 tokens | 3,820t / 2.6s (~1469 t/s) | 140.6 | 2.53 GB |
| 16,000 tokens | 8,261t / 4.2s (~1967 t/s) | 116.1 | 2.89 GB |
| 57,728 tokens | 26,168t / 14.6s (~1792 t/s) | 76.2 | 4.53 GB |
| 115,456 tokens | 58,172t / 44.7s (~1301 t/s) | 53.6 | 6.94 GB |
| 161,638 tokens | 46,361t / 52.8s (~878 t/s) | 43.4 | 8.60 GB |
| 203,202 tokens | 18,929t / 32.9s (~575 t/s) | 36.8 | 10.42 GB |
| 219,110 tokens | 7,118t / 18.3s (~389 t/s) | 34.8 | 11.28 GB |

### Sustained load (20 rounds at ~219K tokens)

- Generation speed: 34.8 tok/s (rock solid)
- VRAM: 11.28 GB (constant)

### Cold-start (8 rounds, fresh KV each time)

- Full prefill: ~219,092 tokens in ~189s (~1159 t/s)
- No VRAM leak detected

### Defrag stress (10 fill→evict cycles)

- Fill speed: ~189s per cycle (stable)
- Evict generation: ~153.2 tok/s
- No peak VRAM drift

### Verdict

Stable but very tight — only 640 MiB free VRAM. The 0.8B model gains ~20K
extra tokens of context compared to the 2B (230K vs 210K) by saving 820 MiB
on model weights, but the KV cache still dominates. Peak VRAM at 11.28 GB is
well into the warning zone. Headless only.

The 0.8B is notably faster than the 2B at short contexts (141 vs 106 tok/s)
but they converge at full context (~35 tok/s) where KV cache bandwidth
dominates over model weight reads.

Full 262K native context remains out of reach on 12 GB — OOM occurs before
the ramp completes at that size.

### Implication

On a 12 GB card, the practical context ceiling for Qwen3.5 models (q8_0/q8_0
KV cache) is approximately **200-230K tokens** depending on model size. To
reach the full 262K native context you would need either:

- A GPU with more VRAM (16+ GB)
- Aggressive KV cache quantization (q4_0/q4_0), accepting the CPU overhead
  penalty and quality loss documented in the tuning guide
- Reduced parallel slots (already at 1)

---

## Qwen3.5-2B UD-Q5_K_XL — Full Vulkan context

**Script:** `Qwen3.5-2B-UD-Q5_K_XL.FULL.vulkan.sh`
**Date:** not yet stress-tested with the current harness
**Note:** this launcher uses `--ctx-size 262144`

---

## Qwen3.5-4B UD-Q5_K_XL — 82K context

**Script:** `Qwen3.5-4B-UD-Q5_K_XL.82k.b8495.sh`
**Date:** not yet tested

---

## Qwen3.5-9B UD-Q5_K_XL — 52K context (ROCm)

**Script:** `Qwen3.5-9B-UD-Q5_K_XL.thinking.general.52k.b8495.sh`
**Date:** previously verified (see `docs/tuning-guide.md` for measured configurations)

Measured VRAM at 53,248 ctx-size: 10.39 GB per-process.
Generation: ~26.5 tok/s. See tuning guide for full breakdown.

---

## Qwen3.5-9B UD-Q5_K_XL — 66K context, ROCm vs Vulkan

**Date:** 2026-04-01
**Result:** ALL PHASES PASSED (both backends)

This run compares ROCm (Docker container) and Vulkan (local, native) at the same
66,048-token context size using identical stress parameters. Both used per-process
VRAM tracking — ROCm via container PIDs, Vulkan via port-matched local PID.

### Configuration (both backends)

```
--ctx-size 66048
--cache-type-k q8_0
--cache-type-v q8_0
--batch-size 1024
--ubatch-size 256
--flash-attn on
--parallel 1
-cram 2048
```

### Phase 1: Ramp

| ~tokens | ROCm prefill | Vulkan prefill | ROCm gen tok/s | Vulkan gen tok/s | ROCm VRAM | Vulkan VRAM |
|---------|-------------|---------------|---------------|-----------------|-----------|-------------|
| 4,000 | 11.5s | 10.6s | 39.7 | 47.7 | 7.21 GB | 6.99 GB |
| 8,000 | 12.3s | 11.5s | 37.8 | 46.9 | 7.43 GB | 6.99 GB |
| 16,000 | 19.2s | 19.7s | 34.7 | 45.4 | 8.01 GB | 6.99 GB |
| 24,000 | 19.7s | 20.9s | 32.4 | 44.0 | 8.59 GB | 6.99 GB |
| 32,000 | 22.4s | 25.7s | 30.4 | 42.7 | 9.15 GB | 6.99 GB |
| 46,200 | 34.9s | 44.9s | 27.3 | 40.4 | 10.40 GB | 6.99 GB |
| 52,800 | 23.6s | 28.5s | 26.1 | 39.6 | 10.81 GB | 6.99 GB |
| 58,080 | 22.2s | 26.4s | 25.2 | 38.9 | 11.26 GB ⚠ | 6.99 GB |
| 60,720 | 16.9s | 17.6s | 24.8 | 38.5 | 11.50 GB ⚠ | 6.99 GB |
| 62,444 | 14.6s | 13.6s | 24.5 | 38.3 | 11.50 GB ⚠ | 6.99 GB |

### Phase 2: Sustained (20 rounds at ~62,444 tokens)

| Metric | ROCm | Vulkan |
|--------|------|--------|
| Generation speed | 24.5 tok/s (rock solid) | 38.3–38.4 tok/s (rock solid) |
| Round time | ~10.6s | ~6.8s |
| VRAM | 11.50 GB ⚠ (stable) | 6.99 GB (stable) |

### Phase 3: Cold-start (8 rounds, fresh KV each time)

| Metric | ROCm | Vulkan |
|--------|------|--------|
| Full prefill time | ~118s per round | ~162s per round |
| Prefill rate | ~527 tok/s | ~385 tok/s |
| Generation speed | 24.5 tok/s | 38.4 tok/s |
| VRAM leak | None | None |

### Phase 4: Defrag stress (10 fill→evict cycles)

| Metric | ROCm | Vulkan |
|--------|------|--------|
| Fill time per cycle | ~117.9s | ~161.4s |
| Evict gen speed | ~42.1 tok/s | ~48.5–49.2 tok/s |
| Peak VRAM drift | None | None |

### Phase 5: Boundary

Both backends returned clean HTTP 400 for ctx_size+1 request. Server stayed alive.

### Verdict

Both backends pass all phases. Key trade-offs:

- **Generation speed:** Vulkan wins decisively (+20% at short context, +57% at 62k tokens)
- **Prefill speed:** ROCm wins for large batches (+37% for a full 62k cold prefill)
- **VRAM:** Vulkan uses 6.99 GB flat at any context fill; ROCm grows to 11.50 GB at 62k
- **Desktop safety:** Vulkan is safe for desktop use at 66k context; ROCm is headless-only

Vulkan is the preferred backend for the 9B model on the RX 6700 XT for interactive use.
Its flat VRAM profile also means it can reach near-full native context within 12 GB — see
the 262k result below.

---

## Qwen3.5-9B UD-Q5_K_XL — 262K context, Vulkan (QUICK=1)

**Script:** `Qwen3.5-9B-UD-Q5_K_XL.FULL.vulkan.sh`
**Date:** 2026-04-01
**Result:** ALL PHASES PASSED
**Flags:** `QUICK=1 CTX_SIZE=262144` (sustained=3, cold=1, defrag=1)

### Configuration

```
--ctx-size 262144
--cache-type-k q8_0
--cache-type-v q8_0
--batch-size 1024
--ubatch-size 256
--flash-attn on
--parallel 1
-cram 2048
```

### VRAM

**10.31 GB flat** — Vulkan pre-commits the entire KV cache at startup. VRAM does not
grow as context fills. ~1.7 GB headroom at all times.

The ~3.32 GB increase vs 66k baseline (6.99 GB → 10.31 GB) is the additional KV cache
for 262k - 66k = 196k tokens × 8 attention layers × q8_0.

### Phase 1: Ramp

| ~tokens | Prefill time | Gen tok/s | VRAM |
|---------|-------------|-----------|------|
| 4,000 | 13.1s | 46.9 | 10.31 GB |
| 8,000 | 13.7s | 46.1 | 10.31 GB |
| 16,000 | 24.0s | 43.8 | 10.31 GB |
| 32,000 | 30.6s | 39.9 | 10.31 GB |
| 65,536 | 138.1s | 33.6 | 10.31 GB |
| 131,072 | 419.3s | 25.7 | 10.31 GB |
| 183,500 | 442.8s | 21.6 | 10.31 GB |
| 209,715 | 254.0s | 20.0 | 10.31 GB |
| 230,686 | 222.3s | 18.9 | 10.31 GB |
| 241,172 | 123.9s | 18.4 | 10.31 GB |
| 248,780 | 99.2s | 18.0 | 10.31 GB |

### Phases 2–4 (at ~248,780 tokens)

| Phase | Result |
|-------|--------|
| Sustained (3 rounds) | 18.0 tok/s, 10.31 GB stable |
| Cold-start (1 round) | 248,429t in **1,689s** (~147 tok/s), no leak |
| Defrag fill | 248,430t in 1,690s, no VRAM drift |
| Defrag evict gen | 49.2 tok/s (empty context, fast) |

### Verdict

The 9B model at full 262k native context is stable on Vulkan with 10.31 GB VRAM.
Generation degrades from 47 tok/s (4k context) to 18 tok/s (248k context) — graceful
and predictable.

The cold-start prefill time (~28 min for 248k tokens) makes full stress-testing
impractical; `QUICK=1` is required. The defrag evict at 49.2 tok/s confirms the
model is compute-bound at small context even at 262k allocation.

ROCm cannot reach this context size on 12 GB — it would exceed VRAM before the
KV cache for 262k context could fit (projected ~14+ GB per-process).
