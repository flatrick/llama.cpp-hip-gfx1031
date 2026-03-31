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

## Qwen3.5-9B UD-Q5_K_XL — 52K context

**Script:** `Qwen3.5-9B-UD-Q5_K_XL.thinking.general.52k.b8495.sh`
**Date:** previously verified (see `docs/tuning-guide.md` for measured configurations)

Measured VRAM at 53,248 ctx-size: 10.39 GB per-process, ~10.64 GB system-wide.
Generation: ~26.5 tok/s. See tuning guide for full breakdown.
