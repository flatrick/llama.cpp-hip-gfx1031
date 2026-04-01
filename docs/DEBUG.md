# Debugging llama-server on ROCm

---

## Memory Usage

When llama-server reports its memory breakdown, a large portion of VRAM can show
up as "unaccounted" — not tracked by llama.cpp but consumed by the ROCm/HIP
runtime. On the RX 6700 XT, the main cause we have now confirmed is KV-cache
dequantization: if `--cache-type-k` or `--cache-type-v` is anything other than
`f16`, ROCm keeps dequantized copies of both K and V and those buffers live in
the HIP caching pool. In the tested `q8_0` runs this adds roughly 8–9 GB of
runtime-managed memory depending on model and context size.

The tools below help trace where that memory goes and confirm whether you are
seeing the same behavior on a different model or setup.

### Quick: GEM buffer objects (host-side, no container changes)

The kernel's GEM (Graphics Execution Manager) tracks every VRAM allocation as a
buffer object. Dump it from the **host** while the server is running:

```bash
sudo cat /sys/kernel/debug/dri/0/amdgpu_gem_info
```

This lists every allocation with its size and placement (VRAM vs GTT). Compare
dumps between two models to see where the extra memory goes.

> **Note:** debugfs must be mounted (it is by default on Arch/EndeavourOS). The
> card number may differ — check `/sys/kernel/debug/dri/*/name` to find the
> right one.

### Quick: sysfs VRAM counters (no root required)

Poll total VRAM usage from sysfs:

```bash
# One-shot
cat /sys/class/drm/card0/device/mem_info_vram_used | numfmt --to=iec

# Live monitoring
watch -n 1 'cat /sys/class/drm/card0/device/mem_info_vram_used | numfmt --to=iec'
```

Available counters at `/sys/class/drm/card0/device/`:

| File | What it shows |
|------|---------------|
| `mem_info_vram_total` | Total VRAM in bytes |
| `mem_info_vram_used` | Currently used VRAM in bytes |
| `mem_info_vis_vram_total` | CPU-visible VRAM total |
| `mem_info_vis_vram_used` | CPU-visible VRAM used |
| `mem_info_gtt_total` | GTT (system memory mapped for GPU) total |
| `mem_info_gtt_used` | GTT used |

### Medium: HIP API tracing via environment variables

Trace all HIP memory API calls (hipMalloc, hipFree, etc.) by passing
environment variables to the container:

```bash
docker run --rm \
  --device /dev/kfd --device /dev/dri \
  --group-add video --group-add render \
  -e AMD_LOG_LEVEL=4 \
  -e AMD_LOG_MASK=0x401 \
  -e AMD_LOG_LEVEL_FILE=/tmp/vram_trace.log \
  -v /tmp:/tmp \
  "$ROCM_IMAGE" \
    -hf "unsloth/Qwen3.5-2B-GGUF:UD-Q5_K_XL" \
    ...
```

The log at `/tmp/vram_trace.log` will contain every HIP API call with arguments,
including allocation sizes. Filter for memory calls:

```bash
grep -i 'malloc\|alloc\|free\|memcpy' /tmp/vram_trace.log
```

`AMD_LOG_MASK` values (combine with bitwise OR):

| Bit | Value | What it logs |
|-----|-------|-------------|
| 0 | `0x1` | API calls (hipMalloc, hipFree, etc.) |
| 1 | `0x2` | Kernel and copy commands |
| 10 | `0x400` | Resource allocation, performance-impacting events |
| 11 | `0x800` | Initialization and shutdown |

> **Note:** `HIP_TRACE_API` is deprecated and removed in current ROCm. Use
> `AMD_LOG_LEVEL` + `AMD_LOG_MASK` instead.

### Detailed: rocprofv3 memory allocation trace

If rocprofv3 is available in the container (it ships with ROCm), this gives the
most detailed allocation timeline:

```bash
rocprofv3 --memory-allocation-trace --output-format csv \
  -d /tmp/profile \
  -- llama-server \
    -hf "unsloth/Qwen3.5-2B-GGUF:UD-Q5_K_XL" \
    ...
```

This generates `memory_allocation_trace.csv` with columns:

| Column | Description |
|--------|-------------|
| Kind | `MEMORY_ALLOCATION` |
| Operation | `MEMORY_ALLOCATE` or `MEMORY_FREE` |
| Agent_Id | Which GPU |
| Allocation_Size | Size in bytes |
| Address | Pointer address |
| Start_Timestamp | When the allocation happened |
| End_Timestamp | When it completed |

For even more detail, use `--sys-trace` which adds HSA-level API traces:

```bash
rocprofv3 --sys-trace --output-format csv -d /tmp/profile -- llama-server ...
# Produces: hip_api_trace.csv, hsa_api_trace.csv, kernel_trace.csv,
#           memory_copy_trace.csv, scratch_memory_trace.csv,
#           memory_allocation_trace.csv
```

### Advanced: LD_PRELOAD hipMalloc interceptor

For custom tracking (running totals, peak detection, stack traces), intercept
HIP allocation calls with a preloaded shim:

```c
// hip_trace_preload.c
#define _GNU_SOURCE
#include <dlfcn.h>
#include <stdio.h>
#include <stdint.h>

static size_t total_allocated = 0;
static size_t peak_allocated = 0;

typedef int (*hipMalloc_fn)(void**, size_t);
typedef int (*hipFree_fn)(void*);

int hipMalloc(void** ptr, size_t size) {
    hipMalloc_fn real = (hipMalloc_fn)dlsym(RTLD_NEXT, "hipMalloc");
    int ret = real(ptr, size);
    total_allocated += size;
    if (total_allocated > peak_allocated) peak_allocated = total_allocated;
    fprintf(stderr, "[TRACE] hipMalloc(%zu bytes) = %p  total=%zu peak=%zu\n",
            size, *ptr, total_allocated, peak_allocated);
    return ret;
}

int hipFree(void* ptr) {
    hipFree_fn real = (hipFree_fn)dlsym(RTLD_NEXT, "hipFree");
    fprintf(stderr, "[TRACE] hipFree(%p)\n", ptr);
    return real(ptr);
}
```

Build and use:

```bash
gcc -shared -fPIC -o hip_trace_preload.so hip_trace_preload.c -ldl
LD_PRELOAD=./hip_trace_preload.so llama-server ...
```

> **Alternative without writing code:** `ltrace -C -e "hip*" llama-server ...`
> shows all HIP API calls with arguments. For lower-level HSA allocations:
> `ltrace -C -e "hsa*" llama-server ...` which shows `hsaKmtAllocMemory`
> calls with exact sizes.

### Comparing two models

To diff VRAM allocation patterns between models (e.g., investigating why the
0.8B uses 880 MiB more ROCm overhead than the 2B despite being smaller):

1. Start model A, wait for it to be ready
2. Dump `amdgpu_gem_info` and/or run the rocprofv3 trace
3. Stop model A
4. Repeat for model B
5. Diff the allocation lists — look for allocations that exist in one but not
   the other, or that are significantly larger

The GEM info dump is the quickest comparison. The rocprofv3 trace gives a
timeline that can reveal *when* during startup the extra allocations happen
(model load vs KV cache init vs first inference).
