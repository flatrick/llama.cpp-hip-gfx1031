# Building llama-server images

Notes on the ROCm and Vulkan container builds, captured so recurring gotchas
don't have to be rediscovered. Both images compile `llama-server` from a
pinned/selected `llama.cpp` source snapshot, via `Dockerfile.rocm` /
`Dockerfile.vulkan` (used by both `llamactl`'s Builds tab and the
`build.docker-*.sh` scripts).

## Build dependencies track the source revision

The build-time dependencies in the Dockerfiles **must match whatever the source
revision requires**. Bumping the source (`latest-tag`, a newer `tag:bNNNN`, …)
can introduce new build-time dependencies that an older snapshot didn't need.

### Vulkan requires `spirv-headers`

Recent `llama.cpp` (b9680 and newer) added this to
`ggml/src/ggml-vulkan/CMakeLists.txt`:

```cmake
find_package(SPIRV-Headers CONFIG REQUIRED)
```

Without the `spirv-headers` package, CMake configure aborts before compiling:

```
CMake Error at ggml/src/ggml-vulkan/CMakeLists.txt:14 (find_package):
  Could not find a package configuration file provided by "SPIRV-Headers"
```

`Dockerfile.vulkan` installs `spirv-headers` in the builder stage to satisfy this
(matching upstream's own `.devops/vulkan.Dockerfile`). The ROCm build is
unaffected because it doesn't build the Vulkan backend (`-DGGML_HIP=ON`, no
`-DGGML_VULKAN=ON`), so it never calls `find_package(SPIRV-Headers)`.

> If a Vulkan build suddenly fails at configure after bumping the pinned source,
> look for a new `find_package(... REQUIRED)` in the ggml-vulkan CMake and add the
> matching Ubuntu package to `Dockerfile.vulkan`. The native-build toolchain check
> lives in `detect_native_toolchain()` in `llamactl/core/builds.py`.

## Docker layer caching ("why did the build skip?")

Rebuilding an image from an **unchanged** source ref finishes in ~1 second with
every step shown as `CACHED` — including the expensive
`RUN cmake ... && cmake --build` layer. This is correct: the source snapshot is
byte-identical (it is produced with `git archive`, which yields deterministic file
contents and timestamps), so Docker reuses the already-compiled layer instead of
recompiling.

This is **not** cross-contamination between images. The ROCm and Vulkan images
share no layers — different base image and different build commands — so a ROCm
build can never seed a Vulkan build's cache. You can confirm an image is what it
claims to be:

| | Vulkan | ROCm |
|---|---|---|
| Base | `ubuntu:24.04` | `rocm/dev-ubuntu-24.04` |
| Size | ~700 MB (slim, multi-stage) | ~12 GB (single-stage) |
| Backend lib | `libggml-vulkan.so` → `libvulkan.so.1` | `libggml-hip.so` → `libamdhip64`/`rocblas` |

```bash
# what backend is actually inside an image?
docker run --rm --entrypoint ldd llama-cpp-vulkan:<tag> /usr/local/bin/llama-server | grep -iE 'vulkan|hip'
```

The Vulkan image is small because `Dockerfile.vulkan` is multi-stage: the compile
happens in a `builder` stage that is discarded, leaving only the binary and its
`.so`s. That can also make a build *look* like "nothing happened."

## Forcing a clean rebuild

To recompile the same ref from scratch (e.g. to rule out a stale cache), bypass
the layer cache:

- **llamactl Builds tab:** tick **Force rebuild (no cache)** before pressing Build.
  This passes `--no-cache` to the build (`BuildRequest(no_cache=True)` →
  `build_image(..., no_cache=True)`). It applies to image builds; native builds
  already run in a fresh temporary context every time, so they are never cached.
- **Shell scripts:** `build.docker-vulkan.sh --force` / `build.docker-rocm.sh --force`
  rebuild even if the image already exists and add `--no-cache`.
- **Raw docker/podman:** `docker build --no-cache -f Dockerfile.vulkan -t <tag> <context>`.
