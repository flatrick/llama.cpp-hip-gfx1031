#!/usr/bin/env bash
set -e

ROCM_IMAGE="llama-cpp-gfx1031:latest"
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
SRC_DIR="$SCRIPT_DIR/llama.cpp-src"
FORCE=0

for arg in "$@"; do
  case "$arg" in
    --force|-f) FORCE=1 ;;
    *) echo "Unknown argument: $arg" >&2; exit 1 ;;
  esac
done

# Ensure the llama.cpp submodule (pinned to b8495) is present.
# On a fresh clone, run: git submodule update --init
if [ ! -f "$SRC_DIR/.git" ] && [ ! -d "$SRC_DIR/.git" ]; then
  echo "Initializing llama.cpp submodule..."
  git -C "$SCRIPT_DIR" submodule update --init --recursive || { echo "ERROR: submodule init failed" >&2; exit 1; }
fi

# Build image
RUNTIME=$(command -v podman || command -v docker)
if [ -z "$RUNTIME" ]; then
  echo "ERROR: neither podman nor docker found in PATH" >&2
  exit 1
fi

if [ "$FORCE" -eq 0 ] && "$RUNTIME" image inspect "$ROCM_IMAGE" > /dev/null 2>&1; then
  echo "Image $ROCM_IMAGE already exists. Use --force to rebuild."
  exit 0
fi

echo "Building $ROCM_IMAGE from Dockerfile.rocm (this takes ~15-20 min)..."
BUILD_FLAGS=()
[ "$FORCE" -eq 1 ] && BUILD_FLAGS+=(--no-cache)
"$RUNTIME" build "${BUILD_FLAGS[@]}" -f "$SCRIPT_DIR/Dockerfile.rocm" -t "$ROCM_IMAGE" "$SCRIPT_DIR"
echo "Done. Image $ROCM_IMAGE is ready."
