#!/usr/bin/env bash
set -euo pipefail

VULKAN_IMAGE="llama-cpp-vulkan:latest"
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
SRC_DIR="$SCRIPT_DIR/llama.cpp-src"
FORCE=0

usage() {
  cat <<'EOF'
Usage: build.docker-vulkan.sh [--force] [--image IMAGE] [--src-dir PATH]

Build the Vulkan llama.cpp image from a local source checkout.

Options:
  --force, -f     Rebuild even if the image already exists
  --image IMAGE   Target image tag (default: llama-cpp-vulkan:latest)
  --src-dir PATH  Source checkout to package (default: ./llama.cpp-src)
  --help, -h      Show this help
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --force|-f)
      FORCE=1
      shift
      ;;
    --image)
      [[ $# -ge 2 ]] || { echo "ERROR: --image requires a value" >&2; exit 1; }
      VULKAN_IMAGE="$2"
      shift 2
      ;;
    --src-dir)
      [[ $# -ge 2 ]] || { echo "ERROR: --src-dir requires a value" >&2; exit 1; }
      SRC_DIR="$2"
      shift 2
      ;;
    --help|-h)
      usage
      exit 0
      ;;
    *)
      echo "Unknown argument: $1" >&2
      usage >&2
      exit 1
      ;;
  esac
done

# Ensure the default llama.cpp submodule checkout is present when using it.
if [[ "$SRC_DIR" == "$SCRIPT_DIR/llama.cpp-src" ]] && [ ! -f "$SRC_DIR/.git" ] && [ ! -d "$SRC_DIR/.git" ]; then
  echo "Initializing llama.cpp submodule..."
  git -C "$SCRIPT_DIR" submodule update --init --recursive || { echo "ERROR: submodule init failed" >&2; exit 1; }
fi

if [ ! -d "$SRC_DIR" ] || [ ! -f "$SRC_DIR/CMakeLists.txt" ]; then
  echo "ERROR: source directory '$SRC_DIR' is not a llama.cpp checkout" >&2
  exit 1
fi

# Build image
RUNTIME=$(command -v podman || command -v docker)
if [ -z "$RUNTIME" ]; then
  echo "ERROR: neither podman nor docker found in PATH" >&2
  exit 1
fi

if [ "$FORCE" -eq 0 ] && "$RUNTIME" image inspect "$VULKAN_IMAGE" > /dev/null 2>&1; then
  echo "Image $VULKAN_IMAGE already exists. Use --force to rebuild."
  exit 0
fi

BUILD_CONTEXT="$(mktemp -d)"
cleanup() {
  rm -rf "$BUILD_CONTEXT"
}
trap cleanup EXIT

mkdir -p "$BUILD_CONTEXT/llama.cpp-src"
cp "$SCRIPT_DIR/Dockerfile.vulkan" "$BUILD_CONTEXT/Dockerfile.vulkan"

# Package the selected working tree without copying .git metadata or prior builds.
tar \
  --exclude=.git \
  --exclude=build \
  -C "$SRC_DIR" \
  -cf - . | tar -C "$BUILD_CONTEXT/llama.cpp-src" -xf -

echo "Building $VULKAN_IMAGE from Dockerfile.vulkan using source in $SRC_DIR..."
BUILD_FLAGS=()
[ "$FORCE" -eq 1 ] && BUILD_FLAGS+=(--no-cache)
"$RUNTIME" build "${BUILD_FLAGS[@]}" -f "$BUILD_CONTEXT/Dockerfile.vulkan" -t "$VULKAN_IMAGE" "$BUILD_CONTEXT"
echo "Done. Image $VULKAN_IMAGE is ready."
