#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_URL="https://github.com/ggml-org/llama.cpp.git"
REF="latest"
IMAGE_PREFIX="llama-cpp-vulkan"
IMAGE_TAG=""
FORCE=0
KEEP_WORKTREE=0
WORKDIR=""

usage() {
  cat <<'EOF'
Usage: build.llama-ref.docker-vulkan.sh [--ref REF] [--image-tag TAG] [--force] [--keep-worktree]

Fetch an upstream llama.cpp ref into a temporary checkout and build it as a
separate Vulkan image without modifying the pinned local submodule.

Options:
  --ref REF         Git ref to test (default: latest upstream default branch)
  --image-tag TAG   Docker tag suffix (default: derived from ref/commit)
  --force, -f       Rebuild even if the image already exists
  --keep-worktree   Keep the temporary checkout under /tmp after the build
  --help, -h        Show this help
EOF
}

sanitize_tag() {
  printf '%s' "$1" | tr '/:@ ' '-' | tr -cd '[:alnum:]._-\n'
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --ref)
      [[ $# -ge 2 ]] || { echo "ERROR: --ref requires a value" >&2; exit 1; }
      REF="$2"
      shift 2
      ;;
    --image-tag)
      [[ $# -ge 2 ]] || { echo "ERROR: --image-tag requires a value" >&2; exit 1; }
      IMAGE_TAG="$2"
      shift 2
      ;;
    --force|-f)
      FORCE=1
      shift
      ;;
    --keep-worktree)
      KEEP_WORKTREE=1
      shift
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

WORKDIR="$(mktemp -d /tmp/llama.cpp-ref.XXXXXX)"
cleanup() {
  if [[ "$KEEP_WORKTREE" -eq 0 ]]; then
    rm -rf "$WORKDIR"
  else
    echo "Kept temporary worktree at $WORKDIR"
  fi
}
trap cleanup EXIT

echo "Cloning llama.cpp into $WORKDIR"
git clone --depth 1 "$REPO_URL" "$WORKDIR/llama.cpp-src"

if [[ "$REF" != "latest" ]]; then
  git -C "$WORKDIR/llama.cpp-src" fetch --depth 1 origin "$REF"
  git -C "$WORKDIR/llama.cpp-src" checkout --detach FETCH_HEAD
fi

COMMIT_LABEL="$(git -C "$WORKDIR/llama.cpp-src" describe --always --tags 2>/dev/null || git -C "$WORKDIR/llama.cpp-src" rev-parse --short HEAD)"
if [[ -z "$IMAGE_TAG" ]]; then
  if [[ "$REF" == "latest" ]]; then
    IMAGE_TAG="latest-test-$(sanitize_tag "$COMMIT_LABEL")"
  else
    IMAGE_TAG="$(sanitize_tag "$REF")"
  fi
fi

IMAGE_NAME="$IMAGE_PREFIX:$IMAGE_TAG"
BUILD_ARGS=(--image "$IMAGE_NAME" --src-dir "$WORKDIR/llama.cpp-src")
if [[ "$FORCE" -eq 1 ]]; then
  BUILD_ARGS+=(--force)
fi

"$SCRIPT_DIR/build.docker-vulkan.sh" "${BUILD_ARGS[@]}"

cat <<EOF

Built image: $IMAGE_NAME

Run an existing launcher against it with:
  VULKAN_IMAGE=$IMAGE_NAME bash Qwen3.5-2B-UD-Q5_K_XL.200k.vulkan.sh

EOF
