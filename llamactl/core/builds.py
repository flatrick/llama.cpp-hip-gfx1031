"""Build subsystem: source-ref resolution, snapshots, image/native builds.

Reimplements the legacy build.*.sh flow in Python. Short git/podman calls use
the blocking injected Runner (runtime.py); long build commands use StreamRunner;
the snapshot pipe uses Extractor. core never imports UI.
"""

from __future__ import annotations

import re

from llamactl.core.lifecycle import DEFAULT_ROCM_IMAGE, DEFAULT_VULKAN_IMAGE

_BUILD_TAG_RE = re.compile(r"^b(\d+)$")

ROCM_IMAGE_PREFIX = DEFAULT_ROCM_IMAGE.split(":")[0]      # "llama-cpp-gfx1031"
VULKAN_IMAGE_PREFIX = DEFAULT_VULKAN_IMAGE.split(":")[0]  # "llama-cpp-vulkan"
_IMAGE_PREFIXES = {"rocm-image": ROCM_IMAGE_PREFIX, "vulkan-image": VULKAN_IMAGE_PREFIX}


def _sanitize_tag(ref: str) -> str:
    """Mirror build.llama-ref.docker-rocm.sh sanitize_tag:
    replace [/:@ ] with '-', drop anything but [A-Za-z0-9._-], then strip
    any leading/trailing dashes produced by the substitution.

    The two passes are order-dependent: separator→dash MUST run before dropping
    disallowed chars so that '/' becomes '-' rather than being silently removed.
    Output is ASCII-only, satisfying Docker's tag character requirements."""
    s = re.sub(r"[/:@ ]", "-", ref)
    s = re.sub(r"[^A-Za-z0-9._-]", "", s)
    return s.strip("-")


def image_tag_for(target: str, ref: str) -> str:
    try:
        prefix = _IMAGE_PREFIXES[target]
    except KeyError:
        raise BuildError(f"not an image target: {target!r}") from None
    tag = _sanitize_tag(ref)
    if not tag:
        raise BuildError(f"ref {ref!r} sanitizes to an empty image tag")
    return f"{prefix}:{tag}"


class BuildError(Exception):
    """A build step failed (resolution, fetch, snapshot, compile, or registry)."""


def _select_latest_build_tag(ls_remote_output: str) -> tuple[str, str]:
    """Parse `git ls-remote --tags` output; return (tag, sha) of the highest
    b<NNNN> tag selected NUMERICALLY (lexical would order b9 after b1000).
    Tags containing but not equal to b<NNNN> (ci_*, fix-*-bNNNN-*) are rejected.
    """
    best_n = -1
    best_tag = ""
    best_sha = ""
    for line in ls_remote_output.splitlines():
        parts = line.split("\t")
        if len(parts) != 2:
            continue
        sha, ref = parts[0].strip(), parts[1].strip()
        if ref.endswith("^{}"):
            ref = ref[:-3]
        if not ref.startswith("refs/tags/"):
            continue
        name = ref[len("refs/tags/"):]
        m = _BUILD_TAG_RE.match(name)
        if not m:
            continue
        n = int(m.group(1))
        if n > best_n:
            best_n, best_tag, best_sha = n, name, sha
    if not best_tag:
        raise BuildError("no build tag (b<NNNN>) found in remote tags")
    return best_tag, best_sha
