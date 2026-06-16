"""Build subsystem: source-ref resolution, snapshots, image/native builds.

Reimplements the legacy build.*.sh flow in Python. Short git/podman calls use
the blocking injected Runner (runtime.py); long build commands use StreamRunner;
the snapshot pipe uses Extractor. core never imports UI.
"""

from __future__ import annotations

import re
import shutil
import subprocess
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

from llamactl.core.lifecycle import DEFAULT_ROCM_IMAGE, DEFAULT_VULKAN_IMAGE
from llamactl.core.runtime import Runner, _default_runner

_BUILD_TAG_RE = re.compile(r"^b(\d+)$")

LLAMA_CPP_REMOTE = "https://github.com/ggml-org/llama.cpp.git"

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
        peeled = ref.endswith("^{}")
        if peeled:
            ref = ref[:-3]
        if not ref.startswith("refs/tags/"):
            continue
        name = ref[len("refs/tags/"):]
        m = _BUILD_TAG_RE.match(name)
        if not m:
            continue
        n = int(m.group(1))
        # Prefer the peeled (^{}) line's sha: for an annotated tag it is the
        # commit we build/checkout, not the tag object.
        if n > best_n or (n == best_n and peeled):
            best_n, best_tag, best_sha = n, name, sha
    if not best_tag:
        raise BuildError("no build tag (b<NNNN>) found in remote tags")
    return best_tag, best_sha


@dataclass(frozen=True)
class ResolvedRef:
    sha: str
    build_number: str   # "" when not a b<NNNN> tag
    fetch_spec: str      # ref/sha to `git fetch`; "" for submodule
    display: str


def _sha_from_ls_remote(output: str) -> str:
    """First field of the (preferably peeled ^{}) line; '' if none."""
    plain = ""
    for line in output.splitlines():
        parts = line.split("\t")
        if len(parts) != 2:
            continue
        sha, ref = parts[0].strip(), parts[1].strip()
        if ref.endswith("^{}"):
            return sha
        if not plain:
            plain = sha
    return plain


def resolve_ref(
    ref: str,
    cache_dir: Path,
    submodule_dir: Path,
    runner: Runner = _default_runner,
) -> ResolvedRef:
    if ref == "submodule":
        res = runner(["git", "-C", str(submodule_dir), "rev-parse", "HEAD"])
        if res.returncode != 0:
            raise BuildError(f"submodule rev-parse failed: {res.stderr}")
        return ResolvedRef(res.stdout.strip(), "", "", "submodule")

    if ref == "latest-tag":
        res = runner(["git", "ls-remote", "--tags", LLAMA_CPP_REMOTE])
        if res.returncode != 0:
            raise BuildError(f"ls-remote failed: {res.stderr}")
        tag, sha = _select_latest_build_tag(res.stdout)
        return ResolvedRef(sha, tag, f"refs/tags/{tag}", f"latest-tag ({tag})")

    if ref.startswith("tag:"):
        name = ref[len("tag:"):]
        res = runner(["git", "ls-remote", LLAMA_CPP_REMOTE, f"refs/tags/{name}"])
        if res.returncode != 0:
            raise BuildError(f"ls-remote failed: {res.stderr}")
        sha = _sha_from_ls_remote(res.stdout)
        if not sha:
            raise BuildError(f"tag '{name}' not found on remote")
        build_number = name if _BUILD_TAG_RE.match(name) else ""
        return ResolvedRef(sha, build_number, f"refs/tags/{name}", ref)

    if ref.startswith("branch:"):
        name = ref[len("branch:"):]
        res = runner(["git", "ls-remote", LLAMA_CPP_REMOTE, f"refs/heads/{name}"])
        if res.returncode != 0:
            raise BuildError(f"ls-remote failed: {res.stderr}")
        sha = _sha_from_ls_remote(res.stdout)
        if not sha:
            raise BuildError(f"branch '{name}' not found on remote")
        return ResolvedRef(sha, "", name, ref)

    if ref.startswith("commit:"):
        sha = ref[len("commit:"):]
        if not sha:
            raise BuildError("commit ref is empty")
        return ResolvedRef(sha, "", sha, ref)

    raise BuildError(f"unrecognized ref spec: {ref!r}")


# StreamRunner: long commands. Yields stdout lines; raises BuildError on nonzero exit.
StreamRunner = Callable[[list[str], "Path | None"], Iterator[str]]
# Extractor: runs archive_cmd | extract_cmd into dest.
Extractor = Callable[[list[str], list[str], Path], None]


def _default_stream_runner(cmd: list[str], cwd: "Path | None" = None) -> Iterator[str]:
    proc = subprocess.Popen(
        cmd, cwd=cwd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
        text=True, bufsize=1,
    )
    assert proc.stdout is not None
    try:
        for line in proc.stdout:
            yield line.rstrip("\n")
    finally:
        proc.stdout.close()
        proc.wait()
    if proc.returncode != 0:
        raise BuildError(f"command failed (exit {proc.returncode}): {' '.join(cmd)}")


def _default_extractor(archive_cmd: list[str], extract_cmd: list[str], dest: Path) -> None:
    p1 = subprocess.Popen(archive_cmd, stdout=subprocess.PIPE)
    p2 = subprocess.Popen(extract_cmd, stdin=p1.stdout)
    if p1.stdout is not None:
        p1.stdout.close()  # allow p1 to receive SIGPIPE if p2 exits
    p2.communicate()
    p1.wait()
    if p1.returncode != 0 or p2.returncode != 0:
        raise BuildError(
            f"git archive | tar extraction failed "
            f"(git={p1.returncode}, tar={p2.returncode})"
        )


def _ensure_cache(cache_dir: Path, stream_runner: StreamRunner) -> None:
    if (cache_dir / ".git").exists():
        return
    cache_dir.parent.mkdir(parents=True, exist_ok=True)
    cmd = ["git", "clone", "--filter=blob:none", "--no-checkout",
           LLAMA_CPP_REMOTE, str(cache_dir)]
    for _ in stream_runner(cmd, None):
        pass


def export_snapshot(
    ref_spec: str,
    resolved: ResolvedRef,
    cache_dir: Path,
    submodule_dir: Path,
    dest: Path,
    stream_runner: StreamRunner = _default_stream_runner,
    extractor: Extractor = _default_extractor,
) -> None:
    if ref_spec == "submodule":
        src = submodule_dir
    else:
        _ensure_cache(cache_dir, stream_runner)
        for _ in stream_runner(
            ["git", "-C", str(cache_dir), "fetch", "--depth", "1",
             "origin", resolved.fetch_spec], None,
        ):
            pass
        src = cache_dir
    archive_cmd = ["git", "-C", str(src), "archive", "--format=tar", resolved.sha]
    extract_cmd = ["tar", "-x", "-C", str(dest)]
    dest.mkdir(parents=True, exist_ok=True)
    extractor(archive_cmd, extract_cmd, dest)


@dataclass(frozen=True)
class ToolchainStatus:
    ok: bool
    missing: tuple[str, ...] = ()


def detect_native_toolchain(
    target: str,
    which: Callable[[str], "str | None"] = shutil.which,
    path_exists: Callable[[str], bool] = lambda p: Path(p).exists(),
) -> ToolchainStatus:
    """rocm-native: cmake/ninja/hipcc + /opt/rocm + hipblas headers + libcurl dev.
    vulkan-native: cmake/ninja/glslc + vulkan headers + libcurl dev."""
    missing: list[str] = []

    def need_bin(name: str) -> None:
        if which(name) is None:
            missing.append(name)

    if target == "rocm-native":
        for b in ("cmake", "ninja", "hipcc"):
            need_bin(b)
        if not path_exists("/opt/rocm"):
            missing.append("/opt/rocm")
        hipblas_headers = ("/opt/rocm/include/hipblas/hipblas.h",
                           "/opt/rocm/include/hipblas.h")
        if not any(path_exists(p) for p in hipblas_headers):
            missing.append("hipblas-dev")
    elif target == "vulkan-native":
        for b in ("cmake", "ninja", "glslc"):
            need_bin(b)
        if not path_exists("/usr/include/vulkan/vulkan.h"):
            missing.append("libvulkan-dev")
    else:
        raise BuildError(f"not a native target: {target}")

    if not path_exists("/usr/include/curl/curl.h"):
        missing.append("libcurl-dev")

    return ToolchainStatus(ok=not missing, missing=tuple(missing))


def build_image(
    target: str,
    context_dir: Path,
    image_tag: str,
    repo_root: Path,
    runtime: str,
    stream_runner: StreamRunner = _default_stream_runner,
) -> Iterator[str]:
    dockerfile = "Dockerfile.rocm" if target == "rocm-image" else "Dockerfile.vulkan"
    cmd = [runtime, "build", "-f", str(repo_root / dockerfile),
           "-t", image_tag, str(context_dir)]
    yield from stream_runner(cmd, None)
