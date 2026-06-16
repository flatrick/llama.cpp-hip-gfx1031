# llamactl Phase 3 — Builds Tab Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a Builds tab that turns a llama.cpp source ref into a container image or native binary, recorded in the artifact registry and selectable from the Serve tab's launch picker.

**Architecture:** A new testable `llamactl/core/builds.py` (no Textual imports) reimplements the build flow in Python using the established injected-`Runner` pattern for short git/podman calls, a new `StreamRunner` for the long `podman build`/`cmake` commands, and an `Extractor` for the `git archive | tar` snapshot pipe. A `llamactl/ui/screens/builds.py` Textual screen drives it via a thread worker, streaming logs into a `RichLog` and listing artifacts in a `DataTable`. The Serve launch form gains an artifact dropdown fed from the same registry.

**Tech Stack:** Python 3.14, Textual, tomlkit/tomllib, pytest + pytest-asyncio, podman/docker, git, cmake/ninja.

**Design basis:** `docs/superpowers/specs/2026-06-16-llamactl-phase3-builds-tab-design.md`

---

## File Structure

**Create:**
- `llamactl/core/builds.py` — build subsystem: ref resolution, snapshot export, image/native builders, toolchain detection, `run_build` orchestrator, artifact delete + in-use helpers.
- `llamactl/ui/screens/builds.py` — `BuildsScreen` widget (request form, streaming log, artifact table).
- `tests/llamactl/test_builds.py` — core unit tests (fake runners).
- `tests/llamactl/test_builds_ui.py` — Textual Pilot smoke tests.

**Modify:**
- `llamactl/ui/app.py:52-53` — replace the Builds placeholder `Static` with `BuildsScreen`.
- `llamactl/ui/screens/serve.py` — add an artifact `Select` to `_LaunchForm` and pass the chosen `image_tag` as the `resolve_image` override.

All build state lives under the gitignored `state/` (`state/src-cache/`, `state/builds/`, `state/registry.toml`).

---

## Task 1: Build tag selection (pure)

The `latest-tag` resolver must pick the numerically-highest `b<NNNN>` tag and reject decoys. This is pure string logic — start here.

**Files:**
- Create: `llamactl/core/builds.py`
- Test: `tests/llamactl/test_builds.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/llamactl/test_builds.py
from __future__ import annotations

import subprocess
from pathlib import Path

import pytest

from llamactl.core.builds import BuildError, _select_latest_build_tag


_LS_REMOTE = "\n".join([
    "1111111111111111111111111111111111111111\trefs/tags/b9",
    "2222222222222222222222222222222222222222\trefs/tags/b100",
    "3333333333333333333333333333333333333333\trefs/tags/b1000",
    "4444444444444444444444444444444444444444\trefs/tags/b99",
    "5555555555555555555555555555555555555555\trefs/tags/ci_cublas-31ff9e2",
    "6666666666666666666666666666666666666666\trefs/tags/fix-release-b7083-foo",
])


def test_select_latest_build_tag_is_numeric_not_lexical():
    tag, sha = _select_latest_build_tag(_LS_REMOTE)
    assert tag == "b1000"
    assert sha == "3333333333333333333333333333333333333333"


def test_select_latest_build_tag_rejects_decoys_only():
    decoys = "\n".join([
        "5555555555555555555555555555555555555555\trefs/tags/ci_cublas-31ff9e2",
        "6666666666666666666666666666666666666666\trefs/tags/fix-release-b7083-foo",
    ])
    with pytest.raises(BuildError, match="no build tag"):
        _select_latest_build_tag(decoys)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/llamactl/test_builds.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'llamactl.core.builds'`

- [ ] **Step 3: Write minimal implementation**

```python
# llamactl/core/builds.py
"""Build subsystem: source-ref resolution, snapshots, image/native builds.

Reimplements the legacy build.*.sh flow in Python. Short git/podman calls use
the blocking injected Runner (runtime.py); long build commands use StreamRunner;
the snapshot pipe uses Extractor. core never imports UI.
"""

from __future__ import annotations

import re

_BUILD_TAG_RE = re.compile(r"^b(\d+)$")


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
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/llamactl/test_builds.py -v`
Expected: PASS (2 passed)

- [ ] **Step 5: Commit**

```bash
git add llamactl/core/builds.py tests/llamactl/test_builds.py
git commit -m "feat: numeric latest-tag selection for llamactl builds"
```

---

## Task 2: Image-tag derivation (pure)

Image tags use the legacy/lifecycle convention as constants — no config field.

**Files:**
- Modify: `llamactl/core/builds.py`
- Test: `tests/llamactl/test_builds.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/llamactl/test_builds.py — append
from llamactl.core.builds import image_tag_for


def test_image_tag_for_rocm_uses_gfx1031_prefix():
    assert image_tag_for("rocm-image", "latest-tag") == "llama-cpp-gfx1031:latest-tag"


def test_image_tag_for_vulkan_uses_vulkan_prefix():
    assert image_tag_for("vulkan-image", "branch:master") == "llama-cpp-vulkan:branch-master"


def test_image_tag_sanitizes_special_chars():
    # commit:<sha> -> colon becomes dash, sha kept
    assert image_tag_for("rocm-image", "commit:abc123") == "llama-cpp-gfx1031:commit-abc123"
    # stray chars dropped
    assert image_tag_for("rocm-image", "tag:b9!@#") == "llama-cpp-gfx1031:tag-b9"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/llamactl/test_builds.py::test_image_tag_for_rocm_uses_gfx1031_prefix -v`
Expected: FAIL with `ImportError: cannot import name 'image_tag_for'`

- [ ] **Step 3: Write minimal implementation**

```python
# llamactl/core/builds.py — add near the top after _BUILD_TAG_RE

ROCM_IMAGE_PREFIX = "llama-cpp-gfx1031"      # matches lifecycle.DEFAULT_ROCM_IMAGE
VULKAN_IMAGE_PREFIX = "llama-cpp-vulkan"     # matches lifecycle.DEFAULT_VULKAN_IMAGE


def _sanitize_tag(ref: str) -> str:
    """Mirror build.llama-ref.docker-rocm.sh sanitize_tag:
    replace [/:@ ] with '-', then drop anything but [A-Za-z0-9._-]."""
    s = re.sub(r"[/:@ ]", "-", ref)
    return re.sub(r"[^A-Za-z0-9._-]", "", s)


def image_tag_for(target: str, ref: str) -> str:
    prefix = ROCM_IMAGE_PREFIX if target == "rocm-image" else VULKAN_IMAGE_PREFIX
    return f"{prefix}:{_sanitize_tag(ref)}"
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/llamactl/test_builds.py -v`
Expected: PASS (5 passed)

- [ ] **Step 5: Commit**

```bash
git add llamactl/core/builds.py tests/llamactl/test_builds.py
git commit -m "feat: image-tag derivation for llamactl builds"
```

---

## Task 3: Source-ref resolution

`resolve_ref` maps a ref spec to a concrete SHA + build_number + fetch spec, using fast `git ls-remote` / `rev-parse` only (no fetch — that happens in Task 4, keeping this under the 30 s Runner timeout).

**Files:**
- Modify: `llamactl/core/builds.py`
- Test: `tests/llamactl/test_builds.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/llamactl/test_builds.py — append
from llamactl.core.builds import ResolvedRef, resolve_ref


def _runner_returning(mapping):
    """Fake Runner: returns CompletedProcess based on a substring match in argv."""
    def run(cmd):
        joined = " ".join(cmd)
        for needle, stdout in mapping.items():
            if needle in joined:
                return subprocess.CompletedProcess(cmd, 0, stdout=stdout, stderr="")
        return subprocess.CompletedProcess(cmd, 1, stdout="", stderr="no match")
    return run


def test_resolve_submodule_uses_rev_parse(tmp_path):
    runner = _runner_returning({"rev-parse": "deadbeef\n"})
    r = resolve_ref("submodule", tmp_path / "cache", tmp_path / "sub", runner)
    assert r == ResolvedRef(sha="deadbeef", build_number="", fetch_spec="", display="submodule")


def test_resolve_latest_tag_picks_highest(tmp_path):
    ls = "aaa\trefs/tags/b10\nbbb\trefs/tags/b2\n"
    runner = _runner_returning({"ls-remote --tags": ls})
    r = resolve_ref("latest-tag", tmp_path / "cache", tmp_path / "sub", runner)
    assert r.sha == "aaa"
    assert r.build_number == "b10"
    assert r.fetch_spec == "refs/tags/b10"


def test_resolve_tag_spec_sets_build_number_when_matching(tmp_path):
    runner = _runner_returning({"ls-remote": "ccc\trefs/tags/b777\n"})
    r = resolve_ref("tag:b777", tmp_path / "cache", tmp_path / "sub", runner)
    assert r.sha == "ccc"
    assert r.build_number == "b777"
    assert r.fetch_spec == "refs/tags/b777"


def test_resolve_branch_spec_has_blank_build_number(tmp_path):
    runner = _runner_returning({"ls-remote": "ddd\trefs/heads/master\n"})
    r = resolve_ref("branch:master", tmp_path / "cache", tmp_path / "sub", runner)
    assert r.sha == "ddd"
    assert r.build_number == ""
    assert r.fetch_spec == "master"


def test_resolve_commit_spec_trusts_sha(tmp_path):
    runner = _runner_returning({})  # no git call needed
    r = resolve_ref("commit:abc123", tmp_path / "cache", tmp_path / "sub", runner)
    assert r.sha == "abc123"
    assert r.build_number == ""
    assert r.fetch_spec == "abc123"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/llamactl/test_builds.py -k resolve -v`
Expected: FAIL with `ImportError: cannot import name 'ResolvedRef'`

- [ ] **Step 3: Write minimal implementation**

```python
# llamactl/core/builds.py — add imports at top
from dataclasses import dataclass, field
from pathlib import Path

from llamactl.core.runtime import Runner, _default_runner

LLAMA_CPP_REMOTE = "https://github.com/ggml-org/llama.cpp.git"


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
        sha = _sha_from_ls_remote(res.stdout) if res.returncode == 0 else ""
        if not sha:
            raise BuildError(f"tag '{name}' not found on remote")
        build_number = name if _BUILD_TAG_RE.match(name) else ""
        return ResolvedRef(sha, build_number, f"refs/tags/{name}", ref)

    if ref.startswith("branch:"):
        name = ref[len("branch:"):]
        res = runner(["git", "ls-remote", LLAMA_CPP_REMOTE, f"refs/heads/{name}"])
        sha = _sha_from_ls_remote(res.stdout) if res.returncode == 0 else ""
        if not sha:
            raise BuildError(f"branch '{name}' not found on remote")
        return ResolvedRef(sha, "", name, ref)

    if ref.startswith("commit:"):
        sha = ref[len("commit:"):]
        if not sha:
            raise BuildError("commit ref is empty")
        return ResolvedRef(sha, "", sha, ref)

    raise BuildError(f"unrecognized ref spec: {ref!r}")
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/llamactl/test_builds.py -k resolve -v`
Expected: PASS (5 passed)

- [ ] **Step 5: Commit**

```bash
git add llamactl/core/builds.py tests/llamactl/test_builds.py
git commit -m "feat: source-ref resolution for llamactl builds"
```

---

## Task 4: Snapshot export

`export_snapshot` fetches the resolved ref into the reused cache clone (StreamRunner — may be slow), then runs `git archive | tar -x` (Extractor) into a clean build context. `submodule` archives directly from the pinned checkout with no fetch.

**Files:**
- Modify: `llamactl/core/builds.py`
- Test: `tests/llamactl/test_builds.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/llamactl/test_builds.py — append
from llamactl.core.builds import StreamRunner, export_snapshot


def _ok_stream(*_a, **_kw):
    """StreamRunner fake that yields nothing and succeeds."""
    if False:
        yield ""  # make it a generator
    return


def test_export_submodule_archives_from_submodule_no_fetch(tmp_path):
    calls = {"fetch": 0, "archive": None, "extract": None}

    def stream(cmd, cwd=None):
        if "fetch" in cmd:
            calls["fetch"] += 1
        return iter(())

    def extractor(archive_cmd, extract_cmd, dest):
        calls["archive"] = archive_cmd
        calls["extract"] = extract_cmd

    resolved = ResolvedRef("deadbeef", "", "", "submodule")
    export_snapshot("submodule", resolved, tmp_path / "cache", tmp_path / "sub",
                    tmp_path / "ctx", stream, extractor)

    assert calls["fetch"] == 0
    assert calls["archive"] == ["git", "-C", str(tmp_path / "sub"),
                                "archive", "--format=tar", "deadbeef"]
    assert calls["extract"] == ["tar", "-x", "-C", str(tmp_path / "ctx")]


def test_export_remote_ref_fetches_then_archives_from_cache(tmp_path):
    (tmp_path / "cache" / ".git").mkdir(parents=True)  # cache already cloned
    fetched = []

    def stream(cmd, cwd=None):
        if "fetch" in cmd:
            fetched.append(cmd)
        return iter(())

    archive_holder = {}

    def extractor(archive_cmd, extract_cmd, dest):
        archive_holder["cmd"] = archive_cmd

    resolved = ResolvedRef("aaa", "b10", "refs/tags/b10", "latest-tag (b10)")
    export_snapshot("latest-tag", resolved, tmp_path / "cache", tmp_path / "sub",
                    tmp_path / "ctx", stream, extractor)

    assert fetched == [["git", "-C", str(tmp_path / "cache"), "fetch",
                        "--depth", "1", "origin", "refs/tags/b10"]]
    assert archive_holder["cmd"] == ["git", "-C", str(tmp_path / "cache"),
                                     "archive", "--format=tar", "aaa"]
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/llamactl/test_builds.py -k export -v`
Expected: FAIL with `ImportError: cannot import name 'StreamRunner'`

- [ ] **Step 3: Write minimal implementation**

```python
# llamactl/core/builds.py — add imports
import subprocess
from collections.abc import Iterator
from typing import Callable

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
    for line in proc.stdout:
        yield line.rstrip("\n")
    proc.wait()
    if proc.returncode != 0:
        raise BuildError(f"command failed (exit {proc.returncode}): {' '.join(cmd)}")


def _default_extractor(archive_cmd: list[str], extract_cmd: list[str], dest: Path) -> None:
    dest.mkdir(parents=True, exist_ok=True)
    p1 = subprocess.Popen(archive_cmd, stdout=subprocess.PIPE)
    p2 = subprocess.Popen(extract_cmd, stdin=p1.stdout)
    if p1.stdout is not None:
        p1.stdout.close()  # allow p1 to receive SIGPIPE if p2 exits
    p2.communicate()
    p1.wait()
    if p1.returncode != 0 or p2.returncode != 0:
        raise BuildError("git archive | tar extraction failed")


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
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/llamactl/test_builds.py -k export -v`
Expected: PASS (2 passed)

- [ ] **Step 5: Commit**

```bash
git add llamactl/core/builds.py tests/llamactl/test_builds.py
git commit -m "feat: clean source snapshot export for llamactl builds"
```

---

## Task 5: Native toolchain detection

Native builds need the build-time deps the Dockerfile base images provide implicitly, not just the compilers.

**Files:**
- Modify: `llamactl/core/builds.py`
- Test: `tests/llamactl/test_builds.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/llamactl/test_builds.py — append
from llamactl.core.builds import ToolchainStatus, detect_native_toolchain


def test_rocm_toolchain_ok_when_all_present():
    which = lambda n: f"/usr/bin/{n}"
    exists = lambda p: True
    st = detect_native_toolchain("rocm-native", which=which, path_exists=exists)
    assert st == ToolchainStatus(ok=True, missing=[])


def test_rocm_toolchain_reports_missing_hipcc_and_hipblas():
    which = lambda n: None if n == "hipcc" else f"/usr/bin/{n}"
    exists = lambda p: p == "/opt/rocm" or p.endswith("curl/curl.h")
    st = detect_native_toolchain("rocm-native", which=which, path_exists=exists)
    assert st.ok is False
    assert "hipcc" in st.missing
    assert "hipblas-dev" in st.missing


def test_vulkan_toolchain_reports_missing_curl_dev():
    which = lambda n: f"/usr/bin/{n}"
    exists = lambda p: p == "/usr/include/vulkan/vulkan.h"  # curl header absent
    st = detect_native_toolchain("vulkan-native", which=which, path_exists=exists)
    assert st.ok is False
    assert "libcurl-dev" in st.missing
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/llamactl/test_builds.py -k toolchain -v`
Expected: FAIL with `ImportError: cannot import name 'ToolchainStatus'`

- [ ] **Step 3: Write minimal implementation**

```python
# llamactl/core/builds.py — add import
import shutil


@dataclass(frozen=True)
class ToolchainStatus:
    ok: bool
    missing: list[str] = field(default_factory=list)


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

    return ToolchainStatus(ok=not missing, missing=missing)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/llamactl/test_builds.py -k toolchain -v`
Expected: PASS (3 passed)

- [ ] **Step 5: Commit**

```bash
git add llamactl/core/builds.py tests/llamactl/test_builds.py
git commit -m "feat: native toolchain detection for llamactl builds"
```

---

## Task 6: Image builder

`build_image` runs `podman build` against the unchanged Dockerfile, streaming output.

**Files:**
- Modify: `llamactl/core/builds.py`
- Test: `tests/llamactl/test_builds.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/llamactl/test_builds.py — append
from llamactl.core.builds import build_image


def test_build_image_rocm_argv_and_streams(tmp_path):
    captured = {}

    def stream(cmd, cwd=None):
        captured["cmd"] = cmd
        yield "Step 1/5"
        yield "Successfully tagged"

    lines = list(build_image(
        "rocm-image", tmp_path / "ctx", "llama-cpp-gfx1031:b10",
        tmp_path / "repo", "podman", stream,
    ))
    assert captured["cmd"] == [
        "podman", "build", "-f", str(tmp_path / "repo" / "Dockerfile.rocm"),
        "-t", "llama-cpp-gfx1031:b10", str(tmp_path / "ctx"),
    ]
    assert lines == ["Step 1/5", "Successfully tagged"]


def test_build_image_vulkan_uses_vulkan_dockerfile(tmp_path):
    captured = {}

    def stream(cmd, cwd=None):
        captured["cmd"] = cmd
        return iter(())

    list(build_image("vulkan-image", tmp_path / "ctx", "llama-cpp-vulkan:b10",
                     tmp_path / "repo", "podman", stream))
    assert "-f" in captured["cmd"]
    assert captured["cmd"][captured["cmd"].index("-f") + 1] == \
        str(tmp_path / "repo" / "Dockerfile.vulkan")
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/llamactl/test_builds.py -k build_image -v`
Expected: FAIL with `ImportError: cannot import name 'build_image'`

- [ ] **Step 3: Write minimal implementation**

```python
# llamactl/core/builds.py — append

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
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/llamactl/test_builds.py -k build_image -v`
Expected: PASS (2 passed)

- [ ] **Step 5: Commit**

```bash
git add llamactl/core/builds.py tests/llamactl/test_builds.py
git commit -m "feat: container image builder for llamactl"
```

---

## Task 7: Native builder

`build_native` replicates the Dockerfile cmake flags, then copies `build/bin/llama-server` to the artifact path.

**Files:**
- Modify: `llamactl/core/builds.py`
- Test: `tests/llamactl/test_builds.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/llamactl/test_builds.py — append
from llamactl.core.builds import ROCM_CMAKE_FLAGS, VULKAN_CMAKE_FLAGS, build_native


def test_build_native_rocm_configure_compile_and_copy(tmp_path):
    cmds = []

    def stream(cmd, cwd=None):
        cmds.append(cmd)
        return iter(())

    copied = {}

    def copier(src, dst):
        copied["src"] = src
        copied["dst"] = dst

    src = tmp_path / "ctx"
    out = tmp_path / "out"
    lines = list(build_native("rocm-native", src, out, stream, copier))

    configure, compile_ = cmds
    assert configure[:6] == ["cmake", "-S", str(src), "-B", str(src / "build"), "-G"]
    assert configure[6] == "Ninja"
    for flag in ROCM_CMAKE_FLAGS:
        assert flag in configure
    assert compile_[:4] == ["cmake", "--build", str(src / "build"), "--target"]
    assert compile_[4] == "llama-server"
    assert copied["src"] == src / "build" / "bin" / "llama-server"
    assert copied["dst"] == out / "llama-server"
    assert any("copied" in line for line in lines)


def test_build_native_vulkan_uses_vulkan_flags(tmp_path):
    cmds = []

    def stream(cmd, cwd=None):
        cmds.append(cmd)
        return iter(())

    list(build_native("vulkan-native", tmp_path / "c", tmp_path / "o",
                      stream, lambda s, d: None))
    assert "-DGGML_VULKAN=ON" in cmds[0]
    assert "-DGGML_HIP=ON" not in cmds[0]
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/llamactl/test_builds.py -k build_native -v`
Expected: FAIL with `ImportError: cannot import name 'ROCM_CMAKE_FLAGS'`

- [ ] **Step 3: Write minimal implementation**

```python
# llamactl/core/builds.py — add import
import os

# cmake flags MUST mirror Dockerfile.rocm / Dockerfile.vulkan exactly.
ROCM_CMAKE_FLAGS = [
    "-DGGML_HIP=ON",
    "-DAMDGPU_TARGETS=gfx1030",
    "-DCMAKE_BUILD_TYPE=Release",
    "-DCMAKE_PREFIX_PATH=/opt/rocm",
    "-DLLAMA_CURL=ON",
    "-DLLAMA_BUILD_BORINGSSL=ON",
]
VULKAN_CMAKE_FLAGS = [
    "-DGGML_VULKAN=ON",
    "-DCMAKE_BUILD_TYPE=Release",
    "-DLLAMA_CURL=ON",
    "-DLLAMA_BUILD_BORINGSSL=ON",
]


def _default_copier(src: Path, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)


def build_native(
    target: str,
    src_dir: Path,
    out_dir: Path,
    stream_runner: StreamRunner = _default_stream_runner,
    copier: Callable[[Path, Path], None] = _default_copier,
) -> Iterator[str]:
    flags = ROCM_CMAKE_FLAGS if target == "rocm-native" else VULKAN_CMAKE_FLAGS
    build_dir = src_dir / "build"
    jobs = str(os.cpu_count() or 1)
    yield from stream_runner(
        ["cmake", "-S", str(src_dir), "-B", str(build_dir), "-G", "Ninja", *flags], None,
    )
    yield from stream_runner(
        ["cmake", "--build", str(build_dir), "--target", "llama-server", "-j", jobs], None,
    )
    out_dir.mkdir(parents=True, exist_ok=True)
    copier(build_dir / "bin" / "llama-server", out_dir / "llama-server")
    yield f"copied llama-server → {out_dir / 'llama-server'}"
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/llamactl/test_builds.py -k build_native -v`
Expected: PASS (2 passed)

- [ ] **Step 5: Commit**

```bash
git add llamactl/core/builds.py tests/llamactl/test_builds.py
git commit -m "feat: native cmake builder for llamactl"
```

---

## Task 8: Build orchestrator

`run_build` ties resolve → snapshot → build → registry together, yields log lines, writes exactly one `Artifact` on success and nothing on failure, and always cleans the temp context.

**Files:**
- Modify: `llamactl/core/builds.py`
- Test: `tests/llamactl/test_builds.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/llamactl/test_builds.py — append
from llamactl.core.builds import BuildRequest, run_build
from llamactl.core.registry import load_registry


def _stub_core(monkeypatch, *, fail_build=False):
    """Stub resolve_ref/export_snapshot/build_* so run_build needs no real git."""
    import llamactl.core.builds as b
    monkeypatch.setattr(b, "resolve_ref",
                        lambda *a, **k: ResolvedRef("sha123456789", "b10", "refs/tags/b10", "x"))
    monkeypatch.setattr(b, "export_snapshot", lambda *a, **k: None)

    def fake_image(*a, **k):
        yield "building image"
        if fail_build:
            raise BuildError("podman build exploded")

    monkeypatch.setattr(b, "build_image", fake_image)
    monkeypatch.setattr(b, "find_runtime", lambda: "podman")


def test_run_build_image_success_writes_one_artifact(tmp_path, monkeypatch):
    _stub_core(monkeypatch)
    reg = tmp_path / "registry.toml"
    lines = list(run_build(
        BuildRequest("latest-tag", "rocm-image"),
        repo_root=tmp_path, state_dir=tmp_path, submodule_dir=tmp_path / "sub",
        registry_path=reg,
    ))
    artifacts = load_registry(reg)
    assert len(artifacts) == 1
    a = artifacts[0]
    assert a.target == "rocm-image"
    assert a.sha == "sha123456789"
    assert a.build_number == "b10"
    assert a.image_tag == "llama-cpp-gfx1031:latest-tag"
    assert any("Registered" in line for line in lines)


def test_run_build_failure_writes_no_artifact(tmp_path, monkeypatch):
    _stub_core(monkeypatch, fail_build=True)
    reg = tmp_path / "registry.toml"
    with pytest.raises(BuildError, match="podman build exploded"):
        list(run_build(
            BuildRequest("latest-tag", "rocm-image"),
            repo_root=tmp_path, state_dir=tmp_path, submodule_dir=tmp_path / "sub",
            registry_path=reg,
        ))
    assert load_registry(reg) == []


def test_run_build_native_refuses_when_toolchain_missing(tmp_path, monkeypatch):
    import llamactl.core.builds as b
    monkeypatch.setattr(b, "resolve_ref",
                        lambda *a, **k: ResolvedRef("sha", "", "sha", "x"))
    monkeypatch.setattr(b, "detect_native_toolchain",
                        lambda *a, **k: ToolchainStatus(ok=False, missing=["hipcc"]))
    with pytest.raises(BuildError, match="toolchain"):
        list(run_build(
            BuildRequest("commit:sha", "rocm-native"),
            repo_root=tmp_path, state_dir=tmp_path, submodule_dir=tmp_path / "sub",
            registry_path=tmp_path / "registry.toml",
        ))
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/llamactl/test_builds.py -k run_build -v`
Expected: FAIL with `ImportError: cannot import name 'BuildRequest'`

- [ ] **Step 3: Write minimal implementation**

```python
# llamactl/core/builds.py — add imports
import datetime
import tempfile

from llamactl.core.registry import (
    Artifact, add_artifact, load_registry, remove_artifact, save_registry,
)
from llamactl.core.runtime import find_runtime


@dataclass(frozen=True)
class BuildRequest:
    ref: str       # ref spec: submodule | latest-tag | tag:.. | branch:.. | commit:..
    target: str    # rocm-image | vulkan-image | rocm-native | vulkan-native


def _now_iso() -> str:
    return datetime.datetime.now(datetime.timezone.utc).isoformat()


def run_build(
    request: BuildRequest,
    repo_root: Path,
    state_dir: Path,
    submodule_dir: Path,
    registry_path: Path,
    runtime: "str | None" = None,
    runner: Runner = _default_runner,
    stream_runner: StreamRunner = _default_stream_runner,
    extractor: Extractor = _default_extractor,
    copier: Callable[[Path, Path], None] = _default_copier,
) -> Iterator[str]:
    """Resolve → snapshot → build → register. Yields log lines. Raises BuildError
    on any failure (and writes no registry row). Always removes the temp context."""
    cache_dir = state_dir / "src-cache" / "llama.cpp"
    is_native = request.target.endswith("-native")

    yield f"Resolving {request.ref}…"
    resolved = resolve_ref(request.ref, cache_dir, submodule_dir, runner)
    yield f"Resolved {resolved.sha[:12]} (build {resolved.build_number or 'unknown'})"

    if is_native:
        status = detect_native_toolchain(request.target)
        if not status.ok:
            raise BuildError(
                f"native toolchain incomplete: missing {', '.join(status.missing)}. "
                f"Use a container target instead."
            )

    tmp_root = state_dir / "builds" / "tmp"
    tmp_root.mkdir(parents=True, exist_ok=True)
    context = Path(tempfile.mkdtemp(dir=tmp_root))
    try:
        yield "Exporting source snapshot…"
        export_snapshot(request.ref, resolved, cache_dir, submodule_dir,
                        context, stream_runner, extractor)

        if is_native:
            out_dir = state_dir / "builds" / resolved.sha / request.target
            yield from build_native(request.target, context, out_dir, stream_runner, copier)
            artifact = Artifact(
                target=request.target, requested_ref=request.ref, sha=resolved.sha,
                build_number=resolved.build_number, built_at=_now_iso(),
                binary_path=str(out_dir / "llama-server"),
            )
        else:
            rt = runtime or find_runtime()
            if rt is None:
                raise BuildError("no container runtime (podman/docker) found")
            tag = image_tag_for(request.target, request.ref)
            yield from build_image(request.target, context, tag, repo_root, rt, stream_runner)
            artifact = Artifact(
                target=request.target, requested_ref=request.ref, sha=resolved.sha,
                build_number=resolved.build_number, built_at=_now_iso(),
                image_tag=tag,
            )

        save_registry(registry_path, add_artifact(load_registry(registry_path), artifact))
        yield f"✓ Registered {request.target} {resolved.sha[:12]}"
    finally:
        shutil.rmtree(context, ignore_errors=True)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/llamactl/test_builds.py -k run_build -v`
Expected: PASS (3 passed)

- [ ] **Step 5: Commit**

```bash
git add llamactl/core/builds.py tests/llamactl/test_builds.py
git commit -m "feat: run_build orchestrator for llamactl builds"
```

---

## Task 9: Artifact delete + in-use helpers

`delete_artifact` removes the image/binary and the registry row (self-healing if already gone); `is_in_use` matches a running server against an artifact.

**Files:**
- Modify: `llamactl/core/builds.py`
- Test: `tests/llamactl/test_builds.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/llamactl/test_builds.py — append
from llamactl.core.builds import delete_artifact, is_in_use
from llamactl.core.lifecycle import ServerInfo
from llamactl.core.registry import Artifact, save_registry


def test_delete_image_artifact_runs_rmi_and_drops_row(tmp_path):
    reg = tmp_path / "registry.toml"
    art = Artifact(target="rocm-image", requested_ref="latest-tag", sha="s1",
                   build_number="b10", built_at="2026-06-16T00:00:00",
                   image_tag="llama-cpp-gfx1031:b10")
    save_registry(reg, [art])
    rmi_calls = []

    def runner(cmd):
        rmi_calls.append(cmd)
        return subprocess.CompletedProcess(cmd, 0, stdout="", stderr="")

    remaining = delete_artifact(art, tmp_path, reg, runtime="podman", runner=runner)
    assert remaining == []
    assert rmi_calls == [["podman", "rmi", "llama-cpp-gfx1031:b10"]]
    assert load_registry(reg) == []


def test_is_in_use_matches_container_image():
    art = Artifact(target="rocm-image", requested_ref="x", sha="s", build_number="",
                   built_at="t", image_tag="llama-cpp-gfx1031:b10")
    info = ServerInfo(model_id="m", backend="rocm", preset="", mode="container",
                      host="0.0.0.0", port=8080, started_at="",
                      container_name="llamactl-m")

    def runner(cmd):
        return subprocess.CompletedProcess(cmd, 0, stdout="llama-cpp-gfx1031:b10\n", stderr="")

    assert is_in_use(art, info, runtime="podman", runner=runner) is True


def test_is_in_use_false_when_no_server():
    art = Artifact(target="rocm-image", requested_ref="x", sha="s", build_number="",
                   built_at="t", image_tag="t:1")
    assert is_in_use(art, None) is False
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/llamactl/test_builds.py -k "delete or in_use" -v`
Expected: FAIL with `ImportError: cannot import name 'delete_artifact'`

- [ ] **Step 3: Write minimal implementation**

```python
# llamactl/core/builds.py — append
from llamactl.core.lifecycle import ServerInfo  # type: ignore[attr-defined]


def delete_artifact(
    artifact: Artifact,
    state_dir: Path,
    registry_path: Path,
    runtime: "str | None" = None,
    runner: Runner = _default_runner,
) -> list[Artifact]:
    """Remove the image (podman rmi) or binary dir, then drop the registry row.
    Tolerates an already-missing image/binary (self-healing)."""
    if artifact.image_tag:
        rt = runtime or find_runtime()
        if rt is not None:
            runner([rt, "rmi", artifact.image_tag])  # ignore failure; may be gone
    if artifact.binary_path:
        shutil.rmtree(state_dir / "builds" / artifact.sha / artifact.target,
                      ignore_errors=True)
    remaining = remove_artifact(load_registry(registry_path), artifact.target, artifact.sha)
    save_registry(registry_path, remaining)
    return remaining


def is_in_use(
    artifact: Artifact,
    server: "ServerInfo | None",
    runtime: "str | None" = None,
    runner: Runner = _default_runner,
) -> bool:
    """Best-effort: container artifact ↔ running container image; native artifact
    ↔ /proc/<pid>/cmdline argv[0]. False (never raises) when it can't tell."""
    if server is None:
        return False
    if artifact.image_tag and server.mode == "container" and server.container_name:
        rt = runtime or find_runtime()
        if rt is None:
            return False
        res = runner([rt, "inspect", "--format", "{{.Config.Image}}", server.container_name])
        return res.returncode == 0 and res.stdout.strip() == artifact.image_tag
    if artifact.binary_path and server.mode == "native" and server.pid:
        try:
            with open(f"/proc/{server.pid}/cmdline", "rb") as f:
                argv0 = f.read().split(b"\0", 1)[0].decode()
            return argv0 == artifact.binary_path
        except OSError:
            return False
    return False
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/llamactl/test_builds.py -k "delete or in_use" -v`
Expected: PASS (3 passed)

- [ ] **Step 5: Commit**

```bash
git add llamactl/core/builds.py tests/llamactl/test_builds.py
git commit -m "feat: artifact delete and in-use detection for llamactl builds"
```

---

## Task 10: Builds tab UI — form, log, table, app wiring

The `BuildsScreen` widget renders the request form, build log, and artifact table, and replaces the placeholder in `app.py`.

**Files:**
- Create: `llamactl/ui/screens/builds.py`
- Modify: `llamactl/ui/app.py:10-12` (import) and `:52-53` (tab body)
- Test: `tests/llamactl/test_builds_ui.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/llamactl/test_builds_ui.py
from __future__ import annotations

from pathlib import Path

import pytest


def _repo(tmp_path: Path) -> Path:
    (tmp_path / "configs" / "models").mkdir(parents=True)
    (tmp_path / "configs" / "llamactl.toml").write_text(
        'default_backend = "rocm"\ndefault_mode = "container"\nport = 8080\n'
        'vram_budget_gb = 11.0\nname_prefix = "llamactl"\n'
    )
    (tmp_path / "state").mkdir()
    return tmp_path


@pytest.mark.asyncio
async def test_builds_tab_has_form_and_table(tmp_path: Path) -> None:
    from llamactl.ui.app import LlamaCtlApp
    from textual.widgets import Button, DataTable, Input, RichLog, Select

    app = LlamaCtlApp(repo_root=_repo(tmp_path))
    async with app.run_test(headless=True) as pilot:
        await pilot.app.query_one("TabbedContent").focus()
        app.query_one("TabbedContent").active = "builds"
        await pilot.pause()
        assert app.query_one("#ref-input", Input) is not None
        assert app.query_one("#target-select", Select) is not None
        assert app.query_one("#btn-build", Button) is not None
        assert app.query_one("#build-log", RichLog) is not None
        assert app.query_one("#artifact-table", DataTable) is not None
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/llamactl/test_builds_ui.py -v`
Expected: FAIL — `#ref-input` not found (Builds tab is still the placeholder `Static`).

- [ ] **Step 3: Write minimal implementation**

```python
# llamactl/ui/screens/builds.py
"""Builds tab — source-ref/target request form, streaming log, artifact table."""
from __future__ import annotations

from typing import TYPE_CHECKING

from textual.app import ComposeResult
from textual.widget import Widget
from textual.widgets import Button, DataTable, Input, Label, RichLog, Select

from llamactl.core.builds import BuildRequest, delete_artifact, is_in_use, run_build

if TYPE_CHECKING:
    from llamactl.ui.app import LlamaCtlApp

_TARGETS = [
    ("ROCm image", "rocm-image"),
    ("Vulkan image", "vulkan-image"),
    ("ROCm native", "rocm-native"),
    ("Vulkan native", "vulkan-native"),
]


class BuildsScreen(Widget):
    DEFAULT_CSS = """
    BuildsScreen { height: 1fr; layout: vertical; }
    BuildsScreen Input, BuildsScreen Select { margin-bottom: 1; }
    BuildsScreen #build-log { height: 1fr; border: solid $panel; margin: 1 0; }
    BuildsScreen #artifact-table { height: 10; border: solid $panel; }
    BuildsScreen .button-row { height: auto; layout: horizontal; }
    BuildsScreen .button-row Button { margin-right: 1; }
    """

    def compose(self) -> ComposeResult:
        yield Label("Source ref (submodule, latest-tag, tag:bNNNN, branch:NAME, commit:SHA):")
        yield Input(value="latest-tag", id="ref-input")
        yield Select(options=_TARGETS, value="rocm-image", id="target-select")
        with Widget(classes="button-row"):
            yield Button("Build", id="btn-build", variant="success")
            yield Button("Delete selected", id="btn-delete", variant="error")
        yield RichLog(id="build-log")
        yield DataTable(id="artifact-table")

    def on_mount(self) -> None:
        table = self.query_one("#artifact-table", DataTable)
        table.add_columns("ref", "sha", "build", "target", "in-use")
        self._refresh_table()

    def _refresh_table(self) -> None:
        from llamactl.core.lifecycle import find_running
        app: LlamaCtlApp = self.app  # type: ignore[assignment]
        table = self.query_one("#artifact-table", DataTable)
        table.clear()
        try:
            server = find_running(app._global_cfg, app._state_dir)
        except Exception:
            server = None
        for art in app._artifacts:
            used = "●" if is_in_use(art, server) else ""
            table.add_row(art.requested_ref, art.sha[:12], art.build_number or "—",
                          art.target, used, key=f"{art.target}:{art.sha}")

    def on_button_pressed(self, event: Button.Pressed) -> None:
        if event.button.id == "btn-build":
            event.button.disabled = True
            ref = self.query_one("#ref-input", Input).value.strip()
            target = str(self.query_one("#target-select", Select).value)
            self.run_worker(
                lambda: self._build_thread(ref, target),
                thread=True, exclusive=True, group="build",
            )
        elif event.button.id == "btn-delete":
            self._delete_selected()

    def _build_thread(self, ref: str, target: str) -> None:
        app: LlamaCtlApp = self.app  # type: ignore[assignment]
        log = self.query_one("#build-log", RichLog)
        try:
            req = BuildRequest(ref=ref, target=target)
            for line in run_build(
                req, app._repo_root, app._state_dir,
                app._repo_root / "llama.cpp-src", app._state_dir / "registry.toml",
            ):
                app.call_from_thread(log.write, line)
        except Exception as exc:
            app.call_from_thread(log.write, f"[red]Build failed:[/red] {exc}")
        finally:
            app.call_from_thread(self._on_build_done)

    def _on_build_done(self) -> None:
        from llamactl.core.registry import load_registry
        app: LlamaCtlApp = self.app  # type: ignore[assignment]
        try:
            app._artifacts = load_registry(app._state_dir / "registry.toml")
        except Exception:
            pass
        self.query_one("#btn-build", Button).disabled = False
        self._refresh_table()

    def _delete_selected(self) -> None:
        app: LlamaCtlApp = self.app  # type: ignore[assignment]
        table = self.query_one("#artifact-table", DataTable)
        if table.cursor_row is None or not app._artifacts:
            self.notify("Select an artifact row first.", severity="warning")
            return
        try:
            row_key = table.coordinate_to_cell_key((table.cursor_row, 0)).row_key
            target, sha = str(row_key.value).split(":", 1)
        except Exception:
            self.notify("Could not identify the selected row.", severity="warning")
            return
        match = next((a for a in app._artifacts
                      if a.target == target and a.sha == sha), None)
        if match is None:
            return
        app._artifacts = delete_artifact(match, app._state_dir,
                                         app._state_dir / "registry.toml")
        self._refresh_table()
```

```python
# llamactl/ui/app.py — change the import block (around line 12)
from llamactl.ui.screens.builds import BuildsScreen
from llamactl.ui.screens.serve import ServeScreen
```

```python
# llamactl/ui/app.py — replace the Builds TabPane body (around line 52-53)
            with TabPane("Builds", id="builds"):
                yield BuildsScreen()
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/llamactl/test_builds_ui.py -v`
Expected: PASS (1 passed)

- [ ] **Step 5: Commit**

```bash
git add llamactl/ui/screens/builds.py llamactl/ui/app.py tests/llamactl/test_builds_ui.py
git commit -m "feat: llamactl Builds tab UI with streaming log and artifact table"
```

---

## Task 11: Builds tab — streaming + button guard test

Verify the build worker streams `run_build`'s lines into the log and re-enables the button when done, with `run_build` faked (no real git/podman).

**Files:**
- Test: `tests/llamactl/test_builds_ui.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/llamactl/test_builds_ui.py — append
@pytest.mark.asyncio
async def test_build_streams_lines_and_reenables_button(tmp_path: Path, monkeypatch) -> None:
    from llamactl.ui.app import LlamaCtlApp
    from textual.widgets import Button
    import llamactl.ui.screens.builds as builds_mod

    def fake_run_build(req, *a, **k):
        yield "line one"
        yield "line two"

    monkeypatch.setattr(builds_mod, "run_build", fake_run_build)

    app = LlamaCtlApp(repo_root=_repo(tmp_path))
    async with app.run_test(headless=True) as pilot:
        app.query_one("TabbedContent").active = "builds"
        await pilot.pause()
        btn = app.query_one("#btn-build", Button)
        app.query_one(builds_mod.BuildsScreen).on_button_pressed(Button.Pressed(btn))
        assert btn.disabled  # synchronous guard
        # let the thread worker finish and post back
        await pilot.pause(0.2)
        assert btn.disabled is False
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/llamactl/test_builds_ui.py -k streams -v`
Expected: FAIL — initially fails if `_build_thread` references break under fakes; iterate until the worker re-enables the button.

- [ ] **Step 3: Implementation**

No new production code expected — Task 10's `BuildsScreen` already implements this. If the test fails, the likely fix is ensuring `_on_build_done` runs via `call_from_thread` in the `finally` (already in Task 10). Adjust only if a real defect surfaces.

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/llamactl/test_builds_ui.py -v`
Expected: PASS (2 passed)

- [ ] **Step 5: Commit**

```bash
git add tests/llamactl/test_builds_ui.py
git commit -m "test: Builds tab streams build output and re-enables button"
```

---

## Task 12: Wire artifact picker into the Serve launch form

Add an artifact `Select` to `_LaunchForm` populated from `app._artifacts`, and pass the chosen `image_tag` as the `resolve_image` override on container launch.

**Files:**
- Modify: `llamactl/ui/screens/serve.py` (`_LaunchForm.compose`, `_refresh_argv_preview`, `get_launch_params`; `ServeScreen._action_launch`)
- Test: `tests/llamactl/test_serve_ui.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/llamactl/test_serve_ui.py — append
@pytest.mark.asyncio
async def test_serve_form_has_artifact_select_from_registry(tmp_path: Path) -> None:
    from llamactl.ui.app import LlamaCtlApp
    from textual.widgets import Select
    from llamactl.core.registry import Artifact, save_registry

    (tmp_path / "configs" / "models").mkdir(parents=True)
    (tmp_path / "configs" / "llamactl.toml").write_text(
        'default_backend = "rocm"\ndefault_mode = "container"\nport = 8080\n'
        'vram_budget_gb = 11.0\nname_prefix = "llamactl"\n'
    )
    (tmp_path / "state").mkdir()
    save_registry(
        tmp_path / "state" / "registry.toml",
        [Artifact(target="rocm-image", requested_ref="latest-tag", sha="s1",
                  build_number="b10", built_at="2026-06-16T00:00:00",
                  image_tag="llama-cpp-gfx1031:b10")],
    )

    app = LlamaCtlApp(repo_root=tmp_path)
    async with app.run_test(headless=True) as pilot:
        artifact_select = app.query_one("#artifact-select", Select)
        values = [v for _label, v in artifact_select._options]
        assert "llama-cpp-gfx1031:b10" in values
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/llamactl/test_serve_ui.py -k artifact_select -v`
Expected: FAIL — `#artifact-select` not found.

- [ ] **Step 3: Write minimal implementation**

In `llamactl/ui/screens/serve.py`, add the select to `_LaunchForm.compose` after the preset select (around line 216):

```python
        yield Select(options=[], id="preset-select", allow_blank=True)

        # Artifact picker — image tags from the registry; blank = default image.
        app = self.app
        artifacts = getattr(app, "_artifacts", [])
        artifact_options = [
            (f"{a.image_tag}  ({a.requested_ref})", a.image_tag)
            for a in artifacts if a.image_tag
        ]
        yield Select(options=artifact_options, id="artifact-select", allow_blank=True)
```

Add a getter and extend `get_launch_params` (replace the existing method, around line 299-306):

```python
    def _get_selected_artifact(self) -> str | None:
        try:
            sel = self.query_one("#artifact-select", Select)
        except NoMatches:
            return None
        value = sel.value
        if value is Select.BLANK or value is None:
            return None
        return str(value)

    def get_launch_params(
        self,
    ) -> tuple[ModelConfig | None, str, str | None, str | None]:
        """Return (model, backend, preset or None, image override or None)."""
        model = self._get_selected_model()
        backend = self._get_selected_backend()
        preset = self._get_selected_preset()
        artifact = self._get_selected_artifact()
        return model, backend, preset, artifact
```

In `ServeScreen._action_launch`, update the unpack and the container `resolve_image` call (around line 447 and 468):

```python
        model, backend, preset, image_override = form.get_launch_params()
```

```python
                image = resolve_image(model, backend, image_override)
```

In `ServeScreen._action_copy_argv`, update the unpack (around line 527):

```python
        model, backend, preset, _image_override = form.get_launch_params()
```

- [ ] **Step 4: Run the Serve + Builds UI suites**

Run: `pytest tests/llamactl/test_serve_ui.py tests/llamactl/test_builds_ui.py -v`
Expected: PASS — including the existing Serve tests (the 4-tuple change must not break them).

- [ ] **Step 5: Commit**

```bash
git add llamactl/ui/screens/serve.py tests/llamactl/test_serve_ui.py
git commit -m "feat: Serve launch form reads build artifacts from the registry"
```

---

## Task 13: Full suite + manual smoke

**Files:** none (verification only)

- [ ] **Step 1: Run the whole llamactl suite (excluding the live-server parity test)**

Run: `pytest tests/llamactl/ -v --ignore=tests/llamactl/test_parity.py`
Expected: all tests pass, including the new `test_builds.py` and `test_builds_ui.py`.

- [ ] **Step 2: Lint**

Run: `ruff check llamactl/core/builds.py llamactl/ui/screens/builds.py`
Expected: no errors.

- [ ] **Step 3: Manual smoke (optional but recommended)**

Run: `python -m llamactl`
Then: switch to the Builds tab, confirm the form/log/table render, type `latest-tag`, pick "ROCm image", press Build, and watch lines stream into the log (this performs a real ls-remote + clone + build — Ctrl-C the app to abort if you only want to verify wiring). Press `q` to quit.

- [ ] **Step 4: Commit any fixups**

```bash
git add -A
git commit -m "chore: Phase 3 Builds tab fixups"
```

---

## Self-Review

**Spec coverage:**
- Two runner abstractions (C1) → Tasks 4, 6, 7 (`StreamRunner`), 3/9 (blocking `Runner`).
- Numeric latest-tag selection (C2) → Task 1.
- Ref resolution table (all 5 specs, build_number rules) → Task 3.
- Snapshot via `git archive | tar`, submodule routing, temp dir + cleanup → Tasks 4, 8.
- Image-tag constants (S1) → Task 2.
- Four targets + Dockerfile-exact cmake flags → Tasks 6, 7.
- Toolchain detection incl. implicit deps (S2) → Task 5.
- Orchestrator: success writes one Artifact, failure writes none → Task 8.
- Delete (rmi / rm -rf, self-healing) + in-use marker via inspect/cmdline (S3) → Task 9.
- Builds tab UI (form, streaming log, table) + one-build-at-a-time (button + `exclusive`/`group`) → Tasks 10, 11.
- Serve-picker wiring (N6) → Task 12.
- Testing surface (numeric sort, failure-writes-nothing, golden argv, toolchain, Pilot smoke) → Tasks 1–12.

**Placeholder scan:** No TBD/TODO; every code step has complete code. Task 11 explicitly states no new production code is expected (verification of Task 10 behavior).

**Type consistency:** `ResolvedRef(sha, build_number, fetch_spec, display)`, `ToolchainStatus(ok, missing)`, `BuildRequest(ref, target)`, `Artifact(...)` (existing) used consistently. `get_launch_params` widened to a 4-tuple and ALL three call sites updated (`_action_launch`, `_action_copy_argv`) in Task 12. `StreamRunner`/`Extractor`/`Runner` signatures match their default impls.

**Note for executor:** `DataTable` cell-key APIs (`coordinate_to_cell_key`, `RowKey.value`) vary across Textual versions — if Task 10's `_delete_selected` raises, adapt to the installed Textual API (verify with `python -c "import textual; print(textual.__version__)"`).
