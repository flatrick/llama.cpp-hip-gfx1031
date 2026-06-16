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
