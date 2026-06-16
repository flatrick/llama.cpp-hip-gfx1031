from __future__ import annotations

import subprocess
from pathlib import Path

import pytest

from llamactl.core.builds import (
    BuildError,
    ResolvedRef,
    StreamRunner,
    ToolchainStatus,
    _select_latest_build_tag,
    detect_native_toolchain,
    export_snapshot,
    image_tag_for,
    resolve_ref,
)


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


def test_image_tag_for_rocm_uses_gfx1031_prefix():
    assert image_tag_for("rocm-image", "latest-tag") == "llama-cpp-gfx1031:latest-tag"


def test_image_tag_for_vulkan_uses_vulkan_prefix():
    assert image_tag_for("vulkan-image", "branch:master") == "llama-cpp-vulkan:branch-master"


def test_image_tag_sanitizes_special_chars():
    # commit:<sha> -> colon becomes dash, sha kept
    assert image_tag_for("rocm-image", "commit:abc123") == "llama-cpp-gfx1031:commit-abc123"
    # stray chars dropped
    assert image_tag_for("rocm-image", "tag:b9!@#") == "llama-cpp-gfx1031:tag-b9"


def test_image_tag_for_rejects_unknown_target():
    with pytest.raises(BuildError, match="image target"):
        image_tag_for("rocm-native", "b1000")


def test_image_tag_for_rejects_empty_sanitized_ref():
    with pytest.raises(BuildError, match="empty image tag"):
        image_tag_for("rocm-image", "@")


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


def test_select_latest_build_tag_prefers_peeled_commit_sha():
    annotated = "\n".join([
        "tagobj1111111111111111111111111111111111\trefs/tags/b500",
        "commitcafe000000000000000000000000000000\trefs/tags/b500^{}",
    ])
    tag, sha = _select_latest_build_tag(annotated)
    assert tag == "b500"
    assert sha == "commitcafe000000000000000000000000000000"


def test_resolve_tag_prefers_peeled_commit_sha(tmp_path):
    # Annotated tags emit two lines; the ^{} (peeled) line is the commit sha we want.
    ls = "tagobj\trefs/tags/b500\ncommitsha\trefs/tags/b500^{}\n"
    runner = _runner_returning({"ls-remote": ls})
    r = resolve_ref("tag:b500", tmp_path / "cache", tmp_path / "sub", runner)
    assert r.sha == "commitsha"


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
