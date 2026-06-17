from __future__ import annotations

import subprocess
from pathlib import Path

import pytest

from llamactl.core.builds import (
    BuildError,
    BuildRequest,
    ROCM_CMAKE_FLAGS,
    ResolvedRef,
    StreamRunner,
    ToolchainStatus,
    VULKAN_CMAKE_FLAGS,
    _select_latest_build_tag,
    build_image,
    build_native,
    delete_artifact,
    detect_native_toolchain,
    export_snapshot,
    image_tag_for,
    is_in_use,
    resolve_ref,
    run_build,
)
from llamactl.core.lifecycle import ServerInfo
from llamactl.core.registry import Artifact, load_registry, save_registry


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
    assert st == ToolchainStatus(ok=True, missing=())


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


def test_vulkan_toolchain_ok_when_all_present():
    which = lambda n: f"/usr/bin/{n}"
    exists = lambda p: True
    st = detect_native_toolchain("vulkan-native", which=which, path_exists=exists)
    assert st == ToolchainStatus(ok=True, missing=())


def test_detect_native_toolchain_rejects_unknown_target():
    with pytest.raises(BuildError, match="not a native target"):
        detect_native_toolchain("gpu-native")


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
                        lambda *a, **k: ToolchainStatus(ok=False, missing=("hipcc",)))
    with pytest.raises(BuildError, match="toolchain"):
        list(run_build(
            BuildRequest("commit:sha", "rocm-native"),
            repo_root=tmp_path, state_dir=tmp_path, submodule_dir=tmp_path / "sub",
            registry_path=tmp_path / "registry.toml",
        ))


def test_run_build_image_refuses_when_no_runtime(tmp_path, monkeypatch):
    import llamactl.core.builds as b
    monkeypatch.setattr(b, "resolve_ref",
                        lambda *a, **k: ResolvedRef("sha123456789", "b10", "refs/tags/b10", "x"))
    monkeypatch.setattr(b, "find_runtime", lambda: None)
    reg = tmp_path / "registry.toml"
    with pytest.raises(BuildError, match="container runtime"):
        list(run_build(
            BuildRequest("latest-tag", "rocm-image"),
            repo_root=tmp_path, state_dir=tmp_path, submodule_dir=tmp_path / "sub",
            registry_path=reg,
        ))
    assert not reg.exists()


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


def test_image_build_context_has_llama_cpp_src_subdir(tmp_path, monkeypatch):
    """Regression: Dockerfile.rocm/.vulkan do `COPY llama.cpp-src /llama.cpp`, so the
    image build context must contain a `llama.cpp-src/` subdir with the source.
    Previously export dropped the source at the context root and `COPY` failed."""
    import llamactl.core.builds as b
    monkeypatch.setattr(b, "resolve_ref",
                        lambda *a, **k: ResolvedRef("sha123", "b1", "refs/tags/b1", "x"))
    monkeypatch.setattr(b, "find_runtime", lambda: "podman")

    def fake_extractor(archive_cmd, extract_cmd, dest):
        # mimic `tar -x -C dest`: lay the source files at `dest`
        dest.mkdir(parents=True, exist_ok=True)
        (dest / "CMakeLists.txt").write_text("x")

    seen = {}

    def fake_build_image(target, context_dir, image_tag, repo_root, runtime, stream_runner,
                         no_cache=False):
        seen["has_src_subdir"] = (Path(context_dir) / "llama.cpp-src" / "CMakeLists.txt").is_file()
        return iter(())

    monkeypatch.setattr(b, "build_image", fake_build_image)

    def fake_stream(cmd, cwd=None):
        return iter(())

    list(run_build(
        BuildRequest("submodule", "rocm-image"),
        repo_root=tmp_path, state_dir=tmp_path, submodule_dir=tmp_path / "sub",
        registry_path=tmp_path / "registry.toml",
        stream_runner=fake_stream, extractor=fake_extractor,
    ))
    assert seen.get("has_src_subdir"), \
        "image build context is missing llama.cpp-src/ (the Dockerfile COPYs it)"


def test_native_build_context_is_source_root(tmp_path, monkeypatch):
    """Native builds cmake from the context root, so the source must sit directly
    in the context (no llama.cpp-src/ subdir)."""
    import llamactl.core.builds as b
    monkeypatch.setattr(b, "resolve_ref",
                        lambda *a, **k: ResolvedRef("sha", "", "sha", "x"))
    monkeypatch.setattr(b, "detect_native_toolchain",
                        lambda *a, **k: ToolchainStatus(ok=True, missing=()))

    def fake_extractor(archive_cmd, extract_cmd, dest):
        dest.mkdir(parents=True, exist_ok=True)
        (dest / "CMakeLists.txt").write_text("x")

    seen = {}

    def fake_build_native(target, src_dir, out_dir, stream_runner, copier):
        seen["src_at_root"] = (Path(src_dir) / "CMakeLists.txt").is_file()
        seen["no_subdir"] = not (Path(src_dir) / "llama.cpp-src").exists()
        return iter(())

    monkeypatch.setattr(b, "build_native", fake_build_native)

    def fake_stream(cmd, cwd=None):
        return iter(())

    list(run_build(
        BuildRequest("submodule", "rocm-native"),
        repo_root=tmp_path, state_dir=tmp_path, submodule_dir=tmp_path / "sub",
        registry_path=tmp_path / "registry.toml",
        stream_runner=fake_stream, extractor=fake_extractor,
    ))
    assert seen.get("src_at_root") and seen.get("no_subdir")


# --- Force rebuild (no-cache) for image builds -------------------------------

def test_build_request_no_cache_defaults_false():
    assert BuildRequest("latest-tag", "rocm-image").no_cache is False


def test_build_image_no_cache_prepends_flag(tmp_path):
    captured = {}

    def stream(cmd, cwd=None):
        captured["cmd"] = cmd
        return iter(())

    list(build_image("rocm-image", tmp_path / "ctx", "llama-cpp-gfx1031:b10",
                     tmp_path / "repo", "docker", stream, no_cache=True))
    cmd = captured["cmd"]
    assert "--no-cache" in cmd
    # placed right after `build`, before the `-f <dockerfile>` args
    assert cmd.index("--no-cache") == cmd.index("build") + 1
    assert cmd.index("--no-cache") < cmd.index("-f")


def test_build_image_omits_no_cache_by_default(tmp_path):
    captured = {}

    def stream(cmd, cwd=None):
        captured["cmd"] = cmd
        return iter(())

    list(build_image("rocm-image", tmp_path / "ctx", "llama-cpp-gfx1031:b10",
                     tmp_path / "repo", "docker", stream))
    assert "--no-cache" not in captured["cmd"]


def test_run_build_threads_no_cache_to_build_image(tmp_path, monkeypatch):
    import llamactl.core.builds as b
    monkeypatch.setattr(b, "resolve_ref",
                        lambda *a, **k: ResolvedRef("sha123456789", "b10", "refs/tags/b10", "x"))
    monkeypatch.setattr(b, "export_snapshot", lambda *a, **k: None)
    monkeypatch.setattr(b, "find_runtime", lambda: "docker")
    captured = {}

    def fake_image(target, context_dir, image_tag, repo_root, runtime,
                   stream_runner=None, no_cache=False):
        captured["no_cache"] = no_cache
        return iter(())

    monkeypatch.setattr(b, "build_image", fake_image)
    list(run_build(
        BuildRequest("latest-tag", "rocm-image", no_cache=True),
        repo_root=tmp_path, state_dir=tmp_path, submodule_dir=tmp_path / "sub",
        registry_path=tmp_path / "registry.toml",
    ))
    assert captured["no_cache"] is True
