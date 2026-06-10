from __future__ import annotations

from pathlib import Path

from llamactl.core.registry import (
    Artifact,
    add_artifact,
    load_registry,
    remove_artifact,
    save_registry,
)

IMAGE = Artifact(
    target="rocm-image",
    requested_ref="tag:b8586",
    sha="7cadbfce10fc16032cfb576ca4607cd2dd183bf1",
    build_number="b8586",
    built_at="2026-06-10T12:00:00",
    image_tag="llama-cpp-gfx1031:b8586",
)
NATIVE = Artifact(
    target="vulkan-native",
    requested_ref="branch:master",
    sha="abc123",
    build_number="",
    built_at="2026-06-10T13:00:00",
    binary_path="state/builds/abc123/vulkan-native/llama-server",
)


def test_load_missing_file_returns_empty(tmp_path: Path) -> None:
    assert load_registry(tmp_path / "registry.toml") == []


def test_round_trip(tmp_path: Path) -> None:
    path = tmp_path / "state" / "registry.toml"
    save_registry(path, [IMAGE, NATIVE])
    assert load_registry(path) == [IMAGE, NATIVE]


def test_add_returns_new_list() -> None:
    artifacts: list[Artifact] = []
    result = add_artifact(artifacts, IMAGE)
    assert result == [IMAGE]
    assert artifacts == []  # input not mutated


def test_add_replaces_same_target_and_sha() -> None:
    rebuilt = Artifact(
        target=IMAGE.target,
        requested_ref=IMAGE.requested_ref,
        sha=IMAGE.sha,
        build_number=IMAGE.build_number,
        built_at="2026-06-11T09:00:00",
        image_tag=IMAGE.image_tag,
    )
    result = add_artifact([IMAGE, NATIVE], rebuilt)
    assert result == [NATIVE, rebuilt]


def test_remove_by_target_and_sha() -> None:
    result = remove_artifact([IMAGE, NATIVE], "rocm-image", IMAGE.sha)
    assert result == [NATIVE]


def test_save_is_atomic_no_temp_file_left(tmp_path: Path) -> None:
    path = tmp_path / "registry.toml"
    save_registry(path, [IMAGE])
    leftovers = [p for p in tmp_path.iterdir() if p != path]
    assert leftovers == []
    assert load_registry(path) == [IMAGE]
