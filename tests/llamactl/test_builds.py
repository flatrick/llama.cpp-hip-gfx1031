from __future__ import annotations

import subprocess
from pathlib import Path

import pytest

from llamactl.core.builds import BuildError, _select_latest_build_tag, image_tag_for


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
