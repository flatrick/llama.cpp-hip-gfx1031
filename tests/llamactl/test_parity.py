"""Phase-1 exit criterion: migrated config + mapper reproduces run.py's argv.

Compares the server-argument portion of `run.py --dry-run` (everything after
the image name) against build_server_argv() for the same model, as unordered
multisets — run.py emits flags in hardcoded order, the mapper in TOML order.
"""

from __future__ import annotations

import shlex
import shutil
import subprocess
import sys
from collections import Counter
from pathlib import Path

import pytest

from llamactl.core.config import load_model, resolve_settings
from llamactl.core.mapper import build_server_argv
from llamactl.core.migrate import migrate

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
MODEL_ID = "qwen3.6-35b-a3b"
IMAGE = "llama-cpp-gfx1031:latest"


def _image_present() -> bool:
    runtime = shutil.which("podman") or shutil.which("docker")
    if not runtime:
        return False
    result = subprocess.run(
        [runtime, "image", "inspect", IMAGE], capture_output=True
    )
    return result.returncode == 0


@pytest.mark.skipif(
    not _image_present(), reason="requires container runtime + rocm image"
)
def test_rocm_container_argv_parity(tmp_path: Path) -> None:
    dry_run = subprocess.run(
        [
            sys.executable, "run.py",
            "--model", MODEL_ID,
            "--backend", "rocm",
            "--container",
            "--dry-run",
        ],
        capture_output=True,
        text=True,
        cwd=REPO_ROOT,
        check=True,
    )
    tokens = shlex.split(dry_run.stdout.strip())
    legacy_args = tokens[tokens.index(IMAGE) + 1 :]

    migrate(REPO_ROOT / "models", tmp_path)
    model = load_model(tmp_path / f"{MODEL_ID}.toml")
    settings = resolve_settings(model, None, "rocm", {})
    new_args = build_server_argv(model.hf, settings, "0.0.0.0", 8080)

    assert Counter(new_args) == Counter(legacy_args)
