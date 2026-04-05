#!/usr/bin/env python3
"""
run.py — unified llama-server launcher

Backends:
  rocm-docker    ROCm Docker/Podman container (llama-cpp-gfx1031:latest)
  vulkan-docker  Vulkan Docker/Podman container (llama-cpp-vulkan:latest)
  vulkan         Native llama-server binary, Vulkan GPU (local install)
  rocm           Native llama-server binary, ROCm GPU (local install)

Settings are resolved in priority order:
  model defaults  →  backend-specific overrides  →  CLI flags

Examples:
  python run.py --model qwen3.5-9b --backend rocm-docker
  python run.py --model qwen3.5-9b --backend vulkan-docker --ctx-size 262144
  python run.py --model qwen3.5-9b --backend vulkan --cache-k q8_0
  python run.py --model models/qwen3.5-9b.json --backend rocm-docker --dry-run
  python run.py --list-models
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import shlex
import shutil
import subprocess
import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
MODELS_DIR = SCRIPT_DIR / "models"

DEFAULT_ROCM_IMAGE   = "llama-cpp-gfx1031:latest"
DEFAULT_VULKAN_IMAGE = "llama-cpp-vulkan:latest"
DEFAULT_PORT = 8080
DEFAULT_HOST = "0.0.0.0"


# ── model config ──────────────────────────────────────────────────────────────

def load_model_config(model_arg: str) -> dict:
    path = Path(model_arg)
    # Treat as a bare name (look up in models/) unless it's an explicit path
    # (contains a path separator or ends in .json)
    if "/" not in model_arg and not model_arg.endswith(".json"):
        path = MODELS_DIR / f"{model_arg}.json"
    if not path.exists():
        candidates = sorted(MODELS_DIR.glob("*.json"))
        names = [p.stem for p in candidates]
        print(f"ERROR: model config not found: {model_arg}", file=sys.stderr)
        if names:
            print(f"  Available models: {', '.join(names)}", file=sys.stderr)
        else:
            print(f"  No configs found in {MODELS_DIR}", file=sys.stderr)
        sys.exit(1)
    with path.open() as f:
        return json.load(f)


def list_models() -> None:
    configs = sorted(MODELS_DIR.glob("*.json"))
    if not configs:
        print(f"No model configs found in {MODELS_DIR}")
        return
    print(f"{'ID':<22}  {'Name':<35}  HF reference")
    print(f"{'─'*22}  {'─'*35}  {'─'*40}")
    for p in configs:
        with p.open() as f:
            cfg = json.load(f)
        print(f"{p.stem:<22}  {cfg.get('name', ''):<35}  {cfg.get('hf', '')}")


def resolve_settings(cfg: dict, backend: str, overrides: dict) -> dict:
    """Merge: model defaults → backend-specific overrides → CLI overrides."""
    settings = dict(cfg.get("defaults", {}))
    settings.update(cfg.get("backends", {}).get(backend, {}))
    settings.update({k: v for k, v in overrides.items() if v is not None})
    return settings


# ── llama-server argument builder ─────────────────────────────────────────────

def build_server_args(hf: str, settings: dict, port: int, host: str) -> list[str]:
    args = [
        "-hf",             hf,
        "--ctx-size",      str(settings["ctx_size"]),
        "--n-gpu-layers",  str(settings.get("n_gpu_layers", -1)),
        "--batch-size",    str(settings.get("batch_size", 1024)),
        "--ubatch-size",   str(settings.get("ubatch_size", 256)),
        "--parallel",      str(settings.get("parallel", 1)),
        "--cache-type-k",  str(settings.get("cache_k", "f16")),
        "--cache-type-v",  str(settings.get("cache_v", "f16")),
        "--top-k",         str(settings.get("top_k", 20)),
        "--top-p",         str(settings.get("top_p", 0.8)),
        "--temp",          str(settings.get("temp", 0.7)),
        "--presence-penalty", str(settings.get("presence_penalty", 1.5)),
        "--host",          host,
        "--port",          str(port),
    ]
    if settings.get("flash_attn", True):
        args += ["--flash-attn", "on"]
    if settings.get("no_warmup", False):
        args.append("--no-warmup")
    if settings.get("no_mmproj", False):
        args.append("--no-mmproj")
    if settings.get("jinja", True):
        args.append("--jinja")
    if settings.get("cram"):
        args += ["-cram", str(settings["cram"])]
    return args


# ── backend helpers ───────────────────────────────────────────────────────────

def find_container_runtime() -> str:
    runtime = shutil.which("podman") or shutil.which("docker")
    if not runtime:
        print("ERROR: neither podman nor docker found in PATH", file=sys.stderr)
        sys.exit(1)
    return runtime


def check_image(runtime: str, image: str) -> None:
    result = subprocess.run(
        [runtime, "image", "inspect", image],
        capture_output=True,
    )
    if result.returncode != 0:
        builder = "build.docker-rocm.sh" if "gfx1031" in image else "build.docker-vulkan.sh"
        print(f"ERROR: image '{image}' not found — run ./{builder} first", file=sys.stderr)
        sys.exit(1)


def dri_passthrough_flags() -> tuple[list[str], list[str]]:
    """Return (device_flags, group_flags) for /dev/dri/* passthrough."""
    if not Path("/dev/dri").is_dir():
        print("ERROR: /dev/dri not present on host; GPU passthrough unavailable", file=sys.stderr)
        sys.exit(1)
    device_flags: list[str] = []
    group_flags: list[str] = []
    seen_gids: set[int] = set()
    for node in glob.glob("/dev/dri/renderD*") + glob.glob("/dev/dri/card*"):
        device_flags += ["--device", f"{node}:{node}"]
        gid = os.stat(node).st_gid
        if gid not in seen_gids:
            group_flags += ["--group-add", str(gid)]
            seen_gids.add(gid)
    if not device_flags:
        print("ERROR: no /dev/dri render or card nodes found", file=sys.stderr)
        sys.exit(1)
    return device_flags, group_flags


def exec_cmd(cmd: list[str], dry_run: bool) -> None:
    if dry_run:
        print(shlex.join(cmd))
        return
    os.execvp(cmd[0], cmd)


# ── backends ──────────────────────────────────────────────────────────────────

def run_rocm(cfg: dict, settings: dict, port: int, host: str, image: str, dry_run: bool) -> None:
    runtime = find_container_runtime()
    check_image(runtime, image)
    cmd = [
        runtime, "run", "--rm",
        "--device", "/dev/kfd",
        "--device", "/dev/dri",
        "--group-add", "video",
        "--group-add", "render",
        "-v", f"{Path.home()}/.cache/huggingface:/root/.cache/huggingface",
        "-v", f"{Path.home()}/.cache/llama.cpp:/root/.cache/llama.cpp",
        "-p", f"{port}:{port}",
        image,
    ] + build_server_args(cfg["hf"], settings, port, host)
    exec_cmd(cmd, dry_run)


def run_vulkan_docker(cfg: dict, settings: dict, port: int, host: str, image: str, dry_run: bool) -> None:
    runtime = find_container_runtime()
    check_image(runtime, image)
    device_flags, group_flags = dri_passthrough_flags()
    cmd = [
        runtime, "run", "--rm",
        *device_flags,
        *group_flags,
        "-v", f"{Path.home()}/.cache/huggingface:/root/.cache/huggingface",
        "-v", f"{Path.home()}/.cache/llama.cpp:/root/.cache/llama.cpp",
        "-p", f"{port}:{port}",
        image,
    ] + build_server_args(cfg["hf"], settings, port, host)
    exec_cmd(cmd, dry_run)


def _resolve_local_binary(binary: str) -> str:
    if not binary:
        binary = shutil.which("llama-server") or ""
    if not binary:
        print(
            "ERROR: llama-server not found in PATH\n"
            "  Set LLAMA_SERVER_BIN=/path/to/llama-server or pass --binary",
            file=sys.stderr,
        )
        sys.exit(1)
    if not os.access(binary, os.X_OK):
        print(f"ERROR: not executable: {binary}", file=sys.stderr)
        sys.exit(1)
    return binary


def run_vulkan_local(cfg: dict, settings: dict, port: int, host: str, binary: str, dry_run: bool) -> None:
    binary = _resolve_local_binary(binary)
    if not Path("/dev/dri").is_dir():
        print("ERROR: /dev/dri not present; Vulkan GPU unavailable", file=sys.stderr)
        sys.exit(1)
    cmd = [binary] + build_server_args(cfg["hf"], settings, port, host)
    exec_cmd(cmd, dry_run)


def run_rocm_local(cfg: dict, settings: dict, port: int, host: str, binary: str, dry_run: bool) -> None:
    binary = _resolve_local_binary(binary)
    if not Path("/dev/kfd").exists():
        print("ERROR: /dev/kfd not present; ROCm GPU unavailable", file=sys.stderr)
        sys.exit(1)
    # gfx1031 (RX 6700 XT) requires override to the stable gfx1030 path
    os.environ.setdefault("HSA_OVERRIDE_GFX_VERSION", "10.3.0")
    cmd = [binary] + build_server_args(cfg["hf"], settings, port, host)
    exec_cmd(cmd, dry_run)


# ── CLI ───────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Unified llama-server launcher — ROCm Docker, Vulkan Docker, or local native.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("--model", "-m",
        help="Model name (looked up in models/) or path to a .json config")
    parser.add_argument("--backend", "-b",
        choices=["rocm-docker", "vulkan-docker", "vulkan", "rocm"],
        help="Backend: rocm-docker, vulkan-docker, vulkan (native), rocm (native)")

    # common overrides
    parser.add_argument("--ctx-size", type=int, default=None,
        metavar="N", help="Context size in tokens (overrides model default)")
    parser.add_argument("--cache-k", default=None,
        metavar="TYPE", help="KV-cache type for K (f16, q8_0, q4_0, ...)")
    parser.add_argument("--cache-v", default=None,
        metavar="TYPE", help="KV-cache type for V (f16, q8_0, q4_0, ...)")
    parser.add_argument("--port", type=int, default=DEFAULT_PORT,
        help=f"Port to expose/bind (default: {DEFAULT_PORT})")
    parser.add_argument("--host", default=DEFAULT_HOST,
        help=f"Host to bind (default: {DEFAULT_HOST})")

    # backend-specific
    parser.add_argument("--image",
        help="Override container image tag (ROCm or Vulkan, depending on backend)")
    parser.add_argument("--binary",
        help="Path to llama-server binary (local backend; default: from PATH or $LLAMA_SERVER_BIN)")

    # utility
    parser.add_argument("--dry-run", action="store_true",
        help="Print the command that would be executed without running it")
    parser.add_argument("--list-models", action="store_true",
        help="List available model configs and exit")

    args = parser.parse_args()

    if args.list_models:
        list_models()
        return

    missing: list[tuple[str, str]] = []
    if not args.model:
        missing.append(("--model  ", "Model name or path to a .json config  (see --list-models)"))
    if not args.backend:
        missing.append(("--backend", "Backend to use: rocm-docker | vulkan-docker | vulkan | rocm"))
    if missing:
        print("Missing required arguments:", file=sys.stderr)
        for flag, desc in missing:
            print(f"  {flag}  {desc}", file=sys.stderr)
        model_ex = args.model if args.model else "<model-name>"
        print(f"\nExample:\n  python run.py --model {model_ex} --backend rocm-docker", file=sys.stderr)
        sys.exit(1)

    cfg = load_model_config(args.model)

    cli_overrides = {
        "ctx_size": args.ctx_size,
        "cache_k":  args.cache_k,
        "cache_v":  args.cache_v,
    }
    settings = resolve_settings(cfg, args.backend, cli_overrides)

    if "ctx_size" not in settings:
        print("ERROR: ctx_size not set — add it to the model config or pass --ctx-size", file=sys.stderr)
        sys.exit(1)

    binary = args.binary or os.environ.get("LLAMA_SERVER_BIN", "")

    if args.backend == "rocm-docker":
        image = args.image or os.environ.get("ROCM_IMAGE", DEFAULT_ROCM_IMAGE)
        run_rocm(cfg, settings, args.port, args.host, image, args.dry_run)

    elif args.backend == "vulkan-docker":
        image = args.image or os.environ.get("VULKAN_IMAGE", DEFAULT_VULKAN_IMAGE)
        run_vulkan_docker(cfg, settings, args.port, args.host, image, args.dry_run)

    elif args.backend == "vulkan":
        run_vulkan_local(cfg, settings, args.port, args.host, binary, args.dry_run)

    elif args.backend == "rocm":
        run_rocm_local(cfg, settings, args.port, args.host, binary, args.dry_run)


if __name__ == "__main__":
    main()
