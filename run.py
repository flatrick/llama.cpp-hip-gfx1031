#!/usr/bin/env python3
"""
run.py — unified llama-server launcher

Backends:
  rocm    AMD ROCm GPU (gfx1031)
  vulkan  Vulkan GPU

Use --container to wrap in a Docker/Podman container instead of running natively.
For native ROCm, HSA_OVERRIDE_GFX_VERSION=10.3.0 is set automatically.

Settings are resolved in priority order:
  model defaults → preset → backend → CLI flags

Container image is resolved in priority order:
  built-in default → model JSON images.{backend} → CLI --image

Examples:
  python run.py --model qwen3.5-9b --backend rocm --container
  python run.py --model qwen3.5-9b --backend rocm --container --preset thinking-unrestricted
  python run.py --model qwen3.5-9b --backend vulkan --preset thinking-budgeted --ctx-size 131072
  python run.py --model models/qwen3.5-9b.json --backend rocm --container --dry-run
  python run.py -l
  python run.py -l --model qwen3.5-9b
  python run.py -l --preset thinking-unrestricted
  python run.py -l --model qwen3.5-9b --preset thinking-unrestricted
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
from typing import Any

SCRIPT_DIR = Path(__file__).resolve().parent
MODELS_DIR = SCRIPT_DIR / "models"

DEFAULT_ROCM_IMAGE = "llama-cpp-gfx1031:latest"
DEFAULT_VULKAN_IMAGE = "llama-cpp-vulkan:latest"
DEFAULT_PORT = 8080
DEFAULT_HOST = "0.0.0.0"


# ── model config ──────────────────────────────────────────────────────────────


def load_model_config(model_arg: str) -> dict[str, Any]:
    path = Path(model_arg)
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


def resolve_image(cfg: dict[str, Any], backend: str, cli_image: str | None) -> str:
    """Resolve container image: built-in default → model JSON → CLI override."""
    default = DEFAULT_ROCM_IMAGE if backend == "rocm" else DEFAULT_VULKAN_IMAGE
    model_image = cfg.get("images", {}).get(backend)
    return cli_image or model_image or default


def resolve_settings(cfg: dict[str, Any], preset: str | None, backend: str, overrides: dict[str, Any]) -> dict[str, Any]:
    """Merge: model defaults → preset → backend → CLI overrides."""
    settings = dict(cfg.get("defaults", {}))
    if preset is not None:
        presets = cfg.get("presets", {})
        if preset not in presets:
            available = ", ".join(presets.keys()) if presets else "(none)"
            print(
                f"ERROR: preset '{preset}' not found for model '{cfg.get('name', '?')}'",
                file=sys.stderr,
            )
            print(f"  Available presets: {available}", file=sys.stderr)
            print("  (was it misspelled?)", file=sys.stderr)
            sys.exit(1)
        settings.update(presets[preset])
    settings.update(cfg.get("backends", {}).get(backend, {}))
    settings.update({k: v for k, v in overrides.items() if v is not None})
    return settings


def list_info(
    model_arg: str | None,
    preset_arg: str | None,
    backend_arg: str | None = None,
    image_arg: str | None = None,
) -> None:
    configs = sorted(MODELS_DIR.glob("*.json"))
    loaded: list[tuple[str, dict[str, Any]]] = []
    for p in configs:
        with p.open() as f:
            cfg = json.load(f)
        loaded.append((p.stem, cfg))

    if model_arg and preset_arg:
        # Show resolved settings for model + preset (+ backend if also given)
        cfg = load_model_config(model_arg)
        settings = resolve_settings(cfg, preset_arg, backend_arg or "", {})
        label_parts = [model_arg, f"preset '{preset_arg}'"]
        if backend_arg:
            label_parts.append(f"backend '{backend_arg}'")
            image = resolve_image(cfg, backend_arg, image_arg)
            print(f"Resolved settings: {' + '.join(label_parts)}")
            print(f"  container image: {image}")
        else:
            print(f"Resolved settings: {' + '.join(label_parts)}")
            print("  (add --backend to see hardware overrides)")
        print()
        for k, v in settings.items():
            print(f"  {k} = {v}")

    elif model_arg:
        # Show that model's presets with their key values
        cfg = load_model_config(model_arg)
        presets = cfg.get("presets", {})
        print(f"Presets for {model_arg} ({cfg.get('name', '')}):")
        if not presets:
            print("  (none)")
        else:
            for pname, pvals in presets.items():
                vals_str = "  ".join(f"{k}={v}" for k, v in pvals.items())
                print(f"\n  {pname}")
                print(f"    {vals_str}")

    elif preset_arg:
        # Show all models that have this preset
        print(f"Models with preset '{preset_arg}':")
        found = False
        for stem, cfg in loaded:
            if preset_arg in cfg.get("presets", {}):
                print(f"  {stem:<26}  {cfg.get('name', '')}")
                found = True
        if not found:
            print(f"  (none — preset '{preset_arg}' not found in any model config)")

    else:
        # Show all models with their preset names
        print(f"{'ID':<26}  {'Name':<38}  Presets")
        print(f"{'─' * 26}  {'─' * 38}  {'─' * 40}")
        for stem, cfg in loaded:
            presets = cfg.get("presets", {})
            preset_names = ", ".join(presets.keys()) if presets else "(none)"
            print(f"{stem:<26}  {cfg.get('name', ''):<38}  {preset_names}")


# ── llama-server argument builder ─────────────────────────────────────────────


def build_server_args(hf: str, settings: dict[str, Any], port: int, host: str) -> list[str]:
    args = [
        "-hf", hf,
        "--ctx-size", str(settings["ctx_size"]),
        "--n-gpu-layers", str(settings.get("n_gpu_layers", -1)),
        "--batch-size", str(settings.get("batch_size", 1024)),
        "--ubatch-size", str(settings.get("ubatch_size", 256)),
        "--parallel", str(settings.get("parallel", 1)),
        "--cache-type-k", str(settings.get("cache_k", "f16")),
        "--cache-type-v", str(settings.get("cache_v", "f16")),
        "--top-k", str(settings.get("top_k", 20)),
        "--top-p", str(settings.get("top_p", 0.8)),
        "--temp", str(settings.get("temp", 0.7)),
        "--presence-penalty", str(settings.get("presence_penalty", 1.5)),
        "--min-p", str(settings.get("min_p", 0.0)),
        "--repeat-penalty", str(settings.get("repeat_penalty", 1.0)),
        "--host", host,
        "--port", str(port),
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
    reasoning = settings.get("reasoning")
    if reasoning is not None:
        args += ["--reasoning", str(reasoning)]
    reasoning_budget = settings.get("reasoning_budget")
    if reasoning_budget is not None:
        args += ["--reasoning-budget", str(reasoning_budget)]
    prefill = settings.get("prefill_assistant")
    if prefill is True:
        args.append("--prefill-assistant")
    elif prefill is False:
        args.append("--no-prefill-assistant")
    return args


# ── backend helpers ───────────────────────────────────────────────────────────


def find_container_runtime() -> str:
    runtime = shutil.which("podman") or shutil.which("docker")
    if not runtime:
        print("ERROR: neither podman nor docker found in PATH", file=sys.stderr)
        sys.exit(1)
    return runtime


def check_image(runtime: str, image: str, backend: str) -> None:
    result = subprocess.run(
        [runtime, "image", "inspect", image],
        capture_output=True,
    )
    if result.returncode != 0:
        builder = "build.docker-rocm.sh" if backend == "rocm" else "build.docker-vulkan.sh"
        print(
            f"ERROR: image '{image}' not found — run ./{builder} first", file=sys.stderr
        )
        sys.exit(1)


def dri_passthrough_flags() -> tuple[list[str], list[str]]:
    """Return (device_flags, group_flags) for /dev/dri/* passthrough."""
    if not Path("/dev/dri").is_dir():
        print(
            "ERROR: /dev/dri not present on host; GPU passthrough unavailable",
            file=sys.stderr,
        )
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


def _build_container_cmd(
    hf: str,
    settings: dict[str, Any],
    port: int,
    host: str,
    runtime: str,
    image: str,
    device_flags: list[str],
    group_flags: list[str],
    volumes: list[tuple[str, str]],
    extra_flags: list[str],
    dry_run: bool,
) -> None:
    cmd = [
        runtime, "run", "--rm",
        *device_flags,
        *group_flags,
        *[item for v in volumes for item in v],
        "-p", str(port) + ":" + str(port),
        *extra_flags,
        image,
    ] + build_server_args(hf, settings, port, host)
    exec_cmd(cmd, dry_run)


def exec_cmd(cmd: list[str], dry_run: bool) -> None:
    if dry_run:
        print(shlex.join(cmd))
        return
    os.execvp(cmd[0], cmd)


# ── backends ──────────────────────────────────────────────────────────────────


def run_container(
    cfg: dict[str, Any],
    settings: dict[str, Any],
    port: int,
    host: str,
    backend: str,
    image: str,
    dry_run: bool,
) -> None:
    runtime = find_container_runtime()
    check_image(runtime, image, backend)
    if backend == "rocm":
        device_flags = ["--device", "/dev/kfd", "--device", "/dev/dri"]
        group_flags = ["--group-add", "video", "--group-add", "render"]
    else:  # vulkan
        device_flags, group_flags = dri_passthrough_flags()
    _build_container_cmd(
        cfg["hf"],
        settings,
        port,
        host,
        runtime,
        image,
        device_flags,
        group_flags,
        [
            ("-v", f"{Path.home()}/.cache/huggingface:/root/.cache/huggingface"),
            ("-v", f"{Path.home()}/.cache/llama.cpp:/root/.cache/llama.cpp"),
        ],
        [],
        dry_run,
    )


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


def run_native(
    cfg: dict[str, Any],
    settings: dict[str, Any],
    port: int,
    host: str,
    backend: str,
    binary: str,
    dry_run: bool,
) -> None:
    binary = _resolve_local_binary(binary)
    if backend == "rocm":
        if not Path("/dev/kfd").exists():
            print("ERROR: /dev/kfd not present; ROCm GPU unavailable", file=sys.stderr)
            sys.exit(1)
        # gfx1031 (RX 6700 XT) requires override to the stable gfx1030 path
        os.environ.setdefault("HSA_OVERRIDE_GFX_VERSION", "10.3.0")
    else:  # vulkan
        if not Path("/dev/dri").is_dir():
            print("ERROR: /dev/dri not present; Vulkan GPU unavailable", file=sys.stderr)
            sys.exit(1)
    cmd = [binary] + build_server_args(cfg["hf"], settings, port, host)
    exec_cmd(cmd, dry_run)


# ── CLI ───────────────────────────────────────────────────────────────────────


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Unified llama-server launcher — ROCm or Vulkan, container or native.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--model", "-m",
        help="Model name (looked up in models/) or path to a .json config",
    )
    parser.add_argument(
        "--backend", "-b",
        choices=["rocm", "vulkan"],
        help="GPU backend: rocm or vulkan",
    )
    parser.add_argument(
        "--container",
        action="store_true",
        help="Run inside a Docker/Podman container (default: run native binary)",
    )
    parser.add_argument(
        "--preset", "-p",
        default=None,
        metavar="NAME",
        help="Named preset from model config (applied after defaults, before backend overrides)",
    )

    # common overrides
    parser.add_argument(
        "--ctx-size", type=int, default=None, metavar="N",
        help="Context size in tokens (overrides model default)",
    )
    parser.add_argument(
        "--cache-k", default=None, metavar="TYPE",
        help="KV-cache type for K (f16, q8_0, q4_0, ...)",
    )
    parser.add_argument(
        "--cache-v", default=None, metavar="TYPE",
        help="KV-cache type for V (f16, q8_0, q4_0, ...)",
    )
    parser.add_argument(
        "--temp", type=float, default=None, metavar="F",
        help="Sampling temperature (overrides model default)",
    )
    parser.add_argument(
        "--top-p", type=float, default=None, metavar="F",
        help="Top-p nucleus sampling (overrides model default)",
    )
    parser.add_argument(
        "--top-k", type=int, default=None, metavar="N",
        help="Top-k sampling (overrides model default)",
    )
    parser.add_argument(
        "--reasoning", default=None, choices=["on", "off", "auto"],
        help="Enable/disable reasoning/thinking: on, off, auto",
    )
    parser.add_argument(
        "--prefill-assistant", dest="prefill_assistant",
        action=argparse.BooleanOptionalAction, default=None,
        help="Prefill assistant response (--prefill-assistant / --no-prefill-assistant)",
    )
    parser.add_argument(
        "--reasoning-budget", dest="reasoning_budget", type=int,
        default=None, metavar="N",
        help="Token budget for thinking: -1 unrestricted, 0 immediate end, N>0 budget",
    )
    parser.add_argument(
        "--min-p", type=float, default=None, metavar="F",
        help="Minimum p-nucleus parameter (overrides model default)",
    )
    parser.add_argument(
        "--repeat-penalty", type=float, default=None, metavar="F",
        help="Repeat penalty (overrides model default)",
    )
    parser.add_argument(
        "--port", type=int, default=DEFAULT_PORT,
        help=f"Port to expose/bind (default: {DEFAULT_PORT})",
    )
    parser.add_argument(
        "--host", default=DEFAULT_HOST,
        help=f"Host to bind (default: {DEFAULT_HOST})",
    )

    # backend-specific
    parser.add_argument(
        "--image",
        help=(
            "Container image tag; implies --container. "
            "Priority: CLI --image > model JSON images.{backend} > built-in default"
        ),
    )
    parser.add_argument(
        "--binary",
        help="Path to llama-server binary (native mode; default: from PATH or $LLAMA_SERVER_BIN)",
    )

    # utility
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Print the command that would be executed without running it",
    )
    parser.add_argument(
        "-l", "--list", dest="list_info", action="store_true",
        help=(
            "List models and presets. Filter with --model and/or --preset:\n"
            "  -l                          all models with preset names\n"
            "  -l --model NAME             that model's presets with values\n"
            "  -l --preset NAME            all models that have this preset\n"
            "  -l --model NAME --preset N  resolved settings for that combination"
        ),
    )

    args = parser.parse_args()

    if args.list_info:
        list_info(args.model, args.preset, args.backend, args.image)
        return

    if args.image:
        args.container = True

    missing: list[tuple[str, str]] = []
    if not args.model:
        missing.append(("--model  ", "Model name or path to a .json config  (see -l)"))
    if not args.backend:
        missing.append(("--backend", "GPU backend: rocm | vulkan"))
    if missing:
        print("Missing required arguments:", file=sys.stderr)
        for flag, desc in missing:
            print(f"  {flag}  {desc}", file=sys.stderr)
        model_ex = args.model if args.model else "<model-name>"
        print(
            f"\nExample:\n  python run.py --model {model_ex} --backend rocm --container",
            file=sys.stderr,
        )
        sys.exit(1)

    cfg = load_model_config(args.model)

    cli_overrides = {
        "ctx_size": args.ctx_size,
        "cache_k": args.cache_k,
        "cache_v": args.cache_v,
        "temp": args.temp,
        "top_p": args.top_p,
        "top_k": args.top_k,
        "reasoning": args.reasoning,
        "reasoning_budget": args.reasoning_budget,
        "prefill_assistant": args.prefill_assistant,
        "min_p": args.min_p,
        "repeat_penalty": args.repeat_penalty,
    }
    settings = resolve_settings(cfg, args.preset, args.backend, cli_overrides)

    if "ctx_size" not in settings:
        print(
            "ERROR: ctx_size not set — add it to the model config or pass --ctx-size",
            file=sys.stderr,
        )
        sys.exit(1)

    binary = args.binary or os.environ.get("LLAMA_SERVER_BIN", "")

    if args.container:
        image = resolve_image(cfg, args.backend, args.image)
        run_container(cfg, settings, args.port, args.host, args.backend, image, args.dry_run)
    else:
        run_native(cfg, settings, args.port, args.host, args.backend, binary, args.dry_run)


if __name__ == "__main__":
    main()
