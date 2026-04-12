#!/usr/bin/env python3
"""
Unit tests for run.py.

Tests cover:
1. Model config loading
2. Settings resolution (defaults → preset → backend → CLI overrides)
3. Server argument building
4. Preset resolution (known preset, unknown preset error)
5. New CLI overrides: temp, top-p, top-k
6. Listing logic
7. Backend-specific overrides
"""

import sys
from pathlib import Path
from typing import Any
import pytest
from run import (
    load_model_config,
    resolve_image,
    resolve_settings,
    build_server_args,
    list_info,
    DEFAULT_ROCM_IMAGE,
    DEFAULT_VULKAN_IMAGE,
)

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))

class TestModelConfigLoading:
    """Test model config loading functionality."""

    def test_load_model_by_name(self):
        cfg = load_model_config("qwen3.5-9b")
        assert cfg["name"] == "Qwen3.5-9B UD-Q5_K_XL"
        assert cfg["hf"] == "unsloth/Qwen3.5-9B-GGUF:UD-Q5_K_XL"

    def test_load_model_by_path(self):
        cfg = load_model_config("models/qwen3.5-9b.json")
        assert cfg["name"] == "Qwen3.5-9B UD-Q5_K_XL"

    def test_load_model_defaults(self):
        cfg = load_model_config("qwen3.5-9b")
        defaults = cfg.get("defaults", {})
        assert defaults["reasoning"] == "auto"
        assert defaults["prefill_assistant"] is False
        assert defaults["temp"] == 0.7
        assert defaults["top_p"] == 0.8
        assert defaults["top_k"] == 20
        assert defaults["presence_penalty"] == 1.5

    def test_load_model_has_presets(self):
        cfg = load_model_config("qwen3.5-9b")
        presets = cfg.get("presets", {})
        assert "thinking-unrestricted" in presets
        assert "thinking-budgeted" in presets
        assert "thinking-disabled" in presets

    def test_load_model_has_new_backends(self):
        cfg = load_model_config("qwen3.5-9b")
        backends = cfg.get("backends", {})
        assert "rocm" in backends
        assert "vulkan" in backends
        assert "rocm-docker" not in backends

    def test_load_model_missing_exits(self):
        with pytest.raises(SystemExit) as exc_info:
            load_model_config("nonexistent-model")
        assert exc_info.value.code == 1


class TestSettingsResolution:
    """Test settings resolution (defaults → preset → backend → CLI)."""

    def test_base_defaults_no_preset(self):
        cfg = load_model_config("qwen3.5-9b")
        settings = resolve_settings(cfg, None, "vulkan", {})
        assert settings["reasoning"] == "auto"
        assert settings["temp"] == 0.7

    def test_preset_overrides_defaults(self):
        cfg = load_model_config("qwen3.5-9b")
        settings = resolve_settings(cfg, "thinking-unrestricted", "vulkan", {})
        assert settings["reasoning"] == "on"
        assert settings["reasoning_budget"] == -1
        assert settings["temp"] == 0.6
        assert settings["presence_penalty"] == 0.0

    def test_preset_thinking_disabled(self):
        cfg = load_model_config("qwen3.5-9b")
        settings = resolve_settings(cfg, "thinking-disabled", "vulkan", {})
        assert settings["reasoning"] == "off"
        assert settings["reasoning_budget"] == 0

    def test_preset_thinking_budgeted(self):
        cfg = load_model_config("qwen3.5-9b")
        settings = resolve_settings(cfg, "thinking-budgeted", "vulkan", {})
        assert settings["reasoning"] == "on"
        assert settings["reasoning_budget"] == 8192
        assert settings["temp"] == 0.6

    def test_backend_overrides_preset(self):
        cfg = load_model_config("qwen3.5-9b")
        settings = resolve_settings(cfg, "thinking-unrestricted", "rocm", {})
        # rocm backend overrides ctx_size and cache types
        assert settings["ctx_size"] == 131072
        assert settings["cache_k"] == "f16"
        assert settings["cache_v"] == "f16"
        # preset values not overridden by backend are preserved
        assert settings["reasoning"] == "on"
        assert settings["temp"] == 0.6

    def test_cli_overrides_backend(self):
        cfg = load_model_config("qwen3.5-9b")
        settings = resolve_settings(cfg, None, "rocm", {"ctx_size": 32768})
        assert settings["ctx_size"] == 32768

    def test_cli_none_does_not_override(self):
        cfg = load_model_config("qwen3.5-9b")
        settings = resolve_settings(cfg, None, "rocm", {"ctx_size": None})
        assert settings["ctx_size"] == 131072  # from rocm backend

    def test_cli_temp_override(self):
        cfg = load_model_config("qwen3.5-9b")
        settings = resolve_settings(cfg, "thinking-unrestricted", "vulkan", {"temp": 0.9})
        assert settings["temp"] == 0.9  # CLI wins over preset

    def test_cli_top_p_override(self):
        cfg = load_model_config("qwen3.5-9b")
        settings = resolve_settings(cfg, None, "vulkan", {"top_p": 0.5})
        assert settings["top_p"] == 0.5

    def test_cli_top_k_override(self):
        cfg = load_model_config("qwen3.5-9b")
        settings = resolve_settings(cfg, None, "vulkan", {"top_k": 40})
        assert settings["top_k"] == 40

    def test_unknown_preset_exits(self, capsys: pytest.CaptureFixture[str]):
        cfg = load_model_config("qwen3.5-9b")
        with pytest.raises(SystemExit) as exc_info:
            resolve_settings(cfg, "nonexistent-preset", "rocm", {})
        assert exc_info.value.code == 1
        captured = capsys.readouterr()
        assert "nonexistent-preset" in captured.err
        assert "was it misspelled" in captured.err

    def test_unknown_preset_shows_available(self, capsys: pytest.CaptureFixture[str]):
        cfg = load_model_config("qwen3.5-9b")
        with pytest.raises(SystemExit):
            resolve_settings(cfg, "typo-preset", "rocm", {})
        captured = capsys.readouterr()
        assert "thinking-unrestricted" in captured.err

    def test_none_preset_on_model_with_no_presets(self):
        cfg = load_model_config("gemma-4-e4b-it")
        settings = resolve_settings(cfg, None, "vulkan", {})
        assert settings["ctx_size"] == 131072  # from defaults

    def test_resolution_order_full_chain(self):
        """Verify full chain: defaults < preset < backend < CLI."""
        cfg = load_model_config("qwen3.5-9b")
        # preset sets temp=0.6, backend sets cache_k=f16, CLI sets ctx_size=65536
        settings = resolve_settings(
            cfg, "thinking-unrestricted", "rocm", {"ctx_size": 65536}
        )
        assert settings["temp"] == 0.6          # from preset
        assert settings["cache_k"] == "f16"     # from backend
        assert settings["ctx_size"] == 65536    # from CLI
        assert settings["reasoning"] == "on"    # from preset


class TestServerArguments:
    """Test server argument building."""

    def test_basic_args_present(self):
        cfg = load_model_config("qwen3.5-9b")
        settings = resolve_settings(cfg, None, "rocm", {})
        args = build_server_args(cfg["hf"], settings, 8080, "0.0.0.0")
        assert "-hf" in args
        assert cfg["hf"] in args
        assert "--ctx-size" in args
        assert "--port" in args
        assert "8080" in args

    def test_rocm_backend_ctx_size(self):
        cfg = load_model_config("qwen3.5-9b")
        settings = resolve_settings(cfg, None, "rocm", {})
        args = build_server_args(cfg["hf"], settings, 8080, "0.0.0.0")
        assert "131072" in args  # rocm backend override

    def test_preset_thinking_unrestricted_args(self):
        cfg = load_model_config("qwen3.5-9b")
        settings = resolve_settings(cfg, "thinking-unrestricted", "rocm", {})
        args = build_server_args(cfg["hf"], settings, 8080, "0.0.0.0")
        assert "--reasoning" in args
        assert "on" in args
        assert "--reasoning-budget" in args
        assert "-1" in args

    def test_preset_thinking_disabled_args(self):
        cfg = load_model_config("qwen3.5-9b")
        settings = resolve_settings(cfg, "thinking-disabled", "rocm", {})
        args = build_server_args(cfg["hf"], settings, 8080, "0.0.0.0")
        assert "--reasoning" in args
        assert "off" in args
        assert "--reasoning-budget" in args
        assert "0" in args

    def test_temp_in_args(self):
        cfg = load_model_config("qwen3.5-9b")
        settings = resolve_settings(cfg, "thinking-unrestricted", "vulkan", {})
        args = build_server_args(cfg["hf"], settings, 8080, "0.0.0.0")
        assert "--temp" in args
        idx = args.index("--temp")
        assert args[idx + 1] == "0.6"

    def test_top_p_in_args(self):
        cfg = load_model_config("qwen3.5-9b")
        settings = resolve_settings(cfg, "thinking-unrestricted", "vulkan", {})
        args = build_server_args(cfg["hf"], settings, 8080, "0.0.0.0")
        assert "--top-p" in args
        idx = args.index("--top-p")
        assert args[idx + 1] == "0.95"

    def test_top_k_in_args(self):
        cfg = load_model_config("qwen3.5-9b")
        settings = resolve_settings(cfg, None, "vulkan", {})
        args = build_server_args(cfg["hf"], settings, 8080, "0.0.0.0")
        assert "--top-k" in args
        idx = args.index("--top-k")
        assert args[idx + 1] == "20"

    def test_temp_cli_override_in_args(self):
        cfg = load_model_config("qwen3.5-9b")
        settings = resolve_settings(cfg, None, "vulkan", {"temp": 1.2})
        args = build_server_args(cfg["hf"], settings, 8080, "0.0.0.0")
        idx = args.index("--temp")
        assert args[idx + 1] == "1.2"

    def test_top_p_cli_override_in_args(self):
        cfg = load_model_config("qwen3.5-9b")
        settings = resolve_settings(cfg, None, "vulkan", {"top_p": 0.5})
        args = build_server_args(cfg["hf"], settings, 8080, "0.0.0.0")
        idx = args.index("--top-p")
        assert args[idx + 1] == "0.5"

    def test_top_k_cli_override_in_args(self):
        cfg = load_model_config("qwen3.5-9b")
        settings = resolve_settings(cfg, None, "vulkan", {"top_k": 64})
        args = build_server_args(cfg["hf"], settings, 8080, "0.0.0.0")
        idx = args.index("--top-k")
        assert args[idx + 1] == "64"

    def test_flash_attn_on(self):
        cfg = load_model_config("qwen3.5-9b")
        settings = resolve_settings(cfg, None, "vulkan", {})
        args = build_server_args(cfg["hf"], settings, 8080, "0.0.0.0")
        assert "--flash-attn" in args
        assert "on" in args

    def test_cram_flag(self):
        cfg = load_model_config("qwen3.5-9b")
        settings = resolve_settings(cfg, None, "vulkan", {})
        args = build_server_args(cfg["hf"], settings, 8080, "0.0.0.0")
        assert "-cram" in args
        assert "2048" in args

    def test_no_mmproj_flag(self):
        cfg = load_model_config("qwen3.5-9b")
        settings = resolve_settings(cfg, None, "vulkan", {})
        args = build_server_args(cfg["hf"], settings, 8080, "0.0.0.0")
        assert "--no-mmproj" in args

    def test_jinja_flag(self):
        cfg = load_model_config("qwen3.5-9b")
        settings = resolve_settings(cfg, None, "vulkan", {})
        args = build_server_args(cfg["hf"], settings, 8080, "0.0.0.0")
        assert "--jinja" in args

    def test_prefill_assistant_true(self):
        cfg = load_model_config("qwen3.5-9b")
        settings = resolve_settings(cfg, None, "vulkan", {"prefill_assistant": True})
        args = build_server_args(cfg["hf"], settings, 8080, "0.0.0.0")
        assert "--prefill-assistant" in args
        assert "--no-prefill-assistant" not in args

    def test_prefill_assistant_false(self):
        cfg = load_model_config("qwen3.5-9b")
        settings = resolve_settings(cfg, None, "vulkan", {"prefill_assistant": False})
        args = build_server_args(cfg["hf"], settings, 8080, "0.0.0.0")
        assert "--no-prefill-assistant" in args
        assert "--prefill-assistant" not in args

    def test_min_p_in_args(self):
        cfg = load_model_config("qwen3.5-9b")
        settings = resolve_settings(cfg, "thinking-unrestricted", "vulkan", {})
        args = build_server_args(cfg["hf"], settings, 8080, "0.0.0.0")
        assert "--min-p" in args

    def test_repeat_penalty_in_args(self):
        cfg = load_model_config("qwen3.5-9b")
        settings = resolve_settings(cfg, "thinking-unrestricted", "vulkan", {})
        args = build_server_args(cfg["hf"], settings, 8080, "0.0.0.0")
        assert "--repeat-penalty" in args
        idx = args.index("--repeat-penalty")
        assert args[idx + 1] == "1.0"


class TestListInfo:
    """Test listing logic."""

    def test_list_all_models(self, capsys: pytest.CaptureFixture[str]) -> None:
        list_info(None, None)
        captured = capsys.readouterr()
        assert "qwen3.5-9b" in captured.out
        assert "phi-4" in captured.out
        assert "gemma-4-e4b-it" in captured.out

    def test_list_all_shows_preset_names(self, capsys: pytest.CaptureFixture[str]) -> None:
        list_info(None, None)
        captured = capsys.readouterr()
        assert "thinking-unrestricted" in captured.out

    def test_list_model_shows_presets(self, capsys: pytest.CaptureFixture[str]) -> None:
        list_info("qwen3.5-9b", None)
        captured = capsys.readouterr()
        assert "thinking-unrestricted" in captured.out
        assert "thinking-budgeted" in captured.out
        assert "thinking-disabled" in captured.out

    def test_list_model_shows_preset_values(self, capsys: pytest.CaptureFixture[str]) -> None:
        list_info("qwen3.5-9b", None)
        captured = capsys.readouterr()
        assert "reasoning=on" in captured.out

    def test_list_preset_shows_matching_models(self, capsys: pytest.CaptureFixture[str]) -> None:
        list_info(None, "thinking-unrestricted")
        captured = capsys.readouterr()
        assert "qwen3.5-9b" in captured.out

    def test_list_preset_excludes_non_matching(self, capsys: pytest.CaptureFixture[str]):
        list_info(None, "thinking-unrestricted")
        captured = capsys.readouterr()
        # gemma-4 has no presets
        assert "gemma-4-e4b-it" not in captured.out

    def test_list_model_and_preset_shows_resolved(self, capsys: pytest.CaptureFixture[str]):
        list_info("qwen3.5-9b", "thinking-unrestricted")
        captured = capsys.readouterr()
        assert "Resolved settings" in captured.out
        assert "thinking-unrestricted" in captured.out

    def test_list_model_no_presets_shows_none(self, capsys: pytest.CaptureFixture[str]):
        list_info("gemma-4-e4b-it", None)
        captured = capsys.readouterr()
        assert "(none)" in captured.out

    def test_list_nonexistent_preset_shows_none(self, capsys: pytest.CaptureFixture[str]):
        list_info(None, "nonexistent-preset")
        captured = capsys.readouterr()
        assert "none" in captured.out.lower()

    def test_list_model_unknown_preset_exits(self):
        with pytest.raises(SystemExit) as exc_info:
            list_info("qwen3.5-9b", "typo")
        assert exc_info.value.code == 1


class TestImageResolution:
    """Test container image resolution priority."""

    def test_rocm_default_image(self):
        cfg: dict[str, Any] = {}
        assert resolve_image(cfg, "rocm", None) == DEFAULT_ROCM_IMAGE

    def test_vulkan_default_image(self):
        cfg: dict[str, Any] = {}
        assert resolve_image(cfg, "vulkan", None) == DEFAULT_VULKAN_IMAGE

    def test_model_json_image_overrides_default(self):
        cfg: dict[str, Any] = {"images": {"rocm": "llama-cpp-gfx1031:custom"}}
        assert resolve_image(cfg, "rocm", None) == "llama-cpp-gfx1031:custom"

    def test_model_json_vulkan_image(self):
        cfg: dict[str, Any] = {"images": {"vulkan": "llama-cpp-vulkan:nightly"}}
        assert resolve_image(cfg, "vulkan", None) == "llama-cpp-vulkan:nightly"

    def test_cli_overrides_model_json_image(self):
        cfg: dict[str, Any] = {"images": {"rocm": "llama-cpp-gfx1031:model-default"}}
        assert resolve_image(cfg, "rocm", "llama-cpp-gfx1031:cli-override") == "llama-cpp-gfx1031:cli-override"

    def test_cli_overrides_builtin_default(self):
        cfg: dict[str, Any] = {}
        assert resolve_image(cfg, "rocm", "my-custom-image:tag") == "my-custom-image:tag"

    def test_missing_backend_in_images_falls_back_to_default(self):
        cfg: dict[str, Any] = {"images": {"vulkan": "llama-cpp-vulkan:custom"}}
        # rocm not in images → falls back to built-in default
        assert resolve_image(cfg, "rocm", None) == DEFAULT_ROCM_IMAGE

    def test_empty_images_section(self):
        cfg: dict[str, Any] = {"images": {}}
        assert resolve_image(cfg, "rocm", None) == DEFAULT_ROCM_IMAGE


class TestEdgeCases:
    """Test edge cases."""

    def test_missing_ctx_size_not_in_settings(self):
        cfg: dict[str, Any] = {"defaults": {}, "backends": {}}
        settings = resolve_settings(cfg, None, "rocm", {})
        assert "ctx_size" not in settings

    def test_empty_model_config(self):
        cfg: dict[str, Any] = {}
        settings = resolve_settings(cfg, None, "rocm", {})
        assert settings == {}

    def test_model_without_presets_none_preset_ok(self):
        cfg: dict[str, Any] = load_model_config("gemma-4-e4b-it")
        settings = resolve_settings(cfg, None, "rocm", {})
        assert "ctx_size" in settings  # from defaults

    def test_vulkan_backend_uses_defaults(self):
        cfg: dict[str, Any] = load_model_config("qwen3.5-9b")
        settings = resolve_settings(cfg, None, "vulkan", {})
        # vulkan backend is empty, so ctx_size comes from defaults
        assert settings["ctx_size"] == 262144

    def test_rocm_backend_overrides_ctx_size(self):
        cfg: dict[str, Any] = load_model_config("qwen3.5-9b")
        settings = resolve_settings(cfg, None, "rocm", {})
        assert settings["ctx_size"] == 131072


if __name__ == "__main__":
    import pytest

    pytest.main([__file__, "-v"])
