from __future__ import annotations

from llamactl.core.mapper import build_server_argv, to_argv


def test_int_value_maps_to_kebab_flag() -> None:
    assert to_argv({"ctx_size": 262144}) == ["--ctx-size", "262144"]


def test_string_value_passes_through() -> None:
    assert to_argv({"flash_attn": "on"}) == ["--flash-attn", "on"]


def test_float_value() -> None:
    assert to_argv({"temp": 0.6, "min_p": 0.0}) == ["--temp", "0.6", "--min-p", "0.0"]


def test_bool_true_emits_bare_flag() -> None:
    assert to_argv({"no_mmap": True}) == ["--no-mmap"]


def test_bool_false_omits_flag() -> None:
    assert to_argv({"no_warmup": False}) == []


def test_none_omits_flag() -> None:
    assert to_argv({"model_draft": None}) == []


def test_list_repeats_flag() -> None:
    assert to_argv({"lora": ["a.gguf", "b.gguf"]}) == [
        "--lora", "a.gguf", "--lora", "b.gguf",
    ]


def test_exceptions_table_single_dash() -> None:
    assert to_argv({"cram": 2048}) == ["-cram", "2048"]


def test_reserved_keys_skipped() -> None:
    assert to_argv({"host": "0.0.0.0", "port": 8080}) == []


def test_unknown_key_flows_through() -> None:
    # The whole point: new upstream flags need zero code changes.
    assert to_argv({"some_new_flag": 5}) == ["--some-new-flag", "5"]


def test_insertion_order_preserved() -> None:
    assert to_argv({"b_key": 1, "a_key": 2}) == ["--b-key", "1", "--a-key", "2"]


def test_build_server_argv_wraps_hf_host_port() -> None:
    argv = build_server_argv("org/model:Q5", {"ctx_size": 4096}, "0.0.0.0", 8080)
    assert argv == [
        "-hf", "org/model:Q5",
        "--ctx-size", "4096",
        "--host", "0.0.0.0",
        "--port", "8080",
    ]


def test_float_scientific_notation_pinned() -> None:
    # str() of small floats yields scientific notation; llama-server's
    # std::stof accepts it. Pin the format so a formatting change is caught.
    assert to_argv({"min_p": 1e-05}) == ["--min-p", "1e-05"]
