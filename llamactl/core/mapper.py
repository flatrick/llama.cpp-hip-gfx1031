"""Convert resolved settings dicts into llama-server argv.

Convention (no allowlist — unknown keys flow through on purpose):
  snake_case key   -> --kebab-case flag
  True             -> bare flag present
  False / None     -> flag omitted
  list             -> flag repeated per element
  anything else    -> str(value) as the flag's argument
"""

from __future__ import annotations

from typing import Any

# Flags that do not follow the double-dash kebab-case convention.
EXCEPTIONS: dict[str, str] = {
    "cram": "-cram",
    "hf": "-hf",
}

# Keys owned by the launcher (injected by lifecycle), never read from settings.
RESERVED: frozenset[str] = frozenset({"host", "port"})


def _flag_name(key: str) -> str:
    return EXCEPTIONS.get(key, "--" + key.replace("_", "-"))


def to_argv(settings: dict[str, Any]) -> list[str]:
    argv: list[str] = []
    for key, value in settings.items():
        if key in RESERVED or value is None:
            continue
        flag = _flag_name(key)
        if isinstance(value, bool):
            if value:
                argv.append(flag)
        elif isinstance(value, list):
            for item in value:
                argv += [flag, str(item)]
        else:
            argv += [flag, str(value)]
    return argv


def build_server_argv(
    hf: str, settings: dict[str, Any], host: str, port: int
) -> list[str]:
    """Full llama-server argument list: model ref, settings, then host/port."""
    return ["-hf", hf, *to_argv(settings), "--host", host, "--port", str(port)]
