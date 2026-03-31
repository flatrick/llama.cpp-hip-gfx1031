from __future__ import annotations


class PromptBuilder:
    def __init__(self, filler_chunk: str, tokens_per_chunk: int) -> None:
        self._filler_chunk = filler_chunk
        self._tokens_per_chunk = tokens_per_chunk

    def build(self, target_tokens: int, prefix: str = "") -> str:
        repeats = max(1, target_tokens // self._tokens_per_chunk)
        body = (self._filler_chunk * repeats).strip()
        return (prefix + " " + body).strip() if prefix else body
