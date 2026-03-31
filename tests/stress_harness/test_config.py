from __future__ import annotations

import unittest

from stress_harness.config import StressConfig


class StressConfigTests(unittest.TestCase):
    def test_from_env_overrides_defaults(self) -> None:
        config = StressConfig.from_env({
            "MAX_TOKENS": "64",
            "REQUEST_TIMEOUT": "900",
            "COLD_ROUNDS": "3",
            "CTX_SIZE": "32768",
            "VRAM_WARN_GB": "9.5",
        })
        self.assertEqual(config.max_tokens, 64)
        self.assertEqual(config.request_timeout, 900)
        self.assertEqual(config.cold_rounds, 3)
        self.assertEqual(config.ctx_size_override, 32768)
        self.assertEqual(config.ctx_size_fallback, 32768)
        self.assertEqual(config.vram_warn_gb, 9.5)

    def test_build_steps_includes_cap(self) -> None:
        config = StressConfig(max_tokens=256)
        steps = config.build_steps(10_000)
        self.assertIn(int(10_000 * 0.95) - 256, steps)


if __name__ == "__main__":
    unittest.main()
