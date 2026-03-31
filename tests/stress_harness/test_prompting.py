from __future__ import annotations

import unittest

from stress_harness.prompting import PromptBuilder


class PromptBuilderTests(unittest.TestCase):
    def test_prefix_is_applied_once(self) -> None:
        builder = PromptBuilder("abc ", 2)
        prompt = builder.build(4, prefix="[tag]")
        self.assertTrue(prompt.startswith("[tag] "))
        self.assertEqual(prompt.count("[tag]"), 1)


if __name__ == "__main__":
    unittest.main()
