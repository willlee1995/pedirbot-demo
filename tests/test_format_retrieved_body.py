"""Top-3 retrieved hits stay full so aftercare is not clipped at 500 chars."""
from __future__ import annotations

import unittest

from src.retrieve_format import SNIPPET_CHARS, format_retrieved_body


class FormatRetrievedBodyTest(unittest.TestCase):
    def test_top_three_keep_full_text(self):
        body = "x" * (SNIPPET_CHARS + 200) + "Bed rest: lie flat for 6 hours."
        for rank in (1, 2, 3):
            self.assertEqual(format_retrieved_body(body, rank), body)
            self.assertIn("Bed rest", format_retrieved_body(body, rank))

    def test_later_hits_are_clipped(self):
        body = "x" * (SNIPPET_CHARS + 200) + "Bed rest: lie flat for 6 hours."
        clipped = format_retrieved_body(body, 4)
        self.assertEqual(len(clipped), SNIPPET_CHARS + 3)
        self.assertTrue(clipped.endswith("..."))
        self.assertNotIn("Bed rest", clipped)


if __name__ == "__main__":
    unittest.main()
