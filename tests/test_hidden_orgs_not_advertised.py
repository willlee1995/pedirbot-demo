"""Prompts, tools, and UI must not advertise hidden orgs as available sources."""
from __future__ import annotations

import ast
import re
import unittest
from pathlib import Path

from config import LIVE_SOURCE_ORGS
from src.source_allowlist import HIDDEN_SOURCE_ORGS, live_orgs_csv
from src import prompts

ROOT = Path(__file__).resolve().parents[1]
_STANDALONE_SIR = re.compile(r"(?<![A-Za-z])SIR(?![A-Za-z])")


def _read(relative: str) -> str:
    return (ROOT / relative).read_text(encoding="utf-8")


class HiddenOrgsNotAdvertisedTest(unittest.TestCase):
    def test_prompts_list_only_live_orgs(self):
        strategy = prompts.RETRIEVAL_STRATEGY_INSTRUCTION
        tool_prompt = prompts.TOOL_SYSTEM_PROMPT_TEMPLATE
        for text in (strategy, tool_prompt):
            self.assertNotIn("SickKids", text)
            self.assertIsNone(_STANDALONE_SIR.search(text))
            for org in LIVE_SOURCE_ORGS:
                self.assertIn(org, text)
        self.assertIn(live_orgs_csv(), strategy)
        self.assertIn("source_org=\"CIRSE\"", strategy)
        self.assertNotIn("source_org=\"SickKids\"", strategy)

    def test_tool_and_retriever_source_files_do_not_name_hidden_orgs(self):
        for relative in ("src/tools.py", "src/retriever.py", "src/prompts.py"):
            text = _read(relative)
            self.assertNotIn("SickKids", text, relative)
            self.assertIsNone(_STANDALONE_SIR.search(text), relative)

    def test_search_kb_docstring_lists_only_live_orgs(self):
        tree = ast.parse(_read("src/tools.py"))
        docstrings = []
        for node in ast.walk(tree):
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                doc = ast.get_docstring(node) or ""
                if "source_org" in doc:
                    docstrings.append((node.name, doc))
        self.assertTrue(docstrings)
        for name, doc in docstrings:
            self.assertNotIn("SickKids", doc, name)
            self.assertIsNone(_STANDALONE_SIR.search(doc), name)
            self.assertIn("HKSIR, HKCH, CIRSE", doc, name)

    def test_streamlit_copy_states_allowlist_and_exclusion(self):
        text = _read("streamlit_app.py")
        self.assertIn("live_orgs_csv()", text)
        self.assertIn("does not include SickKids, SIR", text)
        self.assertNotIn("source_org=\"SickKids\"", text)
        self.assertNotIn("SickKids, SIR, HKSIR, CIRSE", text)
        # Hidden orgs appear only in the T&C exclusion caption, not as filter options.
        for match in re.finditer(r"SickKids|\bSIR\b", text):
            start = max(0, match.start() - 80)
            snippet = text[start:match.end() + 80]
            self.assertTrue(
                "does not include" in snippet or "not include" in snippet,
                f"Hidden org named outside T&C exclusion: {snippet!r}",
            )

    def test_readme_states_allowlist_and_exclusion(self):
        text = _read("README.md")
        self.assertIn("HKSIR, HKCH, CIRSE", text)
        self.assertIn("SickKids, SIR", text)
        self.assertRegex(
            text,
            r"does \*\*not\*\* include SickKids, SIR|SickKids, SIR, or other crawled",
        )


if __name__ == "__main__":
    unittest.main()
