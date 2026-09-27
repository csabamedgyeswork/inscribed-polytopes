"""The [R k] artifact reference group of paper/main.tex.

That group is a second bibliography whose entries are paths in this repository
rather than papers, so its links rot silently whenever a file is renamed or a
citation key is mistyped. These tests keep it closed on both sides: every
\\artref resolves to an entry, every entry is cited at least once, and every
path both exists and would survive a push.
"""

import re
import subprocess
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
PAPER = ROOT / "paper"
MAIN = PAPER / "main.tex"
CITATION = ROOT / "CITATION.cff"

_DEFINITION = re.compile(r"\\artifact\{([^}]*)\}\{([^}]*)\}")
_CITATION = re.compile(r"\\artref\{([^}]*)\}")
_LINK = re.compile(r"\\artpath\{([^}]*)\}")


def _defined_keys() -> list[str]:
    """Entry keys in the order the artifact list declares them."""
    return [key for key, _ in _DEFINITION.findall(MAIN.read_text())]


def _definitions() -> dict[str, str]:
    """Key to repository path, with the LaTeX escaping undone."""
    return {
        key: path.replace(r"\_", "_") for key, path in _DEFINITION.findall(MAIN.read_text())
    }


def _paths() -> dict[str, str]:
    """Every repository path the group links: the entry paths, plus the extra
    links inside descriptions naming the tests that guard an entry. The
    \\artpath{#2} of the macro definition itself is not one of them.
    """
    paths = _definitions()
    paths.update(
        {path.replace(r"\_", "_"): path.replace(r"\_", "_")
         for path in _LINK.findall(MAIN.read_text()) if path != "#2"}
    )
    return paths


def _citations() -> set[str]:
    """Every key cited anywhere in the paper sources."""
    return {
        key
        for tex in PAPER.glob("**/*.tex")
        for key in _CITATION.findall(tex.read_text())
    }


class TestArtifactReferences(unittest.TestCase):
    def test_the_group_is_populated(self):
        self.assertGreater(len(_defined_keys()), 0)

    def test_keys_are_unique(self):
        keys = _defined_keys()
        self.assertEqual(sorted(keys), sorted(set(keys)))

    def test_every_citation_resolves_to_an_entry(self):
        defined = _definitions()
        for key in sorted(_citations()):
            with self.subTest(key=key):
                self.assertIn(key, defined, f"\\artref{{{key}}} has no entry")

    def test_every_entry_is_cited(self):
        """An uncited entry is the failure mode that left horvath-2016-survey
        out of the printed bibliography."""
        cited = _citations()
        for key in _defined_keys():
            with self.subTest(key=key):
                self.assertIn(key, cited, f"entry {key} is never cited")

    def test_every_path_exists(self):
        for key, path in _paths().items():
            with self.subTest(key=key):
                self.assertTrue((ROOT / path).exists(), f"{path} does not exist")

    def test_no_path_is_ignored_by_git(self):
        """An ignored path is absent from the pushed tree, so its link would
        404. Merely uncommitted paths are fine; they are about to be pushed."""
        for key, path in _paths().items():
            with self.subTest(key=key):
                ignored = subprocess.run(
                    ["git", "check-ignore", "-q", path],
                    cwd=ROOT,
                    capture_output=True,
                )
                self.assertNotEqual(ignored.returncode, 0, f"{path} is gitignored")


class TestCitationMetadata(unittest.TestCase):
    """CITATION.cff repeats the paper's abstract, so the two can drift apart."""

    def _paper_abstract(self) -> str:
        """The abstract of main.tex, reduced to the plain text CITATION.cff holds."""
        body = re.search(
            r"\\begin\{abstract\}(.*?)\\end\{abstract\}", MAIN.read_text(), re.S
        )
        self.assertIsNotNone(body, "main.tex has no abstract")
        text = body.group(1).replace("--", "-").replace(r"\leq", "<=")
        text = re.sub(r"\\mathbb\{S\}\^\{(\d)\}", r"S^\1", text)
        text = re.sub(r"\$([^$]*)\$", r"\1", text)
        return re.sub(r"\s+", " ", text).strip()

    def _citation_abstract(self) -> str:
        """The folded scalar under ``abstract:``, unfolded the way YAML would."""
        lines, collecting = [], False
        for line in CITATION.read_text().splitlines():
            if line.startswith("abstract:"):
                collecting = True
                continue
            if collecting:
                if line.startswith(" "):
                    lines.append(line.strip())
                else:
                    break
        self.assertTrue(lines, "CITATION.cff has no abstract")
        return re.sub(r"\s+", " ", " ".join(lines)).strip()

    def test_the_abstract_matches_the_paper(self):
        self.assertEqual(self._citation_abstract(), self._paper_abstract())


class TestArtifactLinks(unittest.TestCase):
    """The URL is built once from \\repobase and \\repocommit."""

    def _macro(self, name: str) -> str:
        found = re.search(rf"\\newcommand\{{\\{name}\}}\{{([^}}]*)\}}", MAIN.read_text())
        self.assertIsNotNone(found, f"\\{name} is not defined")
        return found.group(1)

    def test_the_base_and_commit_are_defined_once(self):
        self.assertTrue(self._macro("repobase").startswith("https://"))
        self.assertTrue(self._macro("repocommit"))

    def test_paths_are_relative_to_the_repository_root(self):
        for key, path in _paths().items():
            with self.subTest(key=key):
                self.assertFalse(path.startswith("/"))
                self.assertNotIn("..", path)


if __name__ == "__main__":
    unittest.main()
