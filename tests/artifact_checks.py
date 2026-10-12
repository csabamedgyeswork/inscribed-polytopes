"""Shared checks for the [R k] artifact reference group of a paper's main.tex.

That group is a second bibliography whose entries are paths in this repository
rather than papers, so its links rot silently whenever a file is renamed or a
citation key is mistyped. These checks keep it closed on both sides: every
\\artref resolves to an entry, every entry is cited at least once, and every
path both exists and would survive a push. Each paper subclasses
``ArtifactChecks`` with its own ``PAPER``.
"""

import re
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

_DEFINITION = re.compile(r"\\artifact\{([^}]*)\}\{([^}]*)\}")
_CITATION = re.compile(r"\\artref\{([^}]*)\}")
_LINK = re.compile(r"\\artpath\{([^}]*)\}")


class ArtifactChecks:
    """Closure checks for one paper's artifact list; mix in before
    ``unittest.TestCase`` and set ``PAPER`` to the paper's directory relative
    to the repository root."""

    PAPER: str

    @classmethod
    def _main(cls) -> str:
        return (ROOT / cls.PAPER / "main.tex").read_text()

    @classmethod
    def _defined_keys(cls) -> list[str]:
        """Entry keys in the order the artifact list declares them."""
        return [key for key, _ in _DEFINITION.findall(cls._main())]

    @classmethod
    def _definitions(cls) -> dict[str, str]:
        """Key to repository path, with the LaTeX escaping undone."""
        return {
            key: path.replace(r"\_", "_") for key, path in _DEFINITION.findall(cls._main())
        }

    @classmethod
    def _paths(cls) -> dict[str, str]:
        """Every repository path the group links: the entry paths, plus the extra
        links inside descriptions naming the tests that guard an entry. The
        \\artpath{#2} of the macro definition itself is not one of them.
        """
        paths = cls._definitions()
        paths.update(
            {path.replace(r"\_", "_"): path.replace(r"\_", "_")
             for path in _LINK.findall(cls._main()) if path != "#2"}
        )
        return paths

    @classmethod
    def _citations(cls) -> set[str]:
        """Every key cited anywhere in the paper sources."""
        return {
            key
            for tex in (ROOT / cls.PAPER).glob("**/*.tex")
            for key in _CITATION.findall(tex.read_text())
        }

    def test_the_group_is_populated(self):
        self.assertGreater(len(self._defined_keys()), 0)

    def test_keys_are_unique(self):
        keys = self._defined_keys()
        self.assertEqual(sorted(keys), sorted(set(keys)))

    def test_every_citation_resolves_to_an_entry(self):
        defined = self._definitions()
        for key in sorted(self._citations()):
            with self.subTest(key=key):
                self.assertIn(key, defined, f"\\artref{{{key}}} has no entry")

    def test_every_entry_is_cited(self):
        """An uncited entry is the failure mode that left horvath-2016-survey
        out of the printed bibliography."""
        cited = self._citations()
        for key in self._defined_keys():
            with self.subTest(key=key):
                self.assertIn(key, cited, f"entry {key} is never cited")

    def test_every_path_exists(self):
        for key, path in self._paths().items():
            with self.subTest(key=key):
                self.assertTrue((ROOT / path).exists(), f"{path} does not exist")

    def test_no_path_is_ignored_by_git(self):
        """An ignored path is absent from the pushed tree, so its link would
        404. Merely uncommitted paths are fine; they are about to be pushed."""
        for key, path in self._paths().items():
            with self.subTest(key=key):
                ignored = subprocess.run(
                    ["git", "check-ignore", "-q", path],
                    cwd=ROOT,
                    capture_output=True,
                )
                self.assertNotEqual(ignored.returncode, 0, f"{path} is gitignored")

    def _macro(self, name: str) -> str:
        found = re.search(rf"\\newcommand\{{\\{name}\}}\{{([^}}]*)\}}", self._main())
        self.assertIsNotNone(found, f"\\{name} is not defined")
        return found.group(1)

    def test_the_base_and_commit_are_defined_once(self):
        """The URL is built once from \\repobase and \\repocommit."""
        self.assertTrue(self._macro("repobase").startswith("https://"))
        self.assertTrue(self._macro("repocommit"))

    def test_paths_are_relative_to_the_repository_root(self):
        for key, path in self._paths().items():
            with self.subTest(key=key):
                self.assertFalse(path.startswith("/"))
                self.assertNotIn("..", path)
