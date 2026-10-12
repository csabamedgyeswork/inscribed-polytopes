"""The [R k] artifact reference group of papers/sweep/main.tex."""

import unittest

from tests.artifact_checks import ArtifactChecks


class TestArtifactReferences(ArtifactChecks, unittest.TestCase):
    PAPER = "papers/sweep"
