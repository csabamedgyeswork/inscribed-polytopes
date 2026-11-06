"""Tests for inscribed.elimination.campaign, on the d = 4, n = 7 case."""

from __future__ import annotations

import json
import shutil
import tempfile
import unittest
from fractions import Fraction
from pathlib import Path
from unittest.mock import patch

import pytest

from inscribed.elimination import Campaign
from inscribed.elimination.campaign import run
from inscribed.elimination.screening import ScreenRow, Screening
from inscribed.enumeration import FirschingSource

TYPES = list(FirschingSource().types(7))
WINNER = "firsching-7-2"  # triangle and square in orthogonal planes
BOUND = Fraction("0.4")


def screening():
    return Screening([ScreenRow(r.label, -1.0, 0.0) for r in TYPES])


class CampaignCase(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp())
        self.addCleanup(shutil.rmtree, self.tmp)
        self.patcher = patch("inscribed.sos.store.certificate_dir", return_value=self.tmp)
        self.patcher.start()
        self.addCleanup(self.patcher.stop)

    def campaign(self, **kwargs):
        return Campaign(4, 7, BOUND, WINNER, source=FirschingSource(),
                        screening=screening(), **kwargs)

    @pytest.mark.slow
    def test_certify_then_skip_cached(self):
        campaign = self.campaign()
        self.assertEqual(len(campaign.work_list()), 4)
        record = campaign.work_list()[0]
        self.assertTrue(campaign.run_one(record))
        self.assertTrue((self.tmp / f"{record.label}.summary.json").exists())
        self.assertNotIn(record.label, [r.label for r in self.campaign().work_list()])

    @pytest.mark.slow
    def test_failure_writes_attempt_and_is_skipped(self):
        campaign = self.campaign(bound=Fraction("0.1"))
        record = campaign.records[0]
        self.assertFalse(campaign.run_one(record))
        attempt = campaign.attempt_path(record.label)
        self.assertTrue(attempt.exists())
        self.assertEqual(json.loads(attempt.read_text())["bound"], "1/10")
        self.assertNotIn(record.label,
                         [r.label for r in self.campaign(bound=Fraction("0.1")).work_list()])
        self.assertIn(record.label,
                      [r.label for r in self.campaign(bound=Fraction("0.1"),
                                                      retry_failed=True).work_list()])

    def test_shard_partitions_work(self):
        zero = self.campaign(shard=(0, 2)).work_list()
        one = self.campaign(shard=(1, 2)).work_list()
        self.assertEqual({r.label for r in zero} | {r.label for r in one},
                         {r.label for r in TYPES} - {WINNER})
        self.assertFalse({r.label for r in zero} & {r.label for r in one})

    @pytest.mark.slow
    def test_not_certifiable_and_status(self):
        rows = [ScreenRow(r.label, -1.0, 0.0) for r in TYPES]
        rows[0] = ScreenRow(TYPES[0].label, -1.0, 1.0)  # signed max above the bound
        campaign = Campaign(4, 7, BOUND, WINNER, source=FirschingSource(),
                            screening=Screening(rows))
        self.assertEqual(campaign.status().not_certifiable, 1)
        status = run(self.campaign(), limit=1, log=lambda line: None)
        self.assertEqual(status.total, 4)
        self.assertEqual(status.proved, 1)
        self.assertEqual(status.not_certifiable, 1)


if __name__ == "__main__":
    unittest.main()
