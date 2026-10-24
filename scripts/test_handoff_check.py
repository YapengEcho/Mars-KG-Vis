"""Unit tests for scripts/handoff_check.py helpers.

Runs without pytest:  python scripts/test_handoff_check.py
Also works under pytest:  pytest scripts/test_handoff_check.py
"""

from __future__ import annotations

import pathlib
import sys
import unittest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))

import handoff_check as hc  # noqa: E402


class ParseDmsTest(unittest.TestCase):
    def test_negative_degrees(self) -> None:
        self.assertAlmostEqual(hc.parse_dms("-106°13′33.88″"), -106.2260777, places=6)

    def test_positive_degrees(self) -> None:
        self.assertAlmostEqual(hc.parse_dms("17°04′06.24″"), 17.0684, places=6)

    def test_zero_degrees_keeps_text_sign(self) -> None:
        # The source data loses the minus sign when degrees == 0; the parser must
        # report what the text says, and the caller compares magnitudes.
        self.assertAlmostEqual(hc.parse_dms("0°25′30.08″"), 0.4250222, places=6)

    def test_rejects_non_dms(self) -> None:
        self.assertIsNone(hc.parse_dms(""))
        self.assertIsNone(hc.parse_dms("106.5"))
        self.assertIsNone(hc.parse_dms(None))


class PlanetographicShiftTest(unittest.TestCase):
    def test_zero_at_equator(self) -> None:
        self.assertAlmostEqual(hc.planetographic_shift(0.0), 0.0, places=6)

    def test_about_quarter_degree_at_45(self) -> None:
        shift = hc.planetographic_shift(45.0)
        self.assertGreater(shift, 0.15)
        self.assertLess(shift, 0.4)

    def test_symmetric_hemispheres(self) -> None:
        self.assertAlmostEqual(hc.planetographic_shift(30.0), hc.planetographic_shift(-30.0), places=9)


class ReportTest(unittest.TestCase):
    def test_summary_counts(self) -> None:
        report = hc.Report()
        report.add("a", "pass", "ok")
        report.add("b", "warn", "hmm")
        report.add("c", "fail", "bad")
        self.assertEqual(report.summary(), {"pass": 1, "warn": 1, "fail": 1})
        self.assertTrue(report.failed)

    def test_no_failure(self) -> None:
        report = hc.Report()
        report.add("a", "pass", "ok")
        report.add("b", "warn", "hmm")
        self.assertFalse(report.failed)


if __name__ == "__main__":
    unittest.main(verbosity=2)
