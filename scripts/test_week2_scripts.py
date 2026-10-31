"""Tests for the week-2 scripts.

Pure functions run everywhere; the integration checks need the teacher's handoff
package and are skipped when ``HANDOFF_DIR`` (or the default path) is missing.

    python -m unittest scripts/test_week2_scripts.py -v
"""

from __future__ import annotations

import json
import math
import os
from pathlib import Path
import sys
import tempfile
import unittest

SCRIPTS = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPTS))

import admission_report  # noqa: E402
import build_case_mapping  # noqa: E402
import export_checked_catalog  # noqa: E402

HANDOFF = Path(os.environ.get("HANDOFF_DIR", r"D:\python_project\teng_jiajie_week1_20261024"))
R = build_case_mapping.MARS_RADIUS_M
DEG = build_case_mapping.DEG_PER_M


class GeoBoundsTest(unittest.TestCase):
    def test_y_grows_southward(self) -> None:
        # 1 degree south of the equator is y = +R·rad(1)
        bounds = build_case_mapping.geo_bounds([0.0, 0.0, R * math.radians(1), R * math.radians(1)])
        self.assertAlmostEqual(bounds["lonMin"], 0.0, places=6)
        self.assertAlmostEqual(bounds["lonMax"], 1.0, places=6)
        self.assertAlmostEqual(bounds["latMin"], -1.0, places=5)
        self.assertAlmostEqual(bounds["latMax"], 0.0, places=5)

    def test_lat_min_below_lat_max(self) -> None:
        bounds = build_case_mapping.geo_bounds([-4_888_710.0, 260_218.0, -4_845_955.0, 302_973.0])
        self.assertLess(bounds["latMin"], bounds["latMax"])
        self.assertLess(bounds["latMin"], -4.0)
        self.assertGreater(bounds["latMax"], -6.0)


class AreaTest(unittest.TestCase):
    def test_spherical_zone(self) -> None:
        square = [[0, 0], [1, 0], [1, 1], [0, 1], [0, 0]]
        expected = R * R * math.radians(1) * (math.sin(math.radians(1)) - math.sin(0)) / 1e6
        self.assertAlmostEqual(build_case_mapping.spherical_area_km2(square), expected, delta=expected * 1e-9)
        self.assertAlmostEqual(expected, 3512.0, delta=2.0)

    def test_admission_area_agrees(self) -> None:
        square = [(0, 0, 0), (1, 0, 0), (1, 1, 0), (0, 1, 0), (0, 0, 0)]
        self.assertAlmostEqual(
            admission_report.polygon_area_km2(square),
            build_case_mapping.spherical_area_km2([list(point) for point in square]),
            places=6,
        )

    def test_area_is_orientation_independent(self) -> None:
        clockwise = [[0, 0], [0, 1], [1, 1], [1, 0], [0, 0]]
        counter = [[0, 0], [1, 0], [1, 1], [0, 1], [0, 0]]
        self.assertAlmostEqual(
            build_case_mapping.spherical_area_km2(clockwise),
            build_case_mapping.spherical_area_km2(counter),
            places=9,
        )


class PolygonTest(unittest.TestCase):
    POLYGON = [(0.0, 0.0), (2.0, 0.0), (2.0, 2.0), (0.0, 2.0)]

    def test_inside(self) -> None:
        self.assertTrue(admission_report.point_in_polygon(1.0, 1.0, self.POLYGON, [0, len(self.POLYGON)]))

    def test_outside(self) -> None:
        self.assertFalse(admission_report.point_in_polygon(3.0, 1.0, self.POLYGON, [0, len(self.POLYGON)]))

    def test_distance_zero_on_edge(self) -> None:
        self.assertEqual(admission_report.distance_to_rings_deg(2.0, 1.0, self.POLYGON, [0, len(self.POLYGON)]), 0.0)

    def test_distance_outside_scales_longitude(self) -> None:
        # one degree of longitude at lat 1° is cos(1°) shorter than a degree of latitude
        distance = admission_report.distance_to_rings_deg(3.0, 1.0, self.POLYGON, [0, len(self.POLYGON)])
        self.assertAlmostEqual(distance, math.cos(math.radians(1.0)), places=9)


class ExportTest(unittest.TestCase):
    RECORD = {
        "id": "ls_0001",
        "source_id": 1,
        "name": "",
        "type": "SL",
        "lon": 10.5,
        "lat": -3.25,
        "area": 12.5,
        "perimeter": 30.25,
        "desc": "",
    }

    def test_staging_payload(self) -> None:
        payload = export_checked_catalog.build_staging_payload([self.RECORD])
        self.assertEqual(
            payload,
            [
                {
                    "id": "ls_0001",
                    "source_id": 1,
                    "name": "",
                    "type": "SL",
                    "lon": 10.5,
                    "lat": -3.25,
                    "area": 12.5,
                    "perimeter": 30.25,
                    "desc": "",
                }
            ],
        )

    def test_digest_is_stable_and_order_sensitive(self) -> None:
        first = export_checked_catalog.records_digest([self.RECORD])
        self.assertEqual(first, export_checked_catalog.records_digest([dict(self.RECORD)]))
        other = dict(self.RECORD, area=13.0)
        self.assertNotEqual(first, export_checked_catalog.records_digest([other]))

    def test_sha256_of_file(self) -> None:
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "sample.bin"
            path.write_bytes(b"mars")
            self.assertEqual(
                export_checked_catalog.sha256_of(path),
                "d158be41db87d7055fa9548fe9c0e3cb685983fedb4f2d648d7a25c046d72d9d",
            )


class CaseMappingUnitTest(unittest.TestCase):
    def test_section_table_covers_every_type(self) -> None:
        self.assertEqual(set(build_case_mapping.SECTION), {"SL", "SP", "FL", "CL"})

    def test_csv_and_table_written(self) -> None:
        mapping = {
            "cases": [
                {
                    "id": "case-ls-0001",
                    "sourceId": 1,
                    "landslideId": "ls_0001",
                    "title": "滑移型案例 · ls_0001",
                    "preview": {"metersPerPixel": 20.0, "tiles": ["E-004_N00"]},
                    "geometry": {"point": {"rawType": "SL"}},
                    "highResolution": {
                        "lonMin": 9.0,
                        "lonMax": 10.0,
                        "latMin": -4.0,
                        "latMax": -3.0,
                        "windowM": 111_000.0,
                        "ctxWidthPx": 18_500,
                        "hiriseWidthPx": 370_000,
                        "requestedDisplayPx": 1024,
                        "status": "pending_request",
                    },
                }
            ]
        }
        with tempfile.TemporaryDirectory() as folder:
            csv_path = Path(folder) / "cases.csv"
            table_path = Path(folder) / "cases.md"
            build_case_mapping.write_csv(mapping, csv_path)
            build_case_mapping.write_table(mapping, table_path)
            text = csv_path.read_text(encoding="utf-8-sig")
            self.assertIn("case-ls-0001", text)
            self.assertIn("pending_request", text)
            self.assertEqual(text.count("\n"), 2)
            self.assertIn("case-ls-0001", table_path.read_text(encoding="utf-8"))


@unittest.skipUnless((HANDOFF / "01_catalog" / "mars_landslides.json").is_file(), f"交接包不存在：{HANDOFF}")
class HandoffIntegrationTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.report = admission_report.run(HANDOFF, check_sha=False)

    def test_admission_has_no_failures(self) -> None:
        failed = [check for check in self.report["checks"] if check["status"] == "fail"]
        self.assertEqual(failed, [], f"准入检查失败：{failed}")

    def test_verdict_is_conditional(self) -> None:
        self.assertEqual(self.report["verdict"], "conditional_pending_coordinate_verification")

    def test_record_count_and_types(self) -> None:
        self.assertEqual(self.report["dataset"]["records"], 2971)
        self.assertEqual(self.report["dataset"]["types"], {"SL": 1787, "SP": 321, "FL": 50, "CL": 813})

    def test_case_mapping_is_complete(self) -> None:
        mapping = build_case_mapping.build_mapping(HANDOFF, HANDOFF / "06_templates" / "cases12.todo.json")
        self.assertEqual(len(mapping["cases"]), 12)
        self.assertEqual(mapping["issues"], [])
        self.assertEqual(mapping["byType"], {"SL": 3, "SP": 3, "FL": 3, "CL": 3})
        for case in mapping["cases"]:
            self.assertEqual(case["status"], "draft")
            self.assertTrue(case["needsExpertConfirmation"])
            self.assertTrue(case["preview"]["sha256"])
            self.assertLess(case["highResolution"]["latMin"], case["highResolution"]["latMax"])

    def test_verdict_is_ready_with_evidence(self) -> None:
        report = admission_report.run(HANDOFF, check_sha=False, coordinate_evidence="unit-test evidence")
        self.assertEqual(report["verdict"], "ready_for_import")
        convention = [check for check in report["checks"] if check["id"] == "coords.latitude_convention"]
        self.assertEqual(len(convention), 1)
        self.assertEqual(convention[0]["status"], "pass")


class LatitudeConversionTest(unittest.TestCase):
    def test_zero_and_poles_are_fixed_points(self) -> None:
        for value in (0.0, 90.0, -90.0):
            self.assertAlmostEqual(admission_report.mars_latitude.planetographic_to_planetocentric(value), value)

    def test_shift_shrinks_the_magnitude(self) -> None:
        for value in (23.5, 45.0, 77.98, -45.0):
            converted = admission_report.mars_latitude.planetographic_to_planetocentric(value)
            self.assertLess(abs(converted), abs(value))
            self.assertEqual(converted > 0, value > 0)

    def test_sixty_degrees_uses_the_iau_ratio(self) -> None:
        expected = math.degrees(math.atan((3376200.0 / 3396190.0) ** 2 * math.tan(math.radians(60.0))))
        self.assertAlmostEqual(
            admission_report.mars_latitude.planetographic_to_planetocentric(60.0), expected, places=12
        )

    def test_round_trip(self) -> None:
        for value in (-70.0, -12.5, 0.0, 33.3, 70.0):
            back = admission_report.mars_latitude.planetocentric_to_planetographic(
                admission_report.mars_latitude.planetographic_to_planetocentric(value)
            )
            self.assertAlmostEqual(back, value, places=9)

    def test_max_shift_is_about_one_third_of_a_degree(self) -> None:
        self.assertAlmostEqual(admission_report.mars_latitude.max_shift_deg(), 0.338, places=3)

    def test_convert_latitudes_reports_the_shift(self) -> None:
        records = [{"id": "ls_0001", "lat": 45.0, "lon": 10.0}, {"id": "ls_0002", "lat": -45.0, "lon": 20.0}]
        converted, summary = export_checked_catalog.convert_latitudes(records, "planetographic")
        self.assertEqual(summary["from"], "planetographic")
        self.assertTrue(summary["applied"])
        self.assertEqual(summary["recordsShifted"], 2)
        self.assertAlmostEqual(summary["maxShiftDeg"], 0.337, places=2)
        self.assertEqual(converted[0]["lat_source"], 45.0)
        self.assertAlmostEqual(converted[0]["lat"], 44.664, places=2)
        self.assertEqual(converted[0]["lon"], 10.0)
        self.assertEqual(converted[0]["id"], "ls_0001")

    def test_convert_latitudes_can_pass_planetocentric_through(self) -> None:
        records = [{"id": "ls_0001", "lat": 45.0, "lon": 10.0}]
        converted, summary = export_checked_catalog.convert_latitudes(records, "planetocentric")
        self.assertFalse(summary["applied"])
        self.assertEqual(summary["recordsShifted"], 0)
        self.assertEqual(converted[0]["lat"], 45.0)

    def test_unknown_convention_is_rejected(self) -> None:
        with self.assertRaises(ValueError):
            export_checked_catalog.convert_latitudes([], "selenographic")


if __name__ == "__main__":
    unittest.main(verbosity=2)
