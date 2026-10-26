"""Synthetic-input checks for the mobile catalog exporter."""

from copy import deepcopy
import json
from pathlib import Path
import unittest
from tempfile import TemporaryDirectory

try:
    from .export_catalog import build_catalog, export_catalog
except ImportError:
    from export_catalog import build_catalog, export_catalog


def sample_rows():
    return [
        {"id": f"ls_{index:04d}", "source_id": index, "name": "", "type": kind,
         "lon": -100 + index, "lat": -20 + index, "area": None, "desc": ""}
        for index, kind in enumerate(("SL", "SP", "FL", "CL"), start=1)
    ]


def convert(rows, **kwargs):
    return build_catalog(rows, source_title="Synthetic test fixture",
                         coordinate_system="planetocentric", version="test-v1", **kwargs)


class ExportCatalogTests(unittest.TestCase):
    def test_preserves_ids_types_and_unknown_values(self):
        rows = sample_rows()
        rows[0]["area"] = 12.5
        rows[1]["name"] = "A sourced name"
        rows[1]["desc"] = "A supplied description"
        catalog = convert(rows)
        self.assertEqual([record["type"] for record in catalog["records"]], ["SL", "SP", "FL", "CL"])
        self.assertEqual([record["sourceId"] for record in catalog["records"]], [1, 2, 3, 4])
        self.assertEqual([record["id"] for record in catalog["records"]], [row["id"] for row in rows])
        self.assertEqual(catalog["records"][0]["name"], "滑坡 1")
        self.assertEqual(catalog["records"][0]["areaKm2"], 12.5)
        self.assertIsNone(catalog["records"][1]["areaKm2"])
        self.assertEqual(catalog["records"][1]["summary"], rows[1]["desc"])
        self.assertTrue(all(record["imageUrl"] is None and record["isDemo"] is False for record in catalog["records"]))
        self.assertEqual(len(convert(rows, limit=2)["records"]), 2)

    def test_rejects_invalid_coordinates_and_unknown_class_even_beyond_limit(self):
        for field, value in (("lon", 181), ("lat", -91), ("lat", float("nan")),
                             ("lon", True), ("type", "slump")):
            with self.subTest(field=field, value=value):
                rows = deepcopy(sample_rows())
                rows[-1][field] = value
                with self.assertRaises(ValueError):
                    convert(rows, limit=1)

    def test_rejects_duplicate_record_or_source_ids(self):
        for field in ("id", "source_id"):
            with self.subTest(field=field):
                rows = sample_rows()
                rows[1][field] = rows[0][field]
                with self.assertRaisesRegex(ValueError, "duplicate"):
                    convert(rows)

    def test_writes_only_explicit_output_and_refuses_input_overwrite(self):
        with TemporaryDirectory() as directory:
            source = Path(directory) / "input.json"
            source.write_text(json.dumps(sample_rows()), encoding="utf-8")
            original = source.read_bytes()
            options = dict(source_title="Fixture", coordinate_system="planetocentric", version="v1")
            with self.assertRaisesRegex(ValueError, "overwrite"):
                export_catalog(source, source, **options)
            self.assertEqual(source.read_bytes(), original)
            target = Path(directory) / "public" / "catalog.json"
            export_catalog(source, target, limit=2, **options)
            self.assertEqual(len(json.loads(target.read_text(encoding="utf-8"))["records"]), 2)
            self.assertEqual(source.read_bytes(), original)


if __name__ == "__main__":
    unittest.main()
