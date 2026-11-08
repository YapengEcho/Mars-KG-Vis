"""Tests for the week-3 atlas content export."""

from __future__ import annotations

import csv
import json
import tempfile
import unittest
from pathlib import Path

from export_atlas_content import (
    DEFAULT_CONTENT,
    case_rows,
    check_content,
    citation_rows,
    export,
    load_content,
    search_blocks,
)


class AtlasContentTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.content = load_content(DEFAULT_CONTENT)

    def test_covers_four_types_with_three_cases_each(self) -> None:
        self.assertEqual([t["type"] for t in self.content["types"]], ["SL", "SP", "FL", "CL"])
        for kind in ("SL", "SP", "FL", "CL"):
            self.assertEqual(sum(1 for c in self.content["cases"] if c["type"] == kind), 3)
        self.assertEqual(len(self.content["cases"]), 12)
        self.assertEqual(len(self.content["comparisons"]), 2)

    def test_content_holds_no_catalogue_values(self) -> None:
        for case in self.content["cases"]:
            for field in ("longitude", "latitude", "areaKm2", "sourceId"):
                self.assertNotIn(field, case)
            self.assertRegex(case["landslideId"], r"^ls_\d{4}$")

    def test_rejects_unregistered_case(self) -> None:
        content = json.loads(json.dumps(self.content, ensure_ascii=False))
        content["types"][0]["caseIds"] = [
            case_id for case_id in content["types"][0]["caseIds"] if case_id != "case-ls-0493"
        ]
        problems = check_content(content)
        self.assertTrue(any("未登记" in problem for problem in problems))

    def test_rejects_duplicate_case_ids(self) -> None:
        content = json.loads(json.dumps(self.content, ensure_ascii=False))
        content["cases"][1]["id"] = content["cases"][0]["id"]
        problems = check_content(content)
        self.assertTrue(any("重复" in problem for problem in problems))

    def test_cases_join_the_catalogue(self) -> None:
        catalogue = {
            "ls_0493": {"id": "ls_0493", "lon": -82.1148, "lat": -4.6952, "area": 200.48},
        }
        rows = case_rows(self.content, catalogue)
        first = next(row for row in rows if row["case_id"] == "case-ls-0493")
        self.assertEqual(first["in_catalog"], "yes")
        self.assertEqual(first["area_km2"], 200.48)
        missing = next(row for row in rows if row["case_id"] == "case-ls-0592")
        self.assertEqual(missing["in_catalog"], "no")

    def test_citations_cover_types_comparisons_and_cases(self) -> None:
        rows = citation_rows(self.content)
        kinds = {row["owner_kind"] for row in rows}
        self.assertEqual(kinds, {"type", "comparison", "case"})
        expected = (
            sum(len(entry["citations"]) for entry in self.content["types"])
            + sum(len(item["citations"]) for item in self.content["comparisons"])
            + len(self.content["cases"])
        )
        self.assertEqual(len(rows), expected)

    def test_search_blocks_are_text_only(self) -> None:
        blocks = search_blocks(self.content)
        self.assertEqual(len(blocks), 4 + 12 + 2)
        joined = "\n".join(block["text"] for block in blocks)
        self.assertIn("表 2", joined)
        self.assertNotRegex(joined, r"°[EW]")

    def test_export_writes_every_artifact(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            out = Path(directory)
            summary = export(DEFAULT_CONTENT, out, None)
            self.assertEqual(summary["cases"], 12)
            self.assertEqual(summary["casesInCatalog"], 0)
            names = {path.name for path in out.iterdir()}
            self.assertEqual(
                names,
                {
                    "atlas-content.json",
                    "atlas-cases.csv",
                    "atlas-citations.csv",
                    "atlas-search.jsonl",
                    "atlas-export-summary.json",
                    "atlas-limits.md",
                },
            )
            with (out / "atlas-cases.csv").open(encoding="utf-8-sig", newline="") as handle:
                rows = list(csv.DictReader(handle))
            self.assertEqual(len(rows), 12)
            lines = (out / "atlas-search.jsonl").read_text(encoding="utf-8").splitlines()
            self.assertEqual(len(lines), 18)
            self.assertEqual(json.loads(lines[0])["kind"], "type")


if __name__ == "__main__":
    unittest.main()
