"""Unit tests for the week-4 content/asset checker (no private handoff needed)."""
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import check_content_assets as checker


def _content() -> dict:
    return {
        "version": "atlas-test-v1",
        "sourceDocument": {"id": "doc", "title": "文献", "url": "https://example.org", "note": ""},
        "types": [{"type": t, "caseIds": []} for t in checker.TYPES],
        "cases": [
            {
                "id": "case-ls-0001",
                "landslideId": "ls_0001",
                "type": "SL",
                "title": "标题",
                "tile": "E-084_N-08",
                "previewScale": "38.4 m/像素",
                "citationLocator": "3.2.1 节（PDF 第 8 页）",
            }
        ],
    }


def _catalog() -> dict:
    return {"ls_0001": {"id": "ls_0001", "type": "SL"}}


def _index() -> dict:
    return {
        "ls_0001": {
            "catalog_id": "ls_0001",
            "type": "SL",
            "source_tiles": ["E-084_N-08"],
            "native_window_world_m": [-1000.0, 0.0, 8595.0, 9000.0],
        }
    }


class WindowMetricsTest(unittest.TestCase):
    def test_metres_per_pixel_uses_the_224_pixel_crop(self):
        width, metres = checker.window_metrics(_index()["ls_0001"])
        self.assertAlmostEqual(width, 9595.0)
        self.assertAlmostEqual(metres, 9595.0 / 224.0)

    def test_scale_is_parsed_from_the_case_text(self):
        self.assertEqual(checker.scale_of({"previewScale": "190.9 m/像素"}), 190.9)
        self.assertIsNone(checker.scale_of({"previewScale": "未登记"}))


class BuildRowsTest(unittest.TestCase):
    def test_flags_a_scale_that_does_not_match_the_crop(self):
        content = _content()
        rows, problems = checker.build_rows(
            content, _catalog(), _index(), {}, set(), Path(".")
        )
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["image"], "暂无公开影像")
        self.assertEqual(rows[0]["catalog"], "仅在完整目录")
        self.assertTrue(any("previewScale" in p for p in problems))

    def test_reports_a_type_mismatch_between_text_and_catalogue(self):
        catalog = {"ls_0001": {"id": "ls_0001", "type": "SP"}}
        _, problems = checker.build_rows(_content(), catalog, _index(), {}, set(), Path("."))
        self.assertTrue(any("类别" in p for p in problems))

    def test_accepts_a_matching_public_sample(self):
        content = _content()
        content["cases"][0]["previewScale"] = "42.8 m/像素"
        with tempfile.TemporaryDirectory() as tmp:
            sample = Path(tmp) / "samples" / "ls_0001.png"
            sample.parent.mkdir()
            sample.write_bytes(b"mars")
            digest = checker.sha256_of(sample)
            samples = {"ls_0001": {"imageUrl": "/samples/ls_0001.png", "sha256": digest, "type": "SL"}}
            index = _index()
            index["ls_0001"]["image_sha256"] = digest
            rows, problems = checker.build_rows(
                content, _catalog(), index, samples, {"ls_0001"}, Path(tmp)
            )
        self.assertEqual(problems, [])
        self.assertTrue(rows[0]["image"].startswith("有"))
        self.assertEqual(rows[0]["catalog"], "已在公开子集")

    def test_flags_a_hash_mismatch(self):
        content = _content()
        content["cases"][0]["previewScale"] = "42.8 m/像素"
        with tempfile.TemporaryDirectory() as tmp:
            sample = Path(tmp) / "samples" / "ls_0001.png"
            sample.parent.mkdir()
            sample.write_bytes(b"mars")
            samples = {"ls_0001": {"imageUrl": "/samples/ls_0001.png", "sha256": "0" * 64, "type": "SL"}}
            _, problems = checker.build_rows(content, _catalog(), _index(), samples, set(), Path(tmp))
        self.assertTrue(any("sha256" in p for p in problems))


if __name__ == "__main__":
    unittest.main()
