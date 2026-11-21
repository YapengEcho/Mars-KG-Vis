"""Tests for the week-5 figure generator's source checks and replacement rule."""

from __future__ import annotations

import json
import shutil
import sys
import tempfile
import unittest
from pathlib import Path

from PIL import Image

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "scripts"))

import capture_demo_figures as figures  # noqa: E402


class VerifyPreviewTest(unittest.TestCase):
    def test_matching_preview_passes(self) -> None:
        sample = figures.sample_index()["ls_0493"]
        source = figures.SAMPLES / "ls_0493.png"
        self.assertTrue(source.exists(), "公开预览缺失，测试需要仓库自带的 8 张样例")
        self.assertEqual(figures.verify_preview(source, sample), [])

    def test_tampered_preview_is_rejected(self) -> None:
        sample = figures.sample_index()["ls_0493"]
        with tempfile.TemporaryDirectory() as tmp:
            copy = Path(tmp) / "ls_0493.png"
            shutil.copyfile(figures.SAMPLES / "ls_0493.png", copy)
            with Image.open(copy) as image:
                image.convert("L").save(copy, optimize=True)
            copy.write_bytes(copy.read_bytes() + b"tampered")
            problems = figures.verify_preview(copy, sample)
        self.assertTrue(problems)
        self.assertIn("SHA-256", problems[0])
        self.assertIn("疑似同名图混入", problems[0])

    def test_wrong_size_is_reported(self) -> None:
        sample = dict(figures.sample_index()["ls_0493"])
        sample["width"], sample["height"] = 512, 512
        problems = figures.verify_preview(figures.SAMPLES / "ls_0493.png", sample)
        self.assertTrue(any("尺寸" in item for item in problems))


class ReplacementTest(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.tmp = Path(self._tmp.name)
        self.name = "demo-ctx-01-ls0493.png"
        self.previous = {self.name: {"sha256": figures.sha256_of(figures.SAMPLES / "ls_0493.png")}}

    def tearDown(self) -> None:
        self._tmp.cleanup()

    def _write(self, path: Path, size: tuple[int, int]) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        Image.new("L", size, 128).save(path)

    def test_generated_file_is_regenerated(self) -> None:
        target = self.tmp / self.name
        self._write(target, (1792, 1792))
        previous = {self.name: {"sha256": figures.sha256_of(target)}}
        self.assertIsNone(figures.replacement_of(self.name, self.tmp, None, previous))

    def test_hand_placed_file_is_kept(self) -> None:
        target = self.tmp / self.name
        self._write(target, (1500, 1500))
        found = figures.replacement_of(self.name, self.tmp, None, self.previous)
        self.assertIsNotNone(found)
        path, mode = found  # type: ignore[misc]
        self.assertEqual(mode, "replacement")
        self.assertEqual(path, target)

    def test_native_dir_wins(self) -> None:
        native = self.tmp / "native"
        self._write(native / self.name, (1600, 1600))
        found = figures.replacement_of(self.name, self.tmp, native, self.previous)
        self.assertIsNotNone(found)
        path, mode = found  # type: ignore[misc]
        self.assertEqual(mode, "native")
        self.assertEqual(path, native / self.name)

    def test_native_scale_uses_window(self) -> None:
        content = json.loads(figures.CONTENT.read_text(encoding="utf-8"))
        case = next(item for item in content["cases"] if item["landslideId"] == "ls_0493")
        path = self.tmp / self.name
        self._write(path, (1800, 1500))
        meta = figures.describe_native(path, case, "native")
        window_m = figures.scale_of(case) * figures.PREVIEW_SIDE
        self.assertIsNone(meta["upscale"])
        self.assertAlmostEqual(meta["fileScale"], round(window_m / 1800, 3), places=3)
        self.assertEqual(meta["size"], [1800, 1500])


if __name__ == "__main__":
    unittest.main()
