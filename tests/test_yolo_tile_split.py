import json
import sys

import numpy as np
import pytest
import tifffile

from data.scripts import make_yolo_dataset as maker


def tile(bounds, *sources):
    return {
        "W": 32, "H": 32, "bounds": bounds,
        "small": [(0, 5, 5, 3, 3, source) for source in sources], "big": [],
    }


def test_auto_plan_keeps_overlapping_tiles_and_shared_landslides_together():
    tiles = {
        "left": tile((0, 0, 4, 4), 10),
        "middle": tile((4, 0, 8, 4), 10),
        "overlapping": tile((7, 0, 11, 4)),
        "right": tile((11, 0, 15, 4), 20),
        "far": tile((15, 0, 19, 4), 30),
        "background": tile((19, 0, 23, 4)),
    }
    positives = {name for name, metadata in tiles.items() if metadata["small"]}
    plan = maker.make_tile_split_plan(tiles, positives, .2, .15, 42)

    assert plan == maker.make_tile_split_plan(dict(reversed(list(tiles.items()))), positives, .2, .15, 42)
    assert {plan[name] for name in positives} == {"train", "val", "test"}
    assert plan["left"] == plan["middle"] == plan["overlapping"]
    assert set(plan) == set(tiles)


def test_plan_rejects_overlapping_footprints_even_without_shared_landslides():
    tiles = {"a": tile((0, 0, 4, 4), 1), "b": tile((3, 0, 7, 4), 2)}
    with pytest.raises(SystemExit, match="实际覆盖范围重叠"):
        maker.assert_tile_split_consistency(tiles, {"a": "train", "b": "test"})


def test_plan_requires_enough_independent_positive_groups():
    tiles = {"a": tile((0, 0, 4, 4), 1), "b": tile((4, 0, 8, 4))}
    with pytest.raises(SystemExit, match="只有 1 个含滑坡的独立图幅组"):
        maker.make_tile_split_plan(tiles, {"a"}, .2, .15, 42)
    assert maker.make_tile_split_plan(tiles, {"a"}, 0, 0, 42) == dict.fromkeys(tiles, "train")


def test_cli_keeps_overlapping_positive_and_background_crops_in_one_split(tmp_path, monkeypatch):
    source = tmp_path / "ctx"
    boxes = {}
    scale = maker.R_MARS * maker.DEG
    for source_id, west in enumerate((0, 4, 8), 1):
        name = f"E{west:03d}_N00"
        folder = source / name
        folder.mkdir(parents=True)
        tifffile.imwrite(
            folder / f"MurrayLab_CTX_V01_{name}.tif",
            np.full((32, 32), 10, dtype=np.uint8),
            extratags=[
                (33550, "d", 3, (scale / 8, scale / 8, 0), False),
                (33922, "d", 6, (0, 0, 0, west * scale, 4 * scale, 0), False),
            ],
        )
        boxes[source_id] = (west + 0.5, 3, west + 1, 3.5)

    (tmp_path / "mars_landslides.json").write_text(
        json.dumps([{"source_id": source_id, "type": "SL"} for source_id in boxes]), encoding="utf-8",
    )
    out = tmp_path / "dataset"
    monkeypatch.setattr(maker, "RAW", tmp_path)
    monkeypatch.setattr(maker, "CTX_DIR", source)
    monkeypatch.setattr(maker, "load_shp_polygons", lambda: boxes)
    monkeypatch.setattr(sys, "argv", ["make_yolo_dataset.py", "--patch", "16", "--overlap", "4",
                                      "--neg-ratio", "100", "--out", str(out)])
    maker.main()

    report = json.loads((out / "quality_report.json").read_text(encoding="utf-8"))
    assert report["split_mode"] == "tile_auto"
    for split, names in report["tiles_per_split"].items():
        assert len(names) == 1
        name, = names
        images = list((out / "images" / split).glob("*.png"))
        assert images and all(path.name.startswith(f"{name}_") for path in images)
        assert (out / "labels" / split / f"{name}_00000_00000.txt").read_text().strip()
        assert not (out / "labels" / split / f"{name}_00012_00000.txt").read_text().strip()
