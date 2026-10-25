import importlib
import io
from pathlib import Path
import sys
import zipfile

import numpy as np
import pytest
from types import SimpleNamespace
import tifffile

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "data/scripts"))
m = importlib.import_module("fetch_v2_train_tiles")


def tif_bytes():
    buf = io.BytesIO()
    tifffile.imwrite(buf, np.zeros((1024, 1024), dtype=np.uint8),
        extratags=[(33550, "d", 3, (5, 5, 0), False), (33922, "d", 6, (0, 0, 0, 0, 100, 0), False)])
    return buf.getvalue()


def test_ranges_cover_bytes_once():
    for n in (1, 2, 7, 101):
        assert [v for a, b in m.segments(n, 4) for v in range(a, b + 1)] == list(range(n))


def test_truncated_tif_is_not_skipped(tmp_path):
    folder = tmp_path / "tile"
    folder.mkdir()
    p = folder / "MurrayLab_CTX_V01_E000_N00_Mosaic.tif"
    p.write_bytes(tif_bytes())
    assert m.existing_tif(tmp_path, "E000_N00")
    p.write_bytes(p.read_bytes()[:-500])
    assert m.existing_tif(tmp_path, "E000_N00") is None


@pytest.mark.parametrize("truncated", [False, True])
def test_zip_deleted_only_after_verified_extraction(tmp_path, truncated):
    p = tmp_path / "tile.zip"
    data = tif_bytes()
    with zipfile.ZipFile(p, "w") as z:
        z.writestr("tile/MurrayLab_CTX_V01_E000_N00_Mosaic.tif", data[:-500] if truncated else data)
    if truncated:
        with pytest.raises(ValueError, match="truncated"):
            m.extract_verified({"tile": "E000_N00"}, p, tmp_path)
        assert p.exists()
        assert not (tmp_path / "tile/MurrayLab_CTX_V01_E000_N00_Mosaic.tif").exists()
    else:
        assert m.extract_verified({"tile": "E000_N00"}, p, tmp_path)["shape"] == [1024, 1024]
        assert not p.exists()


def test_target_is_confined(tmp_path):
    with pytest.raises(ValueError, match="outside"):
        m.confined(tmp_path, tmp_path.parent / "outside.zip")


def test_server_ignoring_range_keeps_original_bytes(monkeypatch, tmp_path):
    p = tmp_path / "part.bin"
    p.write_bytes(b"abc")
    def run(args, **kwargs):
        Path(args[args.index("--dump-header")+1]).write_text("HTTP/1.1 200 OK\n")
        Path(args[args.index("--output")+1]).write_bytes(b"whole file")
        return SimpleNamespace(returncode=0, stderr="")
    monkeypatch.setattr(m.subprocess, "run", run)
    with pytest.raises(ValueError, match="exact byte range"):
        m.download_segment("https://example.test/data", p, 0, 5, 6)
    assert p.read_bytes() == b"abc"


def test_interruption_resumes_from_received_bytes(monkeypatch, tmp_path):
    p = tmp_path / "part.bin"
    calls = []
    def run(args, **kwargs):
        request = args[args.index("--range")+1]
        calls.append(request)
        Path(args[args.index("--dump-header")+1]).write_text(
            f"HTTP/1.1 206 Partial Content\nContent-Range: bytes {request}/6\n")
        Path(args[args.index("--output")+1]).write_bytes(b"abc" if len(calls)==1 else b"def")
        return SimpleNamespace(returncode=56 if len(calls)==1 else 0, stderr="interrupted")
    monkeypatch.setattr(m.subprocess, "run", run)
    monkeypatch.setattr(m.time, "sleep", lambda _: None)
    m.download_segment("https://example.test/data", p, 0, 5, 6)
    assert p.read_bytes() == b"abcdef"
    assert calls == ["0-5", "3-5"]


@pytest.mark.parametrize("valid_partial", [False, True])
def test_stuck_segment_is_bounded_and_resumes_only_verified_bytes(monkeypatch, tmp_path, valid_partial):
    p = tmp_path / "part.bin"
    calls = []
    def run(args, **kwargs):
        assert kwargs["timeout"] == 210
        assert args[args.index("--max-time")+1] == "180"
        request = args[args.index("--range")+1]
        calls.append(request)
        headers = Path(args[args.index("--dump-header")+1])
        chunk = Path(args[args.index("--output")+1])
        if len(calls) == 1:
            headers.write_text(f"HTTP/1.1 206 Partial Content\nContent-Range: bytes {request}/6\n" if valid_partial else "")
            chunk.write_bytes(b"abc")
            raise m.subprocess.TimeoutExpired(args, kwargs["timeout"])
        headers.write_text(f"HTTP/1.1 206 Partial Content\nContent-Range: bytes {request}/6\n")
        chunk.write_bytes(b"def" if valid_partial else b"abcdef")
        return SimpleNamespace(returncode=0, stderr="")
    monkeypatch.setattr(m.subprocess, "run", run)
    monkeypatch.setattr(m.time, "sleep", lambda _: None)
    m.download_segment("https://example.test/data", p, 0, 5, 6)
    assert p.read_bytes() == b"abcdef"
    assert calls == ["0-5", "3-5" if valid_partial else "0-5"]
