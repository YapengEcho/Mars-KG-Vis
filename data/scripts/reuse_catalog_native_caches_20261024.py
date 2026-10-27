"""Finite, ahead-of-queue reuse of lossless caches; never changes labels or deletes originals."""
from collections import defaultdict
from pathlib import Path
import json
import os
import sys
import traceback

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / '.venv/Lib/site-packages'), str(ROOT / 'data/scripts')]
import numpy as np
import tifffile
import run_catalog_expansion_20261024 as W

P = W.P
OUT = W.OUT / 'native_cache_reuse'
BUFFER = 20


def overlap(a, b):
    cut = [max(a[0], b[0]), max(a[1], b[1]), min(a[2], b[2]), min(a[3], b[3])]
    return cut if cut[0] < cut[2] and cut[1] < cut[3] else None


def covered(box, windows):
    cuts = [c for w in windows if (c := overlap(box, w))]
    edges = sorted({box[0], box[2]} | {x for c in cuts for x in [c[0], c[2]]})
    for left, right in zip(edges, edges[1:]):
        intervals = sorted((c[1], c[3]) for c in cuts if c[0] <= left and c[2] >= right)
        y = box[1]
        for low, high in intervals:
            if low > y:
                return False
            y = max(y, high)
        if y < box[3]:
            return False
    return True


def ahead(tile, queue):
    state = P.read(W.OUT / 'worker_status.json')
    return (state['status'] == 'running' and
            queue.index(tile) + 1 >= state['tile_index'] + BUFFER and
            state.get('current_tile') != tile)


def prepare():
    spec, plan = W.check()
    assert not (OUT / 'protocol.json').exists(), 'Preserve frozen reuse protocol'
    pools = defaultdict(list)
    manifests = defaultdict(list)
    for path in sorted((ROOT / 'data/processed').glob('*/native_context/*/manifest.json')):
        d = P.read(path)
        tile = d['tile']
        if tile not in spec['blocks']:
            continue
        assert np.allclose(d['geotransform'], P.tile_gt(tile), atol=1e-6, rtol=0), str(path)
        assert [d['tile_width'], d['tile_height']] == [47420, 47420]
        manifests[tile].append(dict(path=str(path.resolve()), sha256=P.sha(path), original=d['original']))
        for e in d['entries']:
            file = Path(e['path'])
            if e.get('stride') != 1 or not file.is_file():
                continue
            box = list(map(int, e['window_origin']))
            assert box == e['window_origin'] and e['shape'] == [box[3]-box[1], box[2]-box[0]]
            assert 0 <= box[0] < box[2] <= 47420 and 0 <= box[1] < box[3] <= 47420
            stat = file.stat()
            pools[tile].append(dict(path=str(file.resolve()), sha256=e['sha256'], window_px=box,
                                    bytes=stat.st_size, mtime_ns=stat.st_mtime_ns))
    candidates = []
    for tile, sources in pools.items():
        boxes = spec['blocks'][tile]
        if (W.NATIVE / tile).exists() or not ahead(tile, spec['queue']):
            continue
        if W.local_original(tile, plan) is not None:
            continue
        if not all(covered(b, [s['window_px'] for s in sources]) for b in boxes):
            continue
        relevant = {s['path']: s for s in sources if any(overlap(s['window_px'], b) for b in boxes)}
        candidates.append(dict(tile=tile, queue_position=spec['queue'].index(tile)+1, blocks=boxes,
                               sources=list(relevant.values()), manifests=manifests[tile],
                               avoided_archive_bytes=plan['download_rows'][tile]['size_bytes']))
    candidates.sort(key=lambda c: c['queue_position'])
    OUT.mkdir(parents=True, exist_ok=True)
    W.write(OUT / 'protocol.json', dict(status='prepared_finite_cache_reuse', candidates=candidates,
        files_sha256={str(Path(__file__)): P.sha(__file__), str(Path(W.__file__)): P.sha(W.__file__),
                      str(W.OUT / 'worker_protocol.json'): P.sha(W.OUT / 'worker_protocol.json')},
        preserve_original_files=True, create_missing_pixels=False, queue_buffer=BUFFER,
        created_utc=W.utc()))
    W.write(OUT / 'status.json', dict(status='prepared', finite_tiles=len(candidates), completed_tiles=0))
    print('PREPARED', len(candidates), 'fully covered tiles', flush=True)


def assemble(sources, boxes, stage):
    """Decode one strip/tile at a time; require every pixel and consistent overlaps."""
    records = []
    try:
        for i, box in enumerate(boxes):
            shape = (box[3]-box[1], box[2]-box[0])
            raw_path, mask_path = stage / f'block_{i:03d}.raw', stage / f'block_{i:03d}.coverage'
            assert not raw_path.exists() and not mask_path.exists()
            raw = np.memmap(raw_path, dtype=np.uint8, mode='w+', shape=shape)
            mask = np.memmap(mask_path, dtype=np.bool_, mode='w+', shape=shape)
            mask[:] = False
            records.append(dict(box=box, raw=raw, mask=mask, raw_path=raw_path, mask_path=mask_path))
        for src in sources:
            path = Path(src['path'])
            stat = path.stat()
            assert (stat.st_size, stat.st_mtime_ns) == (src['bytes'], src['mtime_ns']), str(path)
            assert P.sha(path) == src['sha256'], 'Old lossless cache hash mismatch: ' + str(path)
            sx, sy, ex, ey = src['window_px']
            with tifffile.TiffFile(path) as tf:
                assert len(tf.pages) == 1
                page = tf.pages[0]
                assert list(page.shape) == [ey-sy, ex-sx] and page.dtype == np.uint8
                assert page.samplesperpixel == 1
                decoded_area = 0
                for segment, index, _ in page.segments(maxworkers=1):
                    assert segment is not None and segment.ndim == 4 and segment.shape[0] == segment.shape[3] == 1
                    assert index[0] == index[1] == index[4] == 0
                    yy, xx = index[2], index[3]
                    hh, ww = min(segment.shape[1], ey-sy-yy), min(segment.shape[2], ex-sx-xx)
                    assert hh > 0 and ww > 0
                    pixels = segment[0, :hh, :ww, 0]
                    region = [sx+xx, sy+yy, sx+xx+ww, sy+yy+hh]
                    decoded_area += hh*ww
                    for record in records:
                        cut = overlap(region, record['box'])
                        if cut is None:
                            continue
                        bx, by = record['box'][:2]
                        target = np.s_[cut[1]-by:cut[3]-by, cut[0]-bx:cut[2]-bx]
                        value = pixels[cut[1]-region[1]:cut[3]-region[1], cut[0]-region[0]:cut[2]-region[0]]
                        previous, seen = record['raw'][target], record['mask'][target]
                        if seen.any():
                            assert np.array_equal(previous[seen], value[seen]), 'Conflicting native pixels in overlapping caches'
                        previous[:] = value
                        seen[:] = True
                assert decoded_area == (ex-sx)*(ey-sy), 'Native decoder did not cover source exactly'
            # Detect source changes while reading, too.
            assert (path.stat().st_size, path.stat().st_mtime_ns) == (src['bytes'], src['mtime_ns'])
        for record in records:
            for row in range(0, record['mask'].shape[0], 128):
                assert record['mask'][row:row+128].all(), 'Missing native pixels; download original instead'
            record['raw'].flush()
        return records
    except BaseException:
        close_records(records)
        raise


def close_records(records):
    for record in records:
        for key in ['raw', 'mask']:
            if not record[key]._mmap.closed:
                record[key]._mmap.close()
        for key in ['raw_path', 'mask_path']:
            path = record[key]
            assert path.resolve().is_relative_to(OUT.resolve()) and path.suffix in ['.raw', '.coverage']
            path.unlink(missing_ok=True)


def smoke():
    stage = OUT / 'smoke'
    assert not stage.exists()
    stage.mkdir(parents=True)
    full = np.random.default_rng(20261024).integers(0, 256, (40, 48), np.uint8)
    sources = []
    for i, box in enumerate([[0, 0, 32, 40], [20, 0, 48, 40]]):
        path = stage / f'source_{i}.tif'
        kwargs = dict(tile=(16, 16)) if i else dict(rowsperstrip=7)
        tifffile.imwrite(path, full[box[1]:box[3], box[0]:box[2]], compression='deflate', metadata=None, **kwargs)
        sources.append(dict(path=str(path), window_px=box, bytes=path.stat().st_size,
                            mtime_ns=path.stat().st_mtime_ns, sha256=P.sha(path)))
    boxes = [[4, 3, 42, 36], [30, 12, 47, 39]]
    assert all(covered(b, [s['window_px'] for s in sources]) for b in boxes)
    records = assemble(sources, boxes, stage)
    try:
        for r in records:
            x0, y0, x1, y1 = r['box']
            assert np.array_equal(r['raw'], full[y0:y1, x0:x1])
    finally:
        close_records(records)
    try:
        assemble(sources[:1], boxes, stage)
        raise RuntimeError('Missing coverage was accepted')
    except AssertionError as error:
        assert 'Missing native pixels' in str(error)
    bad = stage / 'conflicting.tif'
    b = sources[1]['window_px']
    tifffile.imwrite(bad, 255-full[:, b[0]:b[2]], compression='deflate', metadata=None)
    conflicting = dict(sources[1], path=str(bad), bytes=bad.stat().st_size,
                       mtime_ns=bad.stat().st_mtime_ns, sha256=P.sha(bad))
    try:
        assemble([sources[0], conflicting], boxes, stage)
        raise RuntimeError('Conflicting overlap was accepted')
    except AssertionError as error:
        assert 'Conflicting native pixels' in str(error)
    W.write(OUT / 'readiness.json', dict(status='passed_strip_tile_offsets_coverage_and_overlap_checks',
                                       protocol_sha256=P.sha(OUT / 'protocol.json'), utc=W.utc()))
    print('PASSED_CACHE_REUSE_SMOKE', flush=True)


def run():
    protocol = P.read(OUT / 'protocol.json')
    for path, digest in protocol['files_sha256'].items():
        assert P.sha(path) == digest, path
    assert P.read(OUT / 'readiness.json')['protocol_sha256'] == P.sha(OUT / 'protocol.json')
    spec, plan = W.check()
    state = dict(status='running', pid=os.getpid(), finite_tiles=len(protocol['candidates']),
                 completed_tiles=0, reused_tiles=[], skipped_tiles=[], failed_tiles=[], avoided_archive_bytes=0)
    W.write(OUT / 'status.json', state)
    for candidate in protocol['candidates']:
        tile = candidate['tile']
        state['current_tile'] = tile
        W.write(OUT / 'status.json', state)
        stage = OUT / 'staging' / tile
        records = []
        try:
            destination = W.NATIVE / tile
            if destination.exists() or not ahead(tile, spec['queue']) or W.local_original(tile, plan) is not None:
                state['skipped_tiles'].append(tile)
                continue
            for source in candidate['manifests']:
                assert P.sha(source['path']) == source['sha256']
                assert not Path(source['original']['path']).exists(), 'Retained original exists; use main worker'
            assert not stage.exists(), 'Preserve incomplete staging for inspection'
            stage.mkdir(parents=True)
            need = 3*sum((b[2]-b[0])*(b[3]-b[1]) for b in candidate['blocks'])
            assert W.shutil.disk_usage(stage).free > need+(6 << 30), 'Insufficient staging reserve'
            records = assemble(candidate['sources'], candidate['blocks'], stage)
            entries = []
            for i, record in enumerate(records):
                box = record['box']; gt = P.tile_gt(tile)
                block_gt = [gt[0]+box[0]*gt[1], gt[1], 0., gt[3]+box[1]*gt[5], 0., gt[5]]
                h, w = record['raw'].shape
                entry = W.preserve_region(record['raw'], block_gt, [0, 0, w, h], stage / f'block_{i:03d}.tif')
                entry.update(path=str((destination / f'block_{i:03d}.tif').resolve()), window_px=box)
                entries.append(entry)
            original = candidate['manifests'][0]['original']
            manifest = dict(status='verified_complete_native_regions', tile=tile, entries=entries,
                original=dict(path=original['path'], bytes=original['bytes'], sha256=None,
                              note='Historical retired original; full-file hash unavailable, never authorized for deletion here'),
                source_origin='reconstructed_from_existing_lossless_native_contexts',
                source_url=plan['download_rows'][tile]['url'], original_retired_before_current_run=True,
                required_windows_covered=True, required_window_count=spec['required_window_count'][tile],
                coverage_plan_sha256=P.sha(W.OUT / 'worker_protocol.json'),
                input_native_manifests=candidate['manifests'], input_native_sources=candidate['sources'],
                all_source_file_hashes_verified=True, overlapping_native_pixels_equal=True,
                completed_utc=W.utc())
            W.write(stage / 'manifest.json', manifest)
            close_records(records); records = []
            assert ahead(tile, spec['queue']) and W.local_original(tile, plan) is None
            assert not destination.exists()
            # Atomic publication: parent worker sees either no tile or all verified blocks.
            stage.rename(destination)
            state['reused_tiles'].append(tile)
            state['completed_tiles'] += 1
            state['avoided_archive_bytes'] += candidate['avoided_archive_bytes']
            print('REUSED_VERIFIED_NATIVE_CACHE', tile, len(entries), 'blocks', flush=True)
        except Exception as error:
            state['failed_tiles'].append(dict(tile=tile, error=repr(error)))
            traceback.print_exc()
        finally:
            if records:
                close_records(records)
            W.write(OUT / 'status.json', state)
    state.update(status='completed_with_skips' if state['failed_tiles'] else 'completed', finished_utc=W.utc())
    W.write(OUT / 'status.json', state)


if __name__ == '__main__':
    mode = sys.argv[1]
    if mode == 'prepare':
        prepare()
    elif mode == 'smoke':
        smoke()
    elif mode == 'run':
        with (OUT / 'worker.log').open('a', encoding='utf-8', buffering=1) as stream:
            sys.stdout = stream; sys.stderr = stream
            run()
    else:
        raise ValueError(mode)
