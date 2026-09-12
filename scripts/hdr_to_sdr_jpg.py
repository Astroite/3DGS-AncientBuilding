"""Tame blown-highlight / high-contrast JPEG frames down to a flatter SDR look.

Frames extracted from HDR-shot video often keep the harsh highlight clipping
("glow") and punchy contrast of the source even though they are plain 8-bit
sRGB JPEGs. This applies a soft-knee highlight compressor (identity below the
knee, so normally-exposed midtones are untouched; smoothly asymptotic toward
1.0 above it, so blown/near-white regions lose their harsh glow) plus a mild
global contrast reduction, which helps feature matching in photogrammetry
tools (Postshot/RealityScan) that expect consistent, non-clipped exposure
across a trajectory.

The source folder is renamed to `<name>_hdr_backup` (instant, same-volume
rename, not a copy) and a fresh folder with the original name is written with
the tone-mapped output, so the "overwrite in place" result still has an
untouched backup next to it.

Usage:
    python hdr_to_sdr_jpg.py --dir "path\\to\\images"
    python hdr_to_sdr_jpg.py --dir "path\\to\\images" --knee 0.7 --contrast 0.9
"""
from __future__ import annotations

import argparse
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import cv2
import numpy as np


def tonemap_image(bgr_uint8: np.ndarray, knee: float, contrast: float, exposure: float) -> np.ndarray:
    x = bgr_uint8.astype(np.float32) / 255.0 * exposure
    x = np.clip(x, 0.0, None)
    shoulder = 1.0 - knee
    over = x > knee
    compressed = knee + shoulder * (1.0 - np.exp(-(x - knee) / shoulder))
    x = np.where(over, compressed, x)
    x = np.clip(x, 0.0, 1.0)
    x = 0.5 + (x - 0.5) * contrast  # mild global contrast pull-down
    x = np.clip(x, 0.0, 1.0)
    return (x * 255.0 + 0.5).astype(np.uint8)


def process_one(src_path: str, dst_path: str, knee: float, contrast: float, exposure: float, quality: int) -> str | None:
    img = cv2.imread(src_path, cv2.IMREAD_COLOR)
    if img is None:
        return f"skip (unreadable): {src_path}"
    out = tonemap_image(img, knee, contrast, exposure)
    ok = cv2.imwrite(dst_path, out, [cv2.IMWRITE_JPEG_QUALITY, quality])
    if not ok:
        return f"failed to write: {dst_path}"
    return None


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--dir', required=True, type=Path, help='Folder of JPEGs to tone-map, overwritten in place (backup kept alongside)')
    parser.add_argument('--knee', type=float, default=0.75, help='Highlights above this fraction of full range (0-1) get soft-compressed; below it is untouched (default: 0.75)')
    parser.add_argument('--contrast', type=float, default=0.92, help='Global contrast multiplier around mid-gray, <1 flattens contrast (default: 0.92)')
    parser.add_argument('--exposure', type=float, default=1.0, help='Pre-curve exposure multiplier applied before the knee (default: 1.0, no change)')
    parser.add_argument('--quality', type=int, default=95, help='Output JPEG quality (default: 95)')
    parser.add_argument('--workers', type=int, default=None, help='Parallel worker processes (default: CPU count)')
    parser.add_argument('--backup-suffix', default='_hdr_backup', help='Suffix appended to the renamed backup folder (default: _hdr_backup)')
    parser.add_argument('--ext', default='.jpg,.jpeg', help='Comma-separated extensions to process (default: .jpg,.jpeg)')
    args = parser.parse_args()

    target = args.dir.resolve()
    if not target.is_dir():
        parser.error(f'--dir does not exist or is not a folder: {target}')

    backup = target.parent / (target.name + args.backup_suffix)
    if backup.exists():
        print(f'Backup already exists at {backup}, using it as the source (resuming).')
    else:
        target.rename(backup)
        print(f'Backed up original folder to {backup}')

    target.mkdir(exist_ok=True)

    exts = {e.strip().lower() for e in args.ext.split(',') if e.strip()}
    files = sorted(p for p in backup.iterdir() if p.suffix.lower() in exts)
    if not files:
        print(f'No files with extensions {exts} found in {backup}')
        return

    total = len(files)
    print(f'Tone-mapping {total} images (knee={args.knee}, contrast={args.contrast}, exposure={args.exposure}, quality={args.quality}) -> {target}')

    started = time.time()
    done = 0
    errors: list[str] = []
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        futures = {
            pool.submit(process_one, str(p), str(target / p.name), args.knee, args.contrast, args.exposure, args.quality): p
            for p in files
        }
        for fut in as_completed(futures):
            done += 1
            msg = fut.result()
            if msg:
                errors.append(msg)
            if done % 200 == 0 or done == total:
                elapsed = time.time() - started
                print(f'  {done}/{total} ({elapsed:.1f}s)')

    print(f'Done in {time.time() - started:.1f}s. {total - len(errors)} written, {len(errors)} errors.')
    for e in errors[:20]:
        print(f'  ! {e}')
    if len(errors) > 20:
        print(f'  ... and {len(errors) - 20} more errors')


if __name__ == '__main__':
    main()
