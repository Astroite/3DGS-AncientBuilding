"""RealityScan-driven alignment: replaces COLMAP as gsdb's automated reconstruction
backend (see docs/... migration notes). Mirrors postshot.py's shape -- an
`_executable()` resolver, a `build_..._command()` pure function, and a top-level
orchestration function driven through `processes.run_logged()`.

Empirically validated on this machine (2026-09-04) against gsdb's own reprojected
view_XX/frame_NNNNNN.jpg images: -headless -newScene -addFolder -align
-selectMaximalComponent correctly aligns 8 synthetic panorama views into one
component in ~1-2 seconds. -exportRegistration, however, unconditionally requires
a params.xml second argument -- confirmed by testing three different export
formats (COLMAP, and the parameter-free "Internal/External Camera Parameters"
CSV) and getting the identical generic failure ("Operation failed" [err:5618],
visible only in %LOCALAPPDATA%\\Temp\\RealityScan.log, not stdout/stderr) when it
is omitted. There is no CLI-native way to synthesize this file from scratch (see
tools/realityscan-setup/README.md for the one-time GUI export instructions and
exactly what was checked before concluding this).

KNOWN GAP, not yet wired: mask application during alignment (COLMAP's
--ImageReader.mask_path equivalent). RealityScan's setImagesLayer/inpMaskOpts
mechanism (see the design doc) has not been empirically verified yet -- treat
alignment quality without masks as unproven until that lands.
"""

from __future__ import annotations

import json
import os
import shutil
from pathlib import Path
from typing import Any

from .models import ReconstructionAttempt, ReconstructionConfig, ReconstructionConfigV3
from .paths import find_app_root, host_path, wsl_to_windows
from .processes import run_logged
from .reconstruction import (
    _write_colmap_binary_model,
    attach_frame_masks,
    reconstruction_metrics,
)

REALITYSCAN_CLI_ENV = "GSDB_REALITYSCAN_CLI"
REALITYSCAN_EXPORT_PARAMS_ENV = "GSDB_REALITYSCAN_EXPORT_PARAMS"

ReconstructionSettings = ReconstructionConfig | ReconstructionConfigV3


def realityscan_executable() -> Path:
    configured = os.environ.get(REALITYSCAN_CLI_ENV, "").strip()
    candidates: list[Path] = []
    if configured:
        candidates.append(host_path(configured))
    # Pinned to 2.2 (the version whose local Help files were read to derive this
    # module's CLI sequence) rather than whichever RealityScan_* is newest.
    candidates.append(Path(r"C:\Program Files\Epic Games\RealityScan_2.2\RealityScan.exe"))
    found = shutil.which("RealityScan.exe") or shutil.which("RealityScan")
    if found:
        candidates.append(Path(found))
    for candidate in candidates:
        if candidate.is_file():
            return candidate
    raise RuntimeError("RealityScan is unavailable; install it or set GSDB_REALITYSCAN_CLI")


def _default_export_params_path() -> Path:
    return find_app_root() / "tools" / "realityscan-setup" / "colmap-export-params.xml"


def export_params_file() -> Path:
    """The one-time, GUI-exported 'COLMAP' registration-export settings file.

    See this module's docstring: -exportRegistration has no CLI-native way to
    generate this. Checked first as an env var override (for local testing
    against a not-yet-committed file), then the committed default asset.
    """
    configured = os.environ.get(REALITYSCAN_EXPORT_PARAMS_ENV, "").strip()
    candidate = host_path(configured) if configured else _default_export_params_path()
    if not candidate.is_file():
        raise RuntimeError(
            f"RealityScan export params file is missing: {candidate}. Generate it once "
            "via RealityScan's GUI (Export Registration -> COLMAP format -> configure "
            "settings -> Save) -- see tools/realityscan-setup/README.md -- then either "
            f"commit it to that path or point {REALITYSCAN_EXPORT_PARAMS_ENV} at it."
        )
    return candidate


def _windows_argument(path: Path) -> str:
    absolute = path.absolute().resolve(strict=False)
    converted = wsl_to_windows(str(absolute))
    if os.name != "nt" and converted == str(absolute):
        raise RuntimeError(f"RealityScan paths must be Windows-accessible: {absolute}")
    if converted.startswith("\\\\"):
        raise RuntimeError(f"RealityScan paths may not use UNC storage: {converted}")
    return converted


def build_realityscan_align_command(
    executable: Path,
    images_dir: Path,
    crash_dir: Path,
    export_txt: Path,
    project_path: Path,
) -> list[str]:
    return [
        str(executable),
        "-headless",
        "-silent",
        _windows_argument(crash_dir),
        "-set",
        "appQuitOnError=true",
        "-newScene",
        "-set",
        "appIncSubdirs=true",
        "-addFolder",
        _windows_argument(images_dir),
        "-align",
        "-selectMaximalComponent",
        "-exportRegistration",
        _windows_argument(export_txt),
        _windows_argument(export_params_file()),
        "-save",
        _windows_argument(project_path),
        "-quit",
    ]


def convert_colmap_text_to_binary(text_dir: Path, output_dir: Path) -> None:
    """Convert RealityScan's exported COLMAP Text Format into the binary triplet
    postshot.py and the rest of gsdb's pipeline already consume.

    RealityScan's own internal model of "a COLMAP file" is the .txt triplet --
    -exportRegistration has no binary-COLMAP output option, and there is no
    native-Windows COLMAP install on this project to reach for `model_converter`
    (nor should there be -- COLMAP is being dropped). Rather than hand-rolling a
    parser, this reuses nerfstudio's own colmap_parsing_utils (already a hard
    dependency of this project for `colmap_to_json`/`equirect2persp`), whose
    Camera/Image/Point3D namedtuples already match
    reconstruction._write_colmap_binary_model()'s expected input shape exactly.
    """
    from nerfstudio.data.utils.colmap_parsing_utils import (
        read_cameras_text,
        read_images_text,
        read_points3D_text,
    )

    cameras = read_cameras_text(text_dir / "cameras.txt")
    images = read_images_text(text_dir / "images.txt")
    points_path = text_dir / "points3D.txt"
    points = read_points3D_text(points_path) if points_path.is_file() else {}
    for camera in cameras.values():
        if camera.model != "PINHOLE" or len(camera.params) != 4:
            raise RuntimeError(
                "RealityScan exported a non-PINHOLE camera model "
                f"({camera.model}, {len(camera.params)} params); gsdb's reprojected "
                "views are distortion-free by construction (equirect2persp is a pure "
                "pinhole projection), so a non-PINHOLE result means either the export "
                "settings need adjusting or RealityScan estimated real distortion that "
                "needs investigating -- not something to silently coerce."
            )
    _write_colmap_binary_model(output_dir, cameras, images, points)


def run_realityscan_alignment(
    dataset: Path,
    attempt: ReconstructionAttempt,
    log_dir: Path,
    settings: ReconstructionSettings | None = None,
) -> dict[str, Any]:
    """RealityScan-backed equivalent of reconstruction.run_masked_colmap().

    Same contract, so pipeline.py's reconstruct_run() primary/fallback logic and
    every downstream consumer (trajectory QA, postshot.py, mask-finalize) needs
    only a one-line import swap, not a rewrite:
      - same return dict shape (registration_ratio, largest_component_coverage, ...)
      - writes dataset/colmap/selected-attempt.json in the same shape
        _selected_model_dir() already parses
      - lands a valid binary COLMAP model at that path
      - produces dataset/transforms.json via the same nerfstudio colmap_to_json()
        call run_masked_colmap() already makes
    """
    transforms = dataset / "transforms.json"
    if transforms.is_file():
        try:
            existing_metrics = reconstruction_metrics(dataset, attempt, settings)
            threshold = float(getattr(settings, "registration_threshold", 0.70))
            if (
                min(
                    existing_metrics["registration_ratio"],
                    existing_metrics["largest_component_coverage"],
                )
                >= threshold
            ):
                return existing_metrics
        except (RuntimeError, json.JSONDecodeError, KeyError, TypeError):
            # Reuse the completed alignment below and regenerate transforms/masks
            # rather than trusting a possibly-partial previous run.
            pass

    colmap_root = dataset / "colmap"
    log_dir.mkdir(parents=True, exist_ok=True)
    colmap_root.mkdir(parents=True, exist_ok=True)

    export_dir = colmap_root / "realityscan-export"
    export_dir.mkdir(parents=True, exist_ok=True)

    command = build_realityscan_align_command(
        realityscan_executable(),
        dataset / "images",
        log_dir / "crash-reports",
        export_dir / "images.txt",
        colmap_root / "project.rsproj",
    )
    run_logged(command, log_dir / "realityscan-align.log")

    if not (export_dir / "images.txt").is_file():
        raise RuntimeError(
            f"RealityScan did not produce a registration export in {export_dir}; "
            f"see {log_dir / 'realityscan-align.log'}"
        )
    convert_colmap_text_to_binary(export_dir, colmap_root)
    (colmap_root / ".mapping-complete").write_text("complete\n", encoding="utf-8")

    from nerfstudio.process_data.colmap_utils import colmap_to_json

    colmap_to_json(colmap_root, dataset, use_single_camera_mode=False)
    attach_frame_masks(transforms, dataset / "masks")

    (colmap_root / "selected-attempt.json").write_text(
        json.dumps(
            {
                "model": ".",
                "rig": {"enabled": False, "backend": "realityscan"},
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    (colmap_root / ".conversion-complete").write_text("complete\n", encoding="utf-8")
    return reconstruction_metrics(dataset, attempt, settings)
