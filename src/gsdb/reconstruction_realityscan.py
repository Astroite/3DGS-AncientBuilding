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

Schema 5 stages white-keep mask image layers and explicitly enables alignment
masking with inpMaskOpts. The audit checks names, hashes and the executed command;
it does not inspect RealityScan's internal feature selection. Historical settings
retain their original behavior. See docs/CURRENT-WORKFLOW.md for current usage.
"""

from __future__ import annotations

import json
import math
import os
import shutil
from pathlib import Path
from typing import Any

from .masking import image_files
from .models import ReconstructionAttempt, ReconstructionConfig
from .paths import find_app_root, host_path
from .processes import run_logged
from .reconstruction import (
    _recorded_cross_view_pairs,
    _selected_model_dir,
    _write_colmap_binary_model,
    attach_frame_masks,
    validate_frame_masks,
)

REALITYSCAN_CLI_ENV = "GSDB_REALITYSCAN_CLI"
REALITYSCAN_EXPORT_PARAMS_ENV = "GSDB_REALITYSCAN_EXPORT_PARAMS"

ReconstructionSettings = ReconstructionConfig
ReconstructionAttemptSettings = ReconstructionAttempt


def realityscan_executable() -> Path:
    configured = os.environ.get(REALITYSCAN_CLI_ENV, "").strip()
    candidates: list[Path] = []
    if configured:
        candidates.append(host_path(configured))
    # Pinned to 2.2 (the version whose local Help files were read to derive this
    # module's CLI sequence) rather than whichever RealityScan_* is newest.
    candidates.append(
        host_path(r"C:\Program Files\Epic Games\RealityScan_2.2\RealityScan.exe")
    )
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
    converted = str(absolute)
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
    *,
    projection_fov_degrees: float,
    fix_intrinsics: bool = True,
    alignment_masks: bool = False,
) -> list[str]:
    # equirect2persp emits an ideal rectilinear (pinhole) image.  RealityScan
    # otherwise starts from an approximate no-distortion prior and is free to
    # optimize it into Brown3, which COLMAP exports as FULL_OPENCV.  Besides
    # being physically wrong for these generated views, that also leaves the
    # export behind a warning prompt in headless mode.  Pin the known lens model
    # and, by default, the analytically known focal length instead.
    if not 0.0 < projection_fov_degrees < 180.0:
        raise ValueError("projection_fov_degrees must be between 0 and 180")
    focal_35mm = 36.0 / (
        2.0 * math.tan(math.radians(projection_fov_degrees) / 2.0)
    )
    calibration_prior = "2" if fix_intrinsics else "1"
    command = [
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
        "-selectAllImages",
        "-editInputSelection",
        "inpCalibrationGroup=0",
        "-editInputSelection",
        f"inpCalibration={calibration_prior}",
        "-editInputSelection",
        f"inpFocal={focal_35mm:.12g}",
        "-editInputSelection",
        "inpPPX=0",
        "-editInputSelection",
        "inpPPY=0",
        "-editInputSelection",
        "inpLensGroup=0",
        "-editInputSelection",
        "inpDistortion=2",
        "-editInputSelection",
        "inpDistortionModel=0",
        # Clear the input selection used for editing.  Leaving it active makes
        # exportRegistration take the selected-input export path, which writes
        # cameras.txt/points3D.txt but (in RealityScan 2.2) omits images.txt.
        "-deselectAllImages",
        "-align",
        "-selectMaximalComponent",
        # Confirm that exportRegistration should export the currently selected
        # maximal component rather than all components or cancel.  In silent
        # mode an unset choice can report a successful export while omitting
        # images.txt (only cameras.txt/points3D.txt are written).
        "-set",
        "PUS-1-323292754=1",
        # RealityScan records confirmation choices as PUS keys.  This is the
        # documented/observed "continue with current COLMAP export" prompt.  It
        # is normally dormant after fixing the pinhole model, but keeping it in
        # the command makes recovery deterministic if RealityScan warns again.
        "-set",
        "PUS-1-323301212=1",
        "-exportRegistration",
        _windows_argument(export_txt),
        _windows_argument(export_params_file()),
        "-save",
        _windows_argument(project_path),
        "-quit",
    ]
    if alignment_masks:
        position = command.index("-deselectAllImages")
        command[position:position] = ["-editInputSelection", "inpMaskOpts=1"]
    return command


def convert_colmap_text_to_binary(
    text_dir: Path,
    output_dir: Path,
    *,
    image_name_map: dict[str, str] | None = None,
) -> None:
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
    if image_name_map is not None:
        remapped_images = {}
        for image_id, image in images.items():
            exported_name = image.name.replace("\\", "/")
            original_name = image_name_map.get(exported_name)
            if original_name is None:
                raise RuntimeError(
                    "RealityScan exported an image outside its staged input map: "
                    f"{exported_name}"
                )
            remapped_images[image_id] = image._replace(name=original_name)
        images = remapped_images
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


def realityscan_reconstruction_metrics(
    dataset: Path,
    attempt: ReconstructionAttemptSettings,
    settings: ReconstructionSettings | None = None,
    *,
    included_images: set[str],
    projected_image_count: int | None = None,
) -> dict[str, Any]:
    """Measure the single maximal component exported by RealityScan."""
    transforms_path = dataset / "transforms.json"
    if not transforms_path.is_file():
        raise RuntimeError(f"RealityScan conversion did not create {transforms_path}")
    try:
        selected_registered = validate_frame_masks(transforms_path, dataset / "masks")
    except (OSError, RuntimeError, json.JSONDecodeError, KeyError, TypeError) as error:
        raise RuntimeError(f"RealityScan frame-mask validation failed: {error}") from error

    selection_path = dataset / "colmap" / "selected-attempt.json"
    if not selection_path.is_file():
        raise RuntimeError(f"RealityScan model selection is missing: {selection_path}")
    selection = json.loads(selection_path.read_text(encoding="utf-8"))
    model = _selected_model_dir(dataset)

    from nerfstudio.process_data.colmap_utils import read_images_binary

    registered_model = read_images_binary(model / "images.bin")
    registered_names = {image.name.replace("\\", "/") for image in registered_model.values()}
    registered = len(registered_names)
    normalized_included = {name.replace("\\", "/") for name in included_images}
    unexpected = registered_names - normalized_included
    if unexpected:
        raise RuntimeError(
            "RealityScan registered images outside the final inclusion manifest: "
            + ", ".join(sorted(unexpected)[:5])
        )
    expected = len(normalized_included)
    written = (
        int(projected_image_count)
        if projected_image_count is not None
        else len(image_files(dataset / "images"))
    )
    excluded_count = max(0, written - expected)
    registration_ratio = registered / expected if expected else 0.0
    return {
        "expected_planar_images": expected,
        "written_planar_images": written,
        "excluded_images": excluded_count,
        "registered_images": registered,
        "selected_component_images": selected_registered,
        "registration_ratio": registration_ratio,
        "largest_component_coverage": registration_ratio,
        "component_count": 1 if registered else 0,
        "component_sizes": [registered] if registered else [],
        "sift_gpu": bool(getattr(settings, "use_gpu_sift", False)),
        "fixed_intrinsics": bool(getattr(settings, "fix_intrinsics", False)),
        "cross_view_pairs": _recorded_cross_view_pairs(dataset),
        "rig": selection.get("rig", {"enabled": False}),
    }


def run_realityscan_alignment(
    dataset: Path,
    attempt: ReconstructionAttemptSettings,
    log_dir: Path,
    settings: ReconstructionSettings | None = None,
    *,
    included_images: set[str],
    projected_image_count: int | None = None,
) -> dict[str, Any]:
    """Align the masked views with RealityScan and export a binary COLMAP model.

    Returns the usual registration metrics, lands a valid model under
    ``dataset/colmap`` (recorded via ``selected-attempt.json``, which
    ``_selected_model_dir()`` parses) and produces ``dataset/transforms.json``
    through nerfstudio's ``colmap_to_json``. ``included_images`` is the frozen
    input inventory the alignment must not exceed.
    """
    transforms = dataset / "transforms.json"
    alignment_masks = bool(getattr(settings, "alignment_masks", False))
    if transforms.is_file():
        try:
            existing_metrics = realityscan_reconstruction_metrics(
                dataset,
                attempt,
                settings,
                included_images=included_images,
                projected_image_count=projected_image_count,
            )
            threshold = float(getattr(settings, "registration_threshold", 0.70))
            mask_audit = dataset / "alignment-masks.json"
            mask_compatible = not alignment_masks
            if alignment_masks and mask_audit.is_file():
                from .media import sha256_file
                audit = json.loads(mask_audit.read_text(encoding="utf-8"))
                audit_records = audit.get('records',[])
                mask_compatible = audit.get('completed') is True and {r['image'] for r in audit_records} == set(included_images)
                if mask_compatible:
                    mask_compatible = all(sha256_file(dataset/'masks'/f"{r['image']}.png") == r['mask_sha256'] and
                                          sha256_file(dataset/'images'/r['image']) == r['image_sha256'] for r in audit_records)
            if mask_compatible and (alignment_masks or (
                min(
                    existing_metrics["registration_ratio"],
                    existing_metrics["largest_component_coverage"],
                )
                >= threshold
            )):
                return existing_metrics
        except (RuntimeError, json.JSONDecodeError, KeyError, TypeError):
            # Reuse the completed alignment below and regenerate transforms/masks
            # rather than trusting a possibly-partial previous run.
            pass

    colmap_root = dataset / "colmap"
    log_dir.mkdir(parents=True, exist_ok=True)
    colmap_root.mkdir(parents=True, exist_ok=True)

    export_dir = colmap_root / "realityscan-export"
    # Never allow files from a failed export to satisfy a later resume check.
    shutil.rmtree(export_dir, ignore_errors=True)
    export_dir.mkdir(parents=True, exist_ok=True)

    images_dir = dataset / "images"
    staged_images_dir: Path | None = None
    image_name_map: dict[str, str] | None = None
    if included_images is not None:
        available = {
            path.relative_to(images_dir).as_posix(): path
            for path in image_files(images_dir)
        }
        normalized_included = {name.replace("\\", "/") for name in included_images}
        missing = normalized_included - set(available)
        if missing:
            raise RuntimeError(
                "Final RealityScan inclusion manifest references missing images: "
                + ", ".join(sorted(missing)[:5])
            )
        # RealityScan's COLMAP writer drops input subdirectories and records only
        # basenames.  Projection trees deliberately repeat frame filenames under
        # view_XX/, so feed RealityScan unique flat names and restore the original
        # relative names immediately after export.
        staged_images_dir = dataset / "alignment-images"
        shutil.rmtree(staged_images_dir, ignore_errors=True)
        staged_images_dir.mkdir(parents=True, exist_ok=True)
        image_name_map = {}
        mask_records = []
        for index, name in enumerate(sorted(normalized_included)):
            suffix = available[name].suffix.lower() or ".jpg"
            staged_name = f"gsdb_{index:06d}{suffix}"
            destination = staged_images_dir / staged_name
            try:
                os.link(available[name], destination)
            except OSError as error:
                raise RuntimeError(
                    "RealityScan's unique input tree requires same-volume hardlinks; "
                    f"could not link {name}: {error}"
                ) from error
            image_name_map[staged_name] = name
            if alignment_masks:
                from .masking import mask_path_for_image, image_dimensions
                from .media import sha256_file
                import cv2
                import numpy as np
                source_mask = mask_path_for_image(dataset / "masks", available[name])
                pixels = cv2.imread(str(source_mask), cv2.IMREAD_GRAYSCALE)
                if pixels is None or pixels.shape != image_dimensions(available[name]):
                    raise RuntimeError(f"Invalid alignment mask: {source_mask}")
                if not np.all((pixels == 0) | (pixels == 255)):
                    raise RuntimeError(f"Nonbinary alignment mask: {source_mask}")
                target_mask = staged_images_dir / f"{staged_name}.mask.png"
                shutil.copy2(source_mask, target_mask)
                mask_records.append({"image": name, "staged_image": staged_name,
                                     "mask_sha256": sha256_file(source_mask), "image_sha256": sha256_file(available[name])})
        images_dir = staged_images_dir

    command = build_realityscan_align_command(
        realityscan_executable(),
        images_dir,
        log_dir / "crash-reports",
        # The first argument is the export job's chosen filename, not the
        # COLMAP triplet's images.txt member.  Using that reserved member name
        # causes RealityScan 2.2 to omit it while still reporting success.
        export_dir / "registration.txt",
        colmap_root / "project.rsproj",
        projection_fov_degrees=float(attempt.projection_fov_degrees),
        fix_intrinsics=bool(getattr(settings, "fix_intrinsics", True)),
        **({"alignment_masks": True} if alignment_masks else {}),
    )
    if alignment_masks:
        (dataset / "alignment-masks.json").write_text(json.dumps({
            "completed": False, "polarity": "white_keep", "inpMaskOpts": 1,
            "records": mask_records,
        }, indent=2), encoding="utf-8")
    run_logged(command, log_dir / "realityscan-align.log")

    if not (export_dir / "images.txt").is_file():
        raise RuntimeError(
            f"RealityScan did not produce a registration export in {export_dir}; "
            f"see {log_dir / 'realityscan-align.log'}"
        )
    convert_colmap_text_to_binary(
        export_dir,
        colmap_root,
        image_name_map=image_name_map,
    )
    shutil.rmtree(export_dir, ignore_errors=True)
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
    if alignment_masks:
        audit_path = dataset / "alignment-masks.json"
        audit = json.loads(audit_path.read_text(encoding="utf-8"))
        audit["completed"] = True
        audit["verification"] = "paired_binary_masks_staged_and_alignment_command_enabled"
        audit_path.write_text(json.dumps(audit, indent=2), encoding="utf-8")
    metrics = realityscan_reconstruction_metrics(
        dataset,
        attempt,
        settings,
        included_images=included_images,
        projected_image_count=projected_image_count,
    )
    if staged_images_dir is not None:
        shutil.rmtree(staged_images_dir, ignore_errors=True)
    return metrics
