"""Run stages: preprocess -> mask -> reconstruct (primary + one bounded repair) -> qa.

There is one run shape and one implementation per stage. Prepared candidates live
inside the Run (`inputs/primary/<hash>`), identity is pinned by hashes, and resume
never invents a second repair round or resurrects rejected pixels.
"""
from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

from .gpu_lock import gpu_locked
from .manifests import canonical_hash, load_capture_manifest, save_yaml
from .mask_finalize import (
    MaskFinalizationMissingError,
    finalize_mask_dataset,
    validate_mask_finalization,
)
from .mask_review import MaskReviewDataset
from .masking import (
    _recover_mask_filter,
    _recover_mask_records,
    create_mask_contact_sheets,
    filter_masked_images,
    generate_person_masks,
    image_files,
    summarize_detection_records,
    validate_mask_filter,
    validate_mask_set,
)
from .media import (
    analyze_frames,
    create_temporal_subset,
    sha256_file,
    summarize_frame_metrics,
)
from .models import CaptureManifest, RunManifest, RunStatus
from .paths import ensure_run_dir
from .processes import CommandError
from .retention import auto_cleanup, safe_path
from .reconstruction import project_equirectangular_frames
from .reconstruction_realityscan import run_realityscan_alignment
from .runs import begin_stage, complete_stage, fail_stage, save_run
from .run_lock import run_locked
from .segments import analyze_segments, frame_id, write_qa_report, write_segments
from .sources import (
    _validate_frame,
    load_prepared_input,
    probe_capture_source,
    source_adapter,
    validate_source_fingerprints,
)
from .storage import link_or_copy
from .training_data import json_write
from .vision_qa import load_local_mask_qa_review, run_deepseek_mask_qa


def _tree_size(path: Path) -> int:
    if not path.exists():
        return 0
    return sum(item.stat().st_size for item in path.rglob("*") if item.is_file())


def _record_resources(
    run: RunManifest,
    work: Path,
    command_metrics: dict[str, float] | None = None,
) -> None:
    resources = run.metrics.setdefault("resources", {})
    observed = _tree_size(work)
    resources["disk_observed_peak_bytes"] = max(
        int(resources.get("disk_observed_peak_bytes", 0)), observed
    )
    if command_metrics and "gpu_memory_peak_mib" in command_metrics:
        resources["gpu_memory_observed_peak_mib"] = max(
            float(resources.get("gpu_memory_observed_peak_mib", 0)),
            command_metrics["gpu_memory_peak_mib"],
        )


def read_records(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def write_records(path: Path, records: list[dict]) -> None:
    temporary = path.with_suffix(".tmp")
    temporary.write_text(
        "".join(json.dumps(item) + "\n" for item in records), encoding="utf-8"
    )
    temporary.replace(path)


def capture_path(scene_path: Path, capture_id: str) -> Path:
    return scene_path / "captures" / f"{capture_id}.yaml"


def load_capture(scene_path: Path, capture_id: str) -> CaptureManifest:
    return load_capture_manifest(capture_path(scene_path, capture_id))


def ingest_capture(scene_path: Path, capture_id: str) -> CaptureManifest:
    """Re-probe an immutable, explicitly listed source set and refresh its manifest."""
    from .sources import probe_capture_source

    capture = load_capture(scene_path, capture_id)
    probe = probe_capture_source(capture, scene_path / "logs" / "probe-protocol")
    capture.source.probe = type(capture.source.probe)(
        **{
            name: value
            for name, value in probe.items()
            if name in type(capture.source.probe).model_fields
        }
    )
    save_yaml(capture_path(scene_path, capture_id), capture)
    return capture


@run_locked
def preprocess_run(scene_path: Path, run: RunManifest, resume: bool = False) -> RunManifest:
    if not begin_stage(run, "preprocess", resume=resume):
        auto_cleanup(scene_path, run)
        return run
    save_run(scene_path, run)
    work = ensure_run_dir(scene_path, run.id)
    try:
        target = safe_path(work, run.config.input_relative_path)
        prepared = load_prepared_input(target)
        if prepared.dataset_sha256 != run.config.input_dataset_sha256:
            raise RuntimeError("Prepared input changed")
        duration = (
            run.config.input.selection.end_seconds - run.config.input.selection.start_seconds
        )
        records = analyze_frames(
            list(prepared.frame_paths),
            duration,
            work / "frame-metrics.jsonl",
            timestamps_seconds=prepared.timestamps_seconds,
        )
        records = [
            {**record, "source_frame_index": index, "source_sha256": digest}
            for record, index, digest in zip(
                records, prepared.frame_indices, prepared.frame_sha256s
            )
        ]
        write_records(work / "frame-metrics.jsonl", records)
        _, selected = create_temporal_subset(
            prepared.path / "frames",
            records,
            work / "frames-primary",
            run.config.input.selection.start_seconds,
            run.config.preprocess.selected_per_second,
            rank_limit=run.config.reconstruction.primary.temporal_rank_limit,
        )
        for record in selected:
            if sha256_file(work / "frames-primary" / record["file"]) != record["source_sha256"]:
                raise RuntimeError("Selected frame changed; refusing unverified resume input")
        write_records(work / "selected-primary-metrics.jsonl", selected)
        run.metrics["preprocess"] = dict(
            candidate_frame_count=len(records),
            selected_frame_count=len(selected),
            candidate_fps=run.config.preprocess.candidate_fps,
            selected_per_second=run.config.preprocess.selected_per_second,
            candidate=summarize_frame_metrics(records),
            selected=summarize_frame_metrics(selected),
        )
        run.metrics["input"] = prepared.source_probe.model_dump(mode="json")
        run.metrics["selection"] = run.config.input.selection.model_dump(mode="json")
        complete_stage(
            run,
            "preprocess",
            message=f"Prepared {len(records)} Run-owned candidates; selected {len(selected)}",
        )
    except Exception as error:
        fail_stage(run, "preprocess", str(error))
        save_run(scene_path, run)
        raise
    save_run(scene_path, run)
    auto_cleanup(scene_path, run)
    return run


def _attempt_inventory(dataset: Path, run: RunManifest) -> tuple[dict[str, Any], set[str]]:
    filtered = validate_mask_filter(dataset, verify_hashes=False)
    accepted = {str(item["image"]) for item in filtered["accepted"]}
    finalized = validate_mask_finalization(dataset, accepted)
    included = accepted - set(finalized.get("excluded_images", []))
    return finalized, included


def _cached_masking_result(
    scene_path: Path,
    dataset: Path,
    run: RunManifest,
    label: str,
) -> dict[str, Any] | None:
    cached = run.metrics.get("masking", {}).get(label)
    if not isinstance(cached, dict):
        return None
    try:
        filtered = validate_mask_filter(
            dataset, verify_hashes=not run.config.masking.mask_review_required
        )
    except (OSError, RuntimeError, KeyError, TypeError, json.JSONDecodeError):
        return None
    expected_count = len(filtered["accepted"])
    if int(cached.get("reconstruction_input_images", -1)) != expected_count:
        return None
    images_dir = dataset / "images"
    masks_dir = dataset / "masks"
    images = image_files(images_dir)
    image_names = {path.relative_to(images_dir).as_posix() for path in images}
    if len(image_names) != expected_count:
        return None
    expected_masks = {
        (Path(name).parent / f"{Path(name).name}.png").as_posix() for name in image_names
    }
    actual_masks = {
        path.relative_to(masks_dir).as_posix() for path in image_files(masks_dir)
    }
    if actual_masks != expected_masks:
        return None
    sheets = [scene_path / str(path) for path in cached.get("contact_sheets", [])]
    sheets_required = run.config.vision_qa.enabled or run.config.masking.mask_review_required
    if sheets_required and (not sheets or not all(path.is_file() for path in sheets)):
        return None
    if run.config.vision_qa.enabled:
        local_review_path = dataset / "mask-qa" / "codex-local-review.json"
        if not local_review_path.is_file():
            return None
        _, verdict = load_local_mask_qa_review(local_review_path, sheets, run.config.vision_qa)
        if verdict.decision != "pass":
            return None
    return dict(cached)


def _stage_views(source: Path, dataset: Path, attempt) -> list[Path]:
    """Training views under ``dataset/images``: projected views, or the frames themselves.

    A perspective source is already one frame per view, so nothing is projected and
    the prepared frames are linked in under their own names.
    """
    if attempt.panorama is not None:
        return project_equirectangular_frames(source, dataset, attempt)
    target = dataset / "images"
    target.mkdir(parents=True, exist_ok=True)
    frames = image_files(source)
    existing = image_files(target)
    if len(existing) != len(frames):
        for frame in frames:
            link_or_copy(frame, target / frame.name)
        existing = image_files(target)
        if len(existing) != len(frames):
            raise RuntimeError(
                f"Perspective frame staging is incomplete ({len(existing)}/{len(frames)})"
            )
    return existing


def _prepare_masked_dataset(
    scene_path: Path,
    work: Path,
    source: Path,
    dataset: Path,
    attempt,
    run: RunManifest,
    label: str,
) -> dict[str, Any]:
    mask_filter = _recover_mask_filter(
        dataset,
        run.config.masking.mask_discard_threshold,
        review_required=run.config.masking.mask_review_required,
    )
    recovered = mask_filter is not None
    cached = _cached_masking_result(scene_path, dataset, run, label) if not recovered else None
    if cached is not None:
        print(f"Masked dataset: {label} (validated cached result)", flush=True)
        return cached
    if recovered:
        records = _recover_mask_records(
            dataset, mask_filter, work / f"mask-metrics-{label}.jsonl"
        )
        projected_count = mask_filter["original_image_count"]
    else:
        images = _stage_views(source, dataset, attempt)
        projected_count = len(images)
        records = generate_person_masks(
            dataset / "images",
            dataset / "masks",
            run.config.masking,
            work / f"mask-metrics-{label}.jsonl",
        )
    if run.config.retention.mode == "keep" and not recovered:
        for item in records:
            if float(item["masked_fraction"]) > run.config.masking.mask_discard_threshold:
                for folder, key in (("images", "image"), ("masks", "mask")):
                    link_or_copy(
                        dataset / folder / str(item[key]),
                        work / "rejected-inputs" / label / folder / str(item[key]),
                    )
    mask_filter = (
        mask_filter
        if recovered
        else filter_masked_images(dataset, records, run.config.masking.mask_discard_threshold)
    )
    accepted_names = {str(item["image"]) for item in mask_filter["accepted"]}
    accepted_records = [item for item in records if str(item["image"]) in accepted_names]
    deterministic = validate_mask_set(
        dataset / "images",
        dataset / "masks",
        # Review edits can cross the automatic threshold. Finalization still
        # requires explicit exclusion; recovery must not re-filter them.
        1.0
        if recovered and run.config.masking.mask_review_required
        else run.config.masking.mask_discard_threshold,
    )
    if accepted_names and not run.config.masking.mask_review_required:
        finalize_mask_dataset(
            dataset,
            expected_images=accepted_names,
            maximum_included_masked_fraction=run.config.masking.mask_discard_threshold,
        )
    make_sheets = bool(accepted_records) and (
        run.config.masking.mask_review_required or run.config.vision_qa.enabled
    )
    existing_sheets = sorted((dataset / "mask-qa").glob("mask-contact-*.jpg"))
    review_bound = (dataset / "mask-qa" / "codex-local-review.json").is_file()
    sheets = (
        existing_sheets
        if recovered and (existing_sheets or review_bound)
        else (
            create_mask_contact_sheets(
                dataset / "images",
                dataset / "masks",
                accepted_records,
                dataset / "mask-qa",
                run.config.masking.qa_sample_count,
                run.config.vision_qa.max_contact_sheets,
            )
            if make_sheets
            else []
        )
    )
    vision: dict[str, Any]
    if run.config.vision_qa.enabled:
        local_review_path = dataset / "mask-qa" / "codex-local-review.json"
        if local_review_path.is_file():
            local_review, verdict = load_local_mask_qa_review(
                local_review_path, sheets, run.config.vision_qa
            )
            vision = {
                "provider": local_review.reviewer,
                "review_file": local_review_path.relative_to(scene_path).as_posix(),
                "contact_sheet_sha256": local_review.contact_sheet_sha256,
                **verdict.model_dump(mode="json"),
            }
        else:
            verdict = run_deepseek_mask_qa(sheets, run.config.vision_qa)
            vision = {"provider": "deepseek", **verdict.model_dump(mode="json")}
        if verdict.decision != "pass":
            raise RuntimeError(
                f"LLM mask QA rejected {label}: {verdict.rationale}; "
                f"false negatives={verdict.false_negative_views}"
            )
    else:
        vision = {
            "status": "disabled",
            "reason": (
                "Set vision_qa.enabled=true in a new run only after confirming an authorized "
                "DeepSeek multimodal endpoint"
            ),
        }
    result = {
        "planar_images": projected_count,
        "model": run.config.masking.model if run.config.masking.enabled else "disabled-all-white",
        "detection": summarize_detection_records(accepted_records),
        "validation": deterministic,
        "contact_sheets": [path.relative_to(scene_path).as_posix() for path in sheets],
        "vision_qa": vision,
    }
    surviving_frame_ids = sorted(
        {
            int(match.group(1))
            for item in mask_filter["accepted"]
            if (match := re.search(r"frame_(\d+)\.jpg$", str(item["image"])))
        }
    )
    result.update(
        {
            "projected_planar_images": projected_count,
            "automatic_rejected_images": len(mask_filter["rejected"]),
            "automatic_rejected_fraction": (
                len(mask_filter["rejected"]) / projected_count if projected_count else 0.0
            ),
            "reconstruction_input_images": len(mask_filter["accepted"]),
            "surviving_frame_ids": surviving_frame_ids,
            "mask_filter": (dataset / "mask-filter.json").relative_to(scene_path).as_posix(),
        }
    )
    return result


@gpu_locked
@run_locked
def mask_run(scene_path: Path, run: RunManifest, resume: bool = False) -> RunManifest:
    if not begin_stage(run, "mask", resume=resume):
        auto_cleanup(scene_path, run)
        return run
    save_run(scene_path, run)
    work = ensure_run_dir(scene_path, run.id)
    log_path = work / "logs" / "mask.log"
    try:
        primary = _prepare_masked_dataset(
            scene_path,
            work,
            work / "frames-primary",
            work / "reconstruction-primary",
            run.config.reconstruction.primary,
            run,
            "primary",
        )
        run.metrics.setdefault("masking", {})["primary"] = primary
        log_path.parent.mkdir(parents=True, exist_ok=True)
        log_path.write_text(
            json.dumps(primary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )
        _record_resources(run, work)
        complete_stage(
            run,
            "mask",
            message=(
                f"Projected and masked {primary['planar_images']} primary perspective views; "
                f"deterministic QA passed"
            ),
            log_path=log_path.relative_to(scene_path).as_posix(),
        )
        if run.config.masking.mask_review_required and primary.get("reconstruction_input_images", 0):
            run.status = RunStatus.WAITING_REVIEW
            run.stages["mask"].message = (
                f"{run.stages['mask'].message}; awaiting primary mask review/finalization"
            )
    except Exception as error:
        fail_stage(run, "mask", str(error), log_path.relative_to(scene_path).as_posix())
        save_run(scene_path, run)
        raise
    save_run(scene_path, run)
    auto_cleanup(scene_path, run)
    return run


# ------------------------------------------------------------------ single repair


def repair_plan(run: RunManifest, selected, candidates, report) -> dict:
    from .sources import anchored_frame_indices

    selection = run.config.input.selection
    padding = run.config.reconstruction.repair_padding_seconds
    times = {i: float(r["timestamp_seconds"]) for i, r in enumerate(selected, 1)}
    windows = []
    for f in set(report.get("weak_input_frames", [])) | set(report.get("missing_frames", [])):
        windows.append((times[f] - padding, times[f] + padding))
    for issue in report.get("issues", []):
        a = times[issue.get("from_frame", issue.get("frame"))]
        b = times[issue.get("to_frame", issue.get("frame"))]
        windows.append((a - padding, b + padding))
    # Coverage gaps can be absent from the pose-issue list by design.
    ordered = sorted(times.values())
    for a, b in zip(ordered, ordered[1:]):
        if b - a > run.config.segment_qa.max_gap_seconds:
            windows.append((a - padding, b + padding))
    merged = []
    for a, b in sorted(
        (max(selection.start_seconds, a), min(selection.end_seconds, b)) for a, b in windows
    ):
        if a >= b:
            continue
        if merged and a <= merged[-1][1]:
            merged[-1][1] = max(b, merged[-1][1])
        else:
            merged.append([a, b])
    probe = run.config.input.source_probe
    indices, timestamps = anchored_frame_indices(
        int(probe.frame_count),
        float(probe.fps),
        selection.start_seconds,
        selection.end_seconds,
        run.config.reconstruction.repair_candidate_fps,
    )
    seen = {int(r["source_frame_index"]) for r in candidates}
    extras = [
        dict(source_frame_index=i, timestamp_seconds=t)
        for i, t in zip(indices, timestamps)
        if i not in seen and any(a <= t <= b for a, b in merged)
    ]
    return dict(
        schema_version=1,
        attempt=1,
        config_hash=run.config_hash,
        primary_records_sha256=canonical_hash(selected),
        candidate_records_sha256=canonical_hash(candidates),
        source_sha256=run.config.input.source_sha256,
        windows=merged,
        candidate_fps=run.config.reconstruction.repair_candidate_fps,
        frames=extras,
        helper_version=run.config.input.helper_version,
        sdk_version=run.config.input.sdk_version,
    )


def _export_repair(scene_path: Path, run: RunManifest, plan: dict) -> list[Path]:
    work = scene_path / run.id
    capture = load_capture(scene_path, run.config.capture_id)
    if (
        [r.sha256 for r in capture.source.files] != run.config.input.source_sha256
        or capture.normalization != run.config.input.normalization
    ):
        raise RuntimeError("Capture identity changed before repair")
    folder = work / "inputs" / "repair"
    (folder / "frames").mkdir(parents=True, exist_ok=True)
    probe = probe_capture_source(capture, work / "logs" / "repair-protocol")
    for key in ("helper_version", "sdk_version"):
        if probe.get(key) != getattr(run.config.input, key):
            raise RuntimeError("Repair SDK/helper changed; create a new Run")
    for key in ("fps", "frame_count", "width", "height"):
        if probe[key] != getattr(run.config.input.source_probe, key):
            raise RuntimeError(f"Repair source probe changed: {key}")
    indices = [r["source_frame_index"] for r in plan["frames"]]
    outputs = [
        folder / "frames" / f"frame_{i:06d}.jpg" for i in range(1, len(indices) + 1)
    ]
    normalization = capture.normalization
    width = int(normalization.width or probe["width"])
    height = int(normalization.height or probe["height"])
    manifest = folder / "dataset.json"
    if manifest.exists():
        saved = json.loads(manifest.read_text(encoding="utf-8"))
        if saved["plan_sha256"] != canonical_hash(plan):
            raise RuntimeError("Repair extraction identity changed")
        for path, record in zip(outputs, saved["frames"]):
            if _validate_frame(path, width, height) != record["sha256"]:
                raise RuntimeError("Repair extraction changed")
        return outputs
    source_adapter(capture).export(
        capture,
        validate_source_fingerprints(capture),
        probe,
        indices,
        outputs,
        folder,
        work / "logs" / "repair-protocol",
        width,
        height,
        True,
    )
    validate_source_fingerprints(capture)
    records = [
        {**r, "file": p.name, "sha256": _validate_frame(p, width, height)}
        for r, p in zip(plan["frames"], outputs)
    ]
    json_write(
        manifest,
        dict(
            schema_version=1,
            plan_sha256=canonical_hash(plan),
            frames=records,
            helper_version=probe.get("helper_version"),
            sdk_version=probe.get("sdk_version"),
        ),
    )
    return outputs


def merge_repair_inputs(work: Path, run: RunManifest, selected, extra_records, plan) -> list[dict]:
    """Remap both accepted and rejected inventories; never resurrect rejected pixels."""
    dataset = work / "reconstruction-repair"
    all_records = [(r, "primary", i) for i, r in enumerate(selected, 1)] + [
        (r, "new", i) for i, r in enumerate(extra_records, 1)
    ]
    all_records.sort(key=lambda item: item[0]["timestamp_seconds"])
    mapping = {(kind, old): i for i, (_, kind, old) in enumerate(all_records, 1)}
    records = [
        {**r, "file": f"frame_{i:06d}.jpg", "origin": kind, "origin_frame": old}
        for i, (r, kind, old) in enumerate(all_records, 1)
    ]
    identity = dict(plan_sha256=canonical_hash(plan), records_sha256=canonical_hash(records), sources={})
    inventories = {}
    for label, source in (("primary", work / "reconstruction-primary"), ("new", work / "repair-new")):
        inventories[label] = validate_mask_filter(source, verify_hashes=True)
        identity["sources"][label] = sha256_file(source / "mask-filter.json")
    receipt = dataset / "repair-input.json"
    if receipt.exists():
        if json.loads(receipt.read_text(encoding="utf-8")) != identity:
            raise RuntimeError("Repair input identity changed")
        validate_mask_filter(dataset, verify_hashes=True)
        return records
    dataset.mkdir(exist_ok=True)
    (dataset / "images").mkdir(exist_ok=True)
    (dataset / "masks").mkdir(exist_ok=True)
    merged = dict(
        schema_version=1,
        status="complete",
        threshold=run.config.masking.mask_discard_threshold,
        comparison="masked_fraction > threshold",
        original_image_count=0,
        accepted=[],
        rejected=[],
    )
    excluded: set[str] = set()
    storage = []
    for label, source in (("primary", work / "reconstruction-primary"), ("new", work / "repair-new")):
        filtered = inventories[label]
        original_excluded: set[str] = set()
        if (source / "mask-final.json").exists():
            final = validate_mask_finalization(source, {r["image"] for r in filtered["accepted"]})
            original_excluded = set(final["excluded_images"])

        def remap(name, label=label):
            index = mapping[label, frame_id(name)]
            return re.sub(r"frame_\d+", f"frame_{index:06d}", name)

        for kind in ("accepted", "rejected"):
            for row in filtered[kind]:
                changed = {**row, "image": remap(row["image"]), "mask": remap(row["mask"])}
                merged[kind].append(changed)
                if kind == "accepted":
                    for folder, key in (("images", "image"), ("masks", "mask")):
                        storage.append(
                            link_or_copy(source / folder / row[key], dataset / folder / changed[key])
                        )
                    if row["image"] in original_excluded:
                        excluded.add(changed["image"])
        merged["original_image_count"] += filtered["original_image_count"]
    for kind in ("accepted", "rejected"):
        merged[kind].sort(key=lambda r: r["image"])
    json_write(dataset / "mask-filter.json", merged)
    validate_mask_filter(dataset, verify_hashes=True)
    if excluded:
        review = MaskReviewDataset(dataset)
        for item in review.items:
            if item.source_image in excluded:
                review.update_review(item.image_id, "exclude", "Inherited from finalized source attempt")
    if merged["accepted"] and not run.config.masking.mask_review_required:
        finalize_mask_dataset(
            dataset, {r["image"] for r in merged["accepted"]}, run.config.masking.mask_discard_threshold
        )
    json_write(
        dataset / "storage.json",
        dict(files=storage, duplicate_bytes=sum(r["duplicate_bytes"] for r in storage)),
    )
    write_records(work / "selected-repair-metrics.jsonl", records)
    json_write(receipt, identity)
    return records


def record_repair_metrics(run: RunManifest, filtered: dict) -> None:
    total = filtered["original_image_count"]
    run.metrics.setdefault("masking", {})["repair"] = dict(
        projected_planar_images=total,
        planar_images=total,
        reconstruction_input_images=len(filtered["accepted"]),
        automatic_rejected_images=len(filtered["rejected"]),
        automatic_rejected_fraction=len(filtered["rejected"]) / total if total else 0.0,
    )


def prepare_repair(scene_path: Path, run: RunManifest, selected, report) -> list[dict] | None:
    work = scene_path / run.id
    candidates = read_records(work / "frame-metrics.jsonl")
    plan = repair_plan(run, selected, candidates, report)
    path = work / "repair-plan.json"
    if path.exists():
        if json.loads(path.read_text(encoding="utf-8")) != plan:
            raise RuntimeError("Repair plan identity changed; create a new Run")
    else:
        json_write(path, plan)
    if not plan["frames"]:
        return None
    dataset = work / "reconstruction-repair"
    if (dataset / "repair-input.json").exists():
        receipt = json.loads((dataset / "repair-input.json").read_text(encoding="utf-8"))
        records = read_records(work / "selected-repair-metrics.jsonl")
        if (
            receipt["plan_sha256"] != canonical_hash(plan)
            or receipt["records_sha256"] != canonical_hash(records)
        ):
            raise RuntimeError("Committed repair dataset identity changed")
        record_repair_metrics(run, validate_mask_filter(dataset, verify_hashes=True))
        return records
    outputs = _export_repair(scene_path, run, plan)
    extras = analyze_frames(
        outputs,
        run.config.input.selection.end_seconds - run.config.input.selection.start_seconds,
        work / "repair-frame-metrics.jsonl",
        timestamps_seconds=[r["timestamp_seconds"] for r in plan["frames"]],
    )
    extras = [
        {**r, **identity, "source_file": r["file"], "source_sha256": sha256_file(p)}
        for r, identity, p in zip(extras, plan["frames"], outputs)
    ]
    metrics = _prepare_masked_dataset(
        scene_path,
        work,
        outputs[0].parent,
        work / "repair-new",
        run.config.reconstruction.primary,
        run,
        "repair-new",
    )
    run.metrics.setdefault("masking", {})["repair-new"] = metrics
    records = merge_repair_inputs(work, run, selected, extras, plan)
    write_records(work / "selected-repair-metrics.jsonl", records)
    record_repair_metrics(run, validate_mask_filter(dataset, verify_hashes=True))
    return records


# ------------------------------------------------------------------- reconstruct


@gpu_locked
@run_locked
def reconstruct_run(scene_path: Path, run: RunManifest, resume: bool = False) -> RunManifest:
    if not begin_stage(run, "reconstruct", resume=resume):
        auto_cleanup(scene_path, run)
        return run
    work = scene_path / run.id
    save_run(scene_path, run)
    try:
        selected = read_records(work / "selected-primary-metrics.jsonl")
        attempts = run.metrics.setdefault("reconstruction", {})
        viable = []
        report: dict[str, Any] | None = None
        for label, records in (("primary", selected), ("repair", None)):
            if label == "repair":
                if report is None:
                    break
                records = prepare_repair(scene_path, run, selected, report)
                if records is None:
                    break
                save_run(scene_path, run)
                auto_cleanup(scene_path, run, checkpoint=True)
            dataset = work / f"reconstruction-{label}"
            filtered = validate_mask_filter(dataset, verify_hashes=False)
            included = {r["image"] for r in filtered["accepted"]}
            if included:
                _, included = _attempt_inventory(dataset, run)
            if len(included) >= 2:
                try:
                    metrics = run_realityscan_alignment(
                        dataset,
                        run.config.reconstruction.primary,
                        work / "logs" / f"reconstruct-{label}",
                        run.config.reconstruction,
                        included_images=included,
                        projected_image_count=filtered["original_image_count"],
                    )
                except CommandError as error:
                    if label != "repair" or not viable:
                        raise
                    attempts[label] = {
                        "status": "failed",
                        "error": str(error),
                        "retained_valid_primary": True,
                    }
                    break
                attempts[label] = metrics
                report = write_segments(
                    dataset,
                    records,
                    included,
                    run.config.reconstruction.primary,
                    run.config.segment_qa,
                )
            else:
                report = analyze_segments([], records, included, run.config.segment_qa, [])
                attempts[label] = {
                    "registered_images": 0,
                    "reason": "fewer than two included images",
                }
                report["lineage"] = {
                    "records_sha256": canonical_hash(records),
                    "included_sha256": canonical_hash(sorted(included)),
                }
                temporary = dataset / "segments.json.tmp"
                temporary.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
                temporary.replace(dataset / "segments.json")
            run.metrics["segment_qa"] = {
                "attempt": label,
                "training_status": report["training_status"],
                "coverage_status": report["coverage_status"],
                "report": f"{run.id}/reconstruction-{label}/segments.json",
            }
            if report["training_status"] == "passed":
                # Keep independently aligned models separate. Prefer covered time,
                # then usable images; do not discard valid primary segments when the
                # single repair attempt yields a worse component.
                passed = [s for s in report["segments"] if s["status"] == "passed"]
                score = (
                    sum(s["end_seconds"] - s["start_seconds"] for s in passed),
                    sum(len(s["images"]) for s in passed),
                )
                viable.append((score, dataset, report, label))
            save_run(scene_path, run)
        if not viable:
            raise RuntimeError("No valid training segment after the bounded repair attempt")
        _, dataset, report, label = max(viable, key=lambda item: item[0])
        run.metrics["segment_qa"] = {
            "attempt": label,
            "training_status": report["training_status"],
            "coverage_status": report["coverage_status"],
            "report": f"{run.id}/reconstruction-{label}/segments.json",
        }
        run.selected_dataset = dataset.relative_to(scene_path).as_posix()
        run.metrics["selected_attempt"] = dataset.name.removeprefix("reconstruction-")
        complete_stage(
            run, "reconstruct", message=f"Valid segments available; route coverage {report['coverage_status']}"
        )
        save_run(scene_path, run)
        auto_cleanup(scene_path, run)
        return run
    except Exception as error:
        fail_stage(run, "reconstruct", str(error))
        if isinstance(error, MaskFinalizationMissingError):
            run.status = RunStatus.WAITING_REVIEW
        save_run(scene_path, run)
        raise


def review_run(scene_path: Path, run: RunManifest, accepted: bool, notes: str) -> RunManifest:
    if run.status != RunStatus.NEEDS_REVIEW:
        raise RuntimeError(
            f"Only needs_review runs can be reviewed; current status is {run.status.value}"
        )
    run.status = RunStatus.ACCEPTED if accepted else RunStatus.REJECTED
    run.review_notes = notes
    save_run(scene_path, run)
    return run
