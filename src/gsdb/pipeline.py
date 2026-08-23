from __future__ import annotations

import json
import re
import shutil
from pathlib import Path
from typing import Any

from .doctor import collect_tool_versions
from .manifests import load_model, save_yaml
from .masking import (
    create_mask_contact_sheets,
    generate_person_masks,
    summarize_detection_records,
    validate_mask_set,
)
from .media import (
    analyze_frames,
    check_disk_budget,
    create_blur_aware_subset,
    extract_uniform_frames,
    probe_video,
    sha256_file,
    summarize_frame_metrics,
    validate_equirectangular,
)
from .models import (
    ArtifactManifest,
    ArtifactRecord,
    CaptureManifest,
    ReconstructionAttempt,
    RunManifest,
    RunStatus,
    SceneManifest,
    utc_now,
)
from .paths import ensure_within, host_path
from .processes import CommandError, run_logged
from .reconstruction import (
    build_image_pyramid,
    project_equirectangular_frames,
    run_masked_colmap,
)
from .runs import begin_stage, complete_stage, fail_stage, save_run
from .vision_qa import run_mimo_mask_qa


VERSION_PATTERN = re.compile(r"^v\d{3}$")


def _tree_size(path: Path) -> int:
    if not path.exists():
        return 0
    return sum(item.stat().st_size for item in path.rglob("*") if item.is_file())


def _record_resources(
    run: RunManifest,
    work: Path,
    command_metrics: dict[str, float] | None = None,
    extra_path: Path | None = None,
) -> None:
    resources = run.metrics.setdefault("resources", {})
    observed = _tree_size(work) + (_tree_size(extra_path) if extra_path else 0)
    resources["disk_observed_peak_bytes"] = max(
        int(resources.get("disk_observed_peak_bytes", 0)), observed
    )
    if command_metrics and "gpu_memory_peak_mib" in command_metrics:
        resources["gpu_memory_observed_peak_mib"] = max(
            float(resources.get("gpu_memory_observed_peak_mib", 0)),
            command_metrics["gpu_memory_peak_mib"],
        )


def capture_path(scene_path: Path, capture_id: str) -> Path:
    return scene_path / "captures" / f"{capture_id}.yaml"


def load_capture(scene_path: Path, capture_id: str) -> CaptureManifest:
    return load_model(capture_path(scene_path, capture_id), CaptureManifest)


def ingest_capture(
    project_root: Path,
    scene_path: Path,
    capture_id: str,
    stitched_path: Path,
    resume: bool = False,
) -> CaptureManifest:
    capture = load_capture(scene_path, capture_id)
    stitched = ensure_within(stitched_path, project_root)
    if not stitched.is_file():
        raise FileNotFoundError(
            f"Stitched video is missing: {stitched}. "
            "Export a standard 2:1 equirectangular MP4 from Insta360 Studio first."
        )
    raw = host_path(capture.raw_source.windows_path)
    if not raw.is_file():
        raise FileNotFoundError(f"External raw source is missing: {raw}")
    raw_sha256 = sha256_file(raw)
    if capture.raw_source.sha256 and capture.raw_source.sha256 != raw_sha256:
        raise RuntimeError("External raw source hash changed; refusing to ingest a different capture")
    capture.raw_source.sha256 = raw_sha256
    capture.raw_source.byte_size = raw.stat().st_size
    probe = probe_video(stitched)
    validate_equirectangular(probe)
    stitched_sha256 = sha256_file(stitched)
    if capture.stitched_video.sha256:
        if capture.stitched_video.sha256 != stitched_sha256:
            raise RuntimeError(
                "Stitched video hash changed after ingest; create a new capture manifest "
                "instead of silently replacing lineage"
            )
        if not resume:
            raise RuntimeError("Capture was already ingested; use --resume to revalidate it")
    if capture.selection.end_seconds > probe["duration_seconds"] + 0.5:
        raise ValueError(
            f"Selected range ends at {capture.selection.end_seconds:.2f}s but the stitched video "
            f"is only {probe['duration_seconds']:.2f}s"
        )
    relative_to_scene = stitched.relative_to(scene_path).as_posix()
    capture.stitched_video.relative_path = relative_to_scene
    capture.stitched_video.sha256 = stitched_sha256
    capture.stitched_video.byte_size = probe["byte_size"]
    capture.stitched_video.width = probe["width"]
    capture.stitched_video.height = probe["height"]
    capture.stitched_video.fps = probe["fps"]
    capture.stitched_video.duration_seconds = probe["duration_seconds"]
    capture.stitched_video.codec = probe["codec"]
    save_yaml(capture_path(scene_path, capture_id), capture)
    return capture


def preprocess_run(scene_path: Path, run: RunManifest, resume: bool = False) -> RunManifest:
    if not begin_stage(run, "preprocess", resume=resume):
        return run
    save_run(scene_path, run)
    work = scene_path / "work" / run.id
    log_path = work / "logs" / "preprocess.log"
    try:
        if not run.tool_versions:
            run.tool_versions = collect_tool_versions()
        capture = load_capture(scene_path, run.config.capture_id)
        if capture.stitched_video.sha256 != run.config.input_sha256:
            raise RuntimeError("Capture input hash changed after the run was created")
        video = ensure_within(scene_path / capture.stitched_video.relative_path, scene_path)
        if not video.is_file():
            raise FileNotFoundError(
                f"Stitched video is missing: {video}. Export a 2:1 MP4 from the raw 360 media first."
            )
        probe = probe_video(video)
        validate_equirectangular(probe)
        selection_duration = capture.selection.end_seconds - capture.selection.start_seconds
        budget = check_disk_budget(
            scene_path,
            video.stat().st_size,
            run.config.preprocess.minimum_free_gib,
        )
        frames_dir = work / "equirect-primary"
        frames = extract_uniform_frames(
            video,
            frames_dir,
            run.config.preprocess.target_frames,
            selection_duration,
            run.config.preprocess.jpeg_quality,
            log_path,
            start_seconds=capture.selection.start_seconds,
        )
        metrics_path = work / "frame-metrics.jsonl"
        records = analyze_frames(
            frames,
            selection_duration,
            metrics_path,
            start_seconds=capture.selection.start_seconds,
        )
        run.metrics["input"] = probe
        run.metrics["selection"] = capture.selection.model_dump(mode="json")
        run.metrics["disk_budget"] = budget
        run.metrics["preprocess"] = summarize_frame_metrics(records)
        _record_resources(run, work)
        complete_stage(
            run,
            "preprocess",
            message=f"Extracted and analyzed {len(frames)} equirectangular frames",
            log_path=log_path.relative_to(scene_path).as_posix(),
        )
    except Exception as error:
        fail_stage(run, "preprocess", str(error), log_path.relative_to(scene_path).as_posix())
        save_run(scene_path, run)
        raise
    save_run(scene_path, run)
    return run


def _prepare_masked_dataset(
    scene_path: Path,
    work: Path,
    source: Path,
    dataset: Path,
    attempt: ReconstructionAttempt,
    run: RunManifest,
    label: str,
) -> dict[str, Any]:
    images = project_equirectangular_frames(source, dataset, attempt)
    build_image_pyramid(
        dataset / "images", dataset, "images", attempt.num_downscales, is_mask=False
    )
    records = generate_person_masks(
        dataset / "images",
        dataset / "masks",
        run.config.masking,
        work / f"mask-metrics-{label}.jsonl",
    )
    deterministic = validate_mask_set(
        dataset / "images",
        dataset / "masks",
        run.config.masking.max_masked_fraction,
    )
    build_image_pyramid(
        dataset / "masks", dataset, "masks", attempt.num_downscales, is_mask=True
    )
    sheets = create_mask_contact_sheets(
        dataset / "images",
        dataset / "masks",
        records,
        dataset / "mask-qa",
        run.config.masking.qa_sample_count,
        run.config.vision_qa.max_contact_sheets,
    )
    vision: dict[str, Any]
    if run.config.vision_qa.enabled:
        verdict = run_mimo_mask_qa(sheets, run.config.vision_qa)
        vision = verdict.model_dump(mode="json")
        if verdict.decision != "pass":
            raise RuntimeError(
                f"MiMo mask QA rejected {label}: {verdict.rationale}; "
                f"false negatives={verdict.false_negative_views}"
            )
    else:
        vision = {
            "status": "disabled",
            "reason": (
                "Set vision_qa.enabled=true in a new run only after confirming an authorized "
                "MiMo multimodal endpoint"
            ),
        }
    return {
        "planar_images": len(images),
        "model": run.config.masking.model if run.config.masking.enabled else "disabled-all-white",
        "detection": summarize_detection_records(records),
        "validation": deterministic,
        "contact_sheets": [path.relative_to(scene_path).as_posix() for path in sheets],
        "vision_qa": vision,
    }


def mask_run(scene_path: Path, run: RunManifest, resume: bool = False) -> RunManifest:
    if not begin_stage(run, "mask", resume=resume):
        return run
    save_run(scene_path, run)
    work = scene_path / "work" / run.id
    log_path = work / "logs" / "mask.log"
    try:
        primary = _prepare_masked_dataset(
            scene_path,
            work,
            work / "equirect-primary",
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
    except Exception as error:
        fail_stage(run, "mask", str(error), log_path.relative_to(scene_path).as_posix())
        save_run(scene_path, run)
        raise
    save_run(scene_path, run)
    return run


def reconstruct_run(scene_path: Path, run: RunManifest, resume: bool = False) -> RunManifest:
    if not begin_stage(run, "reconstruct", resume=resume):
        return run
    save_run(scene_path, run)
    work = scene_path / "work" / run.id
    metrics_records = [
        json.loads(line)
        for line in (work / "frame-metrics.jsonl").read_text(encoding="utf-8").splitlines()
        if line
    ]
    attempts: dict[str, Any] = {}
    try:
        primary_output = work / "reconstruction-primary"
        primary_metrics = run_masked_colmap(
            primary_output,
            run.config.reconstruction.primary,
            work / "logs" / "reconstruct-primary",
        )
        attempts["primary"] = primary_metrics
        threshold = run.config.reconstruction.registration_threshold
        selected = primary_output

        if min(
            primary_metrics["registration_ratio"],
            primary_metrics["largest_component_coverage"],
        ) < threshold:
            run.fallback_attempted = True
            fallback_frames = work / "equirect-fallback"
            create_blur_aware_subset(
                work / "equirect-primary",
                metrics_records,
                fallback_frames,
                run.config.reconstruction.fallback.frame_count,
            )
            fallback_output = work / "reconstruction-fallback"
            fallback_masking = _prepare_masked_dataset(
                scene_path,
                work,
                fallback_frames,
                fallback_output,
                run.config.reconstruction.fallback,
                run,
                "fallback",
            )
            run.metrics.setdefault("masking", {})["fallback"] = fallback_masking
            fallback_metrics = run_masked_colmap(
                fallback_output,
                run.config.reconstruction.fallback,
                work / "logs" / "reconstruct-fallback",
            )
            attempts["fallback"] = fallback_metrics
            selected = fallback_output
            if min(
                fallback_metrics["registration_ratio"],
                fallback_metrics["largest_component_coverage"],
            ) < threshold:
                raise RuntimeError(
                    "Both reconstruction attempts were below the 70% registration/coverage threshold; "
                    "stop and prepare a recapture report"
                )

        run.metrics["reconstruction"] = attempts
        run.selected_dataset = selected.relative_to(scene_path).as_posix()
        _record_resources(run, work)
        complete_stage(
            run,
            "reconstruct",
            message=f"Selected dataset: {run.selected_dataset}",
            log_path=(work / "logs").relative_to(scene_path).as_posix(),
        )
    except Exception as error:
        run.metrics["reconstruction"] = attempts
        fail_stage(run, "reconstruct", str(error), (work / "logs").relative_to(scene_path).as_posix())
        save_run(scene_path, run)
        raise
    save_run(scene_path, run)
    return run


def _find_training_config(output: Path) -> Path | None:
    configs = sorted(output.rglob("config.yml"), key=lambda item: item.stat().st_mtime)
    return configs[-1] if configs else None


def _train_command(run: RunManifest, dataset: Path, output: Path, downscale: int | None = None) -> list[str]:
    command = [
        "ns-train",
        run.config.train.method,
        "--output-dir",
        str(output),
        "--max-num-iterations",
        str(run.config.train.max_iterations),
        "--viewer.quit-on-train-completion",
        "True",
        "nerfstudio-data",
        "--data",
        str(dataset),
    ]
    if downscale is not None:
        command.extend(["--downscale-factor", str(downscale)])
    return command


def train_run(scene_path: Path, run: RunManifest, resume: bool = False) -> RunManifest:
    if not begin_stage(run, "train", resume=resume):
        return run
    save_run(scene_path, run)
    assert run.selected_dataset is not None
    dataset = ensure_within(scene_path / run.selected_dataset, scene_path)
    work = scene_path / "work" / run.id
    training_root = work / "training"
    attempt_one = training_root / "attempt-1"
    log_one = work / "logs" / "train-attempt-1.log"
    selected_config: Path | None = _find_training_config(attempt_one)
    try:
        if selected_config is None:
            try:
                train_metrics = run_logged(
                    _train_command(run, dataset, attempt_one), log_one, monitor_gpu=True
                )
                run.metrics.setdefault("train", {}).setdefault("attempts", []).append(
                    {"name": "attempt-1", **train_metrics}
                )
            except CommandError as command_error:
                run.metrics.setdefault("train", {}).setdefault("attempts", []).append(
                    {"name": "attempt-1", **command_error.metrics}
                )
                content = log_one.read_text(encoding="utf-8", errors="replace")
                if "out of memory" not in content.lower():
                    raise
                attempt_two = training_root / "attempt-2-downscaled"
                log_two = work / "logs" / "train-attempt-2.log"
                retry_metrics = run_logged(
                    _train_command(
                        run,
                        dataset,
                        attempt_two,
                        downscale=run.config.train.oom_retry_downscale,
                    ),
                    log_two,
                    monitor_gpu=True,
                )
                run.metrics.setdefault("train", {}).setdefault("attempts", []).append(
                    {"name": "attempt-2-downscaled", **retry_metrics}
                )
                selected_config = _find_training_config(attempt_two)
                run.metrics.setdefault("train", {})["oom_retry"] = True
            else:
                selected_config = _find_training_config(attempt_one)
        if selected_config is None:
            raise RuntimeError("Training finished without producing config.yml")
        run.metrics.setdefault("train", {})["config_path"] = selected_config.relative_to(scene_path).as_posix()
        attempts = run.metrics.get("train", {}).get("attempts", [])
        for item in attempts:
            _record_resources(run, work, item)
        complete_stage(
            run,
            "train",
            message=f"Training config: {selected_config.relative_to(scene_path).as_posix()}",
            log_path=(work / "logs").relative_to(scene_path).as_posix(),
        )
    except Exception as error:
        fail_stage(run, "train", str(error), (work / "logs").relative_to(scene_path).as_posix())
        save_run(scene_path, run)
        raise
    save_run(scene_path, run)
    return run


def _artifact_record(path: Path, scene_path: Path, kind: str, version: str) -> ArtifactRecord:
    return ArtifactRecord(
        kind=kind,
        relative_path=path.relative_to(scene_path).as_posix(),
        sha256=sha256_file(path),
        byte_size=path.stat().st_size,
        version=version,
    )


def export_run(
    scene_path: Path,
    run: RunManifest,
    version: str = "v001",
    resume: bool = False,
) -> RunManifest:
    if not VERSION_PATTERN.match(version):
        raise ValueError("Artifact version must look like v001")
    if not begin_stage(run, "export", resume=resume):
        return run
    save_run(scene_path, run)
    export_dir = scene_path / "exports" / version
    manifest_path = export_dir / "artifact.yaml"
    work = scene_path / "work" / run.id
    log_dir = work / "logs"
    try:
        existing_manifest = manifest_path.is_file()
        if export_dir.exists() and any(export_dir.iterdir()) and not (resume and existing_manifest):
            raise FileExistsError(f"Export directory is not empty: {export_dir}")
        export_dir.mkdir(parents=True, exist_ok=True)
        config = ensure_within(scene_path / run.metrics["train"]["config_path"], scene_path)
        ply_files = sorted(export_dir.glob("*.ply"))
        if not ply_files:
            run_logged(
                [
                    "ns-export",
                    "gaussian-splat",
                    "--load-config",
                    str(config),
                    "--output-dir",
                    str(export_dir),
                ],
                log_dir / "export-ply.log",
            )
            ply_files = sorted(export_dir.glob("*.ply"))
        if len(ply_files) != 1:
            raise RuntimeError(f"Expected one exported PLY, found {len(ply_files)}")

        preview = export_dir / "preview.mp4"
        if not preview.is_file():
            run_logged(
                [
                    "ns-render",
                    "spiral",
                    "--load-config",
                    str(config),
                    "--output-path",
                    str(preview),
                    "--seconds",
                    "3",
                    "--frame-rate",
                    "24",
                    "--downscale-factor",
                    "4",
                    "--rendered-output-names",
                    "rgb",
                ],
                log_dir / "export-preview.log",
            )
        thumbnail = export_dir / "thumbnail.jpg"
        if not thumbnail.is_file():
            run_logged(
                [
                    "ffmpeg",
                    "-hide_banner",
                    "-loglevel",
                    "error",
                    "-ss",
                    "1",
                    "-i",
                    str(preview),
                    "-frames:v",
                    "1",
                    str(thumbnail),
                ],
                log_dir / "export-thumbnail.log",
            )

        dataset = ensure_within(scene_path / (run.selected_dataset or ""), scene_path)
        for name in ("transforms.json", "dataparser_transforms.json"):
            source = dataset / name
            if source.is_file():
                shutil.copy2(source, export_dir / name)

        artifacts = [
            _artifact_record(ply_files[0], scene_path, "gaussian_ply", version),
            _artifact_record(preview, scene_path, "preview_video", version),
            _artifact_record(thumbnail, scene_path, "thumbnail", version),
        ]
        for name in ("transforms.json", "dataparser_transforms.json"):
            target = export_dir / name
            if target.is_file():
                artifacts.append(_artifact_record(target, scene_path, name.removesuffix(".json"), version))
        run.artifacts = artifacts
        _record_resources(run, work, extra_path=export_dir)
        artifact_manifest = ArtifactManifest(
            location_id=run.location_id,
            scene_id=run.scene_id,
            run_id=run.id,
            version=version,
            status=RunStatus.NEEDS_REVIEW,
            generated_at=utc_now(),
            artifacts=artifacts,
            qa_metrics=run.metrics,
        )
        save_yaml(manifest_path, artifact_manifest)
        complete_stage(
            run,
            "export",
            message=f"Exported {len(artifacts)} artifacts to {export_dir.relative_to(scene_path)}",
            log_path=log_dir.relative_to(scene_path).as_posix(),
        )
    except Exception as error:
        fail_stage(run, "export", str(error), log_dir.relative_to(scene_path).as_posix())
        save_run(scene_path, run)
        raise
    save_run(scene_path, run)
    return run


def write_qa_report(scene_path: Path, run: RunManifest, resume: bool = False) -> Path:
    if not begin_stage(run, "qa", resume=resume):
        return scene_path / "qa" / f"{run.id}.md"
    save_run(scene_path, run)
    report_path = scene_path / "qa" / f"{run.id}.md"
    try:
        reconstruction = run.metrics.get("reconstruction", {})
        resources = run.metrics.get("resources", {})
        selected_name = "fallback" if run.fallback_attempted else "primary"
        selected = reconstruction.get(selected_name, {})
        masking = run.metrics.get("masking", {}).get(selected_name, {})
        mask_validation = masking.get("validation", {})
        mask_vision = masking.get("vision_qa", {})
        lines = [
            f"# QA: {run.id}",
            "",
            f"- 状态：`needs_review`",
            f"- 配置哈希：`{run.config_hash}`",
            f"- 选中数据集：`{run.selected_dataset}`",
            f"- 注册率：{float(selected.get('registration_ratio', 0)):.2%}",
            f"- 最大连通模型覆盖：{float(selected.get('largest_component_coverage', 0)):.2%}",
            f"- 人像遮罩数：{int(mask_validation.get('mask_count', 0))}",
            f"- 遮罩最大像素占比：{float(mask_validation.get('max_masked_fraction', 0)):.2%}",
            f"- 遮罩确定性 QA：`{mask_validation.get('deterministic_qa', 'missing')}`",
            f"- MiMo 遮罩 QA：`{mask_vision.get('decision', mask_vision.get('status', 'missing'))}`",
            f"- 使用降级重建：{'是' if run.fallback_attempted else '否'}",
            f"- 使用 OOM 降采样重试：{'是' if run.metrics.get('train', {}).get('oom_retry') else '否'}",
            f"- 观测磁盘峰值：{float(resources.get('disk_observed_peak_bytes', 0)) / 1024**3:.2f} GiB",
            f"- 观测显存峰值：{float(resources.get('gpu_memory_observed_peak_mib', 0)):.0f} MiB",
            "",
            "## 最终资产人工检查清单（人物遮罩已在 mask 阶段门禁）",
            "",
            "- [ ] PLY 可在兼容查看器中打开",
            "- [ ] 预览视频不存在明显相机轨迹断裂",
            "- [ ] 记录漂浮噪点、几何断裂和灯光曝光伪影",
            "- [ ] 确认素材与场地使用权限",
            "- [ ] 决定 accepted 或 rejected",
            "",
            "## 说明",
            "",
            "此试点只验证流程，不保证度量尺度或正式参考库质量。",
        ]
        report_path.parent.mkdir(parents=True, exist_ok=True)
        report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
        complete_stage(run, "qa", message=f"QA report: {report_path.relative_to(scene_path)}")
    except Exception as error:
        fail_stage(run, "qa", str(error))
        save_run(scene_path, run)
        raise
    save_run(scene_path, run)
    return report_path


def review_run(scene_path: Path, run: RunManifest, accepted: bool, notes: str) -> RunManifest:
    if run.status != RunStatus.NEEDS_REVIEW:
        raise RuntimeError(f"Only needs_review runs can be reviewed; current status is {run.status.value}")
    run.status = RunStatus.ACCEPTED if accepted else RunStatus.REJECTED
    run.review_notes = notes
    for artifact_path in (scene_path / "exports").glob("*/artifact.yaml"):
        artifact = load_model(artifact_path, ArtifactManifest)
        if artifact.run_id == run.id:
            artifact.status = run.status
            save_yaml(artifact_path, artifact)
    save_run(scene_path, run)
    return run
