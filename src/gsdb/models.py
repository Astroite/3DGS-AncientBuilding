from __future__ import annotations

import ntpath
from datetime import date, datetime, timezone
from enum import Enum
from pathlib import PureWindowsPath
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator


SLUG_PATTERN = r"^[a-z0-9]+(?:-[a-z0-9]+)*$"


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class StringEnum(str, Enum):
    """Python 3.10-compatible equivalent of enum.StrEnum."""

    def __str__(self) -> str:
        return self.value


class RunStatus(StringEnum):
    DRAFT = "draft"
    PROCESSING = "processing"
    FAILED = "failed"
    WAITING_REVIEW = "waiting_review"
    NEEDS_REVIEW = "needs_review"
    ACCEPTED = "accepted"
    REJECTED = "rejected"


class StageStatus(StringEnum):
    PENDING = "pending"
    PROCESSING = "processing"
    SUCCEEDED = "succeeded"
    FAILED = "failed"
    SKIPPED = "skipped"


class Region(StrictModel):
    country_code: str = "CN"
    province: str | None = None
    city: str | None = None


class Rights(StrictModel):
    status: str = "needs_review"
    notes: str | None = None


class LocationManifest(StrictModel):
    schema_version: Literal[1] = 1
    id: str = Field(pattern=SLUG_PATTERN)
    display_name: str = Field(min_length=1)
    status: RunStatus = RunStatus.DRAFT
    capture_date: date | None = None
    region: Region = Field(default_factory=Region)
    tags: list[str] = Field(default_factory=list)
    rights: Rights = Field(default_factory=Rights)
    notes: str | None = None


class SceneManifest(StrictModel):
    schema_version: Literal[1] = 1
    id: str = Field(pattern=SLUG_PATTERN)
    location_id: str = Field(pattern=SLUG_PATTERN)
    display_name: str = Field(min_length=1)
    status: RunStatus = RunStatus.DRAFT
    description: str | None = None
    metric_scale: bool = False
    quality_target: Literal["pipeline_proof", "reference_quality"] = "pipeline_proof"
    default_capture_id: str | None = None
    notes: str | None = None


class RawSource(StrictModel):
    windows_path: str
    immutable: bool = True
    byte_size: int | None = None
    sha256: str | None = Field(default=None, pattern=r"^[0-9a-f]{64}$")


class StitchedVideo(StrictModel):
    relative_path: str
    sha256: str | None = None
    byte_size: int | None = None
    width: int | None = None
    height: int | None = None
    fps: float | None = None
    duration_seconds: float | None = None
    codec: str | None = None


class Camera(StrictModel):
    make: str
    model: str
    firmware: str | None = None


class SourceCharacteristics(StrictModel):
    lens_streams: int | None = None
    lens_resolution: str | None = None
    stitched_resolution: str | None = None
    fps: float | None = None
    duration_seconds: float | None = None
    codec: str | None = None
    approximate_bitrate_mbps: float | None = None


class ExportSettings(StrictModel):
    projection: Literal["equirectangular"] = "equirectangular"
    aspect_ratio: Literal["2:1"] = "2:1"
    resolution: str
    codec: str
    color_space: str
    bitrate: str
    reframing: bool = False
    ai_denoise: bool = False
    sharpening: bool = False
    stabilization: str = "standard"


class TimeSelection(StrictModel):
    start_seconds: float = Field(default=0.0, ge=0)
    end_seconds: float = Field(gt=0)

    @model_validator(mode="after")
    def validate_range(self) -> "TimeSelection":
        if self.end_seconds <= self.start_seconds:
            raise ValueError("end_seconds must be greater than start_seconds")
        return self


class CaptureManifest(StrictModel):
    schema_version: Literal[1] = 1
    id: str = Field(pattern=SLUG_PATTERN)
    location_id: str = Field(pattern=SLUG_PATTERN)
    scene_id: str = Field(pattern=SLUG_PATTERN)
    status: RunStatus = RunStatus.DRAFT
    raw_source: RawSource
    stitched_video: StitchedVideo
    camera: Camera
    source_characteristics: SourceCharacteristics = Field(default_factory=SourceCharacteristics)
    selection: TimeSelection
    export_settings: ExportSettings
    notes: str | None = None

    @field_validator("stitched_video")
    @classmethod
    def validate_relative_video(cls, value: StitchedVideo) -> StitchedVideo:
        if value.relative_path.startswith(("/", "\\")) or ":" in value.relative_path:
            raise ValueError("stitched_video.relative_path must be relative to the scene directory")
        return value


class SourceFile(StrictModel):
    """One immutable source file owned outside the prepared-input cache."""

    windows_path: str
    sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    byte_size: int = Field(ge=0)


class SourceProbe(StrictModel):
    width: int | None = Field(default=None, gt=0)
    height: int | None = Field(default=None, gt=0)
    fps: float | None = Field(default=None, gt=0)
    frame_count: int | None = Field(default=None, gt=0)
    duration_seconds: float | None = Field(default=None, gt=0)
    codec: str | None = None
    camera_model: str | None = None
    helper_version: str | None = None
    sdk_version: str | None = None


class PanoramaSource(StrictModel):
    kind: Literal["equirect_video", "equirect_sequence", "insta360_insv"]
    files: list[SourceFile] = Field(min_length=1)
    projection: Literal["equirectangular", "dual_fisheye"]
    probe: SourceProbe = Field(default_factory=SourceProbe)

    @model_validator(mode="after")
    def validate_source_shape(self) -> "PanoramaSource":
        paths = [
            ntpath.normcase(ntpath.normpath(item.windows_path.replace("/", "\\")))
            for item in self.files
        ]
        if len(set(paths)) != len(paths):
            raise ValueError("source.files must list distinct paths explicitly")
        if self.kind == "equirect_video" and len(self.files) != 1:
            raise ValueError("equirect_video requires exactly one source file")
        if self.kind == "insta360_insv" and len(self.files) not in (1, 2):
            raise ValueError("insta360_insv requires one or two explicitly listed files")
        suffixes = {
            PureWindowsPath(item.windows_path.replace("/", "\\")).suffix.casefold()
            for item in self.files
        }
        if self.kind == "equirect_video" and not suffixes.issubset(
            {".mp4", ".mov", ".mkv", ".avi", ".webm"}
        ):
            raise ValueError("equirect_video source must use a supported video extension")
        if self.kind == "equirect_sequence" and not suffixes.issubset(
            {".jpg", ".jpeg", ".png", ".tif", ".tiff"}
        ):
            raise ValueError("equirect_sequence source files must all be supported images")
        if self.kind == "insta360_insv" and suffixes != {".insv"}:
            raise ValueError("insta360_insv source files must all use the .insv extension")
        expected = "dual_fisheye" if self.kind == "insta360_insv" else "equirectangular"
        if self.projection != expected:
            raise ValueError(f"{self.kind} source projection must be {expected}")
        return self


class NormalizationSettings(StrictModel):
    projection: Literal["equirectangular"] = "equirectangular"
    aspect_ratio: Literal["2:1"] = "2:1"
    width: int | None = Field(default=None, gt=0)
    height: int | None = Field(default=None, gt=0)
    image_format: Literal["jpeg"] = "jpeg"
    jpeg_quality: int = Field(default=95, ge=1, le=100)
    stitching: Literal["dynamic"] = "dynamic"
    flowstate: Literal[True] = True
    direction_lock: Literal[False] = False
    ai_denoise: Literal[False] = False
    sharpening: Literal[False] = False
    color_processing: Literal[False] = False

    @model_validator(mode="after")
    def validate_dimensions(self) -> "NormalizationSettings":
        if (self.width is None) != (self.height is None):
            raise ValueError("normalization width and height must be set together")
        if self.width is not None and self.width != self.height * 2:
            raise ValueError("normalization dimensions must be exactly 2:1")
        return self


class CaptureManifestV2(StrictModel):
    schema_version: Literal[2] = 2
    id: str = Field(pattern=SLUG_PATTERN)
    location_id: str = Field(pattern=SLUG_PATTERN)
    scene_id: str = Field(pattern=SLUG_PATTERN)
    status: RunStatus = RunStatus.DRAFT
    source: PanoramaSource
    camera: Camera
    selection: TimeSelection
    normalization: NormalizationSettings = Field(default_factory=NormalizationSettings)
    prepared_relative_path: str | None = None
    notes: str | None = None

    @field_validator("prepared_relative_path")
    @classmethod
    def validate_prepared_path(cls, value: str | None) -> str | None:
        if value is not None and (value.startswith(("/", "\\")) or ":" in value):
            raise ValueError("prepared_relative_path must be relative to the scene directory")
        return value


class LegacyPreprocessConfigV1(StrictModel):
    target_frames: int = Field(default=270, ge=2, le=5000)
    jpeg_quality: int = Field(default=2, ge=1, le=31)
    minimum_free_gib: float = Field(default=20.0, ge=1)


class PreprocessConfig(LegacyPreprocessConfigV1):
    """Candidate-frame extraction settings for schema v2 runs."""


class PreprocessConfigV3(StrictModel):
    """Prepared-input controls; JPEG encoding lives in normalization."""

    target_frames: int = Field(default=270, ge=2, le=5000)
    minimum_free_gib: float = Field(default=20.0, ge=1)


class MaskingConfig(StrictModel):
    enabled: bool = True
    model: Literal["maskrcnn_resnet50_fpn_v2"] = "maskrcnn_resnet50_fpn_v2"
    weights: Literal["DEFAULT"] = "DEFAULT"
    device: Literal["cuda", "cpu"] = "cuda"
    person_class_id: int = 1
    score_threshold: float = Field(default=0.25, ge=0, le=1)
    probability_threshold: float = Field(default=0.50, ge=0, le=1)
    inference_gamma: float = Field(default=0.75, gt=0, le=2)
    dilation_pixels: int = Field(default=24, ge=0, le=256)
    closing_pixels: int = Field(default=7, ge=0, le=255)
    max_masked_fraction: float = Field(default=0.45, gt=0, lt=1)
    qa_sample_count: int = Field(default=32, ge=1, le=256)

    @field_validator("closing_pixels")
    @classmethod
    def validate_closing_kernel(cls, value: int) -> int:
        if value not in (0, 1) and value % 2 == 0:
            raise ValueError("closing_pixels must be odd (or 0/1 to disable closing)")
        return value


COCO_DYNAMIC_CLASS_IDS = {
    "person": 1,
    "bicycle": 2,
    "car": 3,
    "motorcycle": 4,
    "bus": 6,
    "truck": 8,
}


class MaskingConfigV3(MaskingConfig):
    classes: list[Literal["person", "car", "bus", "truck", "bicycle", "motorcycle"]] = Field(
        default_factory=lambda: ["person"], min_length=1
    )

    @field_validator("classes")
    @classmethod
    def validate_unique_classes(cls, value: list[str]) -> list[str]:
        if len(set(value)) != len(value):
            raise ValueError("masking classes must be unique")
        return value


class VisionQAConfig(StrictModel):
    enabled: bool = False
    provider: Literal["deepseek"] = "deepseek"
    model: Literal["deepseek-v4-flash-vision-exp"] = "deepseek-v4-flash-vision-exp"
    api_key_env: str = "DEEPSEEK_API_KEY"
    base_url: Literal["https://api.deepseek.com"] = "https://api.deepseek.com"
    endpoint_path: str = "/chat/completions"
    image_detail: Literal["original"] = "original"
    timeout_seconds: int = Field(default=120, ge=10, le=600)
    max_contact_sheets: int = Field(default=4, ge=1, le=16)
    minimum_confidence: float = Field(default=0.80, ge=0, le=1)


class LegacyReconstructionAttemptV1(StrictModel):
    frame_count: int
    images_per_equirect: Literal[8, 14]
    crop_bottom: float = Field(ge=0, lt=0.5)
    matching_method: Literal["sequential"] = "sequential"
    # No longer read by the pipeline (the downscale pyramid it configured was
    # removed as dead weight under RealityScan+Postshot). Kept only so existing
    # runs' persisted manifests -- which already serialized this field -- still
    # deserialize under StrictModel's extra="forbid".
    num_downscales: int = Field(default=2, ge=0, le=4)


class LegacyReconstructionConfigV1(StrictModel):
    registration_threshold: float = Field(default=0.70, gt=0, le=1)
    primary: LegacyReconstructionAttemptV1 = Field(
        default_factory=lambda: LegacyReconstructionAttemptV1(
            frame_count=270, images_per_equirect=8, crop_bottom=0.20
        )
    )
    fallback: LegacyReconstructionAttemptV1 = Field(
        default_factory=lambda: LegacyReconstructionAttemptV1(
            frame_count=180, images_per_equirect=14, crop_bottom=0.15
        )
    )


class LegacyTrainConfigV1(StrictModel):
    method: Literal["splatfacto"] = "splatfacto"
    max_iterations: int = Field(default=30_000, ge=1000)
    oom_retry_downscale: int = Field(default=2, ge=2, le=8)


class LegacyRunConfigV1(StrictModel):
    """Exact schema used by existing runs; intentionally has no version field.

    Keeping this shape separate prevents v2 defaults from entering a historical
    run's canonical serialization and changing its immutable config hash.
    """

    capture_id: str = Field(pattern=SLUG_PATTERN)
    input_sha256: str
    preprocess: LegacyPreprocessConfigV1 = Field(default_factory=LegacyPreprocessConfigV1)
    masking: MaskingConfig = Field(default_factory=MaskingConfig)
    vision_qa: VisionQAConfig = Field(default_factory=VisionQAConfig)
    reconstruction: LegacyReconstructionConfigV1 = Field(
        default_factory=LegacyReconstructionConfigV1
    )
    train: LegacyTrainConfigV1 = Field(default_factory=LegacyTrainConfigV1)


class ReconstructionAttempt(StrictModel):
    frame_count: int = Field(ge=2, le=5000)
    images_per_equirect: Literal[8, 14]
    projection_fov_degrees: float = Field(gt=0, lt=180)
    projection_size: int = Field(ge=256, le=8192)
    crop_bottom: float = Field(ge=0, lt=0.5)
    use_rig: bool = True
    matching_method: Literal["sequential"] = "sequential"
    # No longer read by the pipeline (the downscale pyramid it configured was
    # removed as dead weight under RealityScan+Postshot). Kept only so existing
    # runs' persisted manifests -- which already serialized this field -- still
    # deserialize under StrictModel's extra="forbid".
    num_downscales: int = Field(default=2, ge=0, le=4)


class ReconstructionConfig(StrictModel):
    use_gpu_sift: bool = True
    gpu_index: int = Field(default=0, ge=0)
    fix_intrinsics: bool = True
    registration_threshold: float = Field(default=0.70, gt=0, le=1)
    rig_center_spread_ratio_limit: float = Field(default=0.001, gt=0, le=0.1)
    primary: ReconstructionAttempt = Field(
        default_factory=lambda: ReconstructionAttempt(
            frame_count=135,
            images_per_equirect=8,
            projection_fov_degrees=120.0,
            projection_size=2048,
            crop_bottom=0.20,
        )
    )
    fallback: ReconstructionAttempt = Field(
        default_factory=lambda: ReconstructionAttempt(
            frame_count=180,
            images_per_equirect=14,
            projection_fov_degrees=110.0,
            projection_size=1746,
            crop_bottom=0.15,
        )
    )


class LoopClosureConfig(StrictModel):
    enabled: bool = False
    period: int = Field(default=20, ge=1)
    num_images: int = Field(default=5, ge=1)
    vocabulary_tree_path: str | None = None
    vocabulary_tree_sha256: str | None = Field(
        default=None, pattern=r"^[0-9a-f]{64}$"
    )

    @model_validator(mode="after")
    def require_vocabulary_tree(self) -> "LoopClosureConfig":
        if self.enabled and not self.vocabulary_tree_path:
            raise ValueError(
                "loop closure requires an explicit vocabulary_tree_path"
            )
        return self


class ReconstructionConfigV3(ReconstructionConfig):
    loop_closure: LoopClosureConfig = Field(default_factory=LoopClosureConfig)


class TrainConfig(StrictModel):
    method: Literal["splatfacto-big"] = "splatfacto-big"
    max_iterations: int = Field(default=100_000, ge=1000)
    downscale_factor: int = Field(default=1, ge=1, le=8)
    oom_retry_downscale: int = Field(default=2, ge=2, le=8)
    cache_images: Literal["cpu"] = "cpu"
    cache_images_type: Literal["uint8"] = "uint8"
    use_scale_regularization: bool = True
    rasterize_mode: Literal["classic"] = "classic"
    camera_optimizer_mode: Literal["SO3xR3"] = "SO3xR3"
    use_bilateral_grid: bool = True


class RunExportConfig(StrictModel):
    ply_axis: Literal["y_up"] = "y_up"


class RunConfig(StrictModel):
    schema_version: Literal[2] = 2
    capture_id: str = Field(pattern=SLUG_PATTERN)
    input_sha256: str
    preprocess: PreprocessConfig = Field(default_factory=PreprocessConfig)
    masking: MaskingConfig = Field(
        default_factory=lambda: MaskingConfig(qa_sample_count=16)
    )
    vision_qa: VisionQAConfig = Field(default_factory=VisionQAConfig)
    reconstruction: ReconstructionConfig = Field(default_factory=ReconstructionConfig)
    train: TrainConfig = Field(default_factory=TrainConfig)
    export: RunExportConfig = Field(default_factory=RunExportConfig)

    @model_validator(mode="after")
    def validate_reconstruction_subsets(self) -> "RunConfig":
        for label, attempt in (
            ("primary", self.reconstruction.primary),
            ("fallback", self.reconstruction.fallback),
        ):
            if attempt.frame_count > self.preprocess.target_frames:
                raise ValueError(
                    f"reconstruction.{label}.frame_count cannot exceed "
                    "preprocess.target_frames"
                )
        return self


class PreparedInputConfig(StrictModel):
    source_kind: Literal["equirect_video", "equirect_sequence", "insta360_insv"]
    source_sha256: list[str] = Field(min_length=1)
    source_probe: SourceProbe
    normalization: NormalizationSettings
    selection: TimeSelection
    candidate_frame_indices: list[int] = Field(min_length=2)
    helper_version: str | None = None
    sdk_version: str | None = None


class FrameSelectionConfig(StrictModel):
    algorithm: Literal["time_bucket_composite_v1"] = "time_bucket_composite_v1"
    gaussian_kernel: Literal[3] = 3
    sharpness_metric: Literal["tenengrad"] = "tenengrad"
    sharpness_weight: Literal[0.70] = 0.70
    black_fraction_inverse_weight: Literal[0.15] = 0.15
    highlight_fraction_inverse_weight: Literal[0.15] = 0.15
    tie_break: Literal["earlier_frame"] = "earlier_frame"


class RunConfigV3(StrictModel):
    schema_version: Literal[3] = 3
    capture_id: str = Field(pattern=SLUG_PATTERN)
    input_dataset_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    prepared_relative_path: str
    input: PreparedInputConfig
    frame_selection: FrameSelectionConfig = Field(default_factory=FrameSelectionConfig)
    preprocess: PreprocessConfigV3 = Field(default_factory=PreprocessConfigV3)
    masking: MaskingConfigV3 = Field(
        default_factory=lambda: MaskingConfigV3(qa_sample_count=16)
    )
    vision_qa: VisionQAConfig = Field(default_factory=VisionQAConfig)
    reconstruction: ReconstructionConfigV3 = Field(default_factory=ReconstructionConfigV3)
    train: TrainConfig = Field(default_factory=TrainConfig)
    export: RunExportConfig = Field(default_factory=RunExportConfig)

    @field_validator("prepared_relative_path")
    @classmethod
    def validate_prepared_relative_path(cls, value: str) -> str:
        if value.startswith(("/", "\\")) or ":" in value:
            raise ValueError("prepared_relative_path must be relative to the scene directory")
        return value

    @model_validator(mode="after")
    def validate_reconstruction_subsets(self) -> "RunConfigV3":
        if len(self.input.candidate_frame_indices) != self.preprocess.target_frames:
            raise ValueError(
                "input candidate_frame_indices must match preprocess.target_frames"
            )
        for label, attempt in (
            ("primary", self.reconstruction.primary),
            ("fallback", self.reconstruction.fallback),
        ):
            if attempt.frame_count > self.preprocess.target_frames:
                raise ValueError(
                    f"reconstruction.{label}.frame_count cannot exceed "
                    "preprocess.target_frames"
                )
        return self


class PreparedInputConfigV2(StrictModel):
    """Rate-sampled prepared input used by schema-v4 runs."""

    source_kind: Literal["equirect_video", "equirect_sequence", "insta360_insv"]
    source_sha256: list[str] = Field(min_length=1)
    source_probe: SourceProbe
    normalization: NormalizationSettings
    selection: TimeSelection
    candidate_frame_indices: list[int] = Field(min_length=2)
    candidate_fps: float = Field(default=5.0, gt=0, le=120)
    helper_version: str | None = None
    sdk_version: str | None = None


class PreprocessConfigV4(StrictModel):
    candidate_fps: float = Field(default=5.0, gt=0, le=120)
    selected_per_second: int = Field(default=2, ge=1, le=120)
    minimum_free_gib: float = Field(default=20.0, ge=1)


class MaskingConfigV4(StrictModel):
    enabled: bool = True
    model: Literal["maskrcnn_resnet50_fpn_v2"] = "maskrcnn_resnet50_fpn_v2"
    weights: Literal["DEFAULT"] = "DEFAULT"
    device: Literal["cuda", "cpu"] = "cuda"
    person_class_id: int = 1
    score_threshold: float = Field(default=0.25, ge=0, le=1)
    probability_threshold: float = Field(default=0.50, ge=0, le=1)
    inference_gamma: float = Field(default=0.75, gt=0, le=2)
    dilation_pixels: int = Field(default=24, ge=0, le=256)
    closing_pixels: int = Field(default=7, ge=0, le=255)
    mask_discard_threshold: float = Field(default=0.05, gt=0, lt=1)
    qa_sample_count: int = Field(default=16, ge=1, le=256)
    classes: list[Literal["person", "car", "bus", "truck", "bicycle", "motorcycle"]] = Field(
        default_factory=lambda: ["person"], min_length=1
    )
    mask_review_required: bool = False

    @field_validator("closing_pixels")
    @classmethod
    def validate_closing_kernel(cls, value: int) -> int:
        if value not in (0, 1) and value % 2 == 0:
            raise ValueError("closing_pixels must be odd (or 0/1 to disable closing)")
        return value

    @field_validator("classes")
    @classmethod
    def validate_unique_classes(cls, value: list[str]) -> list[str]:
        if len(set(value)) != len(value):
            raise ValueError("masking classes must be unique")
        return value


class ReconstructionAttemptV4(StrictModel):
    temporal_rank_limit: int = Field(ge=1, le=120)
    images_per_equirect: Literal[8, 14]
    projection_fov_degrees: float = Field(gt=0, lt=180)
    projection_size: int = Field(ge=256, le=8192)
    crop_bottom: float = Field(ge=0, lt=0.5)
    use_rig: bool = True
    matching_method: Literal["sequential"] = "sequential"


class ReconstructionConfigV4(StrictModel):
    use_gpu_sift: bool = True
    gpu_index: int = Field(default=0, ge=0)
    fix_intrinsics: bool = True
    registration_threshold: float = Field(default=0.70, gt=0, le=1)
    rig_center_spread_ratio_limit: float = Field(default=0.001, gt=0, le=0.1)
    primary: ReconstructionAttemptV4 = Field(
        default_factory=lambda: ReconstructionAttemptV4(
            temporal_rank_limit=1,
            images_per_equirect=8,
            projection_fov_degrees=120.0,
            projection_size=2048,
            crop_bottom=0.20,
        )
    )
    fallback: ReconstructionAttemptV4 = Field(
        default_factory=lambda: ReconstructionAttemptV4(
            temporal_rank_limit=2,
            images_per_equirect=14,
            projection_fov_degrees=110.0,
            projection_size=1746,
            crop_bottom=0.15,
        )
    )
    loop_closure: LoopClosureConfig = Field(default_factory=LoopClosureConfig)


class RunConfigV4(StrictModel):
    schema_version: Literal[4] = 4
    capture_id: str = Field(pattern=SLUG_PATTERN)
    input_dataset_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    prepared_relative_path: str
    input: PreparedInputConfigV2
    frame_selection: FrameSelectionConfig = Field(default_factory=FrameSelectionConfig)
    preprocess: PreprocessConfigV4 = Field(default_factory=PreprocessConfigV4)
    masking: MaskingConfigV4 = Field(default_factory=MaskingConfigV4)
    vision_qa: VisionQAConfig = Field(default_factory=VisionQAConfig)
    reconstruction: ReconstructionConfigV4 = Field(default_factory=ReconstructionConfigV4)
    train: TrainConfig = Field(default_factory=TrainConfig)
    export: RunExportConfig = Field(default_factory=RunExportConfig)

    @field_validator("prepared_relative_path")
    @classmethod
    def validate_prepared_relative_path(cls, value: str) -> str:
        if value.startswith(("/", "\\")) or ":" in value:
            raise ValueError("prepared_relative_path must be relative to the scene directory")
        return value

    @model_validator(mode="after")
    def validate_temporal_sampling(self) -> "RunConfigV4":
        selected = self.preprocess.selected_per_second
        primary = self.reconstruction.primary.temporal_rank_limit
        fallback = self.reconstruction.fallback.temporal_rank_limit
        if not 1 <= primary <= fallback <= selected <= self.preprocess.candidate_fps:
            raise ValueError(
                "temporal sampling requires 1 <= primary <= fallback <= "
                "selected_per_second <= candidate_fps"
            )
        if abs(self.input.candidate_fps - self.preprocess.candidate_fps) > 1e-9:
            raise ValueError("prepared input candidate_fps must match preprocess.candidate_fps")
        if self.reconstruction.primary.images_per_equirect != 8:
            raise ValueError("RunConfigV4 Primary must use 8 views per panorama")
        if self.reconstruction.fallback.images_per_equirect != 14:
            raise ValueError("RunConfigV4 Fallback must use 14 views per panorama")
        return self


class SegmentQAConfig(StrictModel):
    max_gap_seconds: float = Field(default=2.0, gt=0)
    speed_multiplier: float = Field(default=10.0, gt=1)
    minimum_samples: int = Field(default=10, ge=2)
    minimum_positions: int = Field(default=3, ge=2)
    minimum_views: int = Field(default=3, ge=1)
    registration_threshold: float = Field(default=0.70, gt=0, le=1)
    center_spread_step_ratio: float = Field(default=3.0, gt=0)


class ReconstructionConfigV5(ReconstructionConfigV4):
    primary: ReconstructionAttemptV4 = Field(default_factory=lambda: ReconstructionAttemptV4(
        temporal_rank_limit=2, images_per_equirect=14, projection_fov_degrees=110,
        projection_size=1746, crop_bottom=0.15, use_rig=False,
    ))
    fallback: ReconstructionAttemptV4 = Field(default_factory=lambda: ReconstructionAttemptV4(
        temporal_rank_limit=2, images_per_equirect=14, projection_fov_degrees=110,
        projection_size=1746, crop_bottom=0.15, use_rig=False,
    ))
    alignment_masks: bool = True
    repair_padding_seconds: float = Field(default=2.0, ge=0)
    repair_attempts: Literal[1] = 1


class RunConfigV5(RunConfigV4):
    schema_version: Literal[5] = 5
    reconstruction: ReconstructionConfigV5 = Field(default_factory=ReconstructionConfigV5)
    masking: MaskingConfigV4 = Field(default_factory=lambda: MaskingConfigV4(mask_discard_threshold=0.005))
    segment_qa: SegmentQAConfig = Field(default_factory=SegmentQAConfig)

    @model_validator(mode="after")
    def validate_temporal_sampling(self) -> "RunConfigV5":
        if not 1 <= self.reconstruction.primary.temporal_rank_limit <= self.preprocess.selected_per_second <= self.preprocess.candidate_fps:
            raise ValueError("primary ranks <= selected_per_second <= candidate_fps required")
        if abs(self.input.candidate_fps - self.preprocess.candidate_fps) > 1e-9:
            raise ValueError("prepared input candidate_fps must match preprocess.candidate_fps")
        if self.reconstruction.primary.use_rig or self.reconstruction.fallback.use_rig:
            raise ValueError("RealityScan rig constraints are not implemented")
        return self


class StageRecord(StrictModel):
    status: StageStatus = StageStatus.PENDING
    started_at: datetime | None = None
    completed_at: datetime | None = None
    elapsed_seconds: float | None = None
    message: str | None = None
    log_path: str | None = None


class ArtifactRecord(StrictModel):
    kind: str
    relative_path: str
    sha256: str
    byte_size: int
    version: str


def default_stages() -> dict[str, StageRecord]:
    return {
        name: StageRecord()
        for name in (
            "preprocess",
            "mask",
            "reconstruct",
            "postshot_prepare",
            "train",
            "export",
            "qa",
        )
    }


class RunManifest(StrictModel):
    schema_version: Literal[1] = 1
    id: str
    location_id: str = Field(pattern=SLUG_PATTERN)
    scene_id: str = Field(pattern=SLUG_PATTERN)
    status: RunStatus = RunStatus.DRAFT
    config_hash: str
    config: LegacyRunConfigV1 | RunConfig | RunConfigV3 | RunConfigV4 | RunConfigV5
    created_at: datetime
    updated_at: datetime
    active_stage: str | None = None
    stages: dict[str, StageRecord] = Field(default_factory=default_stages)
    tool_versions: dict[str, str] = Field(default_factory=dict)
    metrics: dict[str, Any] = Field(default_factory=dict)
    selected_dataset: str | None = None
    fallback_attempted: bool = False
    artifacts: list[ArtifactRecord] = Field(default_factory=list)
    review_notes: str | None = None


class ArtifactManifest(StrictModel):
    schema_version: Literal[1] = 1
    location_id: str
    scene_id: str
    run_id: str
    version: str
    status: RunStatus
    metric_scale: bool = False
    generated_at: datetime
    artifacts: list[ArtifactRecord]
    qa_metrics: dict[str, Any] = Field(default_factory=dict)


def utc_now() -> datetime:
    return datetime.now(timezone.utc).replace(microsecond=0)
