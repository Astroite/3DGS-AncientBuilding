from __future__ import annotations

from datetime import date, datetime, timezone
from enum import Enum
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


class LegacyPreprocessConfigV1(StrictModel):
    target_frames: int = Field(default=270, ge=2, le=5000)
    jpeg_quality: int = Field(default=2, ge=1, le=31)
    minimum_free_gib: float = Field(default=20.0, ge=1)


class PreprocessConfig(LegacyPreprocessConfigV1):
    """Candidate-frame extraction settings for schema v2 runs."""


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
        for name in ("preprocess", "mask", "reconstruct", "train", "export", "qa")
    }


class RunManifest(StrictModel):
    schema_version: Literal[1] = 1
    id: str
    location_id: str = Field(pattern=SLUG_PATTERN)
    scene_id: str = Field(pattern=SLUG_PATTERN)
    status: RunStatus = RunStatus.DRAFT
    config_hash: str
    config: LegacyRunConfigV1 | RunConfig
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
