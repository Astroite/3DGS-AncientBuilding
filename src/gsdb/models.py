from __future__ import annotations

import ntpath
from datetime import date, datetime, timezone
from enum import Enum
from pathlib import PureWindowsPath
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator


SLUG_PATTERN = r"^[a-z0-9]+(?:-[a-z0-9]+)*$"

# One capture shape and one run shape. Historical run schemas are not read.
CAPTURE_SCHEMA_VERSION = 2
RUN_SCHEMA_VERSION = 6


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


class Camera(StrictModel):
    make: str
    model: str
    firmware: str | None = None


class TimeSelection(StrictModel):
    start_seconds: float = Field(default=0.0, ge=0)
    end_seconds: float = Field(gt=0)

    @model_validator(mode="after")
    def validate_range(self) -> "TimeSelection":
        if self.end_seconds <= self.start_seconds:
            raise ValueError("end_seconds must be greater than start_seconds")
        return self


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


VIDEO_SUFFIXES = {".mp4", ".mov", ".mkv", ".avi", ".webm"}
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".tif", ".tiff"}

SOURCE_PROJECTIONS = {
    "insta360_insv": "dual_fisheye",
    "perspective_video": "perspective",
}


class CaptureSource(StrictModel):
    """The registered raw media. Panorama kinds are projected into views; a
    perspective_video source is used frame for frame."""

    kind: Literal["equirect_video", "equirect_sequence", "insta360_insv", "perspective_video"]
    files: list[SourceFile] = Field(min_length=1)
    projection: Literal["equirectangular", "dual_fisheye", "perspective"]
    probe: SourceProbe = Field(default_factory=SourceProbe)

    @property
    def is_perspective(self) -> bool:
        return self.kind == "perspective_video"

    @model_validator(mode="after")
    def validate_source_shape(self) -> "CaptureSource":
        paths = [
            ntpath.normcase(ntpath.normpath(item.windows_path.replace("/", "\\")))
            for item in self.files
        ]
        if len(set(paths)) != len(paths):
            raise ValueError("source.files must list distinct paths explicitly")
        suffixes = {
            PureWindowsPath(item.windows_path.replace("/", "\\")).suffix.casefold()
            for item in self.files
        }
        expected_suffixes = IMAGE_SUFFIXES if self.kind == "equirect_sequence" else (
            {".insv"} if self.kind == "insta360_insv" else VIDEO_SUFFIXES
        )
        if not suffixes.issubset(expected_suffixes):
            raise ValueError(f"{self.kind} source must use supported {sorted(expected_suffixes)} files")
        limits = {
            "equirect_video": (1, 1),
            "perspective_video": (1, 1),
            "equirect_sequence": (1, 100000),
            "insta360_insv": (1, 2),
        }[self.kind]
        if not limits[0] <= len(self.files) <= limits[1]:
            raise ValueError(f"{self.kind} requires between {limits[0]} and {limits[1]} source files")
        expected = SOURCE_PROJECTIONS.get(self.kind, "equirectangular")
        if self.projection != expected:
            raise ValueError(f"{self.kind} source projection must be {expected}")
        return self


class NormalizationSettings(StrictModel):
    """Output size and stitching for prepared frames.

    Panorama sources normalise to a 2:1 equirectangular frame; a perspective_video
    source keeps its native dimensions and leaves width/height unset.
    """

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


class CaptureManifest(StrictModel):
    schema_version: Literal[2] = CAPTURE_SCHEMA_VERSION
    id: str = Field(pattern=SLUG_PATTERN)
    location_id: str = Field(pattern=SLUG_PATTERN)
    scene_id: str = Field(pattern=SLUG_PATTERN)
    status: RunStatus = RunStatus.DRAFT
    source: CaptureSource
    camera: Camera
    selection: TimeSelection
    normalization: NormalizationSettings = Field(default_factory=NormalizationSettings)
    # Focal prior for RealityScan only; the training intrinsics come from the
    # exported COLMAP model.
    horizontal_fov_degrees: float | None = Field(default=None, gt=0, lt=180)
    notes: str | None = None

    @model_validator(mode="after")
    def validate_source_settings(self) -> "CaptureManifest":
        if self.source.is_perspective:
            if self.horizontal_fov_degrees is None:
                raise ValueError("perspective_video requires horizontal_fov_degrees")
            if self.normalization.width is not None:
                raise ValueError(
                    "perspective_video keeps the source frame size; normalization dimensions must be unset"
                )
        else:
            if self.horizontal_fov_degrees is not None:
                raise ValueError("horizontal_fov_degrees applies only to perspective_video")
        return self


class PreprocessConfig(StrictModel):
    candidate_fps: float = Field(default=1.0, ge=1, le=120)
    selected_per_second: int = Field(default=1, ge=1, le=120)
    minimum_free_gib: float = Field(default=20.0, ge=1)


COCO_DYNAMIC_CLASS_IDS = {
    "person": 1,
    "bicycle": 2,
    "car": 3,
    "motorcycle": 4,
    "bus": 6,
    "truck": 8,
}


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
    mask_discard_threshold: float = Field(default=0.005, gt=0, lt=1)
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


class LoopClosureConfig(StrictModel):
    enabled: bool = False
    period: int = Field(default=20, ge=1)
    num_images: int = Field(default=5, ge=1)
    vocabulary_tree_path: str | None = None
    vocabulary_tree_sha256: str | None = Field(default=None, pattern=r"^[0-9a-f]{64}$")

    @model_validator(mode="after")
    def require_vocabulary_tree(self) -> "LoopClosureConfig":
        if self.enabled and not self.vocabulary_tree_path:
            raise ValueError("loop closure requires an explicit vocabulary_tree_path")
        return self


class PanoramaProjection(StrictModel):
    """How one prepared panorama becomes training views."""

    views: Literal[8, 14]
    size: int = Field(ge=256, le=8192)
    crop_bottom: float = Field(ge=0, lt=0.5)


class ReconstructionAttempt(StrictModel):
    temporal_rank_limit: int = Field(ge=1, le=120)
    # Equirect sources: horizontal FOV of each projected view. Perspective
    # sources: the capture's horizontal FOV, used as the RealityScan prior.
    projection_fov_degrees: float = Field(gt=0, lt=180)
    # None means the prepared frames are already the training views
    # (perspective_video): one frame is one view.
    panorama: PanoramaProjection | None = None


class ReconstructionConfig(StrictModel):
    use_gpu_sift: bool = True
    fix_intrinsics: bool = True
    registration_threshold: float = Field(default=0.70, gt=0, le=1)
    rig_center_spread_ratio_limit: float = Field(default=0.001, gt=0, le=0.1)
    alignment_masks: bool = True
    primary: ReconstructionAttempt = Field(
        default_factory=lambda: ReconstructionAttempt(
            temporal_rank_limit=1,
            projection_fov_degrees=110.0,
            panorama=PanoramaProjection(views=14, size=1746, crop_bottom=0.15),
        )
    )
    # One bounded local repair round after the primary alignment.
    repair_padding_seconds: float = Field(default=2.0, ge=0)
    repair_attempts: Literal[1] = 1
    repair_candidate_fps: float = Field(default=2.0, ge=1, le=120)
    loop_closure: LoopClosureConfig = Field(default_factory=LoopClosureConfig)


class SegmentQAConfig(StrictModel):
    max_gap_seconds: float = Field(default=2.0, gt=0)
    speed_multiplier: float = Field(default=10.0, gt=1)
    minimum_samples: int = Field(default=10, ge=2)
    minimum_positions: int = Field(default=3, ge=2)
    minimum_views: int = Field(default=3, ge=1)
    registration_threshold: float = Field(default=0.70, gt=0, le=1)
    center_spread_step_ratio: float = Field(default=3.0, gt=0)


class RetentionConfig(StrictModel):
    mode: Literal["minimal", "keep"] = "minimal"


class PreparedInputConfig(StrictModel):
    """Rate-sampled candidate frames prepared from the capture sources."""

    source_kind: Literal["equirect_video", "equirect_sequence", "insta360_insv", "perspective_video"]
    source_sha256: list[str] = Field(min_length=1)
    source_probe: SourceProbe
    normalization: NormalizationSettings
    selection: TimeSelection
    candidate_frame_indices: list[int] = Field(min_length=2)
    candidate_fps: float = Field(default=1.0, gt=0, le=120)
    helper_version: str | None = None
    sdk_version: str | None = None


class RunConfig(StrictModel):
    schema_version: Literal[6] = RUN_SCHEMA_VERSION
    capture_id: str = Field(pattern=SLUG_PATTERN)
    input_dataset_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    # Run-owned candidates, never Capture-shared: inputs/primary/<preparation hash>.
    input_relative_path: str
    input: PreparedInputConfig
    preprocess: PreprocessConfig = Field(default_factory=PreprocessConfig)
    masking: MaskingConfig = Field(default_factory=MaskingConfig)
    vision_qa: VisionQAConfig = Field(default_factory=VisionQAConfig)
    reconstruction: ReconstructionConfig = Field(default_factory=ReconstructionConfig)
    segment_qa: SegmentQAConfig = Field(default_factory=SegmentQAConfig)
    retention: RetentionConfig = Field(default_factory=RetentionConfig)

    @field_validator("input_relative_path")
    @classmethod
    def validate_owned_input(cls, value: str) -> str:
        from pathlib import PurePosixPath

        parts = PurePosixPath(value.replace("\\", "/")).parts
        if len(parts) != 3 or parts[:2] != ("inputs", "primary") or len(parts[2]) != 64:
            raise ValueError("Run input must live at inputs/primary/<preparation hash>")
        if any(character not in "0123456789abcdef" for character in parts[2]):
            raise ValueError("Invalid preparation hash")
        return "/".join(parts)

    @model_validator(mode="after")
    def validate_temporal_sampling(self) -> "RunConfig":
        primary = self.reconstruction.primary.temporal_rank_limit
        if not 1 <= primary <= self.preprocess.selected_per_second <= self.preprocess.candidate_fps:
            raise ValueError(
                "temporal sampling requires 1 <= primary <= selected_per_second <= candidate_fps"
            )
        if abs(self.input.candidate_fps - self.preprocess.candidate_fps) > 1e-9:
            raise ValueError("prepared input candidate_fps must match preprocess.candidate_fps")
        return self


class StageRecord(StrictModel):
    status: StageStatus = StageStatus.PENDING
    started_at: datetime | None = None
    completed_at: datetime | None = None
    elapsed_seconds: float | None = None
    message: str | None = None
    log_path: str | None = None


STAGES = ("preprocess", "mask", "reconstruct", "qa")


def default_stages() -> dict[str, StageRecord]:
    return {name: StageRecord() for name in STAGES}


class RunManifest(StrictModel):
    schema_version: Literal[1] = 1
    id: str
    location_id: str = Field(pattern=SLUG_PATTERN)
    scene_id: str = Field(pattern=SLUG_PATTERN)
    status: RunStatus = RunStatus.DRAFT
    config_hash: str
    config: RunConfig
    created_at: datetime
    updated_at: datetime
    active_stage: str | None = None
    stages: dict[str, StageRecord] = Field(default_factory=default_stages)
    tool_versions: dict[str, str] = Field(default_factory=dict)
    metrics: dict[str, Any] = Field(default_factory=dict)
    # Reconstruction dataset this run trains from: reconstruction-primary|repair.
    selected_dataset: str | None = None
    review_notes: str | None = None


def utc_now() -> datetime:
    return datetime.now(timezone.utc).replace(microsecond=0)
