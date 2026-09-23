"""Services shared by the GS-Studio GUI and the ``gsdb`` CLI.

P1 boundary: read-only status, browsing, dependency views and error
classification. Both front ends call these instead of parsing each other's
output, and neither re-derives stage success from directories or exported
files. Write actions stay on their existing pipeline entry points until the
P2 operation contracts land.
"""
from __future__ import annotations

from .browse import (
    CaptureEntry,
    LocationEntry,
    Problem,
    RunEntry,
    SceneEntry,
    list_captures,
    list_locations,
    list_runs,
    list_scenes,
    load_run_entry,
    read_run_entry,
    require_location,
    require_scene,
    status_label,
)
from .deps import CheckView, collect_checks, format_checks
from .errors import (
    AlreadyExistsError,
    ExternalToolError,
    IntegrityError,
    InvalidInputError,
    LockedError,
    NotFoundError,
    PreconditionError,
    ServiceError,
    StorageError,
    UnsupportedError,
    classify,
)
from .run_status import (
    EvidenceView,
    ExperimentView,
    LogView,
    RunDetail,
    StageView,
    read_run_detail,
)

__all__ = [
    "AlreadyExistsError",
    "CaptureEntry",
    "CheckView",
    "EvidenceView",
    "ExperimentView",
    "ExternalToolError",
    "IntegrityError",
    "InvalidInputError",
    "LocationEntry",
    "LockedError",
    "LogView",
    "NotFoundError",
    "PreconditionError",
    "Problem",
    "RunDetail",
    "RunEntry",
    "SceneEntry",
    "ServiceError",
    "StageView",
    "StorageError",
    "UnsupportedError",
    "classify",
    "collect_checks",
    "format_checks",
    "list_captures",
    "list_locations",
    "list_runs",
    "list_scenes",
    "load_run_entry",
    "read_run_detail",
    "read_run_entry",
    "require_location",
    "require_scene",
    "status_label",
]
