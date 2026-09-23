from __future__ import annotations

import os
from pathlib import Path


def find_app_root(start: Path | None = None) -> Path:
    current = (start or Path.cwd()).resolve()
    if current.is_file():
        current = current.parent
    for candidate in (current, *current.parents):
        if (candidate / "pyproject.toml").is_file():
            return candidate
    return Path(__file__).resolve().parents[3]


DATA_ROOT_ENV = "GSSTUDIO_DATA_ROOT"


def find_data_root(start: Path | None = None) -> Path:
    """Locate the Data tree: ``GSSTUDIO_DATA_ROOT`` or the repository's sibling Data."""
    configured = os.environ.get(DATA_ROOT_ENV, "").strip()
    if configured:
        return host_path(configured)
    candidate = find_app_root(start).parent / "Data"
    if not candidate.is_dir():
        raise RuntimeError(
            f"GS Studio data root is missing: {candidate}. Set {DATA_ROOT_ENV}, or create a "
            f"'Data' folder next to the APP directory ({find_app_root(start)})."
        )
    return candidate


def host_path(value: str | Path) -> Path:
    return Path(value).expanduser()


def _normalized(path: Path) -> Path:
    """Absolute path with ``..`` removed but symlinks left intact.

    ``Path.resolve`` would follow a symlink out of its containing directory and
    reject every path under it, so containment is checked lexically instead.
    Traversal escapes are still blocked because ``normpath`` collapses ``..``
    before the comparison.
    """
    return Path(os.path.normpath(os.path.abspath(path)))


def ensure_within(path: Path, parent: Path) -> Path:
    normalized = _normalized(path)
    base = _normalized(parent)
    try:
        normalized.relative_to(base)
    except ValueError as error:
        raise ValueError(f"Path must stay within {base}: {normalized}") from error
    return normalized


def ensure_run_dir(scene_path: Path, run_id: str, exist_ok: bool = True) -> Path:
    """Return ``<scene>/<run-id>``, the single directory holding a run's manifest and scratch."""
    run_dir = scene_path / run_id
    run_dir.mkdir(parents=True, exist_ok=exist_ok)
    return run_dir


def location_dir(root: Path, location_id: str) -> Path:
    return root / location_id


def scene_dir(root: Path, location_id: str, scene_id: str) -> Path:
    return location_dir(root, location_id) / scene_id
