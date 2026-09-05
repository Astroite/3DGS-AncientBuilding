from __future__ import annotations

import os
import re
from pathlib import Path, PurePosixPath, PureWindowsPath


WINDOWS_DRIVE = re.compile(r"^(?P<drive>[A-Za-z]):[\\/](?P<rest>.*)$")


def find_app_root(start: Path | None = None) -> Path:
    current = (start or Path.cwd()).resolve()
    if current.is_file():
        current = current.parent
    for candidate in (current, *current.parents):
        if (candidate / "pyproject.toml").is_file():
            return candidate
    return Path(__file__).resolve().parents[2]


DATA_ROOT_ENV = "GSDB_DATA_ROOT"


def find_data_root(start: Path | None = None) -> Path:
    """Locate the Data tree: ``GSDB_DATA_ROOT`` if set, else a ``Data`` sibling of APP."""
    configured = os.environ.get(DATA_ROOT_ENV, "").strip()
    if configured:
        return host_path(configured)
    candidate = find_app_root(start).parent / "Data"
    if not candidate.is_dir():
        raise RuntimeError(
            f"gsdb data root is missing: {candidate}. Set {DATA_ROOT_ENV}, or create a "
            f"'Data' folder next to the APP directory ({find_app_root(start)})."
        )
    return candidate


def windows_to_wsl(value: str) -> str:
    match = WINDOWS_DRIVE.match(value)
    if not match:
        return value.replace("\\", "/")
    drive = match.group("drive").lower()
    rest = match.group("rest").replace("\\", "/")
    return str(PurePosixPath("/mnt", drive, rest))


def wsl_to_windows(value: str) -> str:
    path = PurePosixPath(value)
    parts = path.parts
    if len(parts) >= 4 and parts[0] == "/" and parts[1] == "mnt" and len(parts[2]) == 1:
        return str(PureWindowsPath(parts[2].upper() + ":\\", *parts[3:]))
    return value


def host_path(value: str | Path) -> Path:
    text = str(value)
    if os.name != "nt" and WINDOWS_DRIVE.match(text):
        text = windows_to_wsl(text)
    return Path(text).expanduser()


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
