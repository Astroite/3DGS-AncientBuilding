from __future__ import annotations

import os
import re
from pathlib import Path, PurePosixPath, PureWindowsPath


WINDOWS_DRIVE = re.compile(r"^(?P<drive>[A-Za-z]):[\\/](?P<rest>.*)$")


def find_project_root(start: Path | None = None) -> Path:
    current = (start or Path.cwd()).resolve()
    if current.is_file():
        current = current.parent
    for candidate in (current, *current.parents):
        if (candidate / "pyproject.toml").is_file() and (candidate / "locations").is_dir():
            return candidate
    package_root = Path(__file__).resolve().parents[2]
    if (package_root / "pyproject.toml").is_file():
        return package_root
    raise RuntimeError("Not inside a gsdb project (pyproject.toml and locations/ are required)")


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


def ensure_within(path: Path, parent: Path) -> Path:
    resolved = path.resolve()
    base = parent.resolve()
    try:
        resolved.relative_to(base)
    except ValueError as error:
        raise ValueError(f"Path must stay within {base}: {resolved}") from error
    return resolved


def location_dir(root: Path, location_id: str) -> Path:
    return root / "locations" / location_id


def scene_dir(root: Path, location_id: str, scene_id: str) -> Path:
    return location_dir(root, location_id) / "scenes" / scene_id
