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


def _normalized(path: Path) -> Path:
    """Absolute path with ``..`` removed but symlinks left intact.

    ``Path.resolve`` would follow the ``work/<run-id>`` scratch symlink out of the
    scene directory and reject every path under it, so containment is checked
    lexically instead. Traversal escapes are still blocked because ``normpath``
    collapses ``..`` before the comparison.
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


SCRATCH_ROOT_ENV = "GSDB_SCRATCH_ROOT"


def windows_host() -> bool:
    """Whether the CLI is running on Windows rather than inside the WSL environment."""
    return os.name == "nt"


def scratch_root() -> Path | None:
    """Filesystem that should hold run scratch data, or ``None`` to stay in-project.

    WSL2 reaches the Windows project through a 9p mount where every image read
    costs about 50 ms against 13 ms on the VM's own ext4 disk. Pointing this at an
    ext4 path keeps the multi-gigabyte intermediate sets off 9p while the project
    tree still owns the manifests and published artifacts.
    """
    value = os.environ.get(SCRATCH_ROOT_ENV, "").strip()
    if not value:
        return None
    if windows_host():
        raise RuntimeError(
            f"{SCRATCH_ROOT_ENV} redirects run scratch data through a symlink and is "
            "only supported inside the WSL environment"
        )
    return host_path(value)


def ensure_work_dir(scene_path: Path, run_id: str, exist_ok: bool = True) -> Path:
    """Return ``<scene>/work/<run-id>``, backed by the scratch filesystem when configured.

    The returned path always stays inside the scene so manifests keep storing
    scene-relative locations; only the bytes move.
    """
    work = scene_path / "work" / run_id
    root = scratch_root()
    if root is None:
        work.mkdir(parents=True, exist_ok=exist_ok)
        return work
    target = root / scene_path.parent.parent.name / scene_path.name / run_id
    if not exist_ok and (work.is_symlink() or work.exists() or target.exists()):
        raise FileExistsError(f"Run work directory already exists: {work} -> {target}")
    if work.is_symlink():
        current = Path(os.path.realpath(work))
        if current != Path(os.path.realpath(target)):
            raise RuntimeError(
                f"Run scratch symlink {work} already points at {current}; refusing to "
                f"repoint it at {target}"
            )
        target.mkdir(parents=True, exist_ok=True)
        return work
    if work.is_dir():
        # A run started before the redirect existed keeps its in-project data; only
        # a second, conflicting copy on the scratch filesystem is an error.
        if target.is_dir() and any(target.iterdir()):
            raise RuntimeError(
                f"{work} holds an in-project run while {target} holds scratch data for the "
                "same run ID; keep one and remove the other"
            )
        print(
            f"Work directory: reusing in-project {work} (predates {SCRATCH_ROOT_ENV})",
            flush=True,
        )
        return work
    target.mkdir(parents=True, exist_ok=True)
    work.parent.mkdir(parents=True, exist_ok=True)
    os.symlink(target, work, target_is_directory=True)
    return work


def location_dir(root: Path, location_id: str) -> Path:
    return root / "locations" / location_id


def scene_dir(root: Path, location_id: str, scene_id: str) -> Path:
    return location_dir(root, location_id) / "scenes" / scene_id
