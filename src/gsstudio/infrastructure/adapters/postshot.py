"""Postshot CLI discovery and the training command line.

Dataset packaging lives in ``training_data``; dispatch in ``training``. What is
left here is the external program boundary: locating the CLI, reading its
version, and building one training invocation.
"""
from __future__ import annotations

import os
import re
import shutil
import subprocess
from pathlib import Path

from gsstudio.infrastructure.paths import host_path

POSTSHOT_MINIMUM_VERSION = (1, 1, 69)
POSTSHOT_CLI_ENV = "GSSTUDIO_POSTSHOT_CLI"


def postshot_executable() -> Path:
    configured = os.environ.get(POSTSHOT_CLI_ENV, "").strip()
    candidates = []
    if configured:
        candidates.append(host_path(configured))
    if os.name == "nt":
        candidates.append(Path(r"C:\Program Files\Jawset Postshot\bin\postshot-cli.exe"))
    found = shutil.which("postshot-cli") or shutil.which("postshot-cli.exe")
    if found:
        candidates.append(Path(found))
    for candidate in candidates:
        if candidate.is_file():
            return candidate
    raise RuntimeError(
        "Postshot CLI is unavailable; install Postshot or set GSSTUDIO_POSTSHOT_CLI"
    )


def postshot_version(executable: Path | None = None) -> tuple[tuple[int, int, int], str]:
    executable = executable or postshot_executable()
    result = subprocess.run(
        [str(executable), "--help"],
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        timeout=30,
    )
    output = "\n".join((result.stdout, result.stderr)).strip()
    match = re.search(r"Postshot v(\d+)\.(\d+)\.(\d+)", output)
    if result.returncode != 0 or match is None:
        raise RuntimeError(f"Unable to determine Postshot version: {output[:300]}")
    version = tuple(int(value) for value in match.groups())
    return version, ".".join(str(value) for value in version)


def _windows_argument(path: Path) -> str:
    absolute = str(path.absolute().resolve(strict=False))
    if os.name != "nt":
        raise RuntimeError(f"Postshot runs on Windows; got {absolute}")
    if absolute.startswith("\\\\"):
        raise RuntimeError(f"Postshot paths may not use UNC storage: {absolute}")
    return absolute


def build_postshot_train_command(
    executable: Path,
    dataset: Path,
    output: Path,
    *,
    profile: str = "Splat ADC",
    ksteps: int | None = None,
    max_splats: int | None = None,
    gpu: int = 0,
    store_training_context: bool = True,
    export_ply: Path | None = None,
    export_spz: Path | None = None,
) -> list[str]:
    if profile not in {"Splat ADC", "Splat MCMC", "Splat3"}:
        raise ValueError(f"Unsupported Postshot profile: {profile}")
    if output.suffix.casefold() != ".psht":
        raise ValueError("Postshot --output must use a .psht project path")
    if export_ply is not None and export_spz is not None:
        raise ValueError("Postshot accepts only one --export-splat target per training command")
    if export_ply is not None and export_ply.suffix.casefold() != ".ply":
        raise ValueError("--export-ply must use a .ply output path")
    if export_spz is not None and export_spz.suffix.casefold() != ".spz":
        raise ValueError("--export-spz must use a .spz output path")
    command = [
        str(executable),
        "train",
        "--import",
        _windows_argument(dataset / "images"),
        _windows_argument(dataset / "colmap"),
        "--import-masks",
        _windows_argument(dataset / "masks"),
        "--profile",
        profile,
        "--image-select",
        "all",
        "--max-image-size",
        "0",
        "--mask-mode",
        "occluders",
        "--gpu",
        str(gpu),
        "--output",
        _windows_argument(output),
    ]
    if store_training_context:
        command.append("--store-training-context")
    if ksteps is not None:
        if ksteps < 1:
            raise ValueError("Postshot kSteps must be positive")
        command.extend(["--train-steps-limit", str(ksteps)])
    if max_splats is not None:
        if max_splats < 1:
            raise ValueError("Postshot maximum splats must be positive")
        command.extend(["--max-num-splats", str(max_splats)])
    for target in (export_ply, export_spz):
        if target is not None:
            command.extend(["--export-splat", _windows_argument(target)])
    return command
