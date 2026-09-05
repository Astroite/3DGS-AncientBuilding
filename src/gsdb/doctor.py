from __future__ import annotations

import importlib.metadata
import json
import os
import platform
import shutil
import sqlite3
import subprocess
import tempfile
from pathlib import Path
from typing import Any


def _version(command: list[str]) -> tuple[bool, str]:
    executable = shutil.which(command[0])
    if executable is None:
        return False, "missing"
    try:
        result = subprocess.run(
            command,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=30,
        )
        text = (result.stdout or result.stderr).strip().splitlines()
        return result.returncode == 0, text[0] if text else executable
    except Exception as error:  # diagnostic boundary
        return False, str(error)


def collect_tool_versions() -> dict[str, str]:
    versions = {"python": platform.python_version()}
    for name, command in {
        "ffmpeg": ["ffmpeg", "-version"],
        "colmap": ["colmap", "-h"],
    }.items():
        ok, detail = _version(command)
        versions[name] = detail if ok else "unavailable"
    colmap_cuda_ok, _ = _colmap_cuda_build()
    versions["colmap_cuda"] = "enabled" if colmap_cuda_ok else "unavailable"
    for distribution in ("gsplat", "nerfstudio", "torchvision"):
        try:
            versions[distribution] = importlib.metadata.version(distribution)
        except importlib.metadata.PackageNotFoundError:
            versions[distribution] = "unavailable"
    try:
        direct_url = importlib.metadata.distribution("nerfstudio").read_text("direct_url.json")
        if direct_url:
            commit = json.loads(direct_url).get("vcs_info", {}).get("commit_id")
            if commit:
                versions["nerfstudio_commit"] = commit
    except (importlib.metadata.PackageNotFoundError, json.JSONDecodeError):
        pass
    try:
        import torch

        versions["torch"] = str(torch.__version__)
        versions["torch_cuda"] = str(torch.version.cuda)
    except Exception:
        versions["torch"] = "unavailable"
    return versions


def _colmap_cuda_build() -> tuple[bool, str]:
    executable = shutil.which("colmap")
    if executable is None:
        return False, "missing"
    try:
        result = subprocess.run(
            ["colmap", "-h"],
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=30,
        )
        output = "\n".join((result.stdout, result.stderr)).strip()
        first_line = output.splitlines()[0] if output else executable
        cuda_enabled = " with CUDA" in output and " without CUDA" not in output
        return result.returncode == 0 and cuda_enabled, first_line
    except Exception as error:
        return False, str(error)


def _colmap_gpu_sift_smoke(project_root: Path, gpu_index: int = 0) -> tuple[bool, str]:
    try:
        import cv2
        import numpy as np

        with tempfile.TemporaryDirectory(dir=project_root, prefix=".gsdb-colmap-gpu-") as name:
            root = Path(name)
            images = root / "images" / "view_00"
            images.mkdir(parents=True)
            generator = np.random.default_rng(42)
            texture = generator.integers(0, 256, (256, 256), dtype=np.uint8)
            image = cv2.cvtColor(texture, cv2.COLOR_GRAY2BGR)
            cv2.circle(image, (128, 128), 64, (255, 255, 255), 3)
            if not cv2.imwrite(str(images / "frame_000001.jpg"), image):
                raise RuntimeError("Failed to write COLMAP smoke image")
            database = root / "database.db"
            result = subprocess.run(
                [
                    "colmap",
                    "feature_extractor",
                    "--database_path",
                    str(database),
                    "--image_path",
                    str(root / "images"),
                    "--ImageReader.camera_model",
                    "PINHOLE",
                    "--ImageReader.single_camera_per_folder",
                    "1",
                    "--ImageReader.camera_params",
                    "128,128,128,128",
                    "--SiftExtraction.use_gpu",
                    "1",
                    "--SiftExtraction.gpu_index",
                    str(gpu_index),
                    "--SiftExtraction.max_num_features",
                    "512",
                ],
                capture_output=True,
                text=True,
                encoding="utf-8",
                errors="replace",
                timeout=120,
            )
            if result.returncode != 0:
                detail = (result.stderr or result.stdout).strip().splitlines()
                return False, detail[-1] if detail else "feature_extractor failed"
            with sqlite3.connect(database) as connection:
                rows = connection.execute("SELECT rows FROM keypoints").fetchall()
            keypoints = sum(int(row[0]) for row in rows)
            return keypoints > 0, f"CUDA SIFT extracted {keypoints} keypoints on GPU {gpu_index}"
    except Exception as error:
        return False, str(error)


def _wsl_memory_check(minimum_gib: float = 28.0) -> tuple[bool, str]:
    version_path = Path("/proc/version")
    if not version_path.is_file():
        return True, "not running under WSL; WSL memory gate not applicable"
    try:
        version = version_path.read_text(encoding="utf-8", errors="replace")
        if "microsoft" not in version.lower():
            return True, "not running under WSL; WSL memory gate not applicable"
        meminfo = Path("/proc/meminfo").read_text(encoding="utf-8")
        line = next(item for item in meminfo.splitlines() if item.startswith("MemTotal:"))
        total_gib = int(line.split()[1]) * 1024 / 1024**3
        return (
            total_gib >= minimum_gib,
            f"WSL memory {total_gib:.1f} GiB; minimum {minimum_gib:.1f} GiB",
        )
    except Exception as error:
        return False, str(error)


def run_doctor(
    project_root: Path,
    minimum_free_gib: float = 20.0,
    require: set[str] | None = None,
) -> dict[str, Any]:
    required = set(require or ())
    unknown = required - {"mediasdk", "postshot"}
    if unknown:
        raise ValueError(f"Unknown optional doctor requirement(s): {sorted(unknown)}")
    checks: dict[str, Any] = {}
    checks["ffmpeg"] = _version(["ffmpeg", "-version"])
    checks["ffprobe"] = _version(["ffprobe", "-version"])
    checks["colmap_cuda_build"] = _colmap_cuda_build()
    checks["colmap_gpu_sift"] = (
        _colmap_gpu_sift_smoke(project_root)
        if checks["colmap_cuda_build"][0]
        else (False, "CUDA COLMAP build is required before the GPU SIFT smoke test")
    )
    checks["wsl_memory"] = _wsl_memory_check()

    deepseek_key_present = bool(os.environ.get("DEEPSEEK_API_KEY"))
    if deepseek_key_present:
        checks["deepseek_environment"] = (
            True,
            "DEEPSEEK_API_KEY is present; value was not inspected or logged",
        )

    try:
        import torch

        cuda_ok = torch.cuda.is_available()
        detail = f"torch={torch.__version__}, cuda={torch.version.cuda}, available={cuda_ok}"
        if cuda_ok:
            value = torch.randn(32, 32, device="cuda", requires_grad=True)
            (value.square().mean()).backward()
            detail += f", gpu={torch.cuda.get_device_name(0)}"
        checks["torch_cuda"] = (cuda_ok, detail)
    except Exception as error:
        checks["torch_cuda"] = (False, str(error))

    try:
        from torchvision.models.detection import MaskRCNN_ResNet50_FPN_V2_Weights

        weights = MaskRCNN_ResNet50_FPN_V2_Weights.DEFAULT
        checks["person_segmenter"] = (
            True,
            f"torchvision Mask R-CNN available; weights={weights.name} "
            "(download/cache is deferred until gsdb mask)",
        )
    except Exception as error:
        checks["person_segmenter"] = (False, str(error))

    try:
        with tempfile.NamedTemporaryFile(dir=project_root, prefix=".gsdb-doctor-", delete=True) as stream:
            stream.write(b"ok")
            stream.flush()
        checks["project_writable"] = (True, str(project_root))
    except Exception as error:
        checks["project_writable"] = (False, str(error))

    usage = shutil.disk_usage(project_root)
    free_gib = usage.free / 1024**3
    checks["disk_free"] = (
        free_gib >= minimum_free_gib,
        f"{free_gib:.1f} GiB free; minimum reserve {minimum_free_gib:.1f} GiB",
    )
    try:
        from .sources import media_capabilities

        with tempfile.TemporaryDirectory(
            dir=project_root, prefix=".gsdb-mediasdk-doctor-"
        ) as name:
            capabilities = media_capabilities(Path(name))
        checks["mediasdk"] = (
            True,
            f"helper={capabilities.get('helper_version', 'unknown')}, "
            f"sdk={capabilities.get('sdk_version', 'unknown')}",
        )
    except Exception as error:
        checks["mediasdk"] = (False, f"optional unavailable: {error}")
    try:
        from .postshot import POSTSHOT_MINIMUM_VERSION, postshot_version

        version_tuple, version = postshot_version()
        checks["postshot"] = (
            version_tuple >= POSTSHOT_MINIMUM_VERSION,
            (
                f"Postshot {version}"
                if version_tuple >= POSTSHOT_MINIMUM_VERSION
                else f"Postshot {version} is older than required 1.1.69"
            ),
        )
    except Exception as error:
        checks["postshot"] = (False, f"optional unavailable: {error}")

    checks["ok"] = all(
        result[0]
        for key, result in checks.items()
        if key != "ok" and (key not in {"mediasdk", "postshot"} or key in required)
    )
    return checks
