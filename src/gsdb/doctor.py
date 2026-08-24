from __future__ import annotations

import importlib.metadata
import json
import os
import platform
import shutil
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


def run_doctor(project_root: Path, minimum_free_gib: float = 20.0) -> dict[str, Any]:
    checks: dict[str, Any] = {}
    checks["ffmpeg"] = _version(["ffmpeg", "-version"])
    checks["ffprobe"] = _version(["ffprobe", "-version"])
    checks["colmap"] = _version(["colmap", "-h"])
    checks["ns_process_data"] = _version(["ns-process-data", "--help"])
    checks["ns_train"] = _version(["ns-train", "--help"])

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
        import gsplat
        import torch

        device = "cuda"
        means = torch.tensor(
            [[0.0, 0.0, 2.0], [0.15, 0.0, 2.0]],
            device=device,
            requires_grad=True,
        )
        quats = torch.tensor(
            [[1.0, 0.0, 0.0, 0.0], [1.0, 0.0, 0.0, 0.0]], device=device
        )
        scales = torch.full((2, 3), 0.1, device=device)
        opacities = torch.full((2,), 0.8, device=device)
        colors = torch.tensor(
            [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]],
            device=device,
            requires_grad=True,
        )
        viewmats = torch.eye(4, device=device)[None]
        intrinsics = torch.tensor(
            [[[20.0, 0.0, 8.0], [0.0, 20.0, 8.0], [0.0, 0.0, 1.0]]],
            device=device,
        )
        render, alpha, _ = gsplat.rasterization(
            means,
            quats,
            scales,
            opacities,
            colors,
            viewmats,
            intrinsics,
            width=16,
            height=16,
        )
        (render.mean() + alpha.mean()).backward()
        gradient_ok = means.grad is not None and colors.grad is not None
        checks["gsplat"] = (
            gradient_ok,
            f"{getattr(gsplat, '__version__', 'unknown')}; rasterization/backward ok",
        )
    except Exception as error:
        checks["gsplat"] = (False, str(error))

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
    checks["ok"] = all(result[0] for key, result in checks.items() if key != "ok")
    return checks
