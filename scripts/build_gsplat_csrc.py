"""Build gsplat's CUDA extension into the installed gsplat package.

gsplat 1.5.3 has no cu130 wheel, so ``pip install gsplat==1.5.3`` installs the
pure-Python package together with its CUDA sources and no compiled ``csrc``.
This script compiles those sources into ``gsplat/csrc.pyd``, the artifact
``from gsplat import csrc`` loads -- the same thing the official platform wheels
ship. It requires a CUDA 13 toolkit and an MSVC build environment, and it is
idempotent for the selected GPU architecture, and a successful build is cached
under ``wheels/`` so a recreated virtualenv does not recompile.

Run by ``scripts/bootstrap-gsplat-windows.ps1`` after the gsplat package is
installed. See docs/MAINTENANCE.md for the version table and fallback ladder.
"""
from __future__ import annotations

import glob
import hashlib
import json
import os
import re
import shutil
import sys
import tempfile
from importlib.util import find_spec
from pathlib import Path

from torch.utils.cpp_extension import load


def package_root() -> Path:
    spec = find_spec("gsplat")
    if spec is None or not spec.submodule_search_locations:
        raise SystemExit("gsplat is not installed in this interpreter")
    return Path(list(spec.submodule_search_locations)[0])


def _fix_msvc_const_mangling(csrc: Path) -> None:
    """Align explicit-instantiation parameter cv with Rasterization.h.

    MSVC encodes top-level const of by-value parameters in decorated names
    (gcc/clang ignore it), so two upstream ``__INS__`` macros that declare the
    non-const output parameters as ``const at::Tensor`` emit instantiations
    mangled differently from the header declarations Rasterization.cpp calls,
    and the link fails with LNK2019 (38 instantiations across two kernels).
    Idempotent. If a future gsplat upgrade brings the LNK2019 back with the same
    demangled signatures, check here first.
    """
    fixes = {
        "RasterizeToPixels2DGSBwd.cu": [
            ("const at::Tensor v_means2d,", "at::Tensor v_means2d,"),
            ("const at::Tensor v_ray_transforms,", "at::Tensor v_ray_transforms,"),
            ("const at::Tensor v_colors,", "at::Tensor v_colors,"),
            ("const at::Tensor v_opacities,", "at::Tensor v_opacities,"),
            ("const at::Tensor v_normals,", "at::Tensor v_normals,"),
            ("const at::Tensor v_densify", "at::Tensor v_densify"),
        ],
        "RasterizeToPixelsFromWorld3DGSFwd.cu": [
            ("const at::Tensor renders,", "at::Tensor renders,"),
            ("const at::Tensor alphas,", "at::Tensor alphas,"),
            ("const at::Tensor last_ids", "at::Tensor last_ids"),
        ],
    }
    for filename, pairs in fixes.items():
        path = csrc / filename
        text = path.read_text(encoding="utf-8")
        for old, new in pairs:
            text = text.replace(old, new)
        path.write_text(text, encoding="utf-8")


def _architecture() -> str:
    arch = (os.environ.get("GSSTUDIO_CUDA_ARCH_LIST") or
            os.environ.get("TORCH_CUDA_ARCH_LIST") or "").strip()
    if not re.fullmatch(r"\d+\.\d+", arch):
        raise SystemExit("Set GSSTUDIO_CUDA_ARCH_LIST to the target GPU compute capability, e.g. 8.9 or 12.0")
    os.environ["TORCH_CUDA_ARCH_LIST"] = arch
    return arch


def _cache_file(app_root: Path, arch: str) -> Path:
    """Cache name pins Python, torch and GPU architecture."""
    version = f"py{sys.version_info[0]}{sys.version_info[1]}-torch{_torch_tag()}"
    return app_root / "wheels" / f"gsplat-csrc-{version}-sm{arch.replace('.', '')}.pyd"


def _torch_tag() -> str:
    import torch

    return torch.__version__.replace("+", "_").replace(".", "")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while block := stream.read(8 * 1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def _build_identity(arch: str, extension: Path) -> dict:
    import torch

    return {
        "schema_version": 1,
        "compute_capability": arch,
        "python": f"{sys.version_info[0]}.{sys.version_info[1]}",
        "torch": torch.__version__,
        "extension_sha256": _sha256(extension),
    }


def _write_build_identity(metadata: Path, arch: str, extension: Path) -> None:
    temporary = metadata.with_suffix(".tmp")
    temporary.write_text(json.dumps(_build_identity(arch, extension), indent=2) + "\n", encoding="utf-8")
    os.replace(temporary, metadata)


def main() -> None:
    arch = _architecture()
    gsplat_root = package_root()
    app_root = Path(__file__).resolve().parents[1]
    target = gsplat_root / "csrc.pyd"
    metadata = gsplat_root / "csrc-build.json"
    if target.is_file() and metadata.is_file():
        try:
            recorded = json.loads(metadata.read_text(encoding="utf-8"))
            if recorded == _build_identity(arch, target):
                print(f"gsplat CUDA extension already present for sm_{arch.replace('.', '')}: {target}")
                return
        except (OSError, ValueError):
            pass

    cached = _cache_file(app_root, arch)
    if cached.is_file():
        shutil.copy2(cached, target)
        _write_build_identity(metadata, arch, target)
        print(f"gsplat CUDA extension restored from cache: {cached}")
        return

    cuda_root = gsplat_root / "cuda"
    csrc = cuda_root / "csrc"
    _fix_msvc_const_mangling(csrc)
    sources = (
        sorted(glob.glob(str(csrc / "*.cu")))
        + sorted(glob.glob(str(csrc / "*.cpp")))
        + [str(cuda_root / "ext.cpp")]
    )
    if len(sources) < 2:
        raise SystemExit(f"No CUDA sources found under {csrc}")
    build_dir = Path(
        os.environ.get(
            "GSSTUDIO_GSPLAT_BUILD_DIR", str(Path(tempfile.gettempdir()) / f"gsstudio-gsplat-csrc-sm{arch.replace('.', '')}")
        )
    ).resolve()
    build_dir.mkdir(parents=True, exist_ok=True)
    # CUDA 13 bundles CCCL, which rejects MSVC's traditional preprocessor; nvcc
    # does not add the flag itself for this MSVC version, so pass it through.
    load(
        name="csrc",
        sources=sources,
        extra_include_paths=[str(cuda_root / "include"), str(csrc / "third_party" / "glm")],
        extra_cflags=["-O3", "/Zc:preprocessor"],
        extra_cuda_cflags=["-O3", "-use_fast_math", "-Xcompiler=/Zc:preprocessor"],
        build_directory=str(build_dir),
        verbose=True,
    )
    for candidate in (build_dir / "csrc.pyd", build_dir / "csrc.dll"):
        if candidate.is_file():
            shutil.copy2(candidate, target)
            cached.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(candidate, cached)
            _write_build_identity(metadata, arch, target)
            print(f"gsplat CUDA extension installed: {target} (cached at {cached})")
            return
    raise SystemExit(f"No compiled extension found in {build_dir}")


if __name__ == "__main__":
    main()
