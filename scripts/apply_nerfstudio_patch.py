from __future__ import annotations

import hashlib
import importlib.util
from pathlib import Path


EQUIRECT_NEEDLE = (
    b'return torch.nn.functional.grid_sample(img, grid, mode="bicubic", padding_mode="zeros")'
)
EQUIRECT_REPLACEMENT = EQUIRECT_NEEDLE + b".clamp(0.0, 1.0)"

NESTED_DOWNSCALE_NEEDLE = (
    b'            return data_dir / f"{downsample_folder_prefix}{self.downscale_factor}" '
    b"/ filepath.name"
)
NESTED_DOWNSCALE_REPLACEMENT = (
    b"            relative = Path(*filepath.parts[1:])\n"
    b'            return data_dir / f"{downsample_folder_prefix}{self.downscale_factor}" '
    b"/ relative"
)


def patch_module(
    module: str, needle: bytes, replacement: bytes, description: str
) -> None:
    spec = importlib.util.find_spec(module)
    if spec is None or spec.origin is None:
        raise SystemExit(f"{module} was not found")
    target = Path(spec.origin)
    original = target.read_bytes()
    if replacement in original:
        print(f"Nerfstudio {description} already applied: {target}")
        return
    if original.count(needle) != 1:
        digest = hashlib.sha256(original).hexdigest()
        raise SystemExit(
            f"Refusing to patch unexpected nerfstudio source {target}; sha256={digest}"
        )
    target.write_bytes(original.replace(needle, replacement))
    print(f"Applied Nerfstudio {description}: {target}")


def main() -> None:
    patch_module(
        "nerfstudio.process_data.equirect_utils",
        EQUIRECT_NEEDLE,
        EQUIRECT_REPLACEMENT,
        "projection clamp",
    )
    patch_module(
        "nerfstudio.data.dataparsers.nerfstudio_dataparser",
        NESTED_DOWNSCALE_NEEDLE,
        NESTED_DOWNSCALE_REPLACEMENT,
        "nested downscale paths",
    )


if __name__ == "__main__":
    main()
