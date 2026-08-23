from __future__ import annotations

import hashlib
import importlib.util
from pathlib import Path


NEEDLE = b'return torch.nn.functional.grid_sample(img, grid, mode="bicubic", padding_mode="zeros")'
REPLACEMENT = NEEDLE + b".clamp(0.0, 1.0)"


def main() -> None:
    spec = importlib.util.find_spec("nerfstudio.process_data.equirect_utils")
    if spec is None or spec.origin is None:
        raise SystemExit("nerfstudio equirect_utils.py was not found")
    target = Path(spec.origin)
    original = target.read_bytes()
    if REPLACEMENT in original:
        print(f"Nerfstudio projection clamp already applied: {target}")
        return
    if original.count(NEEDLE) != 1:
        digest = hashlib.sha256(original).hexdigest()
        raise SystemExit(
            f"Refusing to patch unexpected nerfstudio source {target}; sha256={digest}"
        )
    target.write_bytes(original.replace(NEEDLE, REPLACEMENT))
    print(f"Applied Nerfstudio projection clamp: {target}")


if __name__ == "__main__":
    main()
