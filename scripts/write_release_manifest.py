"""Write a complete, content-addressed inventory for an assembled Windows release."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import tempfile
from datetime import datetime, timezone
from pathlib import Path


def digest(path: Path) -> str:
    sha = hashlib.sha256()
    with path.open("rb") as stream:
        while block := stream.read(8 * 1024 * 1024):
            sha.update(block)
    return sha.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("root", type=Path)
    parser.add_argument("--version", required=True)
    parser.add_argument("--commit", required=True)
    parser.add_argument("--source-state", choices=("clean", "modified"), required=True)
    parser.add_argument("--gpu-compute-capability", required=True)
    args = parser.parse_args()
    root = args.root.resolve(strict=True)
    required = (
        "GSStudio.exe", "worker/GSStudioWorker.exe",
        "gpu-runtime/GSStudioTrainer.exe", "cli/GSStudioCLI.exe",
        "_internal/gsplat/csrc.pyd", "_internal/gsplat/csrc-build.json",
        "gpu-runtime/_internal/gsplat/csrc.pyd",
        "gpu-runtime/_internal/gsplat/csrc-build.json",
    )
    for relative in required:
        if not (root / relative).is_file():
            raise FileNotFoundError(f"Required release component is missing: {relative}")
    for package in (root / "_internal" / "gsplat",
                    root / "gpu-runtime" / "_internal" / "gsplat"):
        record = json.loads((package / "csrc-build.json").read_text(encoding="utf-8"))
        extension = package / "csrc.pyd"
        if (record.get("compute_capability") != args.gpu_compute_capability or
                record.get("extension_sha256") != digest(extension)):
            raise RuntimeError(f"Release gsplat extension does not match target GPU: {extension}")
    files = []
    for path in sorted(root.rglob("*")):
        if not path.is_file() or path.name == "release-manifest.json":
            continue
        files.append({"path": path.relative_to(root).as_posix(),
                      "bytes": path.stat().st_size, "sha256": digest(path)})
    manifest = {
        "schema_version": 1,
        "product": "GS Studio",
        "version": args.version,
        "git_commit": args.commit,
        "source_state": args.source_state,
        "built_at_utc": datetime.now(timezone.utc).replace(microsecond=0).isoformat(),
        "target": "windows-x64-cuda13",
        "gpu_compute_capability": args.gpu_compute_capability,
        "files": files,
    }
    descriptor, temporary = tempfile.mkstemp(prefix=".release-manifest-", suffix=".tmp", dir=root)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="\n") as stream:
            json.dump(manifest, stream, ensure_ascii=False, indent=2)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, root / "release-manifest.json")
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)
    print(f"Release {args.version}: {len(files)} files inventoried in {root}")


if __name__ == "__main__":
    main()
