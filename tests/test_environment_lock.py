from pathlib import Path

import yaml


def test_environment_lock_contains_fixed_binary_and_training_stack() -> None:
    root = Path(__file__).resolve().parents[1]
    lock = yaml.safe_load((root / "conda-lock.yml").read_text(encoding="utf-8"))
    packages = {item["name"]: item for item in lock["package"]}

    expected = {
        "python": "3.10.12",
        "pytorch": "2.1.2",
        "pytorch-cuda": "11.8",
        "torchvision": "0.16.2",
        "ffmpeg": "6.1.2",
        "colmap": "3.8",
        "gsplat": "1.4.0",
        "nerfstudio": "1.1.5",
    }
    assert {name: packages[name]["version"] for name in expected} == expected
    assert packages["nerfstudio"]["source"]["url"].endswith(
        "@758ea1918e082aa44776009d8e755c2f3a88d2ee"
    )
