from datetime import date
from pathlib import Path

import pytest
from pydantic import ValidationError

from gsdb.manifests import canonical_hash, load_model, save_yaml
from gsdb.models import CaptureManifest, LocationManifest, Region, SceneManifest, TimeSelection


def test_yaml_round_trip_preserves_unicode(tmp_path: Path) -> None:
    path = tmp_path / "location.yaml"
    model = LocationManifest(
        id="yanguan-ancient-town-20260822",
        display_name="盐官古镇",
        capture_date=date(2026, 8, 22),
        region=Region(province="浙江省", city="海宁市"),
        tags=["古建筑", "夜景"],
    )
    save_yaml(path, model)
    loaded = load_model(path, LocationManifest)
    assert loaded == model
    assert "盐官古镇" in path.read_text(encoding="utf-8")


def test_hash_is_stable_for_equivalent_data() -> None:
    first = {"b": 2, "a": {"name": "盐官"}}
    second = {"a": {"name": "盐官"}, "b": 2}
    assert canonical_hash(first) == canonical_hash(second)


def test_slug_must_be_ascii() -> None:
    with pytest.raises(ValidationError):
        LocationManifest(id="盐官古镇", display_name="盐官古镇")


def test_capture_selection_range_must_be_forward() -> None:
    with pytest.raises(ValidationError):
        TimeSelection(start_seconds=12.0, end_seconds=11.0)


def test_repository_manifests_validate() -> None:
    root = Path(__file__).resolve().parents[1]
    location = root / "locations" / "yanguan-ancient-town-20260822"
    load_model(location / "location.yaml", LocationManifest)
    scene = location / "scenes" / "night-pilot-8k"
    load_model(scene / "scene.yaml", SceneManifest)
    load_model(scene / "captures" / "capture-004-8k.yaml", CaptureManifest)
