import sqlite3
from datetime import datetime, timezone
from pathlib import Path

import pytest

from gsdb.catalog import build_catalog
from gsdb.manifests import load_yaml, save_yaml
from gsdb.models import LocationManifest, RunConfig, SceneManifest
from gsdb.runs import create_run


def test_catalog_rebuild_is_idempotent(tmp_path: Path) -> None:
    (tmp_path / "pyproject.toml").write_text("[project]\nname='fixture'\n", encoding="utf-8")
    location_path = tmp_path / "site-001"
    scene_path = location_path / "scene-001"
    (scene_path / "captures").mkdir(parents=True)
    (scene_path / "exports").mkdir()
    save_yaml(
        location_path / "location.yaml",
        LocationManifest(id="site-001", display_name="测试地点", tags=["古建筑"]),
    )
    save_yaml(
        scene_path / "scene.yaml",
        SceneManifest(id="scene-001", location_id="site-001", display_name="测试场景"),
    )
    run = create_run(
        scene_path,
        "site-001",
        "scene-001",
        RunConfig(capture_id="capture-001", input_sha256="c" * 64),
        now=datetime(2026, 8, 23, tzinfo=timezone.utc),
    )
    first = build_catalog(tmp_path)
    second = build_catalog(tmp_path)
    assert first == second == {
        "locations": 1,
        "scenes": 1,
        "captures": 0,
        "runs": 1,
        "artifacts": 0,
    }
    connection = sqlite3.connect(tmp_path / "catalog" / "catalog.sqlite")
    assert connection.execute("SELECT display_name FROM locations").fetchone()[0] == "测试地点"
    assert connection.execute("SELECT count(*) FROM search").fetchone()[0] == 2
    assert connection.execute(
        "SELECT entity_id FROM search WHERE search MATCH ?", ("古建筑",)
    ).fetchone() == ("site-001",)
    connection.close()

    payload = load_yaml(scene_path / run.id / "manifest.yaml")
    payload["config"]["input_sha256"] = "d" * 64
    save_yaml(scene_path / run.id / "manifest.yaml", payload)
    with pytest.raises(RuntimeError, match="config hash mismatch"):
        build_catalog(tmp_path)
