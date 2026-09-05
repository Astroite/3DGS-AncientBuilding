from __future__ import annotations

import json
import os
import sqlite3
from pathlib import Path

from .manifests import load_capture_manifest, load_model
from .models import (
    ArtifactManifest,
    CaptureManifest,
    CaptureManifestV2,
    LocationManifest,
    SceneManifest,
)
from .runs import load_run


SCHEMA = """
PRAGMA foreign_keys = ON;
CREATE TABLE locations (
    id TEXT PRIMARY KEY,
    display_name TEXT NOT NULL,
    status TEXT NOT NULL,
    capture_date TEXT,
    region_json TEXT NOT NULL,
    rights_json TEXT NOT NULL,
    notes TEXT,
    manifest_path TEXT NOT NULL
);
CREATE TABLE scenes (
    id TEXT NOT NULL,
    location_id TEXT NOT NULL REFERENCES locations(id),
    display_name TEXT NOT NULL,
    status TEXT NOT NULL,
    quality_target TEXT NOT NULL,
    metric_scale INTEGER NOT NULL,
    description TEXT,
    manifest_path TEXT NOT NULL,
    PRIMARY KEY (location_id, id)
);
CREATE TABLE captures (
    id TEXT NOT NULL,
    location_id TEXT NOT NULL,
    scene_id TEXT NOT NULL,
    status TEXT NOT NULL,
    source_kind TEXT NOT NULL,
    primary_source_path TEXT NOT NULL,
    prepared_relative_path TEXT,
    media_json TEXT NOT NULL,
    manifest_path TEXT NOT NULL,
    PRIMARY KEY (location_id, scene_id, id),
    FOREIGN KEY (location_id, scene_id) REFERENCES scenes(location_id, id)
);
CREATE TABLE runs (
    id TEXT PRIMARY KEY,
    location_id TEXT NOT NULL,
    scene_id TEXT NOT NULL,
    status TEXT NOT NULL,
    config_hash TEXT NOT NULL,
    selected_dataset TEXT,
    metrics_json TEXT NOT NULL,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL,
    manifest_path TEXT NOT NULL,
    FOREIGN KEY (location_id, scene_id) REFERENCES scenes(location_id, id)
);
CREATE TABLE artifacts (
    run_id TEXT NOT NULL REFERENCES runs(id),
    kind TEXT NOT NULL,
    version TEXT NOT NULL,
    relative_path TEXT NOT NULL,
    sha256 TEXT NOT NULL,
    byte_size INTEGER NOT NULL,
    PRIMARY KEY (run_id, kind, version, relative_path)
);
CREATE TABLE tags (
    location_id TEXT NOT NULL REFERENCES locations(id),
    tag TEXT NOT NULL,
    PRIMARY KEY (location_id, tag)
);
CREATE INDEX idx_scenes_status ON scenes(status);
CREATE INDEX idx_runs_status ON runs(status);
CREATE INDEX idx_artifacts_kind ON artifacts(kind);
CREATE VIRTUAL TABLE search USING fts5(entity_type, entity_id, display_name, tags, notes);
"""


def _relative(path: Path, root: Path) -> str:
    return path.relative_to(root).as_posix()


def build_catalog(root: Path, output_path: Path | None = None) -> dict[str, int]:
    destination = output_path or root / "catalog" / "catalog.sqlite"
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    if temporary.exists():
        temporary.unlink()

    counts = {"locations": 0, "scenes": 0, "captures": 0, "runs": 0, "artifacts": 0}
    connection = sqlite3.connect(temporary)
    try:
        connection.executescript(SCHEMA)
        for location_path in sorted((root / "locations").glob("*/location.yaml")):
            location = load_model(location_path, LocationManifest)
            connection.execute(
                "INSERT INTO locations VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                (
                    location.id,
                    location.display_name,
                    location.status.value,
                    location.capture_date.isoformat() if location.capture_date else None,
                    json.dumps(location.region.model_dump(mode="json"), ensure_ascii=False),
                    json.dumps(location.rights.model_dump(mode="json"), ensure_ascii=False),
                    location.notes,
                    _relative(location_path, root),
                ),
            )
            for tag in sorted(set(location.tags)):
                connection.execute("INSERT INTO tags VALUES (?, ?)", (location.id, tag))
            connection.execute(
                "INSERT INTO search VALUES (?, ?, ?, ?, ?)",
                ("location", location.id, location.display_name, " ".join(location.tags), location.notes or ""),
            )
            counts["locations"] += 1

            location_root = location_path.parent
            for scene_path in sorted((location_root / "scenes").glob("*/scene.yaml")):
                scene = load_model(scene_path, SceneManifest)
                connection.execute(
                    "INSERT INTO scenes VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                    (
                        scene.id,
                        scene.location_id,
                        scene.display_name,
                        scene.status.value,
                        scene.quality_target,
                        int(scene.metric_scale),
                        scene.description,
                        _relative(scene_path, root),
                    ),
                )
                connection.execute(
                    "INSERT INTO search VALUES (?, ?, ?, ?, ?)",
                    ("scene", f"{scene.location_id}/{scene.id}", scene.display_name, "", scene.description or ""),
                )
                counts["scenes"] += 1
                scene_root = scene_path.parent

                for capture_path in sorted((scene_root / "captures").glob("*.yaml")):
                    capture = load_capture_manifest(capture_path)
                    if isinstance(capture, CaptureManifestV2):
                        source_kind = capture.source.kind
                        primary_source_path = capture.source.files[0].windows_path
                        prepared_relative_path = capture.prepared_relative_path
                    else:
                        source_kind = "equirect_video"
                        primary_source_path = capture.raw_source.windows_path
                        prepared_relative_path = capture.stitched_video.relative_path
                    connection.execute(
                        "INSERT INTO captures VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
                        (
                            capture.id,
                            capture.location_id,
                            capture.scene_id,
                            capture.status.value,
                            source_kind,
                            primary_source_path,
                            prepared_relative_path,
                            json.dumps(capture.model_dump(mode="json"), ensure_ascii=False),
                            _relative(capture_path, root),
                        ),
                    )
                    counts["captures"] += 1

                for run_path in sorted((scene_root / "runs").glob("*.yaml")):
                    # Catalogs are derived indexes, so never index a run whose
                    # immutable configuration no longer matches its stored hash.
                    run = load_run(scene_root, run_path.stem)
                    connection.execute(
                        "INSERT INTO runs VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                        (
                            run.id,
                            run.location_id,
                            run.scene_id,
                            run.status.value,
                            run.config_hash,
                            run.selected_dataset,
                            json.dumps(run.metrics, ensure_ascii=False, sort_keys=True),
                            run.created_at.isoformat(),
                            run.updated_at.isoformat(),
                            _relative(run_path, root),
                        ),
                    )
                    counts["runs"] += 1

                for artifact_path in sorted((scene_root / "exports").glob("*/artifact.yaml")):
                    artifact = load_model(artifact_path, ArtifactManifest)
                    for item in artifact.artifacts:
                        connection.execute(
                            "INSERT INTO artifacts VALUES (?, ?, ?, ?, ?, ?)",
                            (
                                artifact.run_id,
                                item.kind,
                                item.version,
                                item.relative_path,
                                item.sha256,
                                item.byte_size,
                            ),
                        )
                        counts["artifacts"] += 1
        connection.commit()
        integrity = connection.execute("PRAGMA integrity_check").fetchone()
        if integrity != ("ok",):
            raise RuntimeError(f"SQLite integrity check failed: {integrity}")
    except Exception:
        connection.close()
        if temporary.exists():
            temporary.unlink()
        raise
    else:
        connection.close()
        os.replace(temporary, destination)
    return counts
