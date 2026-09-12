from pathlib import Path
import json

import cv2
import numpy as np
import pytest

from gsdb.masking import (
    _recover_mask_filter,
    _recover_mask_records,
    PersonPrediction,
    colmap_mask_from_ignored,
    generate_person_masks,
    filter_masked_images,
    image_dimensions,
    mask_path_for_image,
    postprocess_person_mask,
    validate_mask_set,
    validate_mask_filter,
)
from gsdb.models import MaskingConfig, MaskingConfigV3


def _write_image(path: Path, shape: tuple[int, int] = (32, 32)) -> None:
    image = np.full((*shape, 3), 100, dtype=np.uint8)
    assert cv2.imwrite(str(path), image)


def test_person_mask_is_black_for_both_consumers(tmp_path: Path) -> None:
    images = tmp_path / "images"
    masks = tmp_path / "masks"
    images.mkdir()
    image_path = images / "frame_000001_0.jpg"
    _write_image(image_path)
    config = MaskingConfig(dilation_pixels=1, closing_pixels=0)

    def predictor(image: np.ndarray) -> PersonPrediction:
        prediction = np.zeros(image.shape[:2], dtype=bool)
        prediction[14:18, 14:18] = True
        return PersonPrediction(prediction, 1)

    records = generate_person_masks(
        images, masks, config, tmp_path / "metrics.jsonl", predictor=predictor
    )
    stored = cv2.imread(
        str(mask_path_for_image(masks, image_path)), cv2.IMREAD_GRAYSCALE
    )
    assert stored is not None
    assert stored[16, 16] == 0
    assert stored[0, 0] == 255
    assert records[0]["detections"] == 1
    assert not list(masks.glob(".*.tmp"))
    summary = validate_mask_set(images, masks, max_masked_fraction=0.45)
    assert summary["deterministic_qa"] == "passed"
    assert summary["masked_images"] == 1


def test_multiclass_dynamic_masks_record_counts_and_area(tmp_path: Path) -> None:
    images = tmp_path / "images"
    masks = tmp_path / "masks"
    images.mkdir()
    image_path = images / "frame_000001.jpg"
    _write_image(image_path)
    config = MaskingConfigV3(
        classes=["person", "car"], dilation_pixels=0, closing_pixels=0
    )

    def predictor(image: np.ndarray) -> PersonPrediction:
        person = np.zeros(image.shape[:2], dtype=bool)
        car = np.zeros(image.shape[:2], dtype=bool)
        person[1:3, 1:3] = True
        car[10:13, 10:13] = True
        return PersonPrediction(
            mask=person | car,
            detections=3,
            detections_by_class={"person": 1, "car": 2},
            masks_by_class={"person": person, "car": car},
        )

    records = generate_person_masks(
        images, masks, config, tmp_path / "metrics.jsonl", predictor=predictor
    )
    assert records[0]["detections"] == 3
    assert records[0]["classes"] == {
        "person": {"detections": 1, "masked_fraction": round(4 / 1024, 8)},
        "car": {"detections": 2, "masked_fraction": round(9 / 1024, 8)},
    }


def test_mask_resume_reuses_existing_file(tmp_path: Path) -> None:
    images = tmp_path / "images"
    masks = tmp_path / "masks"
    images.mkdir()
    masks.mkdir()
    image_path = images / "view.jpg"
    _write_image(image_path)
    assert cv2.imwrite(
        str(mask_path_for_image(masks, image_path)),
        colmap_mask_from_ignored(np.zeros((32, 32), dtype=bool)),
    )

    def must_not_run(_: np.ndarray) -> PersonPrediction:
        raise AssertionError("predictor should not run for an existing valid mask")

    records = generate_person_masks(
        images,
        masks,
        MaskingConfig(dilation_pixels=0, closing_pixels=0),
        tmp_path / "metrics.jsonl",
        predictor=must_not_run,
    )
    assert records[0]["reused"] is True


def test_deterministic_qa_rejects_destructive_mask(tmp_path: Path) -> None:
    images = tmp_path / "images"
    masks = tmp_path / "masks"
    images.mkdir()
    masks.mkdir()
    image_path = images / "view.jpg"
    _write_image(image_path)
    assert cv2.imwrite(
        str(mask_path_for_image(masks, image_path)), np.zeros((32, 32), dtype=np.uint8)
    )
    with pytest.raises(RuntimeError, match="above the configured"):
        validate_mask_set(images, masks, max_masked_fraction=0.45)


def test_postprocess_rejects_non_2d_mask() -> None:
    with pytest.raises(ValueError, match="2D"):
        postprocess_person_mask(
            np.zeros((4, 4, 1), dtype=np.uint8),
            MaskingConfig(dilation_pixels=0, closing_pixels=0),
        )


def test_nested_view_masks_mirror_image_relative_paths(tmp_path: Path) -> None:
    images = tmp_path / "images"
    masks = tmp_path / "masks"
    view = images / "view_03"
    view.mkdir(parents=True)
    image_path = view / "frame_000007.jpg"
    _write_image(image_path)

    records = generate_person_masks(
        images,
        masks,
        MaskingConfig(enabled=False, dilation_pixels=0, closing_pixels=0),
        tmp_path / "metrics.jsonl",
    )

    assert records[0]["image"] == "view_03/frame_000007.jpg"
    assert records[0]["mask"] == "view_03/frame_000007.jpg.png"
    assert (masks / "view_03" / "frame_000007.jpg.png").is_file()
    assert validate_mask_set(images, masks, 0.45)["mask_count"] == 1


def test_image_dimensions_reads_headers_without_decoding(tmp_path: Path) -> None:
    jpeg = tmp_path / "frame.jpg"
    png = tmp_path / "mask.png"
    assert cv2.imwrite(str(jpeg), np.zeros((48, 96, 3), dtype=np.uint8))
    assert cv2.imwrite(str(png), np.zeros((17, 33), dtype=np.uint8))
    assert image_dimensions(jpeg) == (48, 96)
    assert image_dimensions(png) == (17, 33)


def test_image_dimensions_rejects_truncated_and_foreign_files(tmp_path: Path) -> None:
    jpeg = tmp_path / "frame.jpg"
    assert cv2.imwrite(str(jpeg), np.zeros((48, 96, 3), dtype=np.uint8))
    truncated = tmp_path / "truncated.jpg"
    truncated.write_bytes(jpeg.read_bytes()[:4])
    with pytest.raises(RuntimeError):
        image_dimensions(truncated)
    foreign = tmp_path / "notes.txt"
    foreign.write_bytes(b"not an image at all")
    with pytest.raises(RuntimeError, match="Unsupported image header"):
        image_dimensions(foreign)


def test_mask_filter_keeps_exactly_five_percent_and_deletes_one_pixel_over(
    tmp_path: Path,
) -> None:
    dataset = tmp_path / "dataset"
    images = dataset / "images"
    masks = dataset / "masks"
    images.mkdir(parents=True)
    masks.mkdir()
    records = []
    for name, black_pixels in (("keep.jpg", 20), ("drop.jpg", 21)):
        image = images / name
        _write_image(image, (20, 20))
        mask = np.full((20, 20), 255, dtype=np.uint8)
        mask.flat[:black_pixels] = 0
        mask_path = masks / f"{name}.png"
        assert cv2.imwrite(str(mask_path), mask)
        records.append(
            {
                "image": name,
                "mask": f"{name}.png",
                "masked_fraction": black_pixels / 400,
            }
        )

    result = filter_masked_images(dataset, records, 0.05)
    assert [item["image"] for item in result["accepted"]] == ["keep.jpg"]
    assert [item["image"] for item in result["rejected"]] == ["drop.jpg"]
    assert (images / "keep.jpg").is_file()
    assert (masks / "keep.jpg.png").is_file()
    assert not (images / "drop.jpg").exists()
    assert not (masks / "drop.jpg.png").exists()
    assert all(item["image_sha256"] and item["mask_sha256"] for item in result["accepted"] + result["rejected"])

    pending = {**result, "status": "pending"}
    (dataset / ".mask-filter.pending.json").write_text(
        json.dumps(pending), encoding="utf-8"
    )
    (dataset / "mask-filter.json").unlink()
    resumed = filter_masked_images(dataset, records, 0.05)
    assert resumed["status"] == "complete"
    assert validate_mask_filter(dataset)["accepted"] == resumed["accepted"]


def _recovery_dataset(path: Path):
    (path / "images").mkdir(parents=True)
    (path / "masks").mkdir()
    records, originals = [], {}
    for name, pixels in (("keep.jpg", 20), ("drop.jpg", 21)):
        image = path / "images" / name
        mask = path / "masks" / f"{name}.png"
        _write_image(image, (20, 20))
        values = np.full((20, 20), 255, np.uint8)
        values.flat[:pixels] = 0
        assert cv2.imwrite(str(mask), values)
        originals[image] = image.read_bytes()
        originals[mask] = mask.read_bytes()
        records.append({"image": name, "mask": mask.name, "masked_fraction": pixels / 400})
    payload = filter_masked_images(path, records, 0.05)
    return payload, originals


@pytest.mark.parametrize("pending", [False, True])
@pytest.mark.parametrize("remaining", ["both", "image", "mask", "neither"])
def test_filter_recovery_is_idempotent(tmp_path, pending, remaining):
    payload, originals = _recovery_dataset(tmp_path)
    final = tmp_path / "mask-filter.json"
    completed_bytes = final.read_bytes()
    if pending:
        (tmp_path / ".mask-filter.pending.json").write_text(json.dumps({**payload, "status": "pending"}))
        final.unlink()
    for path, content in originals.items():
        if "drop" in path.name and (remaining == "both" or
                remaining == "image" and path.parent.name == "images" or
                remaining == "mask" and path.parent.name == "masks"):
            path.write_bytes(content)
    for _ in range(2):
        assert _recover_mask_filter(tmp_path, 0.05) == payload
        assert validate_mask_filter(tmp_path) == payload
    if not pending:
        assert final.read_bytes() == completed_bytes
    assert not (tmp_path / ".mask-filter.pending.json").exists()


@pytest.mark.parametrize("failure", ["unknown", "missing", "image", "mask", "threshold",
                                    "duplicate", "overlap", "count", "nan", "conflict", "schema", "status"])
def test_filter_recovery_preflights_before_any_deletion(tmp_path, failure):
    payload, originals = _recovery_dataset(tmp_path)
    for path, content in originals.items():
        path.write_bytes(content)
    match = ""
    threshold = 0.05
    if failure == "unknown":
        (tmp_path / "images/unknown.txt").write_text("unknown")
        match = "unknown.txt"
    elif failure == "missing":
        (tmp_path / "masks/keep.jpg.png").unlink()
        match = "keep.jpg.png"
    elif failure in {"image", "mask"}:
        target = tmp_path / ("images/keep.jpg" if failure == "image" else "masks/keep.jpg.png")
        target.write_bytes(target.read_bytes() + b"changed")
        match = "keep.jpg"
    elif failure == "threshold":
        threshold = 0.06
        match = "threshold"
    elif failure == "duplicate":
        payload["accepted"].append(payload["accepted"][0])
        match = "Duplicate"
    elif failure == "overlap":
        payload["rejected"][0].update(image="keep.jpg", mask="keep.jpg.png")
        match = "Overlapping"
    elif failure == "count":
        payload["original_image_count"] = 99
        match = "count"
    elif failure == "nan":
        payload["accepted"][0]["masked_fraction"] = float("nan")
        match = "keep.jpg"
    elif failure in {"schema", "status"}:
        payload["schema_version" if failure == "schema" else "status"] = "invalid"
        match = "schema/status"
    else:
        (tmp_path / ".mask-filter.pending.json").write_text(json.dumps({**payload, "status": "pending", "extra": 1}))
        match = "Conflicting"
    (tmp_path / "mask-filter.json").write_text(json.dumps(payload))
    before = {p: p.read_bytes() for p in tmp_path.rglob("*") if p.is_file()}
    with pytest.raises(RuntimeError, match=match):
        _recover_mask_filter(tmp_path, threshold)
    assert before == {p: p.read_bytes() for p in tmp_path.rglob("*") if p.is_file()}


@pytest.mark.parametrize("unsafe", ["../escape.jpg", "C:/escape.jpg", "C:\\escape.jpg",
                                   "\\\\server\\share\\escape.jpg", "/escape.jpg", "a/../../escape.jpg"])
def test_filter_recovery_rejects_unsafe_intent(tmp_path, unsafe):
    payload, originals = _recovery_dataset(tmp_path)
    for path, content in originals.items():
        path.write_bytes(content)
    payload["rejected"][0]["image"] = unsafe
    (tmp_path / "mask-filter.json").write_text(json.dumps(payload))
    with pytest.raises(RuntimeError, match="Unsafe"):
        _recover_mask_filter(tmp_path, 0.05)
    assert (tmp_path / "images/drop.jpg").exists()


def test_filter_recovery_interruption_during_cleanup(tmp_path, monkeypatch):
    payload, originals = _recovery_dataset(tmp_path)
    for path, content in originals.items():
        path.write_bytes(content)
    (tmp_path / "mask-filter.json").unlink()
    (tmp_path / ".mask-filter.pending.json").write_text(json.dumps({**payload, "status": "pending"}))
    unlink = Path.unlink
    def interrupt(path, *args, **kwargs):
        if path.name == "drop.jpg.png":
            raise OSError("interrupted")
        return unlink(path, *args, **kwargs)
    with monkeypatch.context() as patch:
        patch.setattr(Path, "unlink", interrupt)
        with pytest.raises(OSError, match="interrupted"):
            _recover_mask_filter(tmp_path, 0.05)
    assert not (tmp_path / "images/drop.jpg").exists()
    assert (tmp_path / "masks/drop.jpg.png").exists()
    assert _recover_mask_filter(tmp_path, 0.05) == payload


def test_filter_recovery_review_hashes_and_finalization(tmp_path):
    from gsdb.mask_review import MaskReviewDataset
    from gsdb.mask_finalize import finalize_mask_dataset
    payload, originals = _recovery_dataset(tmp_path)
    mask = tmp_path / "masks/keep.jpg.png"
    values = np.full((20, 20), 255, np.uint8)
    values.flat[:21] = 0
    assert cv2.imwrite(str(mask), values)
    with pytest.raises(RuntimeError, match="without current review"):
        _recover_mask_filter(tmp_path, 0.05, review_required=True)
    MaskReviewDataset(tmp_path).update_review(1, "exclude", "edited above threshold")
    finalize_mask_dataset(tmp_path, {"keep.jpg"}, maximum_included_masked_fraction=0.05)
    immutable = {p: p.read_bytes() for p in (tmp_path / "mask-review.json", tmp_path / "mask-final.json", tmp_path / "mask-filter.json")}
    for path, content in originals.items():
        if "drop" in path.name:
            path.write_bytes(content)
    assert _recover_mask_filter(tmp_path, 0.05, review_required=True) == payload
    records = _recover_mask_records(tmp_path, payload, tmp_path / "missing.jsonl")
    assert records[0]["masked_fraction"] == 21 / 400
    assert records[0]["detections"] is None and records[0]["classes"] == {}
    assert all(p.read_bytes() == data for p, data in immutable.items())
    with pytest.raises(RuntimeError, match="without current review"):
        _recover_mask_filter(tmp_path, 0.05, review_required=False)
    image = tmp_path / "images/keep.jpg"
    image.write_bytes(image.read_bytes() + b"changed")
    with pytest.raises(RuntimeError, match="image hash changed"):
        _recover_mask_filter(tmp_path, 0.05, review_required=True)


def test_filter_recovery_missing_and_truncated_statistics(tmp_path):
    payload, _ = _recovery_dataset(tmp_path)
    metrics = tmp_path / "metrics.jsonl"
    assert _recover_mask_records(tmp_path, payload, metrics)[0]["detections"] is None
    metrics.write_text(json.dumps({"image": "keep.jpg", "detections": 7,
                                  "masked_fraction": 0.99, "classes": {"person": {"detections": 7}}}) + '\n{"image":')
    record = _recover_mask_records(tmp_path, payload, metrics)[0]
    assert record["detections"] == 7
    assert record["masked_fraction"] == 0.05
    assert record["classes"] == {"person": {"detections": 7}}


def test_filter_recovery_caches_paths_only_within_preflight(tmp_path, monkeypatch, capsys):
    from collections import Counter

    payload, originals = _recovery_dataset(tmp_path)
    capsys.readouterr()
    original_resolve, original_lstat = Path.resolve, Path.lstat
    resolves, lstats = Counter(), Counter()

    def counted_resolve(path, *args, **kwargs):
        resolves[path] += 1
        return original_resolve(path, *args, **kwargs)

    def counted_lstat(path, *args, **kwargs):
        lstats[path] += 1
        return original_lstat(path, *args, **kwargs)

    for _ in range(2):
        # Both records, inventory traversal, and accepted/rejected checks all
        # reference the same paths. The next invocation must check them anew.
        for path, content in originals.items():
            path.write_bytes(content)
        (tmp_path / ".mask-filter.pending.json").write_text(
            json.dumps({**payload, "status": "pending"}))
        resolves.clear()
        lstats.clear()
        with monkeypatch.context() as patch:
            patch.setattr(Path, "resolve", counted_resolve)
            patch.setattr(Path, "lstat", counted_lstat)
            assert _recover_mask_filter(tmp_path, 0.05) == payload
        for path in [tmp_path, tmp_path / "images", tmp_path / "masks", *originals]:
            assert resolves[path] == 1
            assert lstats[path] == 1
        output = capsys.readouterr().out
        assert "beginning images inventory" in output
        assert "hashes/masks 1/1 accepted pairs" in output
        assert output.index("preflight complete") < output.index("cleanup complete")


@pytest.mark.parametrize("linked", [".", "images", "images/keep.jpg", "masks/drop.jpg.png"])
def test_filter_recovery_rejects_reparse_attributes_before_cleanup(tmp_path, monkeypatch, linked):
    import stat
    from types import SimpleNamespace

    # The fixture has already completed one recovery. A fresh invocation must
    # reject newly introduced reparse points, including ancestors, on Python
    # 3.10 where Path.is_junction does not exist.
    _, originals = _recovery_dataset(tmp_path)
    for path, content in originals.items():
        path.write_bytes(content)
    original_lstat = Path.lstat

    def reparse_lstat(path, *args, **kwargs):
        info = original_lstat(path, *args, **kwargs)
        if path == tmp_path / linked:
            return SimpleNamespace(st_mode=info.st_mode,
                                   st_file_attributes=stat.FILE_ATTRIBUTE_REPARSE_POINT)
        return info

    with monkeypatch.context() as patch:
        patch.setattr(Path, "lstat", reparse_lstat)
        with pytest.raises(RuntimeError, match="Unsafe mask-filter link"):
            _recover_mask_filter(tmp_path, 0.05)
    assert all(path.read_bytes() == content for path, content in originals.items())


def test_filter_recovery_rejects_symlink_before_cleanup(tmp_path):
    dataset = tmp_path / "dataset"
    _, originals = _recovery_dataset(dataset)
    for path, content in originals.items():
        path.write_bytes(content)
    image = dataset / "images/keep.jpg"
    outside = tmp_path / "outside.jpg"
    outside.write_bytes(image.read_bytes())
    image.unlink()
    try:
        image.symlink_to(outside)
    except OSError:
        pytest.skip("Creating symlinks is unavailable on this host")
    with pytest.raises(RuntimeError, match="link|Escaping"):
        _recover_mask_filter(dataset, 0.05)
    assert (dataset / "images/drop.jpg").exists()
    assert outside.read_bytes() == originals[image]


def test_filter_recovery_preserves_reviewed_qa_and_rejects_tampering(tmp_path):
    import hashlib
    payload, originals = _recovery_dataset(tmp_path)
    qa = tmp_path / "mask-qa"
    qa.mkdir()
    sheet = qa / "mask-contact-01.jpg"
    sheet.write_bytes(b"reviewed sheet")
    review = qa / "codex-local-review.json"
    review.write_text(json.dumps({"contact_sheet_sha256": {sheet.name: hashlib.sha256(sheet.read_bytes()).hexdigest()}}))
    before = review.read_bytes(), sheet.read_bytes()
    assert _recover_mask_filter(tmp_path, 0.05) == payload
    assert (review.read_bytes(), sheet.read_bytes()) == before
    for path, content in originals.items():
        if "drop" in path.name:
            path.write_bytes(content)
    sheet.unlink()
    with pytest.raises(RuntimeError, match="mask-contact-01.jpg"):
        _recover_mask_filter(tmp_path, 0.05)
    assert (tmp_path / "images/drop.jpg").exists()


def test_filter_recovery_stale_finalization_fails_before_cleanup(tmp_path):
    from gsdb.mask_finalize import finalize_mask_dataset
    _, originals = _recovery_dataset(tmp_path)
    finalize_mask_dataset(tmp_path)
    final = tmp_path / "mask-final.json"
    payload = json.loads(final.read_text())
    payload["mask_sha256"] = {}
    final.write_text(json.dumps(payload))
    for path, content in originals.items():
        if "drop" in path.name:
            path.write_bytes(content)
    with pytest.raises(RuntimeError, match="mask-final.json"):
        _recover_mask_filter(tmp_path, 0.05)
    assert (tmp_path / "images/drop.jpg").exists()
