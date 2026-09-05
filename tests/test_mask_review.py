from __future__ import annotations

import csv
import json
import threading
from pathlib import Path
from urllib.request import Request, urlopen

import cv2
import numpy as np

from gsdb.mask_review import MaskReviewDataset, create_mask_review_server


def _dataset(path: Path) -> Path:
    images = path / "images"
    masks = path / "masks"
    images.mkdir(parents=True)
    masks.mkdir()
    image = np.full((24, 32, 3), 90, dtype=np.uint8)
    image[:, 16:, 1] = 180
    mask = np.zeros((24, 32), dtype=np.uint8)
    mask[4:20, 18:30] = 255
    assert cv2.imwrite(str(images / "image_00000007.jpg"), image)
    assert cv2.imwrite(str(masks / "image_00000007.png"), mask)
    with (path / "image-map.csv").open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(
            stream,
            fieldnames=(
                "image_id",
                "source_image",
                "output_image",
                "source_mask",
                "output_mask",
                "width",
                "height",
                "ignored_pixels",
            ),
        )
        writer.writeheader()
        writer.writerow(
            {
                "image_id": 7,
                "source_image": "view_00/frame_000001.jpg",
                "output_image": "image_00000007.jpg",
                "source_mask": "view_00/frame_000001.jpg.png",
                "output_mask": "image_00000007.png",
                "width": 32,
                "height": 24,
                "ignored_pixels": 192,
            }
        )
    return path


def test_mask_review_dataset_persists_non_destructive_labels(tmp_path: Path) -> None:
    path = _dataset(tmp_path / "postshot")
    dataset = MaskReviewDataset(path)

    assert dataset.payload()["items"][0]["masked_fraction"] == 0.25
    assert dataset.review_for(7) == {"status": "unreviewed", "note": ""}
    dataset.update_review(7, "false_negative", "right arm")

    stored = json.loads((path / "mask-review.json").read_text(encoding="utf-8"))
    assert stored["reviews"]["7"] == {
        "status": "false_negative",
        "note": "right arm",
    }
    assert MaskReviewDataset(path).review_for(7)["status"] == "false_negative"
    assert (path / "images" / "image_00000007.jpg").is_file()
    assert (path / "masks" / "image_00000007.png").is_file()


def test_mask_review_http_api_serves_media_and_updates_review(tmp_path: Path) -> None:
    server = create_mask_review_server(_dataset(tmp_path / "postshot"), port=0)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    host, port = server.server_address[:2]
    base = f"http://{host}:{port}"
    try:
        with urlopen(base + "/", timeout=5) as response:
            page = response.read()
            assert response.headers["Cache-Control"] == "no-store"
            assert b"Mask Review" in page
            assert b"mask-mode: luminance" in page
            assert b'data-workspace="gallery"' in page
            assert b'id="gallery"' in page
        with urlopen(base + "/api/items", timeout=5) as response:
            payload = json.load(response)
        assert payload["counts"]["total"] == 1
        assert payload["items"][0]["image_id"] == 7
        with urlopen(base + "/media/thumbnail?id=7", timeout=5) as response:
            assert response.headers["Cache-Control"] == "no-store"
            assert response.read(2) == b"\xff\xd8"
        for mode in ("overlay", "image", "mask"):
            with urlopen(base + f"/media/thumbnail?id=7&mode={mode}", timeout=5) as response:
                assert response.read(2) == b"\xff\xd8"
        request = Request(
            base + "/api/review",
            data=json.dumps(
                {"image_id": 7, "status": "false_positive", "note": "roof"}
            ).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urlopen(request, timeout=5) as response:
            updated = json.load(response)
        assert updated["review"] == {"status": "false_positive", "note": "roof"}
        with urlopen(base + "/api/reviews.csv", timeout=5) as response:
            assert "false_positive" in response.read().decode("utf-8-sig")
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)
