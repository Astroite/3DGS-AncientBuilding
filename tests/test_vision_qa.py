import base64
import hashlib
import json
from pathlib import Path

import pytest

from gsdb.models import VisionQAConfig
from gsdb.vision_qa import (
    build_deepseek_request,
    load_local_mask_qa_review,
    parse_deepseek_verdict,
    resolve_deepseek_endpoint,
    validate_deepseek_gate,
)


def test_deepseek_request_embeds_contact_sheet_and_requires_json(tmp_path: Path) -> None:
    sheet = tmp_path / "sheet.jpg"
    sheet.write_bytes(b"jpeg-bytes")
    request = build_deepseek_request([sheet], VisionQAConfig(enabled=True))
    content = request["messages"][1]["content"]
    encoded = content[1]["image_url"]["url"].split(",", 1)[1]
    assert base64.b64decode(encoded) == b"jpeg-bytes"
    assert content[1]["image_url"]["detail"] == "original"
    assert request["model"] == "deepseek-v4-flash-vision-exp"
    assert request["response_format"] == {"type": "json_object"}
    assert request["max_tokens"] == 4096


def test_deepseek_verdict_is_strictly_parsed() -> None:
    payload = {
        "choices": [
            {
                "message": {
                    "content": json.dumps(
                        {
                            "decision": "pass",
                            "confidence": 0.92,
                            "false_negative_views": [],
                            "false_positive_views": [],
                            "rationale": "People are covered and architecture is retained.",
                        }
                    )
                }
            }
        ]
    }
    verdict = parse_deepseek_verdict(payload)
    assert verdict.decision == "pass"
    assert verdict.confidence == 0.92


def test_deepseek_verdict_accepts_reasoning_prefix_but_keeps_strict_schema() -> None:
    content = """<think>Checked all sixteen tiles.</think>
```json
{"decision":"pass","confidence":0.95,"false_negative_views":[],"false_positive_views":[],"rationale":"All visible people are masked."}
```"""
    verdict = parse_deepseek_verdict(
        {"choices": [{"message": {"content": content}}]}
    )
    assert verdict.decision == "pass"
    assert verdict.confidence == 0.95


def test_deepseek_endpoint_requires_https() -> None:
    assert (
        resolve_deepseek_endpoint("https://api.deepseek.com", "/chat/completions")
        == "https://api.deepseek.com/chat/completions"
    )
    with pytest.raises(RuntimeError, match="HTTPS"):
        resolve_deepseek_endpoint("http://api.deepseek.com", "/chat/completions")


def test_deepseek_gate_is_fail_closed() -> None:
    config = VisionQAConfig(enabled=True, minimum_confidence=0.8)
    low_confidence = parse_deepseek_verdict(
        {
            "choices": [
                {
                    "message": {
                        "content": json.dumps(
                            {
                                "decision": "pass",
                                "confidence": 0.2,
                                "false_negative_views": [],
                                "false_positive_views": [],
                                "rationale": "uncertain",
                            }
                        )
                    }
                }
            ]
        }
    )
    with pytest.raises(RuntimeError, match="below"):
        validate_deepseek_gate(low_confidence, config)

    reported_miss = low_confidence.model_copy(
        update={"confidence": 0.95, "false_negative_views": ["view.jpg"]}
    )
    with pytest.raises(RuntimeError, match="false negatives"):
        validate_deepseek_gate(reported_miss, config)


def test_local_llm_review_is_bound_to_exact_contact_sheet_bytes(tmp_path: Path) -> None:
    sheet = tmp_path / "mask-contact-01.jpg"
    sheet.write_bytes(b"contact-sheet")
    review_path = tmp_path / "codex-local-review.json"
    review_path.write_text(
        json.dumps(
            {
                "reviewer": "codex-local-llm",
                "contact_sheet_sha256": {
                    sheet.name: hashlib.sha256(sheet.read_bytes()).hexdigest()
                },
                "verdict": {
                    "decision": "pass",
                    "confidence": 0.95,
                    "false_negative_views": [],
                    "false_positive_views": [],
                    "rationale": "All visible people are covered.",
                },
            }
        ),
        encoding="utf-8",
    )

    review, verdict = load_local_mask_qa_review(
        review_path, [sheet], VisionQAConfig(enabled=True)
    )

    assert review.reviewer == "codex-local-llm"
    assert verdict.decision == "pass"
    sheet.write_bytes(b"changed")
    with pytest.raises(RuntimeError, match="does not match"):
        load_local_mask_qa_review(
            review_path, [sheet], VisionQAConfig(enabled=True)
        )
