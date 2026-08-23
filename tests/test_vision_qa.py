import base64
import json
from pathlib import Path

import pytest

from gsdb.models import VisionQAConfig
from gsdb.vision_qa import (
    build_mimo_request,
    parse_mimo_verdict,
    resolve_mimo_endpoint,
    validate_mimo_gate,
)


def test_mimo_request_embeds_contact_sheet_and_requires_json(tmp_path: Path) -> None:
    sheet = tmp_path / "sheet.jpg"
    sheet.write_bytes(b"jpeg-bytes")
    request = build_mimo_request([sheet], VisionQAConfig(enabled=True))
    content = request["messages"][1]["content"]
    encoded = content[1]["image_url"]["url"].split(",", 1)[1]
    assert base64.b64decode(encoded) == b"jpeg-bytes"
    assert request["model"] == "mimo-v2.5"
    assert request["response_format"] == {"type": "json_object"}


def test_mimo_verdict_is_strictly_parsed() -> None:
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
    verdict = parse_mimo_verdict(payload)
    assert verdict.decision == "pass"
    assert verdict.confidence == 0.92


def test_mimo_endpoint_requires_https() -> None:
    assert (
        resolve_mimo_endpoint("https://api.example.test/v1", "/chat/completions")
        == "https://api.example.test/v1/chat/completions"
    )
    with pytest.raises(RuntimeError, match="HTTPS"):
        resolve_mimo_endpoint("http://api.example.test/v1", "/chat/completions")


def test_mimo_gate_is_fail_closed() -> None:
    config = VisionQAConfig(enabled=True, minimum_confidence=0.8)
    low_confidence = parse_mimo_verdict(
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
        validate_mimo_gate(low_confidence, config)

    reported_miss = low_confidence.model_copy(
        update={"confidence": 0.95, "false_negative_views": ["view.jpg"]}
    )
    with pytest.raises(RuntimeError, match="false negatives"):
        validate_mimo_gate(reported_miss, config)
