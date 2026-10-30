"""The SAIA OCR reply is accepted with keys beyond the required ones (audit AG-05).

_parse_ocr_payload compared the reply's keys to the required six for equality,
so a transcription with one extra key ("reasoning", "notes") was thrown away:
a repair call followed, and if the model repeated the key the page came back
empty with INVALID_OCR_JSON. Extra keys are now ignored and named in a warning.
"""

from __future__ import annotations

import json
from types import SimpleNamespace
from typing import Any

from app.agents.saia_ocr_agent import REQUIRED_JSON_KEYS, SaiaOCRAgent, _parse_ocr_payload

REPLY = {
    "lines": ["In nomine domini amen"],
    "text": "In nomine domini amen",
    "script_hint": "latin",
    "detected_language": "latin",
    "confidence": 0.8,
    "warnings": [],
}


def test_the_required_keys_alone_parse_without_a_warning():
    parsed = _parse_ocr_payload(json.dumps(REPLY))
    assert parsed is not None
    assert parsed["text"] == "In nomine domini amen"
    assert not any(w.startswith("SCHEMA_EXTRA_KEYS") for w in parsed["warnings"])


def test_extra_keys_are_ignored_and_named():
    parsed = _parse_ocr_payload(json.dumps({**REPLY, "reasoning": "clear hand", "notes": []}))
    assert parsed is not None
    assert parsed["text"] == "In nomine domini amen"
    assert parsed["lines"] == ["In nomine domini amen"]
    assert "SCHEMA_EXTRA_KEYS: notes, reasoning" in parsed["warnings"]
    assert "reasoning" not in parsed


def test_a_missing_required_key_is_still_rejected():
    for key in REQUIRED_JSON_KEYS:
        reply = {k: v for k, v in REPLY.items() if k != key}
        assert _parse_ocr_payload(json.dumps({**reply, "notes": "x"})) is None, key


def test_an_extra_key_costs_no_repair_call():
    calls: list[Any] = []

    def complete(*, model: str, messages: Any) -> dict[str, Any]:
        calls.append(messages)
        return {"text": json.dumps({**REPLY, "reasoning": "clear hand"})}

    agent = SimpleNamespace(_chat_completion_with_optional_json_object=complete)
    result = SaiaOCRAgent._request_json_from_model(
        agent, model="m", image_b64="aGk=", location_suggestions=[]
    )
    assert len(calls) == 1
    assert result["text"] == "In nomine domini amen"
    assert "INVALID_OCR_JSON" not in result["warnings"]
