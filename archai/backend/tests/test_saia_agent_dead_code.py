"""Dead code in the SAIA OCR agent hot path is gone (audit AG-18 a, c)."""

from __future__ import annotations

import inspect

from app.agents import saia_ocr_agent
from app.agents.saia_ocr_agent import SaiaOCRAgent
from app.schemas.agents_ocr import SaiaOCRRequest
from tests.test_saia_ocr_agent import SAMPLE_B64, _FakeSaiaClient


def test_an_unparseable_reply_is_reported_as_invalid_json_not_an_empty_response() -> None:
    """The EMPTY_MODEL_RESPONSE branch could never run: _request_json_from_model never returns None."""
    fake = _FakeSaiaClient(models=["internvl3.5-30b-a3b"], responses_by_model={"internvl3.5-30b-a3b": ["not json", "still not json"]})
    agent = SaiaOCRAgent(client=fake, model_prefs=["internvl3.5-30b-a3b"])
    result = agent.extract(SaiaOCRRequest(image_b64=SAMPLE_B64, apply_proofread=False))
    assert "INVALID_OCR_JSON" in " ".join(result.warnings)
    assert all(fallback.error != "EMPTY_MODEL_RESPONSE" for fallback in result.fallbacks)
    assert "EMPTY_MODEL_RESPONSE" not in inspect.getsource(SaiaOCRAgent._extract)


def test_the_unread_vision_hint_list_is_gone() -> None:
    assert not hasattr(saia_ocr_agent, "VISION_MODEL_HINTS")
