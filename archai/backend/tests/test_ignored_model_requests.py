"""Model choices the locked OCR policy overrides are reported, not swallowed (audits AG-07, AG-08).

Which models run is unchanged: the policy (InternVL only, one locked model in
ocr_agent) is the user's decision, recorded by the strict xfails in
test_saia_ocr_agent.py. What changes is that a request or configuration naming
another model now gets a logged warning and a response warning naming it.
"""

from __future__ import annotations

import logging
from typing import Any

import pytest

from app.agents import ocr_agent
from app.agents.saia_ocr_agent import SaiaOCRAgent
from app.schemas.agents_ocr import OCRExtractOptions, OCRExtractRequest, SaiaOCRRequest
from tests.test_ocr_agent import _FakeSaiaClient as _FakeOcrClient
from tests.test_saia_ocr_agent import SAMPLE_B64, _FakeSaiaClient, _valid_json

ONE_PIXEL = "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADElEQVR42mNk+M/wHwAE/wJ/lxNn4QAAAABJRU5ErkJggg=="
TILE_JSON = '{"lines":["fauente"],"text":"fauente","script_hint":"latin","confidence":0.9,"warnings":[]}'


def _request(**options: Any) -> OCRExtractRequest:
    prefer = options.pop("prefer_model", None)
    return OCRExtractRequest(
        page_id="p", image_b64=ONE_PIXEL, prefer_model=prefer, options=OCRExtractOptions(backend="saia", quality_floor=0.0, **options)
    )


def test_ignored_requests_are_named() -> None:
    request = _request(prefer_model="qwen3-vl-30b-a3b-instruct", model_preference=["gemma-3-27b-it", "internvl3.5-30b-a3b"], max_fallbacks=2)
    assert ocr_agent.ignored_model_requests(request) == [
        "MODEL_PREFERENCE_IGNORED:qwen3-vl-30b-a3b-instruct",
        "MODEL_PREFERENCE_IGNORED:gemma-3-27b-it",
        "MAX_FALLBACKS_IGNORED:2",
    ]
    assert ocr_agent.ignored_model_requests(_request(max_fallbacks=0, prefer_model="internvl3.5-30b-a3b")) == []


def test_the_locked_model_still_runs_and_the_response_says_what_was_ignored(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    fake = _FakeOcrClient(
        models=["qwen3-vl-30b-a3b-instruct", "internvl3.5-30b-a3b"],
        responses_by_model={"internvl3.5-30b-a3b": [TILE_JSON] * 4},
    )
    monkeypatch.setattr(ocr_agent, "SaiaClient", lambda: fake)
    with caplog.at_level(logging.WARNING, logger=ocr_agent.__name__):
        result = ocr_agent.run_ocr_extraction(_request(prefer_model="qwen3-vl-30b-a3b-instruct", max_fallbacks=0))
    assert {call["model"] for call in fake.calls} == {"internvl3.5-30b-a3b"}  # policy unchanged
    assert "MODEL_PREFERENCE_IGNORED:qwen3-vl-30b-a3b-instruct" in result.warnings
    assert "qwen3-vl-30b-a3b-instruct" in caplog.text


def test_a_configured_non_internvl_model_is_reported_by_the_saia_agent(caplog: pytest.LogCaptureFixture) -> None:
    fake = _FakeSaiaClient(
        models=["qwen3-vl-30b-a3b-instruct", "internvl3.5-30b-a3b"],
        responses_by_model={"internvl3.5-30b-a3b": [_valid_json("Linea prima")]},
    )
    with caplog.at_level(logging.WARNING):
        agent = SaiaOCRAgent(client=fake, model_prefs=["qwen3-vl-30b-a3b-instruct", "internvl3.5-30b-a3b"])
    assert agent.model_prefs == ["internvl3.5-30b-a3b"]  # policy unchanged
    assert agent.ignored_model_prefs == ["qwen3-vl-30b-a3b-instruct"]
    assert "qwen3-vl-30b-a3b-instruct" in caplog.text

    result = agent.extract(SaiaOCRRequest(image_b64=SAMPLE_B64, apply_proofread=False))
    assert result.model_used == "internvl3.5-30b-a3b"
    assert "OCR_MODEL_PREFERENCE_IGNORED:qwen3-vl-30b-a3b-instruct" in result.warnings


def test_an_internvl_only_configuration_adds_nothing() -> None:
    fake = _FakeSaiaClient(models=["internvl3.5-30b-a3b"], responses_by_model={"internvl3.5-30b-a3b": [_valid_json("Linea")]})
    agent = SaiaOCRAgent(client=fake, model_prefs=["internvl3.5-30b-a3b"])
    assert agent.ignored_model_prefs == []
    result = agent.extract(SaiaOCRRequest(image_b64=SAMPLE_B64, apply_proofread=False))
    assert not any("IGNORED" in warning for warning in result.warnings)


def test_the_default_max_fallbacks_is_not_reported() -> None:
    """The schema default (2) is on every request; only a value the caller set counts."""
    request = OCRExtractRequest(page_id="p", image_b64=ONE_PIXEL, options=OCRExtractOptions(backend="saia"))
    assert request.options.max_fallbacks == 2
    assert ocr_agent.ignored_model_requests(request) == []


def test_max_fallbacks_is_not_reported_on_the_segmented_route() -> None:
    from app.schemas.agents_ocr import OCRRegionInput

    request = OCRExtractRequest(
        page_id="p",
        image_b64=ONE_PIXEL,
        regions=[OCRRegionInput(region_id="r1", bbox_xyxy=[0, 0, 1, 1])],
        options=OCRExtractOptions(max_fallbacks=3),
    )
    assert ocr_agent.ignored_model_requests(request) == []
