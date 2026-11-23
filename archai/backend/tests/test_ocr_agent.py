from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

import pytest

_backend_src = Path(__file__).resolve().parent.parent / "app"
if str(_backend_src.parent) not in sys.path:
    sys.path.insert(0, str(_backend_src.parent))

from app.agents import ocr_agent  # type: ignore[import-untyped]
from app.agents import ocr_text_parsing  # type: ignore[import-untyped]
from app.schemas.agents_ocr import (  # type: ignore[import-untyped]
    OCRExtractOptions,
    OCRExtractRequest,
    OCRExtractSimpleResponse,
    OCRRegionInput,
)


class _FakeSaiaClient:
    def __init__(
        self,
        *,
        models: list[str],
        responses_by_model: dict[str, list[str]] | None = None,
        failures_by_model: dict[str, Exception] | None = None,
    ) -> None:
        self._models = models
        self._responses_by_model = responses_by_model or {}
        self._failures_by_model = failures_by_model or {}
        self.calls: list[dict[str, Any]] = []

    def list_models(self, force_refresh: bool = False) -> list[str]:
        return list(self._models)

    def chat_completion(self, **kwargs: Any) -> dict[str, Any]:
        self.calls.append(dict(kwargs))
        model = str(kwargs.get("model") or "")
        failure = self._failures_by_model.get(model)
        if failure is not None:
            raise failure

        queue = self._responses_by_model.get(model, [])
        if not queue:
            return {"text": ""}
        return {"text": queue.pop(0)}


def _sample_request(options: OCRExtractOptions | None = None) -> OCRExtractRequest:
    return OCRExtractRequest(
        page_id="page-1",
        image_b64="iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADElEQVR42mNk+M/wHwAE/wJ/lxNn4QAAAABJRU5ErkJggg==",
        regions=[OCRRegionInput(region_id="r1", bbox_xyxy=[0, 0, 1, 1])],
        options=options or OCRExtractOptions(),
    )


def test_choose_models_prefers_declared_order() -> None:
    available = [
        "gemma-3-27b-it",
        "internvl3.5-30b-a3b",
        "qwen3-vl-30b-a3b-instruct",
    ]
    selected = ocr_agent.choose_models(
        available,
        ["qwen3-vl-30b-a3b-instruct", "internvl3.5-30b-a3b", "gemma-3-27b-it"],
    )

    assert selected == ["qwen3-vl-30b-a3b-instruct", "internvl3.5-30b-a3b", "gemma-3-27b-it"]


def test_sanitize_ocr_text_strips_json_and_code_fences() -> None:
    assert ocr_text_parsing.sanitize_ocr_text('```json\n{"text":"Linea prima\\nLinea secunda"}\n```') == (
        "Linea prima\nLinea secunda"
    )
    assert ocr_text_parsing.sanitize_ocr_text('"Linea tertia"') == "Linea tertia"


def test_ocr_user_prompt_declares_schema_and_reading_order_rules() -> None:
    assert "Keys must be exactly: lines, text, script_hint, confidence, warnings" in ocr_agent.PALEO_OCR_USER_PROMPT
    assert "If page is two-column: output left column fully first, then right column." in ocr_agent.PALEO_OCR_USER_PROMPT


def test_build_ocr_messages_is_single_turn_system_and_user_with_image() -> None:
    messages = ocr_agent.build_ocr_messages("ZmFrZV9iNjQ=")
    assert len(messages) == 2
    assert messages[0]["role"] == "system"
    assert messages[1]["role"] == "user"
    user_content = messages[1]["content"]
    assert isinstance(user_content, list)
    assert len(user_content) == 2
    assert user_content[0]["type"] == "text"
    assert user_content[1]["type"] == "image_url"
    assert "history" not in str(messages).lower()


@pytest.mark.xfail(
    strict=True,
    reason="the SAIA backend reads with InternVL only (see test_saia_ocr_agent.INTERNVL_ONLY); Qwen is never tried",
)
def test_ocr_agent_falls_back_from_qwen_to_internvl(monkeypatch: Any) -> None:
    fake = _FakeSaiaClient(
        models=["qwen3-vl-30b-a3b-instruct", "internvl3.5-30b-a3b"],
        responses_by_model={"internvl3.5-30b-a3b": ["Linea una"]},
        failures_by_model={"qwen3-vl-30b-a3b-instruct": RuntimeError("Model Not Found")},
    )
    monkeypatch.setattr(ocr_agent, "SaiaClient", lambda: fake)

    result = ocr_agent.run_ocr_extraction(
        _sample_request(
            OCRExtractOptions(
                backend="saia",
                model_preference=["qwen3-vl-30b-a3b-instruct", "internvl3.5-30b-a3b"],
                max_fallbacks=2,
                quality_floor=0.0,
            )
        )
    )

    assert result.model == "internvl3.5-30b-a3b"
    assert result.text == "Linea una"
    assert result.fallbacksUsed == ["qwen3-vl-30b-a3b-instruct"]
    assert fake.calls
    sent_messages = fake.calls[0]["messages"]
    assert isinstance(sent_messages, list)
    assert len(sent_messages) == 2
    assert sent_messages[0]["role"] == "system"
    assert sent_messages[1]["role"] == "user"


def test_ocr_agent_returns_empty_string_for_no_readable_response(monkeypatch: Any) -> None:
    fake = _FakeSaiaClient(
        models=["internvl3.5-30b-a3b"],
        responses_by_model={"internvl3.5-30b-a3b": ["No readable text is visible in this image."]},
    )
    monkeypatch.setattr(ocr_agent, "SaiaClient", lambda: fake)

    result = ocr_agent.run_ocr_extraction(_sample_request(OCRExtractOptions(backend="saia", max_fallbacks=0, quality_floor=0.0)))

    assert result.model == "internvl3.5-30b-a3b"
    assert result.text == ""
    assert result.regions[0].text == ""
    assert "EMPTY_TEXT" in result.regions[0].flags


def test_parse_ocr_json_selects_expected_region_id() -> None:
    raw = (
        '{"document_script_hint":"latin","regions":['
        '{"region_id":"r0","category":"text","bbox_xywh":[0,0,10,10],"lines":["Alpha"],"text":"Alpha","confidence":0.7,"warnings":[]},'
        '{"region_id":"r1","category":"text","bbox_xywh":[0,0,10,10],"lines":["Reuerendi"],"text":"Reuerendi","confidence":0.9,"warnings":[]}'
        '],"full_text":"Alpha\\n\\nReuerendi"}'
    )
    parsed = ocr_agent.parse_ocr_json(raw, expected_region_id="r1", latin_lock=True)

    assert parsed["text"] == "Reuerendi"
    assert parsed["lines"] == ["Reuerendi"]


def test_parse_ocr_json_replaces_non_latin_when_latin_locked() -> None:
    raw = '{"lines":["Αρχαίος"],"text":"Αρχαίος","script_hint":"latin","confidence":0.8,"warnings":[]}'
    parsed = ocr_agent.parse_ocr_json(raw, latin_lock=True)

    assert parsed["text"] == ""
    assert "non_latin_chars_replaced" in parsed["warnings"]
    assert "pattern_junk_removed" in parsed["warnings"]


def test_ocr_agent_drops_non_latin_hallucinated_region(monkeypatch: Any) -> None:
    fake = _FakeSaiaClient(
        models=["internvl3.5-30b-a3b"],
        responses_by_model={
            "internvl3.5-30b-a3b": [
                '{"lines":["Αρχαίος"],"text":"Αρχαίος","script_hint":"latin","confidence":0.9,"warnings":[]}'
            ]
        },
    )
    monkeypatch.setattr(ocr_agent, "SaiaClient", lambda: fake)

    result = ocr_agent.run_ocr_extraction(_sample_request(OCRExtractOptions(backend="saia", max_fallbacks=0, quality_floor=0.0)))

    assert result.text == ""
    assert any(flag == "REGION_TEXT_DROPPED" for flag in result.regions[0].flags)
    assert any(msg.startswith("NON_LATIN_CHARS_DETECTED:") for msg in result.warnings)


def test_ocr_agent_simple_mode_returns_only_final_text(monkeypatch: Any) -> None:
    fake = _FakeSaiaClient(
        models=["internvl3.5-30b-a3b"],
        responses_by_model={
            "internvl3.5-30b-a3b": [
                '{"lines":["faunente"],"text":"faunente","script_hint":"latin","confidence":0.9,"warnings":[]}',
                "fauente",
            ]
        },
    )
    monkeypatch.setattr(ocr_agent, "SaiaClient", lambda: fake)

    request = _sample_request(OCRExtractOptions(backend="saia", max_fallbacks=0, quality_floor=0.0)).model_copy(
        update={"mode": "simple"}
    )
    result = ocr_agent.run_ocr_extraction(request)

    assert isinstance(result, OCRExtractSimpleResponse)
    assert result.text == "fauente"
    assert result.script_hint == "latin_medieval"


# ── tile concurrency (audit AG-11) ──────────────────────────────────────


def test_tiles_read_concurrently_give_the_sequential_results_in_order(monkeypatch: Any) -> None:
    """Every tile is read; results, order and the merged page match the serial loop exactly."""
    import threading
    import time

    tiles_seen: list[str] = []
    active = {"now": 0, "peak": 0}
    lock = threading.Lock()

    def tile_reader(**kwargs: Any) -> Any:
        tile = kwargs["tile"]
        with lock:
            tiles_seen.append(tile.tile_id)
            active["now"] += 1
            active["peak"] = max(active["peak"], active["now"])
        # Later tiles answer first, so completion order differs from tile order.
        time.sleep(0.02 * (len(tiles_seen) % 3))
        try:
            return ocr_agent.OCRTileResult(
                tile_id=tile.tile_id,
                bbox_xyxy=tile.bbox_xyxy,
                lines=[f"line of {tile.tile_id}"],
                text=f"line of {tile.tile_id}",
                confidence=0.9,
                script_hint="latin",
                warnings=[],
                flags=[],
                model="internvl3.5-30b-a3b",
            ), [], []
        finally:
            with lock:
                active["now"] -= 1

    monkeypatch.setattr(ocr_agent, "_ocr_tile_with_model_candidates", tile_reader)
    monkeypatch.setattr(ocr_agent, "SaiaClient", lambda: _FakeSaiaClient(models=["internvl3.5-30b-a3b"]))
    import base64
    import io

    from PIL import Image

    page = io.BytesIO()
    Image.new("RGB", (800, 1200), (200, 190, 170)).save(page, format="PNG")  # cut into 6 tiles
    request = OCRExtractRequest(
        page_id="page-1",
        image_b64=base64.b64encode(page.getvalue()).decode(),
        options=OCRExtractOptions(backend="saia", max_fallbacks=0, quality_floor=0.0),
    )

    monkeypatch.setattr(ocr_agent.settings, "ocr_tile_concurrency", 1)
    sequential = ocr_agent.run_ocr_extraction(request)
    assert active["peak"] == 1
    active["peak"] = 0
    tiles_seen.clear()

    monkeypatch.setattr(ocr_agent.settings, "ocr_tile_concurrency", 4)
    concurrent = ocr_agent.run_ocr_extraction(request)

    def stable(result: Any) -> dict[str, Any]:
        data = result.model_dump()
        for key in ("created_at", "timestamp", "duration_ms", "elapsed_ms"):
            data.pop(key, None)
        return data

    assert len(tiles_seen) == 6
    assert stable(concurrent)["raw_ocr"] == stable(sequential)["raw_ocr"]
    assert concurrent.raw_ocr.lines == sequential.raw_ocr.lines
    assert concurrent.regions == sequential.regions
    assert concurrent.text == sequential.text and concurrent.warnings == sequential.warnings
    assert active["peak"] > 1  # tiles really overlapped


def test_map_tiles_keeps_order_and_raises_the_first_failure(monkeypatch: Any) -> None:
    import time

    monkeypatch.setattr(ocr_agent.settings, "ocr_tile_concurrency", 3)

    def slow_reverse(n: int) -> int:
        time.sleep(0.01 * (5 - n))
        return n * 10

    assert ocr_agent._map_tiles(slow_reverse, [1, 2, 3, 4]) == [10, 20, 30, 40]

    started: list[int] = []

    def fails_on_two(n: int) -> int:
        started.append(n)
        if n == 2:
            raise ocr_agent.OCRAgentError("tile 2 failed")
        time.sleep(0.05)
        return n

    with pytest.raises(ocr_agent.OCRAgentError, match="tile 2"):
        ocr_agent._map_tiles(fails_on_two, list(range(1, 30)))
    assert len(started) < 29  # tiles not yet started were cancelled


def test_the_default_tile_concurrency_is_cautious() -> None:
    from app.config import Settings

    assert Settings.model_fields["ocr_tile_concurrency"].default == 2
