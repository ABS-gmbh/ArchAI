"""Dead and unreachable normalisation code is gone (audit AL-15)."""

from __future__ import annotations

from app.services import lexicon_trust, text_normalization
from app.services.text_normalization import _OCR_CONFUSION_MAP, ocr_confusion_fixes


def test_vne_is_still_rewritten_by_the_vn_rule_alone() -> None:
    assert ("vne", "une") not in _OCR_CONFUSION_MAP  # unreachable after ("vn", "un")
    assert ocr_confusion_fixes("vne vn vnes") == "une un unes"


def test_the_uncalled_helpers_are_gone() -> None:
    for name in ("agreement_score", "line_length_mismatch_ratio"):
        assert not hasattr(lexicon_trust, name), name
    assert not hasattr(text_normalization, "ocr_aware_similarity")
