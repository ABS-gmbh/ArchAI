"""The quality label's thresholds live in ocr_quality_config, values unchanged (audit SVC-23)."""

from __future__ import annotations

import ast
import inspect
import textwrap

from app.services import ocr_quality, ocr_quality_config


def test_derive_quality_label_compares_against_no_bare_number() -> None:
    tree = ast.parse(textwrap.dedent(inspect.getsource(ocr_quality._derive_quality_label)))
    literals = [
        node.value
        for compare in ast.walk(tree)
        if isinstance(compare, ast.Compare)
        for node in [compare.left, *compare.comparators]
        if isinstance(node, ast.Constant) and isinstance(node.value, float)
    ]
    assert literals == []  # 0 (int), the "not measured" sentinel check, is not a threshold


def test_the_moved_values_are_the_ones_the_code_used() -> None:
    assert (
        ocr_quality_config.ENTROPY_UNRELIABLE_LIMIT,
        ocr_quality_config.HIGH_GIBBERISH_MAX,
        ocr_quality_config.HIGH_NON_WORDLIKE_MAX,
        ocr_quality_config.HIGH_LEADING_FRAG_MAX,
        ocr_quality_config.HIGH_SEAM_FRAG_MAX,
        ocr_quality_config.HIGH_UNCERTAINTY_MAX,
        ocr_quality_config.HIGH_ENTROPY_MIN,
        ocr_quality_config.HIGH_ENTROPY_MAX,
        ocr_quality_config.HIGH_REPETITION_MAX,
    ) == (1.5, 0.10, 0.15, 0.06, 0.05, 0.03, 2.5, 5.0, 0.10)
