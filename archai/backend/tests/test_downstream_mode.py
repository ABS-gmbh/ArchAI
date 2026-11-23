"""The downstream mode says what the pipeline does when token search is blocked (audit SVC-18).

It reported "vision_fallback" for RISKY and UNRELIABLE pages, but no
shape/layout path exists: token-based stages are skipped and nothing runs in
their place, and the shape-search helpers had no callers.
"""

from __future__ import annotations

import pytest

from app.services import pipeline_hardening
from app.services.pipeline_hardening import decide_downstream_mode


@pytest.mark.parametrize(("label", "mode"), [("HIGH", "token_based"), ("OK", "token_based"), ("RISKY", "token_search_disabled"), ("UNRELIABLE", "token_search_disabled")])
def test_the_mode_names_what_happens(label: str, mode: str) -> None:
    assert decide_downstream_mode(label) == mode


def test_the_shape_search_that_never_ran_is_gone() -> None:
    for name in ("generate_shape_based_candidates", "should_use_shape_based_search"):
        assert not hasattr(pipeline_hardening, name), name
    assert "vision_fallback" not in (pipeline_hardening.decide_downstream_mode.__doc__ or "")
