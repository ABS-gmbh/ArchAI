"""Request lists and free-text fields are bounded (security review S2).

Each region or location suggestion costs one recogniser or provider call, and
none of the lists had a length limit; evidence text and meta_json had none
either.
"""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from app.schemas import agents_ocr as schemas

REGION = {"bbox_xyxy": [0, 0, 10, 10]}
SUGGESTION = {"region_id": "r", "bbox_xywh": [0, 0, 10, 10], "category": "text"}


@pytest.mark.parametrize(
    ("model", "field", "item", "extra"),
    [
        (schemas.OCRExtractRequest, "regions", REGION, {"image_b64": "eA=="}),
        (schemas.SaiaFullPageExtractRequest, "regions", REGION, {"image_b64": "eA=="}),
        (schemas.SaiaFullPageExtractRequest, "location_suggestions", SUGGESTION, {"image_b64": "eA=="}),
        (schemas.SaiaOCRRequest, "location_suggestions", SUGGESTION, {"image_b64": "eA=="}),
    ],
)
def test_region_lists_are_bounded(model, field: str, item: dict, extra: dict) -> None:
    limit = schemas.MAX_REGIONS_PER_REQUEST
    model(**extra, **{field: [item] * limit})
    with pytest.raises(ValidationError):
        model(**extra, **{field: [item] * (limit + 1)})


def test_evidence_text_and_meta_are_bounded() -> None:
    limit = schemas.MAX_EVIDENCE_TEXT_CHARS
    schemas.EvidenceSpanCreateRequest(run_id="r", text="a" * limit, meta_json={"note": "x"})
    for field in ("text", "raw_text", "normalized_text"):
        with pytest.raises(ValidationError):
            schemas.EvidenceSpanCreateRequest(run_id="r", **{field: "a" * (limit + 1)})
    with pytest.raises(ValidationError):
        schemas.EvidenceSpanCreateRequest(run_id="r", meta_json={"blob": "a" * schemas.MAX_EVIDENCE_META_BYTES})


def test_polygons_are_bounded() -> None:
    # Second review, L: an evidence span with a 500,000-point polygon was stored.
    limit = schemas.MAX_POLYGON_POINTS
    square = [[0.0, 0.0], [10.0, 0.0], [10.0, 10.0]]
    points = (square * limit)[:limit]
    schemas.EvidenceSpanCreateRequest(run_id="r", text="x", polygon=points)
    schemas.OCRRegionInput(polygon=points)
    with pytest.raises(ValidationError):
        schemas.EvidenceSpanCreateRequest(run_id="r", text="x", polygon=points + [[1.0, 1.0]])
    with pytest.raises(ValidationError):
        schemas.OCRRegionInput(polygon=points + [[1.0, 1.0]])
