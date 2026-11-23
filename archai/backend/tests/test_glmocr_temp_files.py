"""The local GLM-OCR backend leaves no temp file behind when writing the crop fails.

The crop is written with NamedTemporaryFile(delete=False), and the file was
removed only after recognition; a failed save left it in the temp directory.
glmocr itself is not installed in CI, so a stub module stands in for it.
"""

from __future__ import annotations

import base64
import io
import sys
import tempfile
import types
from pathlib import Path

import pytest
from PIL import Image

from app.services import ocr_backends


def _crop_b64() -> str:
    buffer = io.BytesIO()
    Image.new("RGB", (40, 20), "white").save(buffer, format="PNG")
    return base64.b64encode(buffer.getvalue()).decode("ascii")


def test_a_failed_crop_write_leaves_no_temp_file(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setitem(sys.modules, "glmocr", types.SimpleNamespace(GlmOcr=object))
    monkeypatch.setattr(tempfile, "tempdir", str(tmp_path))

    class Unwritable:
        def save(self, fp, format: str) -> None:
            fp.write(b"\x89PNG partial")
            raise OSError("No space left on device")

    monkeypatch.setattr(ocr_backends, "_preprocess_glmocr_crop", lambda image: (Unwritable(), {}))
    metadata = ocr_backends.OCRRecognitionMetadata(page_id="p", image_id=None, region_id="r1")

    with pytest.raises(ocr_backends.OCRBackendError, match="No space left on device"):
        ocr_backends.GlmOcrBackend().recognize(_crop_b64(), metadata)
    assert list(tmp_path.iterdir()) == []
