from __future__ import annotations

import unicodedata
from collections.abc import Sequence
from pathlib import Path
from typing import Literal

NormalizationForm = Literal["NFC", "NFD", "NFKC", "NFKD"]

REGION_SEPARATOR = "\n\n"


def assemble_text(
    region_texts: Sequence[str],
    output_path: Path,
    *,
    normalize: NormalizationForm | None = "NFC",
    drop_empty_regions: bool = True,
) -> Path:
    """Join per-region transcriptions into the final .txt output.

    normalize applies Unicode normalization (default NFC). Kraken models emit a
    mix of precomposed and decomposed forms for accented medieval Latin/French;
    without normalization the same word can compare unequal across regions,
    which breaks downstream lexicon lookup and diffing against ground truth.

    drop_empty_regions removes regions that recognized nothing, so a failed or
    blank region does not leave a run of blank lines in the transcription.
    """
    texts = [text.strip("\n") for text in region_texts]
    if drop_empty_regions:
        texts = [text for text in texts if text.strip()]

    payload = REGION_SEPARATOR.join(texts)
    if normalize:
        payload = unicodedata.normalize(normalize, payload)
    # A trailing newline makes the file well-formed for line-oriented tools.
    if payload and not payload.endswith("\n"):
        payload += "\n"

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(payload, encoding="utf-8")
    return output_path
