from __future__ import annotations

import unicodedata
from collections.abc import Sequence
from pathlib import Path
from typing import TYPE_CHECKING, Literal

if TYPE_CHECKING:
    from archai_ocr.pipeline.page import TranscribedRegion

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


def assemble_secondary_text(
    regions: Sequence[TranscribedRegion],
    output_path: Path,
    *,
    normalize: NormalizationForm | None = "NFC",
) -> Path | None:
    """Write the text found outside the body, grouped under a heading per zone.

    Marginalia, running titles, foliation and quire marks each get a section,
    in the order they first occur on the page. Returns None and removes any
    file left by an earlier run when there is nothing to write, so a stale
    file never outlives the page it described.
    """
    from archai_ocr.pipeline.zones import zone_kind

    sections: dict[str, list[str]] = {}
    for region in regions:
        text = region.text.strip("\n")
        if text.strip():
            sections.setdefault(zone_kind(region.detection.class_name).label, []).append(text)

    if not sections:
        output_path.unlink(missing_ok=True)
        return None

    payload = REGION_SEPARATOR.join(
        f"[{label}]\n" + REGION_SEPARATOR.join(texts) for label, texts in sections.items()
    )
    if normalize:
        payload = unicodedata.normalize(normalize, payload)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(payload + "\n", encoding="utf-8")
    return output_path
