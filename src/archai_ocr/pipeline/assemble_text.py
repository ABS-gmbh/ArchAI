from __future__ import annotations

from pathlib import Path
from typing import Sequence


def assemble_text(region_texts: Sequence[str], output_path: Path) -> Path:
    normalized = [text.rstrip("\n") for text in region_texts]
    payload = "\n\n".join(normalized)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(payload, encoding="utf-8")
    return output_path
