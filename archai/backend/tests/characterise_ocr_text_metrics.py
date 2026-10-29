"""What the OCR quality metrics give on every reading of the reference pages.

``tests/test_ocr_text_metrics_characterisation.py`` compares them with
``tests/fixtures/ocr_text_metrics_characterisation.json``, recorded before the
metrics were moved out of ``app/services/ocr_quality.py``. To re-record after
an intended change::

    python tests/characterise_ocr_text_metrics.py --write
"""

from __future__ import annotations

import dataclasses
import importlib
import json
import sys
from pathlib import Path
from typing import Any

BACKEND = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BACKEND))
FIXTURE = BACKEND / "tests" / "fixtures" / "ocr_text_metrics_characterisation.json"
READINGS = BACKEND / "eval" / "quality_gate" / "readings.json"
LANGUAGES = ("latin", "old_french", "middle_french", None)


def helper(name: str) -> Any:
    try:
        found = importlib.import_module("app.services.ocr_text_metrics")
        if hasattr(found, name):
            return getattr(found, name)
    except ModuleNotFoundError:
        pass
    return getattr(importlib.import_module("app.services.ocr_quality"), name)


def _clean(value: Any) -> Any:
    if dataclasses.is_dataclass(value) and not isinstance(value, type):
        return _clean(dataclasses.asdict(value))
    if isinstance(value, dict):
        return {str(k): _clean(v) for k, v in sorted(value.items(), key=lambda kv: str(kv[0]))}
    if isinstance(value, (list, tuple)):
        return [_clean(v) for v in value]
    if isinstance(value, float):
        return round(value, 6)
    return value


def record() -> dict[str, Any]:
    pages = json.loads(READINGS.read_text(encoding="utf-8"))["pages"]
    out: dict[str, Any] = {}
    for page in pages:
        reference = page["reference"]
        for name, text in {"reference": reference, **page["readings"]}.items():
            lines = [line for line in text.splitlines() if line.strip()]
            tokens = text.split()
            script = helper("detect_script_family")(text)
            sample = tokens[:: max(1, len(tokens) // 25)][:25]
            out[f"{page['page_id']}:{name}"] = {
                "script": script,
                "entropy": helper("char_entropy")(text),
                "token_stats": helper("token_length_stats")(tokens),
                "rare_bigrams": helper("rare_bigram_ratio")(text, script),
                "tokens": {t: [helper("vowel_ratio")(t, script), helper("token_vowel_anomaly")(t, script), helper("non_wordlike_score")(t, script)] for t in sample},
                "set_apart_initial": helper("_is_set_apart_initial")(lines[0].split() if lines else []),
                "leading": helper("leading_fragment_ratio")(lines, script),
                "seam": helper("seam_fragment_ratio")(lines, list(range(0, len(lines), 5)), script),
                "seam_default": helper("seam_fragment_ratio")(lines, None, script),
                "trailing": helper("trailing_fragment_ratio")(lines),
                "stability_vs_reference": helper("token_stability_score")(tokens, reference.split()),
                "similarity_vs_reference": helper("normalized_levenshtein_similarity")(text[:300], reference[:300]),
                "cross_pass": helper("compute_cross_pass_stability")(text, reference),
                "uncertainty": helper("uncertainty_density")(text),
                "repetition": helper("repetition_score")(text),
                "by_language": {
                    str(lang): {
                        "contrast": helper("lexical_contrast")(text, lang),
                        "has_profile": helper("_has_profile")(lang),
                        "implausibility": helper("lexical_implausibility")(text, lang),
                        "gibberish": helper("gibberish_score")(text, script, lang),
                    }
                    for lang in LANGUAGES
                },
            }
    out["implausibility_from_contrast"] = {str(r): helper("implausibility_from_contrast")(r) for r in (0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 5.0)}
    return _clean(out)


if __name__ == "__main__":
    if "--write" in sys.argv:
        FIXTURE.write_text(json.dumps(record(), ensure_ascii=False, indent=1, sort_keys=True) + "\n", encoding="utf-8")
        print(f"wrote {FIXTURE}")
    else:
        print(json.dumps(record(), ensure_ascii=False)[:1200])
