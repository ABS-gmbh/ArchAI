"""What authority linking's scoring and candidate filtering give on fixed inputs.

``tests/test_authority_scoring_characterisation.py`` compares them with
``tests/fixtures/authority_scoring_characterisation.json``, recorded before
they were moved out of ``app/services/authority_linking.py``. To re-record
after an intended change::

    python tests/characterise_authority_scoring.py --write
"""

from __future__ import annotations

import importlib
import json
import random
import sys
from pathlib import Path
from typing import Any

BACKEND = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BACKEND))
FIXTURE = BACKEND / "tests" / "fixtures" / "authority_scoring_characterisation.json"
GOLD = BACKEND / "eval" / "entities" / "gold.json"
READINGS = BACKEND / "eval" / "quality_gate" / "readings.json"
EXTRA = [
    ("Arthur", "person"), ("Lausanne", "place"), ("Paris", "place"), ("Rome", "place"), ("Iohannes", "person"),
    ("Merlin", "person"), ("Lancelot", "person"), ("Roman de la Rose", "work"), ("mil cinq cens", "date"),
    ("uilain", "place"), ("dominus", "person"), ("Hyerusalem", "place"), ("Saint Denis", "place"), ("xp̄o", "person"),
]


def helper(name: str) -> Any:
    try:
        found = importlib.import_module("app.services.authority_scoring")
        if hasattr(found, name):
            return getattr(found, name)
    except ModuleNotFoundError:
        pass
    return getattr(importlib.import_module("app.services.authority_linking"), name)


def _clean(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): _clean(v) for k, v in sorted(value.items(), key=lambda kv: str(kv[0]))}
    if isinstance(value, (list, tuple)):
        return [_clean(v) for v in value]
    if isinstance(value, (set, frozenset)):
        return sorted(_clean(v) for v in value)
    if isinstance(value, float):
        return round(value, 6)
    return value


def _surfaces() -> list[tuple[str, str, str]]:
    pages = {p["page_id"]: p["reference"] for p in json.loads(READINGS.read_text(encoding="utf-8"))["pages"]}
    gold = json.loads(GOLD.read_text(encoding="utf-8"))
    out = [(e["surface"], e["type"], pages[p["page_id"]]) for p in gold["pages"] for e in p["entities"]]
    text = pages["middle-french-antitus"]
    return out + [(s, t, text) for s, t in EXTRA]


def _candidates(surface: str, ent_type: str, seed: int) -> list[dict[str, Any]]:
    rng = random.Random(seed)
    labels = [surface, surface.lower(), surface + "us", "Saint " + surface, "Unrelated Name", surface[: max(1, len(surface) // 2)]]
    types = {"person": ["human", "fictional human"], "place": ["city", "commune of France"], "work": ["literary work"], "date": ["year"]}
    out = []
    for i, label in enumerate(labels):
        out.append(
            {
                "source": rng.choice(["wikidata", "viaf", "geonames"]),
                "id": f"Q{seed}{i}",
                "label": label,
                "description": rng.choice(["French king", "city in Switzerland", "medieval romance", "", "saint and bishop"]),
                "aliases": [{"value": a} for a in rng.sample(labels, 2)],
                "types": rng.sample(types.get(ent_type, ["thing"]) + ["thing"], 1),
                "score": round(rng.random(), 3),
            }
        )
    return out


def record() -> dict[str, Any]:
    out: dict[str, Any] = {}
    for n, (surface, ent_type, evidence) in enumerate(_surfaces()):
        key = f"{n:02d}:{surface}:{ent_type}"
        start = max(0, evidence.find(surface))
        candidates = _candidates(surface, ent_type, n)
        canonical = helper("_check_canonical_match")(surface)
        mention = {"surface": surface, "ent_type": ent_type, "label": ent_type.upper(), "start_offset": start, "end_offset": start + len(surface)}
        scored = []
        for cand in candidates:
            for type_ok in (True, False):
                scored.append(
                    helper("_score_candidate")(
                        mention=mention,
                        candidate=cand,
                        context=evidence[:400],
                        canonical_norm=str((canonical or {}).get("canonical") or ""),
                        domain_bonus=0.05,
                        type_ok=type_ok,
                        resolved_tokens={surface.lower()},
                    )
                )
        out[key] = {
            "name_likeness": helper("_compute_name_likeness")(surface, ent_type),
            "canonical": canonical,
            "place_context": helper("_has_strong_place_context")(surface, evidence),
            "variants": {m: helper("_build_query_variants")(surface, m) for m in ("rule:salvage_trigger", "rule:anchor", "unknown")},
            "deferred": helper("_deferred_reason_for_mention")(surface, ent_type, evidence, "low_quality"),
            "summary": helper("_summarize_authority_reason")("ACCEPT score=0.81 margin=0.2", surface, ent_type),
            "prefiltered": helper("_prefilter_candidates")(candidates, surface, ent_type),
            "scored": scored,
            "levenshtein": helper("_levenshtein_quick")(surface.lower(), "arthur"),
        }
    return _clean(out)


if __name__ == "__main__":
    if "--write" in sys.argv:
        FIXTURE.write_text(json.dumps(record(), ensure_ascii=False, indent=1, sort_keys=True) + "\n", encoding="utf-8")
        print(f"wrote {FIXTURE}")
    else:
        print(json.dumps(record(), ensure_ascii=False)[:1200])
