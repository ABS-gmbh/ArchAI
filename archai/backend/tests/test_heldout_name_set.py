"""The held-out name set is well formed: every gold span is in its page, and no manuscript is in both halves."""

from __future__ import annotations

import json
from pathlib import Path

BACKEND = Path(__file__).resolve().parents[1]
PAGES = json.loads((BACKEND / "eval/entities/heldout_pages.json").read_text(encoding="utf-8"))["pages"]
GOLD = json.loads((BACKEND / "eval/entities/heldout_gold.json").read_text(encoding="utf-8"))["pages"]


def test_every_gold_mention_is_in_its_page() -> None:
    texts = {p["page_id"]: p["reference"] for p in PAGES}
    assert sorted(texts) == sorted(g["page_id"] for g in GOLD)
    for page in GOLD:
        for entity in page["entities"]:
            start = -1
            for _ in range(entity.get("occurrence", 1)):
                start = texts[page["page_id"]].find(entity["surface"], start + 1)
            assert start >= 0, (page["page_id"], entity["surface"], entity.get("occurrence", 1))
            assert entity["type"] in ("person", "place", "work")
            assert entity["authority"]["status"] in ("linkable", "none", "uncertain")
            assert bool(entity["authority"]["qids"]) == (entity["authority"]["status"] == "linkable")


def test_the_halves_share_no_manuscript() -> None:
    halves = {"dev": set(), "test": set()}
    for page in PAGES:
        halves[page["split"]].add(page["shelfmark"])
    assert halves["dev"] and halves["test"] and not (halves["dev"] & halves["test"])
