"""The trigram profiles list each trigram once, and the gate does not depend on their sizes (audit AL-14)."""

from __future__ import annotations

import ast
from pathlib import Path

from app.services import lexicon_trust

SOURCE = Path(lexicon_trust.__file__).read_text(encoding="utf-8")


def _literal_profiles() -> dict[str, list[str]]:
    for node in ast.walk(ast.parse(SOURCE)):
        if isinstance(node, ast.AnnAssign) and getattr(node.target, "id", "") == "_TRIGRAM_PROFILES":
            return {
                key.value: [elt.value for elt in value.args[0].elts] if value.args else []
                for key, value in zip(node.value.keys, node.value.values)
            }
    raise AssertionError("_TRIGRAM_PROFILES literal not found")


def test_no_trigram_is_listed_twice() -> None:
    for language, trigrams in _literal_profiles().items():
        assert len(trigrams) == len(set(trigrams)), language


def test_the_profile_sizes_are_the_documented_ones() -> None:
    sizes = {language: len(lexicon_trust._TRIGRAM_PROFILES[language]) for language in ("latin", "old_french", "middle_french", "french")}
    assert sizes == {"latin": 95, "old_french": 94, "middle_french": 102, "french": 73}


def test_the_gate_score_does_not_flip_with_the_french_label() -> None:
    """The audit's failure: the same passage scored lower under 'french' (73) than 'middle_french' (102)."""
    passage = "Mout par furent les nocez granz quer ml't fu sagez et vaillans et a toute honeur entendans"
    raw = {lang: lexicon_trust.lexical_plausibility(passage, lang) for lang in ("french", "middle_french")}
    assert raw["french"] != raw["middle_french"]  # the raw profile scores do differ
    best = {lang: lexicon_trust.best_fit_plausibility(passage, lang) for lang in ("french", "middle_french")}
    assert best["french"] == best["middle_french"]  # what the gate reads does not
