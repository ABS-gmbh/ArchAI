"""Medieval Spanish and Middle Dutch get their own lexical profiles in the quality gate.

Spanish was scored against the Latin profile, and Dutch had none, so Dutch text
got no lexical check at all. The passages below were written for this test in
the CATMuS conventions (graphematic, abbreviations unexpanded); they are not
taken from the held-out benchmark's lines.
"""

from __future__ import annotations

import importlib.util
import json
import random
import sys
from pathlib import Path

import pytest

BACKEND = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BACKEND / "scripts"))

import calibrate_quality_gate as gate  # noqa: E402
from app.services import lexicon_trust  # noqa: E402
from app.services.ocr_quality_config import LEXICAL_PLAUSIBILITY_FLOOR  # noqa: E402

SPANISH = """y luego el fijo mayor llego ayna muy ayrado a su hermano
y dixole hermano yo vos ruego mucho q̃ me fagades ayuda
ca muy grand cuyta me ha llegado agora desta guerra
y el hermano le dixo yo lo fare de grado mucho ayna
y luego cabalgo con su gente y llego a vna aldea pequeña
y fallo y muchos labradores q̃ estauan llorando mucho
por q̃ les auian llevado sus bueyes y sus ganados
y el les dixo no ayades miedo q̃ yo vos lo fare cobrar
y fue en pos de aquellos q̃ lo llevauan y alcançolos
y luchó con ellos y tomoles todo lo q̃ llevauan
y tornolo a los labradores y ellos le besaron las manos
y le dixieron señor dios vos lo gradezca q̃ mucho
bien nos auedes fecho oy y mientra biuamos lo diremos
y el fijo mayor quando lo sopo plogole mucho y dixo
bien paresçe q̃ eres mi hermano y fijo de mi padre
y desde aquel dia en adelante fueron muy amigos
y ayudaronse el vno al otro en todas sus cuytas
y nunca ouo entrellos desabenencia ninguna
fasta q̃ murieron y fueron enterrados juntos
en vna yglesia pequeña cerca de aquella aldea"""

DUTCH = """Doe die coninc dat vernam dat sine viande quamen
so sende hi boden in al sijn lant ende ontboet
sinen ridders ende sinen mannen dat si quamen
met haren wapenen ende met haren perden te hove
ende si quamen alle ghereet ende wel ghewapent
ende die coninc sprac te hem ende seide ghi heren
ghi hebt wel vernomen hoe dat mine viande comen
in mijn lant ende willen mi nemen mijn erve
ende mine steden ende mine borghe die ic hebbe
nu biddic u allen dat ghi mi helpt alse goede
ridders ende ghetrouwe manne sculdich sijn te doene
doe antwoerden si alle met eenre stemmen ende seiden
here wi selen u helpen also langhe als wi leven
ende wi selen met u varen waer dat ghi wilt
doe was die coninc blide ende dancte hem seere
ende hi gaf hem cleder ende ghelt ende goede perde
ende des anderen daghes voeren si ute der stede
met groten ghedrange ende met vele volcs
ende si reden so langhe dat si quamen bi den viande
ende daer was een groot strijt ende vele doder lieden"""


def _reversed(text: str) -> str:
    return "\n".join(" ".join(word[::-1] for word in line.split()) for line in text.splitlines())


def _random_letters(text: str, seed: int = 3) -> str:
    rng = random.Random(seed)
    letters = "abcdefghilmnopqrstuvxyz"
    return "\n".join(" ".join("".join(rng.choice(letters) for _ in word) for word in line.split()) for line in text.splitlines())


def _builder():
    spec = importlib.util.spec_from_file_location("build_trigram_profiles", BACKEND / "scripts" / "build_trigram_profiles.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_the_builder_takes_the_most_frequent_gate_trigrams() -> None:
    # casa casa mesa: cas 2, asa 2, mes 1, esa 1; words under three letters are not scored.
    assert _builder().profile_from_texts(["casa casa mesa", "de la"], top=2) == ["asa", "cas"]


@pytest.mark.parametrize("language", ["spanish", "dutch"])
def test_the_committed_profiles_are_the_generated_ones(language: str) -> None:
    generated = json.loads((BACKEND / "eval" / "quality_gate" / "trigram_profiles.json").read_text(encoding="utf-8"))
    assert lexicon_trust._TRIGRAM_PROFILES[language] == frozenset(generated["profiles"][language]["trigrams"])
    assert len(lexicon_trust._TRIGRAM_PROFILES[language]) == generated["top"] == 100


def test_spanish_is_scored_against_its_own_profile_and_allowed(monkeypatch: pytest.MonkeyPatch) -> None:
    # Against the Latin profile, which Spanish used to share, the passage's hit
    # rate is under the floor the gate blocks at; against its own profile it is
    # far above it.
    assert lexicon_trust.lexical_plausibility(SPANISH, "latin") < LEXICAL_PLAUSIBILITY_FLOOR
    assert lexicon_trust.lexical_plausibility(SPANISH, "spanish") > 2 * LEXICAL_PLAUSIBILITY_FLOOR
    verdict = gate.verdict(SPANISH)
    assert verdict.language == "spanish" and verdict.allowed


@pytest.mark.parametrize("text", [DUTCH, SPANISH], ids=["dutch", "spanish"])
def test_reversed_and_random_letter_text_is_blocked(text: str) -> None:
    assert gate.verdict(text).allowed
    assert not gate.verdict(_reversed(text)).allowed
    assert not gate.verdict(_random_letters(text)).allowed


def test_dutch_text_now_gets_a_lexical_check(monkeypatch: pytest.MonkeyPatch) -> None:
    verdict = gate.verdict(DUTCH)
    assert verdict.language == "dutch" and verdict.allowed and verdict.contrast > 0  # measured, not -1
    # Without the profile, Dutch was not measured at all and its reversed text passed.
    monkeypatch.delitem(lexicon_trust._TRIGRAM_PROFILES, "dutch")
    assert gate.verdict(_reversed(DUTCH)).allowed
