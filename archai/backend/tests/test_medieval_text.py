"""Tests for scribal abbreviation expansion and search-key construction.

The Kraken models here transcribe diplomatically, emitting the scribe's own
abbreviation signs. normalize_for_search's NFKD pass deleted combining marks, so
the macron carrying a suspension simply vanished and 2 of 21 medieval/modern
spelling pairs collapsed to a shared search key. A query could not match the
indexed text.
"""

from __future__ import annotations

import pytest

from app.services.medieval_text import (
    ABBREVIATION_LETTERS,
    _fold,
    WORD_SUSPENSIONS,
    build_search_key,
    expand_abbreviations,
    expand_abbreviations_without_nasals,
    rejoin_line_breaks,
)

# (diplomatic as recognised, modern form a reader would type)
SPELLING_PAIRS = [
    ("ꝓpter", "propter"), ("ꝑ manus", "per manus"), ("ꝗdam", "quidam"),
    ("ꝙ", "quod"), ("ꝯuersus", "conuersus"), ("eiꝰ", "eius"),
    ("⁊", "et"), ("ſanctus", "sanctus"), ("oīs", "omnis"),
    ("dñs", "dominus"), ("dñi", "domini"), ("ihſ", "iesus"),
    ("xpſ", "christus"), ("ſcs", "sanctus"), ("epſ", "episcopus"),
    ("nrā", "nostra"), ("ſpū", "spiritu"), ("Iohannes", "johannes"),
    ("uita", "vita"), ("cęlum", "celum"), ("præ", "pre"),
]


@pytest.mark.parametrize(("diplomatic", "modern"), SPELLING_PAIRS)
def test_diplomatic_and_modern_share_a_search_key(diplomatic: str, modern: str) -> None:
    assert build_search_key(diplomatic) == build_search_key(modern)


def test_every_abbreviation_letter_expands_to_letters_only() -> None:
    for sign, expansion in ABBREVIATION_LETTERS.items():
        assert expand_abbreviations(sign) == expansion
        assert sign not in build_search_key(sign)


def test_nasal_bar_becomes_m_before_a_labial() -> None:
    """A bar over a vowel is m before b/p/m, n elsewhere."""
    assert expand_abbreviations("cōpanus") == "companus"
    assert expand_abbreviations("cāpus") == "campus"
    assert expand_abbreviations("mōte") == "monte"


def test_nasal_bar_over_a_vowel_is_dropped_in_the_classic_reading() -> None:
    """The classic reading omits the nasal the bar stands for."""
    assert expand_abbreviations_without_nasals("ſpū") == "spu"
    assert expand_abbreviations_without_nasals("cōpanus") == "copanus"


def test_classic_reading_folds_to_the_suspension_key() -> None:
    """What matters is that folding the classic reading yields the table key.

    A bar over a CONSONANT (the n of "dñs") is not treated as a positional nasal,
    because that rule would also rewrite Spanish "ñ". The mark is then removed by
    the diacritic fold, so the lookup still lands on "dns".
    """
    assert _fold("dñs") == ["dns"]
    assert _fold(expand_abbreviations_without_nasals("dñs")) == ["dns"]
    assert build_search_key("dñs") == "dominus"


def test_word_suspensions_resolve_through_the_classic_reading() -> None:
    """dñs must reach 'dominus' even though the bar positionally gives 'dnns'."""
    for abbrev, expansion in WORD_SUSPENSIONS.items():
        assert build_search_key(abbrev) == build_search_key(expansion), abbrev


def test_tironian_et_is_expanded_not_dropped() -> None:
    assert build_search_key("pater ⁊ filius") == "pater et filius"


def test_long_s_folds_to_s() -> None:
    assert build_search_key("ſpiritus ſanctus") == "spiritus sanctus"


def test_medieval_orthography_is_folded() -> None:
    """Scribes did not distinguish i/j or u/v."""
    assert build_search_key("iohannes") == build_search_key("johannes")
    assert build_search_key("uita") == build_search_key("vita")
    assert build_search_key("æternum") == build_search_key("eternum")


def test_expansion_can_be_disabled() -> None:
    assert "per" not in build_search_key("ꝑ", expand=False)


def test_empty_input_is_safe() -> None:
    assert expand_abbreviations("") == ""
    assert build_search_key("") == ""
    assert rejoin_line_breaks("") == ""


def test_diplomatic_text_is_never_mutated() -> None:
    """The citable reading must survive; only a derived key is normalised."""
    original = "ꝓpter dñs ⁊ ſanctus"
    build_search_key(original)
    expand_abbreviations(original)
    assert original == "ꝓpter dñs ⁊ ſanctus"


# ─────────────────────────────────────────── line-break rejoining ──


def test_explicit_hyphen_break_is_rejoined() -> None:
    assert rejoin_line_breaks("sanc-\ntus") == "sanctus"


@pytest.mark.parametrize("mark", ["-", "‐", "‑", "–", "­", "¬"])
def test_all_break_marks_are_consumed(mark: str) -> None:
    assert rejoin_line_breaks(f"sanc{mark}\ntus") == "sanctus"


def test_unmarked_break_is_rejoined_only_with_lexicon_support() -> None:
    lexicon = frozenset({"sanctus", "spiritus"})
    # Without a lexicon an unmarked break is left alone: guessing would corrupt text.
    assert rejoin_line_breaks("sanc\ntus") == "sanc\ntus"
    # With one, the join is accepted because 'sanctus' is a word and neither half is.
    assert rejoin_line_breaks("sanc\ntus", lexicon=lexicon) == "sanctus"


def test_unmarked_break_between_two_real_words_is_not_joined() -> None:
    lexicon = frozenset({"pater", "noster", "paternoster"})
    assert rejoin_line_breaks("pater\nnoster", lexicon=lexicon) == "pater\nnoster"


def test_rejoining_preserves_unbroken_lines() -> None:
    text = "prima linea\nsecunda linea\ntertia linea"
    assert rejoin_line_breaks(text) == text


def test_single_line_is_returned_unchanged() -> None:
    assert rejoin_line_breaks("una linea") == "una linea"


def test_trailing_break_at_end_of_text_does_not_lose_the_fragment() -> None:
    assert rejoin_line_breaks("sanc-") == "sanc-"
    assert rejoin_line_breaks("prima\nsanc-") == "prima\nsanc"


def test_suspension_keys_survive_folding() -> None:
    """Every WORD_SUSPENSIONS key must be reachable after normalisation.

    build_search_key folds case, diacritics, digraphs and i/j-u/v before looking
    a token up. A key that changes under folding can never match: "noe" folds to
    "ne" via oe -> e, so it silently did nothing.
    """
    unreachable = {k: _fold(k) for k in WORD_SUSPENSIONS if _fold(k) != [k]}
    assert not unreachable, f"keys altered by folding: {unreachable}"


# Ordinary words in the languages this pipeline handles. A suspension key that is
# also a real word would silently rewrite genuine text.
REAL_WORDS = frozenset({
    "aut", "et", "in", "de", "ad", "non", "est", "cum", "per", "pro", "qui", "quod",
    "sed", "ut", "si", "ex", "sunt", "esse", "hic", "ille", "iam", "nunc", "post",
    "li", "le", "la", "les", "que", "qui", "ne", "pas", "mout", "rois", "chevalier",
})


def test_no_suspension_key_is_an_ordinary_word() -> None:
    """Regression: 'aut' means 'or' in Latin; expanding it to 'autem' corrupted text."""
    collisions = sorted(set(WORD_SUSPENSIONS) & REAL_WORDS)
    assert not collisions, f"suspension keys that are real words: {collisions}"


def test_ordinary_latin_survives_the_search_key_unchanged() -> None:
    """A sentence of common words must pass through with no rewriting."""
    sentence = "aut est in domo et non pro me"
    assert build_search_key(sentence) == sentence


def test_search_key_is_idempotent() -> None:
    """Re-normalising an already-normalised key must not change it."""
    for raw in ("ꝓpter dñs ⁊ ſanctus", "oīs cōpanus", "aut est in domo"):
        once = build_search_key(raw)
        assert build_search_key(once) == once, raw


# ─────────────────────── regressions found by adversarial review ──


def test_stray_mark_does_not_shift_later_token_lookups() -> None:
    """Critical regression.

    Two whole-string readings were paired by token index; a nasal mark on a
    non-Latin base produced a token in one reading and not the other, shifting
    every later lookup. "Iesus ᾱ dns ht" became "iesus dominus habet habet" -
    'dns' rewritten to 'habet' and the Greek token to 'dominus'.
    """
    assert build_search_key("Iesus ᾱ dns ht") == "iesus α dominus habet"
    assert build_search_key("x ̄ ht") == "x habet"
    assert build_search_key("ᾱ ee") == "α esse"


@pytest.mark.parametrize(
    "text",
    [
        "Ἀρχὴ σοφίας φόβος Κυρίου",
        "الحمد لله رب العالمين",
        "Въ начѧлѣ бѣ слово",
        "וַיֹּאמֶר אֱלֹהִים",
    ],
)
def test_non_latin_scripts_are_not_deleted(text: str) -> None:
    """An ASCII-only tokenizer erased whole scripts; the pipeline transcribes them."""
    assert build_search_key(text).strip()


def test_latin_and_greek_in_one_line_both_survive() -> None:
    key = build_search_key("Sicut dicit Aristoteles in libro Ηθικων")
    assert "sicut" in key and "ηθικων" in key


@pytest.mark.parametrize(
    ("text", "must_not_contain"),
    [("España", "espanna"), ("mañana", "mannana"), ("Nguyễn", "nguyenn")],
)
def test_generic_combining_marks_are_not_read_as_suspensions(text: str, must_not_contain: str) -> None:
    """U+0303/0304 also occur in Spanish, Portuguese, Vietnamese and Greek.

    Treating them as nasal bars turned "España" into "espanna".
    """
    assert build_search_key(text) != must_not_contain


def test_greek_long_vowel_macron_is_not_turned_into_a_latin_n() -> None:
    assert "n" not in expand_abbreviations("ᾱῑῡ")


@pytest.mark.parametrize("word", ["dñs", "ſpū", "nrā", "oīs", "cōpanus", "ꝓpter"])
def test_decomposed_and_precomposed_input_agree(word: str) -> None:
    """The Kraken models contain zero precomposed letters, so NFD is the norm."""
    import unicodedata as _ud

    assert build_search_key(word) == build_search_key(_ud.normalize("NFD", word))


def test_a_second_diacritic_does_not_flip_the_nasal_decision() -> None:
    """Regression: 'cōpanus' gave 'companus' but 'cō̈panus' gave 'conpanus'."""
    assert build_search_key("cō̈panus") == build_search_key("cōpanus")


def test_literal_sentinel_text_is_not_rewritten() -> None:
    """The in-band '~NASAL~' marker rewrote source text containing it."""
    assert expand_abbreviations("a~NASAL~b") == "a~NASAL~b"


def test_tironian_et_is_tokenized_despite_being_punctuation() -> None:
    """U+204A is category Po, so a word-character tokenizer dropped it."""
    assert build_search_key("pater ⁊ filius") == "pater et filius"


def test_modifier_letter_signs_are_expanded() -> None:
    """U+A770 is named 'MODIFIER LETTER US' with no 'LATIN', so a name test missed it."""
    assert build_search_key("eiꝰ") == "eius"


def test_no_abbreviation_sign_survives_into_a_search_key() -> None:
    survivors = {sign for sign in ABBREVIATION_LETTERS if sign in build_search_key(sign)}
    assert not survivors, f"signs passed through unexpanded: {survivors}"
