"""Tests for the OCR and retrieval evaluation harness.

The edit distance is checked against a plain dynamic-programming reference on
random inputs, because a vectorised recurrence that is wrong by one on some
inputs would silently shift every CER this module ever reports.
"""

from __future__ import annotations

import importlib.util
import json
import math
import random
from pathlib import Path

import pytest

from app.services import evaluation
from app.services.evaluation import (
    DIPLOMATIC,
    LENIENT,
    NormalizationPolicy,
    RetrievalQuery,
    align,
    alignment_offsets,
    bootstrap_ci,
    cer,
    edit_distance,
    evaluate_corpus,
    evaluate_retrieval,
    ndcg_at_k,
    normalize_for_eval,
    precision_at_k,
    recall_at_k,
    reciprocal_rank,
    wer,
)

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "evaluate_ocr.py"


def _naive_distance(a, b) -> int:
    previous = list(range(len(b) + 1))
    for i, x in enumerate(a, start=1):
        current = [i]
        for j, y in enumerate(b, start=1):
            current.append(min(previous[j] + 1, current[j - 1] + 1, previous[j - 1] + (x != y)))
        previous = current
    return previous[-1]


def _random_pairs(count: int, seed: int = 7):
    rng = random.Random(seed)
    for _ in range(count):
        a = "".join(rng.choice("abcd ") for _ in range(rng.randrange(0, 30)))
        b = "".join(rng.choice("abcd ") for _ in range(rng.randrange(0, 30)))
        yield a, b


# ── edit distance ───────────────────────────────────────────────────────


@pytest.mark.parametrize(
    ("ref", "hyp", "expected"),
    [
        ("kitten", "sitting", 3),
        ("", "", 0),
        ("", "abc", 3),
        ("abc", "", 3),
        ("abc", "abc", 0),
        ("a", "b", 1),
        ("flaw", "lawn", 2),
    ],
)
def test_edit_distance_known_values(ref: str, hyp: str, expected: int) -> None:
    assert edit_distance(ref, hyp) == expected
    assert align(ref, hyp).distance == expected


def test_vectorised_distance_matches_reference_implementation() -> None:
    for a, b in _random_pairs(400):
        expected = _naive_distance(a, b)
        assert edit_distance(a, b) == expected, (a, b)
        assert align(a, b).distance == expected, (a, b)


def test_breakdown_accounts_for_every_edit() -> None:
    """S + D + I is the distance, and applying the edits yields the hypothesis length."""
    for a, b in _random_pairs(400, seed=11):
        counts = align(a, b)
        assert counts.substitutions + counts.deletions + counts.insertions == counts.distance
        assert len(a) - counts.deletions + counts.insertions == len(b)


def test_breakdown_prefers_substitution_on_ties() -> None:
    counts = align("kitten", "sitting")
    assert (counts.substitutions, counts.deletions, counts.insertions) == (2, 0, 1)


def test_breakdown_for_empty_sides() -> None:
    assert (align("", "abc").insertions, align("", "abc").deletions) == (3, 0)
    assert (align("abc", "").insertions, align("abc", "").deletions) == (0, 3)


def test_works_on_word_tokens() -> None:
    assert edit_distance("in principio erat".split(), "in principio est".split()) == 1


def test_oversized_alignment_keeps_exact_distance(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(evaluation, "_MAX_ALIGNMENT_CELLS", 10)
    counts = align("kitten", "sitting")
    assert counts.distance == 3
    assert not counts.has_breakdown


# ── offset alignment ────────────────────────────────────────────────────


@pytest.mark.parametrize(
    ("ref", "hyp", "expected"),
    [
        ("abc", "abc", [0, 1, 2, 3]),
        ("abc", "aXbc", [0, 2, 3, 4]),  # the offset after an insertion maps past it
        ("abc", "ac", [0, 1, 1, 2]),
        ("", "xy", [2]),
        ("ab", "", [0, 0, 0]),
    ],
)
def test_alignment_offsets_known_cases(ref: str, hyp: str, expected: list[int]) -> None:
    assert alignment_offsets(ref, hyp).tolist() == expected


def test_a_span_projects_onto_another_reading() -> None:
    ref, hyp = "the kyng of fraunce", "the king of france"
    offsets = alignment_offsets(ref, hyp)
    start = ref.index("kyng")
    assert hyp[offsets[start] : offsets[start + 4]] == "king"
    start = ref.index("fraunce")
    assert hyp[offsets[start] : offsets[len(ref)]] == "france"


def test_alignment_offsets_are_monotone_and_bounded() -> None:
    for a, b in _random_pairs(300, seed=13):
        offsets = alignment_offsets(a, b).tolist()
        assert offsets == sorted(offsets)
        assert offsets[-1] == len(b)
        assert all(0 <= value <= len(b) for value in offsets)


def test_alignment_offsets_refuse_oversized_inputs(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(evaluation, "_MAX_ALIGNMENT_CELLS", 10)
    with pytest.raises(ValueError):
        alignment_offsets("kitten", "sitting")


# ── rates ───────────────────────────────────────────────────────────────


def test_cer_can_exceed_one_when_the_hypothesis_hallucinates() -> None:
    assert cer("ab", "abcdefgh").rate == pytest.approx(3.0)


def test_empty_reference_rates() -> None:
    assert cer("", "").rate == 0.0
    assert math.isinf(cer("", "text").rate)


def test_wer_counts_whole_words() -> None:
    assert wer("in principio erat verbum", "in principio erat uerbum").rate == pytest.approx(0.25)


# ── normalisation ───────────────────────────────────────────────────────


def test_unicode_composition_is_not_an_error() -> None:
    composed, decomposed = "caf\u00e9", "cafe\u0301"
    assert cer(composed, decomposed).distance == 0
    raw = NormalizationPolicy(unicode_form=None)
    assert cer(composed, decomposed, raw).distance > 0


def test_whitespace_layout_is_not_an_error() -> None:
    assert cer("in  principio\n erat", "in principio erat").distance == 0


def test_diplomatic_policy_keeps_case_and_punctuation() -> None:
    assert cer("Dominus", "dominus").distance == 1
    assert cer("dominus.", "dominus").distance == 1


def test_lenient_policy_folds_case_and_punctuation() -> None:
    assert cer("Dominus, deus.", "dominus deus", LENIENT).distance == 0


def test_lenient_policy_keeps_abbreviation_marks_inside_words() -> None:
    """A combining macron is not punctuation; stripping it would split "dn̄s" in two."""
    text = "dn\u0304s"
    assert normalize_for_eval(text, LENIENT) == text
    assert wer(text, text, LENIENT).reference_length == 1


def test_lenient_policy_keeps_word_signs_for_et() -> None:
    assert normalize_for_eval("pater \u204a filius & spiritus", LENIENT).split() == [
        "pater",
        "\u204a",
        "filius",
        "&",
        "spiritus",
    ]


def test_policy_description_is_recorded() -> None:
    assert DIPLOMATIC.describe() == "NFC+collapse-whitespace"
    assert LENIENT.describe() == "NFC+casefold+strip-punctuation+collapse-whitespace"


# ── corpus aggregation ──────────────────────────────────────────────────


def _two_page_corpus():
    short = ("short", "a" * 10, "a" * 9 + "b")  # 1 error in 10
    long = ("long", "a" * 100, "a" * 50 + "b" * 50)  # 50 errors in 100
    return [short, long]


def test_micro_and_macro_answer_different_questions() -> None:
    report = evaluate_corpus(_two_page_corpus())
    assert report.cer.micro == pytest.approx(51 / 110)
    assert report.cer.macro == pytest.approx((0.1 + 0.5) / 2)


def test_worst_pages_are_ordered_by_cer() -> None:
    report = evaluate_corpus(_two_page_corpus())
    assert [p.page_id for p in report.worst_pages(2)] == ["long", "short"]


def test_corpus_breakdown_sums_pages() -> None:
    report = evaluate_corpus(_two_page_corpus())
    assert report.char_breakdown == {"substitutions": 51, "deletions": 0, "insertions": 0}


def test_single_page_has_no_interval() -> None:
    report = evaluate_corpus([("only", "abc", "abd")])
    assert all(math.isnan(v) for v in report.cer.micro_ci + report.cer.macro_ci)


def test_report_is_strict_json_even_with_non_finite_values() -> None:
    report = evaluate_corpus([("hallucinated", "", "text"), ("single", "abc", "abd")])
    payload = json.dumps(report.to_dict(), allow_nan=False)
    assert json.loads(payload)["per_page"][0]["cer"] is None


# ── bootstrap ───────────────────────────────────────────────────────────


def _mean(values):
    return sum(values) / len(values)


def test_bootstrap_is_reproducible_for_a_seed() -> None:
    values = [random.Random(3).random() for _ in range(40)]
    assert bootstrap_ci(values, _mean, seed=5) == bootstrap_ci(values, _mean, seed=5)


def test_bootstrap_interval_brackets_the_estimate() -> None:
    rng = random.Random(1)
    values = [rng.gauss(0.2, 0.05) for _ in range(60)]
    low, high = bootstrap_ci(values, _mean)
    assert low < _mean(values) < high


def test_bootstrap_interval_narrows_with_more_pages() -> None:
    rng = random.Random(2)
    values = [rng.gauss(0.2, 0.05) for _ in range(200)]
    few = bootstrap_ci(values[:10], _mean)
    many = bootstrap_ci(values, _mean)
    assert (many[1] - many[0]) < (few[1] - few[0])


def test_bootstrap_degenerate_inputs() -> None:
    assert all(math.isnan(v) for v in bootstrap_ci([0.3], _mean))
    assert all(math.isnan(v) for v in bootstrap_ci([], _mean))
    assert bootstrap_ci([0.3, 0.3, 0.3], _mean) == pytest.approx((0.3, 0.3))
    with pytest.raises(ValueError):
        bootstrap_ci([0.1, 0.2], _mean, confidence=1.0)


# ── retrieval metrics ───────────────────────────────────────────────────

RANKED = ["d1", "d2", "d3", "d4", "d5"]
RELEVANT = {"d2", "d4", "d9"}


def test_recall_at_k() -> None:
    assert recall_at_k(RANKED, RELEVANT, 1) == 0.0
    assert recall_at_k(RANKED, RELEVANT, 2) == pytest.approx(1 / 3)
    assert recall_at_k(RANKED, RELEVANT, 5) == pytest.approx(2 / 3)


def test_precision_divides_by_k_not_by_results_returned() -> None:
    assert precision_at_k(RANKED, RELEVANT, 5) == pytest.approx(2 / 5)
    assert precision_at_k(RANKED, RELEVANT, 10) == pytest.approx(2 / 10)


def test_reciprocal_rank() -> None:
    assert reciprocal_rank(RANKED, RELEVANT) == pytest.approx(0.5)
    assert reciprocal_rank(RANKED, {"absent"}) == 0.0


def test_ndcg_matches_hand_computation() -> None:
    gains = {item: 1.0 for item in RELEVANT}
    dcg = 1 / math.log2(3) + 1 / math.log2(5)
    ideal = 1 + 1 / math.log2(3) + 1 / math.log2(4)
    assert ndcg_at_k(RANKED, gains, 5) == pytest.approx(dcg / ideal)


def test_ndcg_rewards_putting_the_best_item_first() -> None:
    gains = {"a": 3.0, "b": 1.0}
    assert ndcg_at_k(["a", "b"], gains, 2) == pytest.approx(1.0)
    assert ndcg_at_k(["b", "a"], gains, 2) < 1.0


@pytest.mark.parametrize("metric", [recall_at_k, precision_at_k])
def test_non_positive_k_is_rejected(metric) -> None:
    with pytest.raises(ValueError):
        metric(RANKED, RELEVANT, 0)


def test_evaluate_retrieval_skips_unjudged_queries() -> None:
    queries = [
        RetrievalQuery("q1", ("a", "b"), frozenset({"a"})),
        RetrievalQuery("q2", ("a", "b"), frozenset({"b"})),
        RetrievalQuery("unjudged", ("a",), frozenset()),
    ]
    report = evaluate_retrieval(queries, ks=(1,))
    assert report["mrr"]["mean"] == pytest.approx(0.75)
    assert report["recall@1"]["mean"] == pytest.approx(0.5)
    assert report["mrr"]["n"] == 2
    assert report["_meta"] == {"queries": 3, "judged": 2, "confidence": 0.95}


def test_evaluate_retrieval_with_nothing_judged_is_not_a_zero_score() -> None:
    report = evaluate_retrieval([RetrievalQuery("q", ("a",), frozenset())], ks=(1,))
    assert math.isnan(report["mrr"]["mean"])
    assert report["mrr"]["n"] == 0


# ── command-line tool ───────────────────────────────────────────────────


@pytest.fixture(scope="module")
def cli():
    spec = importlib.util.spec_from_file_location("evaluate_ocr", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _write_corpus(root: Path) -> Path:
    (root / "gold").mkdir()
    (root / "runs").mkdir()
    lines = []
    for page, (reference, hypothesis) in {
        "001r": ("in principio erat verbum", "in principio erat uerbum"),
        "001v": ("et verbum erat apud deum", "et verbum erat apud deum"),
    }.items():
        (root / "gold" / f"{page}.gt.txt").write_text(reference, encoding="utf-8")
        (root / "runs" / f"{page}.txt").write_text(hypothesis, encoding="utf-8")
        record = {"page_id": page, "reference": f"gold/{page}.gt.txt", "hypothesis": f"runs/{page}.txt"}
        lines.append(json.dumps(record))
    manifest = root / "manifest.jsonl"
    manifest.write_text("# comment lines are ignored\n" + "\n".join(lines) + "\n", encoding="utf-8")
    return manifest


def test_cli_reports_a_corpus(cli, tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    manifest = _write_corpus(tmp_path)
    assert cli.main(["--manifest", str(manifest)]) == 0
    output = capsys.readouterr().out
    assert "pages scored     2" in output
    assert "CER" in output and "WER" in output


def test_cli_json_output_is_strict_json(cli, tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    manifest = _write_corpus(tmp_path)
    assert cli.main(["--manifest", str(manifest), "--json"]) == 0
    payload = json.loads(capsys.readouterr().out, parse_constant=lambda c: pytest.fail(f"non-JSON constant {c}"))
    assert payload["pages"] == 2
    assert payload["wer"]["micro"] == pytest.approx(1 / 9)  # one error in nine words


@pytest.mark.parametrize(
    "content",
    [
        "not json\n",
        json.dumps({"page_id": "p", "reference": "missing.txt"}) + "\n",
        json.dumps({"page_id": "p", "reference": "missing.txt", "hypothesis": "missing.txt"}) + "\n",
        "# only a comment\n",
    ],
)
def test_cli_rejects_bad_manifests(cli, tmp_path: Path, capsys: pytest.CaptureFixture[str], content: str) -> None:
    manifest = tmp_path / "manifest.jsonl"
    manifest.write_text(content, encoding="utf-8")
    assert cli.main(["--manifest", str(manifest)]) == 2
    assert capsys.readouterr().err.startswith("error:")


def test_cli_missing_manifest(cli, tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    assert cli.main(["--manifest", str(tmp_path / "absent.jsonl")]) == 2
    assert "manifest not found" in capsys.readouterr().err
