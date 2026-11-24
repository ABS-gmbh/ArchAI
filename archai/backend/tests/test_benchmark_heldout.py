"""The held-out benchmark's arithmetic, on rows written by hand.

scripts/benchmark_heldout.py reads line images from the CATMuS Medieval test
split through the Kraken models, which the test suite has neither of. Its
report is plain arithmetic over the rows that run writes, so that part is
checked here: micro CER, intervals resampled over manuscripts, rank
correlation, and the gate table on pseudo-pages.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

import benchmark_heldout as heldout  # noqa: E402


def _row(shelfmark: str, engine: str, errors: int, chars: int, **extra: object) -> dict:
    return {"shelfmark": shelfmark, "engine": engine, "errors": errors, "chars": chars, **extra}


def test_micro_cer_is_errors_over_characters_not_a_mean_of_line_rates() -> None:
    rows = [_row("A", "kraken_catmus", 1, 10), _row("A", "kraken_catmus", 0, 90)]
    assert heldout._micro(rows) == pytest.approx(0.01)
    assert heldout._micro([]) != heldout._micro([])  # no characters: not a number


def test_the_interval_is_resampled_over_manuscripts() -> None:
    # One manuscript read perfectly and one read badly: the interval must span
    # both, however many lines each has. Resampling lines would give a narrow one.
    rows_by_ms = {
        "good": [_row("good", "kraken_catmus", 0, 50) for _ in range(200)],
        "bad": [_row("bad", "kraken_catmus", 25, 50) for _ in range(200)],
    }
    low, high = heldout._boot(rows_by_ms, heldout._micro)
    assert low == pytest.approx(0.0) and high == pytest.approx(0.5)
    assert heldout._boot(rows_by_ms, heldout._micro) == (low, high)  # seeded


def test_rank_correlation_handles_ties_and_needs_three_points() -> None:
    assert heldout._spearman([1, 2, 3, 4], [10, 20, 30, 40]) == pytest.approx(1.0)
    assert heldout._spearman([1, 2, 3, 4], [4, 3, 2, 1]) == pytest.approx(-1.0)
    assert heldout._ranks([5, 1, 5, 3]) == [3.5, 1.0, 3.5, 2.0]
    assert heldout._spearman([1, 2], [2, 1]) is None
    assert heldout._spearman([1, 1, 1], [1, 2, 3]) is None  # no variation to rank


def test_the_report_scores_each_block_as_one_page(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    latin = [
        "In principio erat verbum et verbum erat apud deum",
        "et deus erat verbum hoc erat in principio apud deum",
        "omnia per ipsum facta sunt et sine ipso factum est nihil",
        "quod factum est in ipso vita erat et vita erat lux hominum",
    ]
    rows = []
    for shelfmark in ("MS A", "MS B"):
        for index, reference in enumerate(latin * 3):
            for engine in heldout.ENGINES:
                text = reference if engine == "kraken_catmus" else reference[::-1]
                rows.append(
                    {
                        "line_id": f"{shelfmark}.parquet#{index}",
                        "block": f"{shelfmark}#0",
                        "language": "Latin",
                        "century": 12,
                        "script_type": "Caroline",
                        "shelfmark": shelfmark,
                        "engine": engine,
                        "reference": reference,
                        "text": text,
                        "confidence": 0.99 if engine == "kraken_catmus" else 0.5 + index / 100,
                        "errors": 0 if engine == "kraken_catmus" else len(reference) // 2,
                        "chars": len(reference),
                        "error": None,
                    }
                )
    lines = tmp_path / "lines.jsonl"
    lines.write_text("\n".join(json.dumps(row) for row in rows) + "\n", encoding="utf-8")
    out = tmp_path / "report.json"

    heldout.report(lines, out)

    result = json.loads(out.read_text(encoding="utf-8"))
    assert result["lines"] == 24 and result["manuscripts"] == 2
    cell = result["cer"]["all"]["all"]
    assert cell["kraken_catmus"]["cer"] == 0 and cell["kraken_catmus"]["manuscripts"] == 2
    assert cell["kraken_mccatmus"]["minus_catmus"]["mean"] == pytest.approx(cell["kraken_mccatmus"]["cer"])
    # Two manuscripts, one block each, three recognisers: six pseudo-pages.
    assert len(result["pages"]) == 6
    perfect = [page for page in result["pages"] if page["engine"] == "kraken_catmus"]
    assert all(page["cer"] == 0 and page["lines"] == 12 for page in perfect)
    assert "Quality gate on pseudo-pages" in capsys.readouterr().out
