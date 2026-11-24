"""Run retention (audit DB-07): opt-in, conservative, and only ever on temp databases here."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import pytest

from app.db import pipeline_db
from app.services import run_retention

NOW = datetime(2026, 11, 23, 12, 0, tzinfo=timezone.utc)
CUTOFF = (NOW - timedelta(days=30)).isoformat()


@pytest.fixture(autouse=True)
def fresh_db(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("ARCHAI_DB_PATH", str(tmp_path / "archai.sqlite"))
    monkeypatch.setattr(pipeline_db, "_DB_READY", False)


def _run(page: str, *, days_old: float, status: str = "COMPLETED") -> str:
    run_id = pipeline_db.create_run(asset_ref=page, asset_sha256=f"sha-{page}")
    stamp = (NOW - timedelta(days=days_old)).isoformat()
    with pipeline_db._connect() as conn:
        conn.execute(
            "UPDATE pipeline_runs SET status=?, created_at=?, updated_at=? WHERE run_id=?",
            (status, stamp, stamp, run_id),
        )
        conn.commit()
    return run_id


def _plan(**kwargs: Any) -> run_retention.RetentionPlan:
    return run_retention.plan_retention(30, now=NOW, protected=kwargs.pop("protected", set()), **kwargs)


def test_only_an_old_finished_superseded_run_expires() -> None:
    superseded = _run("p1", days_old=90)
    newest = _run("p1", days_old=60)
    recent = _run("p2", days_old=1)
    plan = _plan()
    assert plan.expired == [superseded]
    assert plan.kept["newest_of_page"] == [newest]
    assert recent in plan.kept["recent"]


def _make_searchable(run_id: str) -> None:
    pipeline_db.insert_chunks(
        run_id, [{"chunk_id": f"{run_id}-c", "idx": 0, "start_offset": 0, "end_offset": 6, "text": "Arthur"}]
    )
    with pipeline_db._connect() as conn:
        conn.execute("UPDATE pipeline_runs SET search_allowed=1 WHERE run_id=?", (run_id,))
        conn.commit()


def test_a_failed_newest_run_does_not_cost_the_page_its_transcription() -> None:
    """The newest run was kept by creation time alone; the completed one it superseded expired."""
    completed = _run("p1", days_old=90)
    _make_searchable(completed)
    failed = _run("p1", days_old=60, status="FAILED")
    plan = _plan()
    assert plan.expired == []
    assert plan.kept["newest_of_page"] == [failed]
    assert plan.kept["newest_completed_of_page"] == [completed]


def test_only_runs_the_page_does_not_rely_on_expire() -> None:
    oldest = _run("p1", days_old=120)
    newer = _run("p1", days_old=90)
    newest = _run("p1", days_old=60, status="FAILED_QUALITY")
    plan = _plan()
    assert plan.expired == [oldest]
    assert plan.kept["newest_completed_of_page"] == [newer]
    assert plan.kept["newest_of_page"] == [newest]


def test_the_run_search_serves_is_kept(monkeypatch: pytest.MonkeyPatch) -> None:
    served = _run("p1", days_old=120)
    unused = _run("p1", days_old=90)
    _run("p1", days_old=60)
    monkeypatch.setattr(pipeline_db, "searchable_runs", lambda: [{"run_id": served}])
    plan = _plan()
    assert plan.kept["searchable"] == [served] and plan.expired == [unused]


def test_runs_without_a_page_identity_are_kept() -> None:
    """With neither hash nor asset_ref they all shared one page key, and all but one expired."""
    anonymous = []
    for days_old in (120, 90, 60):
        run_id = pipeline_db.create_run(asset_ref="", asset_sha256=None)
        stamp = (NOW - timedelta(days=days_old)).isoformat()
        with pipeline_db._connect() as conn:
            conn.execute(
                "UPDATE pipeline_runs SET status='COMPLETED', created_at=?, updated_at=? WHERE run_id=?",
                (stamp, stamp, run_id),
            )
            conn.commit()
        anonymous.append(run_id)
    plan = _plan()
    assert plan.expired == []
    assert sorted(plan.kept["no_page_key"]) == sorted(anonymous)


def test_a_run_updated_after_planning_is_not_deleted(monkeypatch: pytest.MonkeyPatch) -> None:
    from app.services import rag_store

    monkeypatch.setattr(rag_store, "delete_run", lambda run_id: 0)
    old = _run("p1", days_old=90)
    _run("p1", days_old=60)
    plan = _plan()
    assert plan.expired == [old]
    with pipeline_db._connect() as conn:
        conn.execute("UPDATE pipeline_runs SET updated_at=? WHERE run_id=?", (NOW.isoformat(), old))
        conn.commit()
    assert run_retention.delete_runs(plan.expired, cutoff=plan.cutoff) == {"runs": 0, "vectors": 0}
    assert pipeline_db.get_run(old) is not None


def test_a_run_in_progress_is_never_expired() -> None:
    stuck = _run("p1", days_old=400, status="RUNNING")
    _run("p1", days_old=1)
    assert _plan().kept["not_finished"] == [stuck]


def test_failed_runs_expire_like_completed_ones() -> None:
    failed = _run("p1", days_old=90, status="FAILED_QUALITY")
    _run("p1", days_old=60)
    assert _plan().expired == [failed]


def test_a_benchmarked_run_is_kept() -> None:
    old = _run("p1", days_old=90)
    _run("p1", days_old=60)
    pipeline_db.insert_ocr_benchmark_reference(old, page_id="p1", source_label="gold", reference_text="text")
    plan = _plan()
    assert plan.expired == [] and plan.kept["benchmark_reference"] == [old]


def test_a_showcase_run_is_kept() -> None:
    old = _run("p1", days_old=90)
    _run("p1", days_old=60)
    plan = _plan(protected={old})
    assert plan.expired == [] and plan.kept["thesis_showcase"] == [old]


def test_the_showcase_bundle_ids_are_found(tmp_path: Path) -> None:
    bundle = tmp_path / "bundle"
    bundle.mkdir()
    (bundle / "m.json").write_text('{"run_id": "c10d2520-51f2-4321-a022-1fbb1cd57dda"}', encoding="utf-8")
    assert run_retention.showcase_run_ids(bundle) == {"c10d2520-51f2-4321-a022-1fbb1cd57dda"}
    assert run_retention.showcase_run_ids(tmp_path / "absent") == set()


def test_a_window_under_a_day_is_refused() -> None:
    with pytest.raises(ValueError, match="at least 1"):
        run_retention.plan_retention(0, now=NOW, protected=set())


def test_deleting_a_run_removes_what_hangs_off_it_and_nothing_else(monkeypatch: pytest.MonkeyPatch) -> None:
    gone, kept = _run("p1", days_old=90), _run("p1", days_old=60)
    for run_id in (gone, kept):
        pipeline_db.insert_chunks(
            run_id, [{"chunk_id": f"{run_id}-c", "idx": 0, "start_offset": 0, "end_offset": 6, "text": "Arthur"}]
        )
        pipeline_db.upsert_evidence_span({"run_id": run_id, "chunk_id": f"{run_id}-c", "raw_text": "Arthur"})
    vector_calls: list[str] = []
    from app.services import rag_store

    monkeypatch.setattr(rag_store, "delete_run", lambda run_id: vector_calls.append(run_id) or 3)

    assert run_retention.delete_runs([gone], cutoff=CUTOFF) == {"runs": 1, "vectors": 3}
    assert vector_calls == [gone]
    assert pipeline_db.get_run(gone) is None and pipeline_db.list_chunks(gone) == []
    assert pipeline_db.get_run(kept) is not None and len(pipeline_db.list_chunks(kept)) == 1
    with pipeline_db._connect() as conn:
        spans = {row[0] for row in conn.execute("SELECT run_id FROM evidence_spans")}
    assert spans == {kept}


def test_a_run_whose_vectors_cannot_be_deleted_is_kept(monkeypatch: pytest.MonkeyPatch) -> None:
    run_id = _run("p1", days_old=90)
    from app.services import rag_store

    def broken(run_id: str) -> int:
        raise RuntimeError("chroma unavailable")

    monkeypatch.setattr(rag_store, "delete_run", broken)
    assert run_retention.delete_runs([run_id], cutoff=CUTOFF) == {"runs": 0, "vectors": 0}
    assert pipeline_db.get_run(run_id) is not None


def _span(run_id: str, source: str | None) -> None:
    pipeline_db.upsert_evidence_span(
        {"run_id": run_id, "raw_text": f"span {source}", "meta_json": {"source": source} if source else None}
    )


def test_an_annotated_run_is_kept_unless_asked() -> None:
    annotated = _run("p1", days_old=90)
    _run("p1", days_old=60)
    _span(annotated, None)
    assert _plan().kept["annotated"] == [annotated]
    assert _plan(include_annotated=True).expired == [annotated]


def test_spans_the_pipeline_wrote_are_not_annotations() -> None:
    old = _run("p1", days_old=90)
    _run("p1", days_old=60)
    for source in run_retention.PIPELINE_SPAN_SOURCES:
        _span(old, source)
    assert _plan().expired == [old]


def _prune_runs() -> Any:
    import importlib.util
    import sys

    path = Path(__file__).resolve().parents[1] / "scripts" / "prune_runs.py"
    spec = importlib.util.spec_from_file_location("prune_runs", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules.setdefault("prune_runs", module)
    spec.loader.exec_module(module)
    return module


def _ancient_superseded_run() -> str:
    """A run old enough to expire whenever the suite runs."""
    old = _run("p1", days_old=4000)
    _run("p1", days_old=3990)
    return old


def test_the_script_reports_without_deleting(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    old = _ancient_superseded_run()
    db = tmp_path / "archai.sqlite"
    assert _prune_runs().main(["--db", str(db), "--older-than-days", "30"]) == 0
    assert "expired: 1" in capsys.readouterr().out
    assert pipeline_db.get_run(old) is not None


def test_the_script_deletes_with_apply(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from app.services import rag_store

    monkeypatch.setattr(rag_store, "delete_run", lambda run_id: 0)
    old = _ancient_superseded_run()
    assert _prune_runs().main(["--db", str(tmp_path / "archai.sqlite"), "--older-than-days", "30", "--apply"]) == 0
    assert pipeline_db.get_run(old) is None


def test_the_script_requires_a_window() -> None:
    with pytest.raises(SystemExit) as exit_info:
        _prune_runs().main(["--apply"])
    assert exit_info.value.code == 2


# ───────────────────────────────────────────── the scheduled sweep ──


@pytest.fixture
def no_vectors(monkeypatch: pytest.MonkeyPatch) -> None:
    from app.services import rag_store

    monkeypatch.setattr(rag_store, "delete_run", lambda run_id: 0)


def test_the_default_configuration_deletes_nothing(no_vectors: None) -> None:
    from app.config import Settings

    assert Settings(_env_file=None).run_retention_days == 0
    old = _ancient_superseded_run()
    sweeper = run_retention.RetentionSweeper()
    assert sweeper.sweep(now=NOW) is None
    assert sweeper.sweep(now=NOW) is None
    assert pipeline_db.get_run(old) is not None


def test_the_first_sweep_only_previews(
    no_vectors: None, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    from app.config import settings

    monkeypatch.setattr(settings, "run_retention_days", 30)
    monkeypatch.setattr(run_retention, "showcase_run_ids", lambda: set())
    old = _run("p1", days_old=90)
    _run("p1", days_old=60)
    sweeper = run_retention.RetentionSweeper()
    with caplog.at_level("INFO"):
        first = sweeper.sweep(now=NOW)
        assert first is not None and first["runs"] == 0 and first["previewed"] == 1
        assert pipeline_db.get_run(old) is not None
        assert f"run {old} (page sha-p1, last updated 90 days ago)" in caplog.text
        second = sweeper.sweep(now=NOW)
    assert second is not None and second["runs"] == 1
    assert pipeline_db.get_run(old) is None
    assert f"Retention: deleted run {old}" in caplog.text
    assert "Retention sweep: deleted 1 runs" in caplog.text


def test_a_sweep_deletes_at_most_the_cap_oldest_first(no_vectors: None, monkeypatch: pytest.MonkeyPatch) -> None:
    from app.config import settings

    monkeypatch.setattr(settings, "run_retention_days", 30)
    monkeypatch.setattr(settings, "run_retention_max_per_sweep", 2)
    monkeypatch.setattr(run_retention, "showcase_run_ids", lambda: set())
    oldest, older, old = _run("p1", days_old=300), _run("p1", days_old=200), _run("p1", days_old=100)
    _run("p1", days_old=1)
    sweeper = run_retention.RetentionSweeper()
    sweeper.sweep(now=NOW)
    assert sweeper.sweep(now=NOW)["runs"] == 2
    assert [pipeline_db.get_run(r) is None for r in (oldest, older, old)] == [True, True, False]


def test_the_sweep_never_deletes_an_annotated_run(no_vectors: None, monkeypatch: pytest.MonkeyPatch) -> None:
    from app.config import settings

    monkeypatch.setattr(settings, "run_retention_days", 30)
    monkeypatch.setattr(run_retention, "showcase_run_ids", lambda: set())
    annotated = _run("p1", days_old=90)
    _run("p1", days_old=60)
    _span(annotated, None)
    sweeper = run_retention.RetentionSweeper()
    sweeper.sweep(now=NOW)
    sweeper.sweep(now=NOW)
    assert pipeline_db.get_run(annotated) is not None


@pytest.mark.parametrize(
    ("name", "value", "message"),
    [
        ("RUN_RETENTION_DAYS", "-1", "RUN_RETENTION_DAYS must be 0"),
        ("RUN_RETENTION_MAX_PER_SWEEP", "0", "RUN_RETENTION_MAX_PER_SWEEP must be at least 1"),
    ],
)
def test_bad_retention_settings_are_refused_by_name(
    monkeypatch: pytest.MonkeyPatch, name: str, value: str, message: str
) -> None:
    from pydantic import ValidationError

    from app.config import Settings

    monkeypatch.setenv(name, value)
    with pytest.raises(ValidationError, match=message):
        Settings(_env_file=None)


def test_startup_runs_the_preview_sweep(monkeypatch: pytest.MonkeyPatch) -> None:
    """The sweep is wired into the lifespan; with the default setting it does nothing."""
    import asyncio

    import app.main as main

    calls: list[str] = []

    class Sweeper:
        def sweep(self) -> None:
            calls.append("sweep")

    monkeypatch.setattr(main, "init_model_pool", lambda: None)
    monkeypatch.setattr(main, "shutdown_model_pool", lambda: None)
    monkeypatch.setattr(main, "cleanup_expired_tasks", lambda: {})
    monkeypatch.setattr(main, "configure_logging", lambda *args, **kwargs: None)
    monkeypatch.setattr(main, "RetentionSweeper", Sweeper)

    async def start_and_stop() -> None:
        async with main.lifespan(main.app):
            pass

    asyncio.run(start_and_stop())
    assert calls == ["sweep"]
