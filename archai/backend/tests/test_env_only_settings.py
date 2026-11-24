"""Variables read with os.getenv must work from .env too, as .env.example says.

ARCHAI_DB_PATH, ARCHAI_WIKIDATA_CACHE, ARCHAI_DEBUG_RAG and
ARCHAI_DISABLE_PROCESS_POOL are read with os.getenv where they are used, not
through Settings. Settings reads .env into its own fields only (and ignores
names it has no field for), so the four were ignored when set in .env, the
way .env.example and docs/deployment.md tell an operator to set them: a
server configured with ARCHAI_DB_PATH in .env kept its database inside the
code checkout.
"""

from __future__ import annotations

import os
import subprocess
import sys
from collections.abc import Iterator
from pathlib import Path

import pytest

from app import config

KEYS = ("ARCHAI_DB_PATH", "ARCHAI_WIKIDATA_CACHE", "ARCHAI_DEBUG_RAG", "ARCHAI_DISABLE_PROCESS_POOL")
BACKEND = Path(__file__).resolve().parents[1]


@pytest.fixture
def clean_env(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> Iterator[Path]:
    """The four keys unset and the working directory empty; both restored afterwards."""
    saved = {key: os.environ.get(key) for key in KEYS}
    for key in KEYS:
        os.environ.pop(key, None)
    monkeypatch.chdir(tmp_path)
    try:
        yield tmp_path
    finally:
        for key, value in saved.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value


def test_values_in_env_file_reach_the_code_that_reads_them(tmp_path: Path) -> None:
    """A fresh process started in a directory with a .env, as the server is."""
    db = tmp_path / "data" / "archai.sqlite"
    cache = tmp_path / "data" / "wikidata.sqlite"
    (tmp_path / ".env").write_text(
        f"ARCHAI_DB_PATH={db}\nARCHAI_WIKIDATA_CACHE={cache}\nARCHAI_DEBUG_RAG=1\nARCHAI_DISABLE_PROCESS_POOL=1\n"
    )
    probe = (
        "import os, app.config\n"
        "from app.db import pipeline_db\n"
        "from app.services import rag_debug, wikidata_client\n"
        "print(pipeline_db._db_config().path)\n"
        "print(wikidata_client._cache_path())\n"
        "print(rag_debug.is_rag_debug_enabled())\n"
        "print(os.environ.get('ARCHAI_DISABLE_PROCESS_POOL'))\n"
    )
    env = {key: value for key, value in os.environ.items() if key not in KEYS}
    env["PYTHONPATH"] = str(BACKEND)
    result = subprocess.run([sys.executable, "-c", probe], cwd=tmp_path, env=env, capture_output=True, text=True, timeout=120)

    assert result.returncode == 0, result.stderr
    assert result.stdout.splitlines() == [str(db), str(cache), "True", "1"]


def test_the_environment_wins_over_env_files_and_env_local_over_env(clean_env: Path) -> None:
    (clean_env / ".env").write_text("ARCHAI_DB_PATH=/from/env-file.sqlite\nARCHAI_WIKIDATA_CACHE=/from/env-file-cache.sqlite\n")
    (clean_env / ".env.local").write_text("ARCHAI_WIKIDATA_CACHE=/from/env-local-cache.sqlite\n")
    os.environ["ARCHAI_DB_PATH"] = "/from/process.sqlite"

    config.export_env_only_variables()

    assert os.environ["ARCHAI_DB_PATH"] == "/from/process.sqlite"
    assert os.environ["ARCHAI_WIKIDATA_CACHE"] == "/from/env-local-cache.sqlite"


def test_without_env_files_nothing_is_set(clean_env: Path) -> None:
    config.export_env_only_variables()

    assert all(key not in os.environ for key in KEYS)
