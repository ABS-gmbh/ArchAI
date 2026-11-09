"""Authority cache calls close their SQLite connections (audit AL-11).

_cache_conn returned a bare connection used as a context manager, which
commits but does not close: a run left hundreds of connections and file
descriptors open until the garbage collector reached them (measured: 6 open
descriptors grew to 59 over 1,000 calls).
"""

from __future__ import annotations

import gc
import os

from app.services import wikidata_client


def _open_descriptors() -> int:
    return len(os.listdir("/dev/fd"))


def test_cache_calls_do_not_leave_connections_open():
    wikidata_client.cache_put("wikidata", "leak check", [{"qid": "Q1"}])
    gc.collect()
    gc.disable()
    try:
        before = _open_descriptors()
        for _ in range(200):
            wikidata_client.cache_get("wikidata", "leak check")
            wikidata_client.cache_check("wikidata", "leak check")
        after = _open_descriptors()
    finally:
        gc.enable()
    assert after - before <= 2


def test_the_cache_still_reads_back_what_it_stored():
    wikidata_client.cache_put("wikidata", "round trip", [{"qid": "Q45720", "label": "Arthur"}])
    assert wikidata_client.cache_get("wikidata", "round trip") == [{"qid": "Q45720", "label": "Arthur"}]
