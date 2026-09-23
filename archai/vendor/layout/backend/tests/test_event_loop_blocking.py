"""Guards against putting blocking authority-linking work on the event loop.

_run_authority_linking_stage is synchronous: urllib with 10-15s timeouts per
request, plus a 0.25s time.sleep rate limiter between them, under a threading
lock. Called directly from an async function it blocks the single event loop for
the whole stage, so every other request on the process is frozen - the trace
endpoint, the SSE stream, /health - not just the run doing the linking.

Measured with a 50ms heartbeat over a stage of 8 mentions at an optimistic 0.4s
per response: worst event-loop lag 5.233s before, 0.004s after. Real timeouts are
10-15s, so an unreachable Wikidata is far worse than this.

One call site is legitimately synchronous: _run_post_ocr_pipeline_for_glm is a
plain def already dispatched through asyncio.to_thread by its callers.
"""

from __future__ import annotations

import ast
import asyncio
import time
from pathlib import Path

import pytest

ROUTER = Path(__file__).resolve().parents[1] / "app" / "routers" / "ocr.py"
SOURCE = ROUTER.read_text()
TREE = ast.parse(SOURCE)


def enclosing_function(line: int):
    best = None
    for node in ast.walk(TREE):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            if node.lineno <= line <= node.end_lineno:
                if best is None or node.lineno > best.lineno:
                    best = node
    return best


def linking_call_lines() -> list[int]:
    """Every line that invokes the linking stage, direct or via to_thread.

    Matching only the direct form "_run_authority_linking_stage(run_id)" would
    miss the dispatched form "to_thread(_run_authority_linking_stage, run_id)",
    so the check below would silently skip exactly the sites it exists to guard.
    """
    return [
        i + 1
        for i, text in enumerate(SOURCE.splitlines())
        if "_run_authority_linking_stage" in text
        and "linking_result =" in text
        and "def " not in text
    ]


def test_the_call_sites_are_still_where_we_think() -> None:
    assert len(linking_call_lines()) == 3


@pytest.mark.parametrize("line", linking_call_lines())
def test_async_callers_dispatch_the_blocking_stage_to_a_thread(line: int) -> None:
    """The invariant: inside `async def`, this call must not be direct."""
    function = enclosing_function(line)
    statement = SOURCE.splitlines()[line - 1]
    if isinstance(function, ast.AsyncFunctionDef):
        assert "asyncio.to_thread" in statement, (
            f"{function.name} is async but calls the blocking linking stage directly "
            f"at line {line}; wrap it in await asyncio.to_thread(...)"
        )
        assert statement.strip().startswith("linking_result = await")
    else:
        # A plain def may call it directly; its own callers are responsible for
        # keeping it off the loop.
        assert "asyncio.to_thread" not in statement


def test_the_synchronous_call_site_is_reached_only_through_a_thread() -> None:
    """_run_post_ocr_pipeline_for_glm is dispatched via to_thread by its callers."""
    assert "await asyncio.to_thread(\n            _run_post_ocr_pipeline_for_glm," in SOURCE


def test_asyncio_is_imported() -> None:
    assert "import asyncio" in SOURCE


# ───────────────────────────────────────── the behaviour it protects ──


def blocking_work(iterations: int = 4, per_call: float = 0.05) -> str:
    for _ in range(iterations):
        time.sleep(per_call)
    return "done"


async def worst_loop_lag(run_stage, *, interval: float = 0.01) -> float:
    """Largest gap between heartbeat ticks while the stage runs."""
    gaps: list[float] = []
    stop = asyncio.Event()

    async def heartbeat() -> None:
        last = time.perf_counter()
        while not stop.is_set():
            await asyncio.sleep(interval)
            now = time.perf_counter()
            gaps.append(now - last - interval)
            last = now

    task = asyncio.create_task(heartbeat())
    await asyncio.sleep(interval * 3)
    await run_stage()
    stop.set()
    await task
    return max(gaps)


def test_direct_call_starves_the_loop() -> None:
    async def direct() -> None:
        blocking_work()

    assert asyncio.run(worst_loop_lag(direct)) > 0.1


def test_to_thread_keeps_the_loop_responsive() -> None:
    async def threaded() -> None:
        await asyncio.to_thread(blocking_work)

    assert asyncio.run(worst_loop_lag(threaded)) < 0.05
