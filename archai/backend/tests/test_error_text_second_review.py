"""Exception text stays in the log on the paths the second security review found (C).

The first pass (test_error_text_stays_in_the_log.py) covered run events,
run.error, batch progress and fallback reasons. These paths still returned
the exception's text: a label-analysis warning carrying the Ollama host and a
urllib3 pool repr, a batch result naming a missing module, search notes ending
with the embedding provider's error, and the OCR agents' fallback warnings.
The last test reads app/ for the pattern, so the next one is caught.
"""

from __future__ import annotations

import ast
import base64
import io
import logging
import zipfile
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
from PIL import Image

SECRET = "http://10.9.8.7:11434 HTTPConnectionPool(host='10.9.8.7') /srv/archai-private"


def _png_bytes(size: tuple[int, int] = (120, 90)) -> bytes:
    buffer = io.BytesIO()
    Image.new("RGB", size, "white").save(buffer, format="PNG")
    return buffer.getvalue()


def test_a_page_that_fails_in_a_batch_is_named_by_its_error_type(monkeypatch: pytest.MonkeyPatch, tmp_path: Path, caplog: pytest.LogCaptureFixture) -> None:
    from app.routers import predict
    from app.services import file_manager

    monkeypatch.setattr(file_manager, "TASK_BASE_DIR", str(tmp_path / "tasks"))
    task = file_manager.create_task()
    archive = tmp_path / "b.zip"
    with zipfile.ZipFile(archive, "w") as zf:
        zf.writestr("p0.png", _png_bytes())

    def missing(*_args: object, **_kwargs: object) -> None:
        raise ModuleNotFoundError(f"No module named 'ultralytics' ({SECRET})")

    monkeypatch.setattr(predict, "run_models_parallel", missing)
    with caplog.at_level(logging.WARNING):
        predict._run_batch(task, str(archive), 0.25, 0.45, ["Main script black"])
    assert task.errors == ["Processing p0.png failed (ModuleNotFoundError). The details are in the server log."]
    assert SECRET in caplog.text


def _label_payload(label: str = "Page Number") -> SimpleNamespace:
    buffer = base64.b64encode(_png_bytes()).decode("ascii")
    return SimpleNamespace(
        question="What does it say?",
        label_name=label,
        image_b64=buffer,
        filename="folio.png",
        regions=[SimpleNamespace(region_id="r1", bbox_xyxy=[20, 15, 50, 45], polygons=[])],
    )


def test_a_label_ocr_failure_warning_does_not_carry_the_ollama_error(monkeypatch: pytest.MonkeyPatch) -> None:
    from app.agents import label_analysis_agent
    from app.services.glm_ollama_ocr import GlmOllamaOcrError

    def down(*_args: object, **_kwargs: object) -> None:
        raise GlmOllamaOcrError(f"Ollama is not reachable at {SECRET}")

    monkeypatch.setattr(label_analysis_agent, "run_glm_ollama_ocr", down)
    result = label_analysis_agent.LabelAnalysisAgent(client=SimpleNamespace(chat_completion=None)).run(_label_payload())
    text = repr(result)
    assert SECRET not in text and "10.9.8.7" not in text
    assert any(w.startswith("LABEL_OCR_FAILED:Label OCR failed (GlmOllamaOcrError)") for w in result.warnings)


def test_a_failed_provider_call_is_named_by_type_in_the_label_analysis_error() -> None:
    from app.agents.label_analysis_agent import LabelAnalysisAgent

    message = LabelAnalysisAgent._friendly_error(ConnectionError(f"Connection error. {SECRET}"))
    assert SECRET not in message
    assert message == "The provider call failed (ConnectionError). The details are in the server log."


def test_the_saia_ocr_agent_fallback_and_tile_warnings_hold_no_error_text() -> None:
    from app.agents.saia_ocr_agent import SaiaOCRAgent
    from app.schemas.agents_ocr import SaiaOCRRequest

    class Failing:
        def list_models(self) -> list[str]:
            return ["internvl3.5-30b-a3b"]

        def chat_completion(self, **_kwargs: Any) -> dict[str, Any]:
            raise RuntimeError(f"upstream said {SECRET}")

    agent = SaiaOCRAgent(client=Failing(), model_prefs=["internvl3.5-30b-a3b"])  # type: ignore[arg-type]
    result = agent.extract(SaiaOCRRequest(image_b64=base64.b64encode(_png_bytes()).decode("ascii"), apply_proofread=False))
    text = result.model_dump_json()
    assert SECRET not in text and "10.9.8.7" not in text
    assert any(w.startswith("MODEL_ERROR:internvl3.5-30b-a3b:") for w in result.warnings)


# ── the pattern, read from the source ───────────────────────────────────

APP = Path(__file__).resolve().parents[1] / "app"
_LOG_METHODS = {"debug", "info", "warning", "error", "exception", "critical", "log"}
# Exceptions this code base raises with fixed sentences meant for the client:
# a handler that catches only these may pass their text on.
_OWN_REFUSALS = {
    "RequestRefused",  # app/services/public_errors.py, and its subclasses:
    "ImageTooLarge",
    "UnreadableImage",
    "ChatConfigError",  # configuration errors naming the variable to set
    "SaiaConfigError",
    "LabelAnalysisAgentError",  # its provider text goes through public_failure
    "_SegmentedPageUnread",  # a code ("no_text"), not a message
}
# Places that pass on a caught exception's text outside such a handler, and why.
_ALLOWED = {
    ("routers/predict.py", "_process_batch"),  # `except Exception`, then only for BatchArchiveRefused (isinstance)
    # Into the GlmOllamaOcrError it raises, which every caller turns into a
    # sentence (_glm_failure, public_failure) and the log keeps.
    ("services/glm_ollama_ocr.py", "run_glm_ollama_ocr"),
}


def _caught(handler: ast.ExceptHandler) -> set[str]:
    types = handler.type.elts if isinstance(handler.type, ast.Tuple) else [handler.type]
    return {t.id if isinstance(t, ast.Name) else t.attr if isinstance(t, ast.Attribute) else "?" for t in types if t is not None}


def _exception_names(handler: ast.ExceptHandler) -> set[str]:
    """The name a handler binds, unless it catches only this code base's own refusals."""
    if not handler.name or (handler.type is not None and _caught(handler) <= _OWN_REFUSALS):
        return set()
    return {handler.name}


def _mentions(node: ast.AST, names: set[str]) -> bool:
    """*node* puts the text of one of *names* into a string: f"{exc}" or str(exc)."""
    for sub in ast.walk(node):
        if isinstance(sub, ast.FormattedValue) and isinstance(sub.value, ast.Name) and sub.value.id in names:
            return True
        if (
            isinstance(sub, ast.Call)
            and isinstance(sub.func, ast.Name)
            and sub.func.id == "str"
            and sub.args
            and isinstance(sub.args[0], ast.Name)
            and sub.args[0].id in names
        ):
            return True
    return False


def _is_log_call(node: ast.AST) -> bool:
    return isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute) and node.func.attr in _LOG_METHODS


def _is_comparison(node: ast.AST) -> bool:
    """str(exc).lower() and "x" in str(exc): a test of the text, not a copy of it."""
    return isinstance(node, ast.Compare) or (
        isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute) and node.func.attr in {"lower", "casefold", "startswith", "endswith"}
    )


def _offences(tree: ast.Module) -> list[tuple[str, int]]:
    found: list[tuple[str, int]] = []

    def visit(node: ast.AST, function: str, names: set[str], parents: list[ast.AST]) -> None:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            function, names = node.name, set()
        if isinstance(node, ast.ExceptHandler):
            names = names | _exception_names(node)
        if names and isinstance(node, (ast.JoinedStr, ast.Call)) and _mentions(node, names) and not any(
            _is_log_call(p) or _is_comparison(p) for p in [*parents, node]
        ):
            raise_parent = next((p for p in reversed(parents) if isinstance(p, ast.Raise)), None)
            # Raising a new exception with the old text is internal: the
            # route turns it into a sentence (unexpected_error). Raising an
            # HTTPException with it is a response.
            raises_response = raise_parent is not None and "HTTPException" in ast.unparse(raise_parent)
            if raise_parent is None or raises_response:
                found.append((function, node.lineno))
                return
        for child in ast.iter_child_nodes(node):
            visit(child, function, names, [*parents, node])

    visit(tree, "<module>", set(), [])
    return found


def test_no_response_field_or_warning_carries_the_text_of_a_caught_exception() -> None:
    offences = []
    for path in sorted(APP.rglob("*.py")):
        rel = path.relative_to(APP).as_posix()
        for function, line in _offences(ast.parse(path.read_text(encoding="utf-8"))):
            if (rel, function) not in _ALLOWED:
                offences.append(f"app/{rel}:{line} in {function}")
    assert offences == [], "exception text put into a string outside a log call; use public_failure:\n" + "\n".join(offences)


def test_the_scan_catches_the_patterns_it_is_for() -> None:
    source = '''
def f(warnings, notes, task):
    try:
        pass
    except Exception as exc:
        warnings.append(f"FAILED:{exc}")
        notes.append("x: " + str(exc))
        log.warning("kept in the log: %s", exc)
        if "timed out" in str(exc).lower():
            pass
        raise RuntimeError(f"internal {exc}") from exc

def g():
    try:
        pass
    except ValueError as err:
        raise HTTPException(status_code=422, detail=str(err)) from err
'''
    assert [line for _function, line in _offences(ast.parse(source))] == [6, 7, 17]


def test_every_allowed_place_still_exists() -> None:
    """An entry whose code moved or was fixed is removed, so the list cannot hide a new one."""
    seen = set()
    for path in sorted(APP.rglob("*.py")):
        rel = path.relative_to(APP).as_posix()
        seen |= {(rel, function) for function, _line in _offences(ast.parse(path.read_text(encoding="utf-8")))}
    assert sorted(_ALLOWED - seen) == []


def test_a_handler_of_this_code_base_s_own_refusals_may_pass_their_sentence_on() -> None:
    source = '''
def h():
    try:
        pass
    except RequestRefused as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    except (ChatConfigError, ValueError) as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
'''
    assert [line for _function, line in _offences(ast.parse(source))] == [8]  # a tuple with ValueError is not only our own
