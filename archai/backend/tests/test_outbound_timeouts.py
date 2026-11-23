"""Every outbound HTTP call in app/ names its timeout.

Without one, requests and urllib wait forever and the OpenAI SDK waits 600 s
per attempt; the chat client ran that way until CHAT_AI_TIMEOUT_SECONDS. This
parses app/ and fails on a call that opens a connection - requests.get/post/...,
urlopen, httpx, or an OpenAI client being built - without an explicit,
non-None timeout= argument. SaiaClient carries its own (SAIA_TIMEOUT_SECONDS)
and is not listed here.
"""

from __future__ import annotations

import ast
from pathlib import Path

APP = Path(__file__).resolve().parents[1] / "app"

_REQUESTS_METHODS = {"get", "post", "put", "patch", "delete", "head", "options", "request"}
_HTTPX_CALLS = _REQUESTS_METHODS | {"stream", "Client", "AsyncClient"}


def _needs_timeout(call: ast.Call) -> bool:
    func = call.func
    if isinstance(func, ast.Attribute) and isinstance(func.value, ast.Name):
        module, name = func.value.id, func.attr
        if module == "requests" and name in _REQUESTS_METHODS:
            return True
        if module == "httpx" and name in _HTTPX_CALLS:
            return True
    name = func.attr if isinstance(func, ast.Attribute) else func.id if isinstance(func, ast.Name) else ""
    return name in {"urlopen", "OpenAI", "AsyncOpenAI", "openai_class"}


def _has_timeout(call: ast.Call) -> bool:
    for keyword in call.keywords:
        if keyword.arg == "timeout":
            return not (isinstance(keyword.value, ast.Constant) and keyword.value.value is None)
    return False


def calls_without_timeout(source: str, label: str = "<source>") -> list[str]:
    return [
        f"{label}:{node.lineno} {ast.unparse(node.func)}"
        for node in ast.walk(ast.parse(source))
        if isinstance(node, ast.Call) and _needs_timeout(node) and not _has_timeout(node)
    ]


def test_no_outbound_call_in_app_lacks_a_timeout() -> None:
    found = []
    for path in sorted(APP.rglob("*.py")):
        found.extend(calls_without_timeout(path.read_text(encoding="utf-8"), str(path.relative_to(APP))))
    assert found == []


def test_the_check_finds_what_it_looks_for() -> None:
    offenders = """
import requests, httpx, urllib.request
from openai import OpenAI
requests.get(url)
requests.post(url, timeout=None)
urllib.request.urlopen(req)
httpx.Client()
OpenAI(api_key=key)
requests.get(url, timeout=5)
OpenAI(api_key=key, timeout=settings.chat_ai_timeout_seconds)
"""
    assert [line.split(" ", 1)[1] for line in calls_without_timeout(offenders)] == [
        "requests.get",
        "requests.post",
        "urllib.request.urlopen",
        "httpx.Client",
        "OpenAI",
    ]
