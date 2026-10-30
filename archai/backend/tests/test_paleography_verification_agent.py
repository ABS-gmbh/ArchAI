from __future__ import annotations

import json
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any

_backend_src = Path(__file__).resolve().parent.parent / "app"
if str(_backend_src.parent) not in sys.path:
    sys.path.insert(0, str(_backend_src.parent))

from app.agents.paleography_verification_agent import PaleographyVerificationAgent  # type: ignore[import-untyped]
from app.config import settings  # type: ignore[import-untyped]


class _FakeVerifierClient:
    def __init__(self, response_text: str) -> None:
        self.response_text = response_text
        self.calls: list[dict[str, Any]] = []
        self.chat = SimpleNamespace(completions=SimpleNamespace(create=self._create_completion))

    def _create_completion(self, **kwargs: Any) -> Any:
        self.calls.append(kwargs)
        return SimpleNamespace(
            choices=[SimpleNamespace(message=SimpleNamespace(content=self.response_text))]
        )


def test_paleography_verification_agent_uses_dedicated_verifier_model(monkeypatch: Any) -> None:
    fake = _FakeVerifierClient(
        '{"assessment":"supported","corrected_answer":"Verified answer.","notes":["Matches transcript."],"citations_checked":["chunk-1"]}'
    )
    monkeypatch.setattr(settings, "paleography_verification_model", "qwen3-235b-a22b")
    monkeypatch.setattr(settings, "chat_rag_model", "qwen3-30b-a3b-instruct-2507")
    monkeypatch.setattr("app.services.chat_ai._create_client", lambda: fake)

    result = PaleographyVerificationAgent().run(
        SimpleNamespace(
            question="What does the passage say?",
            draft_answer="Draft answer.",
            transcript="Li rois Artus",
            authority_report="Arthur -> Q45720",
            evidence_text="[EVIDENCE]\ntext: Li rois Artus\n[/EVIDENCE]",
            ocr_run_id="run-1",
        )
    )

    assert result.model_used == "qwen3-235b-a22b"
    assert fake.calls[0]["model"] == "qwen3-235b-a22b"
    assert result.stage_metadata["stage_name"] == "verification"
    assert result.stage_metadata["model_used"] == "qwen3-235b-a22b"
    assert result.duration_ms >= 0


def test_paleography_verification_agent_sanitizes_reasoning_leaks(monkeypatch: Any) -> None:
    fake = _FakeVerifierClient(
        """
        {
          "assessment": "supported",
          "corrected_answer": "<think>hidden</think> Verified answer grounded in the text.",
          "notes": ["Reasoning: private chain of thought", "Transcript supports the reading."],
          "citations_checked": ["```chunk-1```", "ENTITY_EVIDENCE Arthur"]
        }
        """
    )
    monkeypatch.setattr(settings, "paleography_verification_model", "qwen3-235b-a22b")
    monkeypatch.setattr("app.services.chat_ai._create_client", lambda: fake)

    result = PaleographyVerificationAgent().run(
        SimpleNamespace(
            question="Who is mentioned?",
            draft_answer="Draft answer.",
            transcript="Arthur",
            authority_report="Arthur -> Q45720",
            evidence_text="[ENTITY_EVIDENCE]\ncanonical_label: King Arthur\n[/ENTITY_EVIDENCE]",
            ocr_run_id="run-1",
        )
    )

    assert result.corrected_answer == "Verified answer grounded in the text."
    assert result.notes == ["Transcript supports the reading."]
    assert result.citations_checked == ["chunk-1", "ENTITY_EVIDENCE Arthur"]
    assert result.inspection["confidence_or_assessment"] == "supported"


def test_paleography_verification_agent_handles_malformed_output_without_leaking_raw_text(monkeypatch: Any) -> None:
    fake = _FakeVerifierClient("<think>private scratchpad</think> unsupported because maybe something")
    monkeypatch.setattr(settings, "paleography_verification_model", "qwen3-235b-a22b")
    monkeypatch.setattr("app.services.chat_ai._create_client", lambda: fake)

    result = PaleographyVerificationAgent().run(
        SimpleNamespace(
            question="What does it say?",
            draft_answer="Draft answer retained.",
            transcript="Arthur",
            authority_report="",
            evidence_text="[EVIDENCE]\ntext: Arthur\n[/EVIDENCE]",
            ocr_run_id="run-1",
        )
    )

    assert result.assessment == "unavailable"
    assert result.corrected_answer == "Draft answer retained."
    assert result.notes == ["Verifier response could not be parsed cleanly."]
    assert "scratchpad" not in result.notes[0].lower()
    assert result.inspection["final_output"]["verified_answer"] == "Draft answer retained."


# ── notes and citations given as one string (audit AG-03) ───────────────


def _normalized(payload: dict[str, Any]) -> tuple[str, str, list[str], list[str]]:
    import json

    from app.agents.paleography_verification_agent import _normalized_verification_payload

    return _normalized_verification_payload(json.dumps(payload), draft_answer="draft")


def test_notes_given_as_one_string_are_sentences_not_characters() -> None:
    _assessment, _answer, notes, _citations = _normalized(
        {"assessment": "supported", "notes": "Supported by chunk 3, which names Arthur; Arthur appears twice."}
    )
    assert notes == ["Supported by chunk 3, which names Arthur", "Arthur appears twice."]


def test_citations_given_as_one_string_are_split_into_citations() -> None:
    *_rest, citations = _normalized({"assessment": "supported", "notes": ["ok"], "citations_checked": "chunk-3, wikidata:Q45720"})
    assert citations == ["chunk-3", "wikidata:Q45720"]


def test_arrays_are_taken_as_they_are() -> None:
    _assessment, _answer, notes, citations = _normalized({"assessment": "supported", "notes": ["a note", "another"], "citations_checked": ["c1"]})
    assert (notes, citations) == (["a note", "another"], ["c1"])


def _verdict(raw: str) -> tuple[str, str, list[str], list[str]]:
    from app.agents.paleography_verification_agent import _normalized_verification_payload

    return _normalized_verification_payload(raw, draft_answer="draft")


def test_a_verdict_wrapped_in_an_array_is_read() -> None:
    """Audit AG-02: a list reached payload.get() and raised AttributeError."""
    assessment, _answer, notes, citations = _verdict(
        json.dumps([{"assessment": "supported", "notes": ["names Arthur"], "citations_checked": ["chunk-3"]}])
    )
    assert (assessment, notes, citations) == ("supported", ["names Arthur"], ["chunk-3"])


def test_a_fenced_verdict_wrapped_in_an_array_is_read() -> None:
    raw = 'Verdict:\n```json\n[{"assessment": "unsupported", "notes": ["no date"]}]\n```'
    assessment, _answer, notes, _citations = _verdict(raw)
    assert (assessment, notes) == ("unsupported", ["no date"])


def test_json_that_is_not_an_object_reads_as_no_verdict() -> None:
    for raw in ('"supported"', "[1, 2]", "[]", "42", "null", '[{"a": 1}, {"b": 2}]'):
        assessment, answer, notes, citations = _verdict(raw)
        assert assessment == "unavailable", raw
        assert answer == "draft"


def test_the_verifier_call_has_its_own_timeout(monkeypatch: Any) -> None:
    """Audit AG-09: no timeout meant the SDK's 600 s default, retried twice."""
    fake = _FakeVerifierClient('{"assessment":"supported","notes":["ok"]}')
    monkeypatch.setattr(settings, "paleography_verification_timeout_seconds", 45)
    monkeypatch.setattr("app.services.chat_ai._create_client", lambda: fake)

    PaleographyVerificationAgent().run(
        SimpleNamespace(
            question="q", draft_answer="d", transcript="t", authority_report="",
            evidence_text="", ocr_run_id="run-1",
        )
    )

    assert fake.calls[0]["timeout"] == 45.0


def test_the_default_timeout_is_a_minute() -> None:
    from app.config import Settings  # type: ignore[import-untyped]

    assert Settings.model_fields["paleography_verification_timeout_seconds"].default == 60
