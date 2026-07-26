"""
tests/test_escalation_auto_redact.py

Covers the auto-redact-before-escalation split in
agents/compliance_validation_agent.py's escalate_to_review():
near-deterministic, regex-backed PHI types (AUTO_REDACT_ESCALATION_TYPES,
derived from PHIDetectionAgent's own CONFIDENCE_THRESHOLDS) get redacted
immediately without a human pause; noisy NER types still go through
escalation_review_agent's interrupt().

Same mocking approach as tests/test_retry_escalation.py: scripts
agents.phi_detection_agent._detect_fallback to force sustained FAIL
through max_retries=1, and stubs redaction to a no-op so the
_is_inside_placeholder guard doesn't collide with a static mocked span
position (see that file's force_fallback_detector docstring for why).

Run with: pytest -v tests/test_escalation_auto_redact.py
"""
from __future__ import annotations

import os
import sys
import uuid

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pytest

from agents import phi_detection_agent as phi_detection_module
from agents import redaction_agent as redaction_module
from agents.compliance_validation_agent import AUTO_REDACT_ESCALATION_TYPES
from graph.workflow import compiled_graph

_MRN_SPAN = {
    "start": 0, "end": 5, "text": "mrn-dummy", "phi_type": "MRN",
    "confidence": 0.9, "source_agent": "test_mock",
}
_PERSON_SPAN = {
    "start": 10, "end": 15, "text": "person-dummy", "phi_type": "PERSON",
    "confidence": 0.95, "source_agent": "test_mock",
}


@pytest.fixture
def force_fallback_detector(monkeypatch):
    # Also pins ClassificationAgent to heuristic -- see
    # tests/test_integration.py's matching fixture docstring: a live
    # PHI_DEID_CLASSIFICATION_BACKEND=llm .env value can otherwise route
    # this file's thin synthetic notes to not_applicable/END instead of
    # actually reaching PHIDetectionAgent.
    monkeypatch.setenv("PHI_DEID_CLASSIFICATION_BACKEND", "heuristic")
    original_use_fallback_only = phi_detection_module.USE_FALLBACK_ONLY
    original_presidio_available = redaction_module._PRESIDIO_AVAILABLE
    phi_detection_module.USE_FALLBACK_ONLY = True
    monkeypatch.setattr(redaction_module, "_PRESIDIO_AVAILABLE", False)
    monkeypatch.setattr(redaction_module, "_redact_fallback", lambda text, spans: text)
    yield
    phi_detection_module.USE_FALLBACK_ONLY = original_use_fallback_only
    redaction_module._PRESIDIO_AVAILABLE = original_presidio_available


def _invoke(initial_state, config):
    return compiled_graph.invoke(initial_state, config=config)


def _resume(decisions, config):
    from langgraph.types import Command
    return compiled_graph.invoke(Command(resume=decisions), config=config)


def test_auto_redact_types_match_near_deterministic_thresholds():
    """Sanity check on the derivation itself: the near-deterministic
    regex-backed types PHIDetectionAgent documents (threshold 0.6) should
    be exactly what escalation auto-redacts -- and noisy NER types
    (PERSON/DATE_TIME/LOCATION/NRP, threshold 0.8+) should not be."""
    assert "MRN" in AUTO_REDACT_ESCALATION_TYPES
    assert "EMAIL_ADDRESS" in AUTO_REDACT_ESCALATION_TYPES
    assert "SSN" in AUTO_REDACT_ESCALATION_TYPES
    assert "PERSON" not in AUTO_REDACT_ESCALATION_TYPES
    assert "DATE_TIME" not in AUTO_REDACT_ESCALATION_TYPES
    assert "LOCATION" not in AUTO_REDACT_ESCALATION_TYPES


def test_pure_auto_redact_never_pauses_for_human(monkeypatch, force_fallback_detector):
    """An MRN-only residual should be auto-redacted at escalation time with
    no interrupt at all -- the whole point of the auto-redact split."""
    call_count = {"n": 0}

    def scripted_detect(text):
        call_count["n"] += 1
        return [dict(_MRN_SPAN)] if call_count["n"] <= 4 else []

    monkeypatch.setattr(phi_detection_module, "_detect_fallback", scripted_detect)

    thread_id = str(uuid.uuid4())
    config = {"configurable": {"thread_id": thread_id}}
    initial_state = {
        "raw_text": "Patient note: chief complaint follow-up visit, vital signs stable.",
        "filename": "auto_redact_note.txt",
        "retry_count": 0,
        "max_retries": 1,
        "audit_log": [],
    }

    final = _invoke(initial_state, config)

    assert "__interrupt__" not in final, "pure auto-redact should never pause for a human"
    assert final["escalation_attempted"] is True
    assert final["validation_status"] == "PASS"
    assert final["compliance_report"]["escalated_to_human_review"] is True
    assert final["compliance_report"]["retries_exhausted_while_failing"] is False

    auto_entries = [e for e in final["audit_log"] if e["action"] == "auto_redacted"]
    assert len(auto_entries) == 1
    assert auto_entries[0]["phi_type"] == "MRN"
    # No EscalationReviewAgent interrupt/decision entries at all -- confirms
    # escalation_review_agent's node was genuinely skipped, not just that
    # the interrupt was somehow auto-approved.
    assert not any(e["agent"] == "EscalationReviewAgent" for e in final["audit_log"])


def test_mixed_auto_redact_and_human_review(monkeypatch, force_fallback_detector):
    """MRN + PERSON residual: only PERSON should reach the human reviewer;
    MRN gets auto-redacted. After approving PERSON, both should show up as
    redacted in the audit trail."""
    call_count = {"n": 0}

    def scripted_detect(text):
        call_count["n"] += 1
        return [dict(_MRN_SPAN), dict(_PERSON_SPAN)] if call_count["n"] <= 4 else []

    monkeypatch.setattr(phi_detection_module, "_detect_fallback", scripted_detect)

    thread_id = str(uuid.uuid4())
    config = {"configurable": {"thread_id": thread_id}}
    initial_state = {
        "raw_text": "Patient note: chief complaint follow-up visit, vital signs stable.",
        "filename": "mixed_note.txt",
        "retry_count": 0,
        "max_retries": 1,
        "audit_log": [],
    }

    result = _invoke(initial_state, config)
    assert "__interrupt__" in result, "PERSON span should still require human review"
    interrupt_payload = result["__interrupt__"][0].value
    # Only the noisy PERSON span should be presented -- MRN was already
    # auto-redacted in escalate_to_review before the interrupt ever fires.
    assert len(interrupt_payload["spans"]) == 1
    assert interrupt_payload["spans"][0]["phi_type"] == "PERSON"

    decisions = [{"span_index": 0, "approved": True, "reviewer": "test_reviewer"}]
    final = _resume(decisions, config)

    assert "__interrupt__" not in final
    assert final["validation_status"] == "PASS"

    redacted_types = {e["phi_type"] for e in final["audit_log"] if e["agent"] in ("RedactionAgent",) and e["action"] == "redacted"}
    assert "MRN" in redacted_types
    assert "PERSON" in redacted_types
    auto_entries = [e for e in final["audit_log"] if e["action"] == "auto_redacted"]
    assert len(auto_entries) == 1 and auto_entries[0]["phi_type"] == "MRN"
    human_entries = [e for e in final["audit_log"] if e["agent"] == "EscalationReviewAgent" and e["action"] == "human_approved"]
    assert len(human_entries) == 1 and human_entries[0]["phi_type"] == "PERSON"


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
