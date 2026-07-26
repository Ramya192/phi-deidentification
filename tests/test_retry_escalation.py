"""
tests/test_retry_escalation.py

Covers the retry-cap escalation path added to
agents/compliance_validation_agent.py (escalate_to_review,
escalation_review_agent) and agents/redaction_agent.py
(escalation_redaction_agent), wired into graph/workflow.py between
compliance_validation and audit_report.

Rather than relying on a specific real PHI pattern's redaction accuracy
to force retries to exhaust (fragile -- depends on regex edge cases
this suite already had to fix once), these tests monkeypatch the
underlying fallback detector (agents.phi_detection_agent._detect_fallback,
used by both PHIDetectionAgent's node and ComplianceValidationAgent's
re-scan) to return a scripted sequence of results. This isolates what's
actually new here -- the graph CONTROL FLOW (does exhausting retries
correctly escalate instead of dropping straight to audit_report, does
escalation-approval actually clear validation, does a still-failing
escalation terminate instead of looping again) -- from PHI-detection
accuracy, which the rest of the suite (and eval/evaluate.py) already
covers.

Invokes graph.workflow.compiled_graph directly (rather than run_sync(),
which hardcodes max_retries=2) with max_retries=1, so the exact call
sequence needed to reach escalation is small and easy to reason about.

Run with: pytest -v tests/test_retry_escalation.py
"""
from __future__ import annotations

import os
import sys
import uuid

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pytest

from agents import phi_detection_agent as phi_detection_module
from agents import redaction_agent as redaction_module
from graph.workflow import compiled_graph

_FIXED_SPAN = {
    "start": 0,
    "end": 5,
    "text": "dummy",
    # PERSON has CONFIDENCE_THRESHOLDS 0.85 and is NOT in
    # AUTO_REDACT_ESCALATION_TYPES (a noisy NER type, not regex-backed) --
    # these tests are specifically about the human-review escalation path,
    # so they need a type that doesn't get auto-redacted before ever
    # reaching escalation_review_agent. See test_escalation_auto_redact.py
    # for the auto-redact-only and mixed-type paths.
    "phi_type": "PERSON",
    "confidence": 0.95,
    "source_agent": "test_mock",
}


@pytest.fixture
def force_fallback_detector(monkeypatch):
    """Pins detection to the fallback backend AND stubs redaction to a
    no-op (returns text unchanged). The no-op is what actually matters
    here: real redaction inserts a [TYPE] tag at exactly the coordinates
    a span was detected at, and compliance_validation_agent's
    _is_inside_placeholder guard (correctly) filters out any "residual"
    span that lands inside a tag its own prior redaction just created.
    Since these tests script a detector that reports the SAME fixed
    coordinates on every call to force sustained FAIL, without this
    no-op that guard would immediately swallow the scripted span the
    moment it collides with its own tag -- collapsing the intended
    4-call FAIL sequence down to 1-2 calls before an accidental PASS.
    Stubbing redaction removes that interaction entirely so the scripted
    detector's return value is what actually decides PASS/FAIL each
    call, matching this file's traced call sequences exactly.
    """
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


def test_escalation_triggers_after_retries_exhausted_and_clears_on_approval(monkeypatch, force_fallback_detector):
    """4 calls (detect, redact, re-scan-FAIL-retry, detect, redact,
    re-scan-FAIL) exhaust max_retries=1, escalating instead of dropping to
    audit_report. Approving the escalation span, then a clean final
    re-scan, should PASS."""
    call_count = {"n": 0}

    def scripted_detect(text):
        call_count["n"] += 1
        # Calls 1-4: initial detect, initial re-scan (FAIL->retry), retry
        # detect, retry re-scan (FAIL, retries exhausted -> escalate).
        # Call 5+ (the post-escalation-redaction re-scan): clean.
        return [dict(_FIXED_SPAN)] if call_count["n"] <= 4 else []

    monkeypatch.setattr(phi_detection_module, "_detect_fallback", scripted_detect)

    # Unique per test run -- the SQLite checkpointer persists across
    # process runs (that's its whole point), so a reused thread_id would
    # resume/no-op against leftover state from a previous run instead of
    # starting fresh.
    thread_id = str(uuid.uuid4())
    config = {"configurable": {"thread_id": thread_id}}
    initial_state = {
        "raw_text": "Patient note: chief complaint follow-up visit, vital signs stable.",
        "filename": "escalation_note.txt",
        "retry_count": 0,
        "max_retries": 1,
        "audit_log": [],
    }

    result = _invoke(initial_state, config)
    assert "__interrupt__" in result, "expected the graph to pause at escalation_review"
    interrupt_payload = result["__interrupt__"][0].value
    assert interrupt_payload["reason"] == "compliance_validation_escalation"
    assert len(interrupt_payload["spans"]) == 1
    assert interrupt_payload["spans"][0]["phi_type"] == "PERSON"

    decisions = [{"span_index": 0, "approved": True, "reviewer": "test_reviewer"}]
    final = _resume(decisions, config)

    assert "__interrupt__" not in final
    assert final["validation_status"] == "PASS"
    assert final["escalation_attempted"] is True
    assert final["compliance_report"]["escalated_to_human_review"] is True
    assert final["compliance_report"]["validation_status"] == "PASS"
    # requires_manual_followup also factors in PHIValidationAgent's
    # separate schema-completeness check (expected identifier types for
    # this doc_type) -- unrelated to escalation, and this test's mocked
    # detector only ever returns an MRN span, so that check will
    # legitimately report PERSON/DATE_TIME as missing regardless of
    # escalation outcome. What escalation actually controls is
    # retries_exhausted_while_failing, which must be False once
    # validation_status flips to PASS.
    assert final["compliance_report"]["retries_exhausted_while_failing"] is False

    # Escalation-specific audit trail entries should be present.
    agents_in_log = {entry["agent"] for entry in final["audit_log"]}
    assert "EscalationReviewAgent" in agents_in_log
    escalation_entries = [e for e in final["audit_log"] if e["agent"] == "ComplianceValidationAgent" and e["action"] == "escalated_to_human_review"]
    assert len(escalation_entries) == 1


def test_escalation_still_failing_terminates_instead_of_looping(monkeypatch, force_fallback_detector):
    """If the escalation redaction pass STILL leaves the document failing
    validation, the graph must terminate at audit_report (flagged) rather
    than escalating a second time -- escalation_attempted guards this."""
    monkeypatch.setattr(phi_detection_module, "_detect_fallback", lambda text: [dict(_FIXED_SPAN)])

    thread_id = str(uuid.uuid4())
    config = {"configurable": {"thread_id": thread_id}}
    initial_state = {
        "raw_text": "Patient note: chief complaint follow-up visit, vital signs stable.",
        "filename": "escalation_note_2.txt",
        "retry_count": 0,
        "max_retries": 1,
        "audit_log": [],
    }

    result = _invoke(initial_state, config)
    assert "__interrupt__" in result

    decisions = [{"span_index": 0, "approved": True, "reviewer": "test_reviewer"}]
    final = _resume(decisions, config)

    assert "__interrupt__" not in final, "must not escalate a second time"
    assert final["validation_status"] == "FAIL"
    assert final["escalation_attempted"] is True
    assert final["compliance_report"]["requires_manual_followup"] is True
    assert final["compliance_report"]["escalated_to_human_review"] is True


def test_escalation_rejected_span_is_not_redacted_but_still_terminates(monkeypatch, force_fallback_detector):
    """Rejecting the escalation span (reviewer says "not real PHI") should
    leave it un-redacted and record the rejection, same contract as the
    normal HumanReviewAgent path -- and still terminate rather than loop,
    since the rejected span is filtered out of the next re-scan."""
    call_count = {"n": 0}

    def scripted_detect(text):
        call_count["n"] += 1
        return [dict(_FIXED_SPAN)] if call_count["n"] <= 4 else []

    monkeypatch.setattr(phi_detection_module, "_detect_fallback", scripted_detect)

    thread_id = str(uuid.uuid4())
    config = {"configurable": {"thread_id": thread_id}}
    initial_state = {
        "raw_text": "Patient note: chief complaint follow-up visit, vital signs stable.",
        "filename": "escalation_note_3.txt",
        "retry_count": 0,
        "max_retries": 1,
        "audit_log": [],
    }

    result = _invoke(initial_state, config)
    assert "__interrupt__" in result

    decisions = [{"span_index": 0, "approved": False, "reviewer": "test_reviewer"}]
    final = _resume(decisions, config)

    assert "__interrupt__" not in final
    assert {"phi_type": "PERSON", "text": "dummy"} in final["rejected_spans"]
    rejection_entries = [e for e in final["audit_log"] if e["agent"] == "EscalationReviewAgent" and e["action"] == "human_rejected"]
    assert len(rejection_entries) == 1


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
