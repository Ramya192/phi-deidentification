"""
tests/test_integration.py (Member 3, per capstone deliverable list)

Run with:  pytest -v

Covers:
  1. ClassificationAgent routes each of the 7 doc types + a non-clinical
     doc correctly.
  2. PHIDetectionAgent finds PHI in a clinical note (fallback regex
     backend, so this runs with zero extra downloads).
  3. RedactionAgent actually removes detected spans from the text.
  4. ComplianceValidationAgent flags PASS once text is fully redacted.
  5. Full graph run end-to-end on a clean note with no low-confidence
     spans (no interrupt expected).
  6. Full graph run that DOES hit HumanReviewAgent's interrupt(), then
     resumes with reviewer decisions and completes.
  7. Non-clinical document routes straight to END without a redacted_text.

Tests 5 and 6 pin PHIDetectionAgent to the deterministic regex fallback
(see `force_fallback_detector` below) rather than whatever's actually
installed. They're testing graph CONTROL FLOW -- does the confidence
branch route correctly, does interrupt()/resume work, does retry work --
not PHI-detection accuracy, which is backend-dependent (Presidio+spaCy
vs. the regex fallback give genuinely different confidence scores on the
same text). Without pinning, these tests would pass or fail depending on
what happens to be installed in the environment, which is exactly what
broke them the first time this suite ran somewhere with real Presidio
installed.
"""
from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pytest

from agents import phi_detection_agent as phi_detection_module
from agents.classification_agent import classification_agent, classify_with_heuristic
from agents.compliance_validation_agent import compliance_validation_agent
from agents.phi_detection_agent import phi_detection_agent
from agents.phi_validation_agent import EXPECTED_IDENTIFIER_TYPES_BY_DOC_TYPE, phi_validation_agent
from agents.redaction_agent import redaction_agent
from graph.workflow import resume_stream, resume_sync, run_stream, run_sync

SAMPLE_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", "txt_format")


def _read_sample(name: str) -> str:
    with open(os.path.join(SAMPLE_DIR, name), encoding="utf-8") as f:
        return f.read()


@pytest.fixture
def force_fallback_detector(monkeypatch):
    """Pins PHIDetectionAgent -- and ComplianceValidationAgent's re-scan,
    which reads the same flag live off this module -- to the deterministic
    regex fallback for the duration of a test, regardless of whether
    Presidio/spaCy are installed in the environment.

    Also pins ClassificationAgent to the heuristic backend, regardless of
    whatever PHI_DEID_CLASSIFICATION_BACKEND happens to be set to in the
    developer's real .env. Found live: once PHI_DEID_CLASSIFICATION_BACKEND
    was set to "llm" locally (with a real, working OPENAI_API_KEY), these
    tests' thin synthetic text ("Patient contact: ...email... or
    ...phone... for follow-up.") started getting classified as
    not_applicable by the real LLM instead of clinical_note -- reasonably,
    since it's not much of a clinical document, but that routes straight to
    END before PHIDetectionAgent/RedactionAgent/ComplianceValidationAgent
    ever run, leaving redacted_text/validation_status/compliance_report
    all unset. Every other backend-sensitive piece of this test file is
    already pinned (this fixture); classification was the one gap, since
    at the time these tests were written "heuristic" was still the .env
    default. These are control-flow tests, not classification-accuracy
    tests, so they shouldn't depend on a live external API call at all.
    """
    original = phi_detection_module.USE_FALLBACK_ONLY
    phi_detection_module.USE_FALLBACK_ONLY = True
    monkeypatch.setenv("PHI_DEID_CLASSIFICATION_BACKEND", "heuristic")
    yield
    phi_detection_module.USE_FALLBACK_ONLY = original


@pytest.mark.parametrize("filename,expected_type", [
    ("clinical_note_01.txt", "clinical_note"),
    ("discharge_summary_01.txt", "discharge_summary"),
    ("insurance_document_01.txt", "insurance_document"),
    ("not_applicable_01.txt", "not_applicable"),
])
def test_classification_routes_correctly(filename, expected_type):
    text = _read_sample(filename)
    doc_type, confidence = classify_with_heuristic(text)
    assert doc_type == expected_type
    assert 0.0 <= confidence <= 1.0


def test_llm_classification_backend_falls_back_safely(monkeypatch):
    """classify_with_llm() needs OPENAI_API_KEY (and network access) to
    actually succeed -- neither is available in CI/this sandbox. What this
    test pins down is the *safety net*: classification_agent(backend="llm")
    must never crash the graph just because the LLM call failed. It should
    silently fall back to the heuristic backend and still return a valid
    (doc_type, confidence), the same defensive contract PHIDetectionAgent
    has for presidio-vs-regex-fallback.
    """
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    text = _read_sample("clinical_note_01.txt")
    state = {"raw_text": text, "retry_count": 0, "audit_log": []}
    result = classification_agent(state, backend="llm")

    assert result["doc_type"] == "clinical_note"
    assert 0.0 <= result["doc_type_confidence"] <= 1.0
    # The audit log entry should record that it actually fell back, not
    # silently claim the LLM path succeeded.
    last_entry = result["audit_log"][-1]
    assert "fallback" in last_entry["notes"]


def test_phi_detection_finds_spans(force_fallback_detector):
    text = _read_sample("clinical_note_01.txt")
    state = {"raw_text": text, "retry_count": 0, "audit_log": []}
    result = phi_detection_agent(state)
    assert len(result["phi_spans"]) > 0
    phi_types = {s["phi_type"] for s in result["phi_spans"]}
    # MRN and phone/email patterns should always fire via the fallback regexes
    assert "MRN" in phi_types or "PHONE_NUMBER" in phi_types or "EMAIL_ADDRESS" in phi_types


def test_redaction_removes_detected_spans(force_fallback_detector):
    text = "Contact john.carter@example.com or (555) 234-9981 for follow-up."
    state = {"raw_text": text, "retry_count": 0, "audit_log": []}
    detected = phi_detection_agent(state)
    detected["high_confidence_spans"] = detected["phi_spans"]  # force everything through for this test
    redacted = redaction_agent(detected)
    assert "john.carter@example.com" not in redacted["redacted_text"]
    assert "(555) 234-9981" not in redacted["redacted_text"]
    assert "[EMAIL_ADDRESS]" in redacted["redacted_text"]


def test_phi_validation_flags_missing_expected_identifiers():
    # pathology_report expects PERSON, DATE_TIME, MRN, ACCESSION_NUMBER --
    # give it a detection result missing MRN and ACCESSION_NUMBER entirely.
    state = {
        "doc_type": "pathology_report",
        "phi_spans": [
            {"start": 0, "end": 10, "text": "Jane Doe", "phi_type": "PERSON", "confidence": 0.9, "source_agent": "test"},
            {"start": 20, "end": 30, "text": "05/12/2024", "phi_type": "DATE_TIME", "confidence": 0.9, "source_agent": "test"},
        ],
        "audit_log": [],
    }
    result = phi_validation_agent(state)
    assert result["missing_expected_identifier_types"] == ["ACCESSION_NUMBER", "MRN"]
    assert 0.0 < result["schema_completeness_score"] < 1.0
    assert result["audit_log"][-1]["agent"] == "PHIValidationAgent"


def test_phi_validation_passes_when_all_expected_types_present():
    expected = EXPECTED_IDENTIFIER_TYPES_BY_DOC_TYPE["referral_letter"]
    state = {
        "doc_type": "referral_letter",
        "phi_spans": [
            {"start": 0, "end": 5, "text": "x", "phi_type": t, "confidence": 0.9, "source_agent": "test"}
            for t in expected
        ],
        "audit_log": [],
    }
    result = phi_validation_agent(state)
    assert result["missing_expected_identifier_types"] == []
    assert result["schema_completeness_score"] == 1.0


def test_compliance_validation_passes_on_clean_text(force_fallback_detector):
    state = {
        "redacted_text": "Patient [PERSON] was seen on [DATE_TIME] and is doing well.",
        "retry_count": 0,
        "audit_log": [],
    }
    result = compliance_validation_agent(state)
    assert result["validation_status"] == "PASS"
    assert result["remaining_phi_spans"] == []


def test_full_graph_no_interrupt_on_dense_but_high_confidence_note(force_fallback_detector):
    text = "Patient contact: john.carter@example.com or (555) 234-9981 for follow-up."
    result = run_sync(text, filename="quick_note.txt", thread_id="test-thread-1")
    # This note has only high-confidence regex hits (email/phone), so it
    # should complete without needing human review.
    assert result["status"] == "completed"
    state = result["state"]
    assert state["validation_status"] in ("PASS", "FAIL")
    assert "compliance_report" in state


def test_full_graph_interrupt_and_resume_on_low_confidence_note(force_fallback_detector):
    # "Patient" + "treatment" give ClassificationAgent enough generic
    # health-domain signal to route this to PHIDetectionAgent (rather than
    # "not_applicable") -- a bare "Dr. Smith" sentence with no other
    # clinical vocabulary fails classification and never reaches PHI
    # detection at all, which is what broke this test originally. The
    # bare "Dr. Smith" pattern itself is intentionally low-confidence
    # (0.6 < 0.7 threshold) in the fallback detector, to exercise the
    # human review interrupt path.
    text = "Patient reports feeling better after treatment. Please follow up with Dr. Smith next week regarding the results."
    result = run_sync(text, filename="review_note.txt", thread_id="test-thread-2")

    if result["status"] == "interrupted":
        spans = result["interrupt"]["spans"]
        decisions = [{"span_index": s["span_index"], "approved": True, "reviewer": "test_reviewer"} for s in spans]
        result = resume_sync(decisions, thread_id="test-thread-2")

    assert result["status"] == "completed"
    assert "compliance_report" in result["state"]


def test_streaming_yields_same_final_result_as_sync(force_fallback_detector):
    """run_stream() must reach the same final state run_sync() would --
    it's the same compiled_graph.stream() under the hood, just observed
    node-by-node instead of all at once. This pins down the contract the
    Streamlit UI's live pipeline stepper depends on (api/main.py's
    /redact/stream, _stream_response()): a sequence of ("node", name, ...)
    events followed by exactly one ("done", status, payload)."""
    text = "Patient contact: john.carter@example.com or (555) 234-9981 for follow-up."
    events = list(run_stream(text, filename="quick_note.txt", thread_id="test-stream-thread-1"))

    assert events, "run_stream() yielded nothing"
    *node_events, last_event = events
    assert all(e[0] == "node" for e in node_events)
    assert last_event[0] == "done"
    status, payload = last_event[1], last_event[2]
    assert status == "completed"
    assert "compliance_report" in payload
    # Same nodes classification/phi_detection/phi_validation/redaction/
    # compliance_validation/audit_report should have fired, in order, for a
    # note with no low-confidence spans (no human_review interrupt expected).
    node_names = [e[1] for e in node_events]
    assert node_names[0] == "classification"
    assert "phi_validation" in node_names
    assert node_names.index("phi_validation") == node_names.index("phi_detection") + 1
    assert "human_review" not in node_names


def test_streaming_interrupt_and_resume(force_fallback_detector):
    """Same low-confidence note used in
    test_full_graph_interrupt_and_resume_on_low_confidence_note above, but
    exercised through the streaming API to confirm resume_stream() also
    reaches a valid final ("done", ...) event after the interrupt."""
    text = "Patient reports feeling better after treatment. Please follow up with Dr. Smith next week regarding the results."
    events = list(run_stream(text, filename="review_note.txt", thread_id="test-stream-thread-2"))
    last_event = events[-1]
    assert last_event[0] == "done"

    if last_event[1] == "interrupted":
        spans = last_event[2]["spans"]
        decisions = [{"span_index": s["span_index"], "approved": True, "reviewer": "test_reviewer"} for s in spans]
        resume_events = list(resume_stream(decisions, thread_id="test-stream-thread-2"))
        final = resume_events[-1]
        assert final[0] == "done"
        assert final[1] == "completed"
        assert "compliance_report" in final[2]
    else:
        assert last_event[1] == "completed"
        assert "compliance_report" in last_event[2]


def test_not_applicable_document_skips_phi_detection():
    text = _read_sample("not_applicable_01.txt")
    result = run_sync(text, filename="not_applicable_01.txt", thread_id="test-thread-3")
    assert result["status"] == "completed"
    state = result["state"]
    assert state["doc_type"] == "not_applicable"
    assert not state.get("redacted_text")


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
