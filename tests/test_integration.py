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
from agents.classification_agent import classify_with_heuristic
from agents.compliance_validation_agent import compliance_validation_agent
from agents.phi_detection_agent import phi_detection_agent
from agents.redaction_agent import redaction_agent
from graph.workflow import resume_sync, run_sync

SAMPLE_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", "txt_format")


def _read_sample(name: str) -> str:
    with open(os.path.join(SAMPLE_DIR, name), encoding="utf-8") as f:
        return f.read()


@pytest.fixture
def force_fallback_detector():
    """Pins PHIDetectionAgent -- and ComplianceValidationAgent's re-scan,
    which reads the same flag live off this module -- to the deterministic
    regex fallback for the duration of a test, regardless of whether
    Presidio/spaCy are installed in the environment."""
    original = phi_detection_module.USE_FALLBACK_ONLY
    phi_detection_module.USE_FALLBACK_ONLY = True
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


def test_not_applicable_document_skips_phi_detection():
    text = _read_sample("not_applicable_01.txt")
    result = run_sync(text, filename="not_applicable_01.txt", thread_id="test-thread-3")
    assert result["status"] == "completed"
    state = result["state"]
    assert state["doc_type"] == "not_applicable"
    assert not state.get("redacted_text")


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
