"""
tests/test_ambiguous_identifier_routing.py

Covers the low-confidence routing path exercised by the new
"Internal case reference: 048261953." line added to
data/txt_format/discharge_summary_01.txt (all 3 formats) this session.

Real-world background: on the Presidio backend, a bare, unlabeled 9-digit
number with no surrounding phone/SSN/account context words is caught by
Presidio's *built-in* PhoneRecognizer (the `phonenumbers`-library-backed
one, not this project's custom phone_ext_recognizer -- that one requires
a full 10 digits) at a flat score of 0.4. Verified directly against the
installed presidio-analyzer package, not assumed:

    >>> from presidio_analyzer.predefined_recognizers.generic.phone_recognizer import PhoneRecognizer
    >>> PhoneRecognizer().analyze(
    ...     text="Internal case reference: 048261953. Please retain for records.",
    ...     entities=["PHONE_NUMBER"], nlp_artifacts=None,
    ... )
    [type: PHONE_NUMBER, start: 25, end: 34, score: 0.4]

0.4 is below this project's PHONE_NUMBER threshold (0.65), so it should
land in low_confidence_spans and route to human_review -- this file tests
that routing decision directly (backend-agnostic, via scripted spans),
the same way tests/test_escalation_auto_redact.py already tests the
auto-redact split. It does not require a real spaCy/Presidio install to
run, matching every other test in this suite.

This also covers the CONFIDENCE_THRESHOLDS "SSN" vs "US_SSN" key fix in
agents/phi_detection_agent.py: the regex fallback backend emits type
"SSN", but Presidio's real UsSsnRecognizer entity type is "US_SSN" --
before this session's fix, the latter silently fell back to
HIGH_CONFIDENCE_THRESHOLD (0.7) instead of the intended 0.6.
"""
from __future__ import annotations

import os
import sys
import uuid

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pytest

from agents import phi_detection_agent as phi_detection_module
from agents import redaction_agent as redaction_module
from agents.phi_detection_agent import _threshold_for, route_after_detection
from graph.workflow import compiled_graph

_AMBIGUOUS_PHONE_SPAN = {
    "start": 26, "end": 35, "text": "048261953", "phi_type": "PHONE_NUMBER",
    "confidence": 0.4, "source_agent": "test_mock",
}


def test_ssn_and_us_ssn_share_the_same_threshold():
    """Regression test for the key-mismatch bug: the fallback backend's
    "SSN" and Presidio's real "US_SSN" entity type must resolve to the
    same intended threshold (0.6), not silently diverge to the 0.7
    default for one of them."""
    assert _threshold_for("SSN") == 0.6
    assert _threshold_for("US_SSN") == 0.6


def test_route_after_detection_sends_low_confidence_phone_to_human_review():
    """A 0.4-confidence PHONE_NUMBER span (below the 0.65 threshold) must
    route to human_review, not straight to redaction."""
    state = {"low_confidence_spans": [_AMBIGUOUS_PHONE_SPAN]}
    assert route_after_detection(state) == "human_review"


def test_route_after_detection_sends_high_confidence_phone_to_redaction():
    """Sanity check on the other side of the same threshold: a
    high-confidence PHONE_NUMBER span should skip human review entirely."""
    state = {"low_confidence_spans": []}
    assert route_after_detection(state) == "redaction"


@pytest.fixture
def force_fallback_detector(monkeypatch):
    # Also pins ClassificationAgent to heuristic -- see
    # tests/test_integration.py's matching fixture docstring: a live
    # PHI_DEID_CLASSIFICATION_BACKEND=llm .env value can otherwise route
    # this test's synthetic note to not_applicable/END before it ever
    # reaches PHIDetectionAgent.
    monkeypatch.setenv("PHI_DEID_CLASSIFICATION_BACKEND", "heuristic")
    original_use_fallback_only = phi_detection_module.USE_FALLBACK_ONLY
    original_presidio_available = redaction_module._PRESIDIO_AVAILABLE
    phi_detection_module.USE_FALLBACK_ONLY = True
    monkeypatch.setattr(redaction_module, "_PRESIDIO_AVAILABLE", False)
    monkeypatch.setattr(redaction_module, "_redact_fallback", lambda text, spans: text)
    yield
    phi_detection_module.USE_FALLBACK_ONLY = original_use_fallback_only
    redaction_module._PRESIDIO_AVAILABLE = original_presidio_available


def test_full_graph_pauses_for_ambiguous_identifier(monkeypatch, force_fallback_detector):
    """End-to-end: scripting _detect_fallback to return exactly the
    low-confidence span Presidio produces in real life for
    discharge_summary_01's new "Internal case reference" line -- confirms
    the compiled graph actually pauses for human review rather than
    silently auto-redacting or silently passing it through."""
    call_count = {"n": 0}

    def scripted_detect(text):
        call_count["n"] += 1
        return [dict(_AMBIGUOUS_PHONE_SPAN)] if call_count["n"] <= 4 else []

    monkeypatch.setattr(phi_detection_module, "_detect_fallback", scripted_detect)

    thread_id = str(uuid.uuid4())
    config = {"configurable": {"thread_id": thread_id}}
    initial_state = {
        # Needs real clinical signal words so ClassificationAgent's
        # heuristic backend routes this into the graph instead of
        # "not_applicable" -- the ambiguous reference line alone has no
        # clinical vocabulary for the heuristic scorer to key off of.
        "raw_text": (
            "Discharge Diagnosis: Acute exacerbation of COPD, resolved.\n"
            "Hospital Course: patient improved steadily.\n"
            "Internal case reference: 048261953. Please retain for records."
        ),
        "filename": "discharge_summary_01.txt",
        "retry_count": 0,
        "max_retries": 1,
        "audit_log": [],
    }

    result = compiled_graph.invoke(initial_state, config=config)

    assert "__interrupt__" in result, "ambiguous low-confidence identifier should pause for human review"
    interrupt_payload = result["__interrupt__"][0].value
    assert len(interrupt_payload["spans"]) == 1
    assert interrupt_payload["spans"][0]["phi_type"] == "PHONE_NUMBER"
    assert interrupt_payload["spans"][0]["confidence"] == 0.4


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
