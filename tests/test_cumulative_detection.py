"""
tests/test_cumulative_detection.py

Regression test for a real bug found via manual testing (uploading the PDF
sample through the Streamlit UI): on any run that triggers at least one
retry, PHIDetectionAgent's second pass deliberately re-scans only the
*current* redacted_text for residual PHI (see that module's docstring) --
by design, a small, round-scoped list. But PHIValidationAgent's
completeness check and AuditReportAgent's final totals were both reading
straight off state["phi_spans"], which gets *replaced* (not merged) every
time PHIDetectionAgent runs. So a document that found PERSON/MRN/PHONE in
its first pass, then triggered one retry that only found 1 residual
DATE_TIME span, ended up reporting total_phi_spans_detected=1 and
missing_expected_identifier_types=["MRN", "PERSON"] in the final
compliance_report -- even though PERSON/MRN were genuinely found and
redacted in the first pass. The actual redacted output was correct; only
the bookkeeping was wrong.

Fix: GraphState.all_detected_spans accumulates every span found across
every pass (graph/state.py, agents/phi_detection_agent.py), and
PHIValidationAgent / AuditReportAgent now read from that instead of the
round-scoped phi_spans.
"""
from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from agents.audit_report_agent import audit_report_agent
from agents.phi_detection_agent import phi_detection_agent
from agents.phi_validation_agent import phi_validation_agent


def test_phi_detection_agent_accumulates_across_retry_passes(monkeypatch):
    import agents.phi_detection_agent as phi_detection_module

    # Round 1 (retry_count=0): finds 3 spans against raw_text.
    round1_spans = [
        {"start": 0, "end": 5, "text": "Alice", "phi_type": "PERSON", "confidence": 0.9, "source_agent": "test"},
        {"start": 10, "end": 15, "text": "12345", "phi_type": "MRN", "confidence": 0.9, "source_agent": "test"},
        {"start": 20, "end": 30, "text": "1/1/2020", "phi_type": "DATE_TIME", "confidence": 0.75, "source_agent": "test"},
    ]
    monkeypatch.setattr(phi_detection_module, "_detect_fallback", lambda text: round1_spans)
    phi_detection_module.USE_FALLBACK_ONLY = True

    state1 = {"raw_text": "Alice MRN 12345 seen 1/1/2020", "retry_count": 0, "rejected_spans": []}
    result1 = phi_detection_agent(state1)

    assert result1["phi_spans"] == round1_spans
    assert result1["all_detected_spans"] == round1_spans

    # Round 2 (retry_count=1): re-scans redacted_text, finds just 1 new
    # residual span PHIDetectionAgent missed the first time.
    round2_span = {"start": 0, "end": 4, "text": "5/5/2020", "phi_type": "DATE_TIME", "confidence": 0.7, "source_agent": "test"}
    monkeypatch.setattr(phi_detection_module, "_detect_fallback", lambda text: [round2_span])

    state2 = {
        **result1,
        "retry_count": 1,
        "redacted_text": "[PERSON] MRN [MRN] seen [DATE_TIME], also 5/5/2020",
    }
    result2 = phi_detection_agent(state2)

    # phi_spans (this pass only) is just the residual span -- unchanged,
    # round-scoped behavior, still correct and still needed for the
    # confidence-split/redaction this specific pass does.
    assert result2["phi_spans"] == [round2_span]

    # all_detected_spans must have ALL FOUR spans across both passes, not
    # just the second pass's one residual finding.
    assert result2["all_detected_spans"] == round1_spans + [round2_span]


def test_validation_and_audit_report_use_cumulative_spans_not_last_pass_only():
    """Directly reproduces the reported symptom: a state where phi_spans
    (last pass only) is small, but all_detected_spans (the whole run) is
    complete. PHIValidationAgent and AuditReportAgent must both report
    against the complete picture, not the last pass."""
    round1_spans = [
        {"start": 0, "end": 5, "text": "Alice", "phi_type": "PERSON", "confidence": 0.9, "source_agent": "test"},
        {"start": 10, "end": 15, "text": "12345", "phi_type": "MRN", "confidence": 0.9, "source_agent": "test"},
    ]
    round2_span = {"start": 0, "end": 8, "text": "5/5/2020", "phi_type": "DATE_TIME", "confidence": 0.7, "source_agent": "test"}

    state = {
        "doc_type": "clinical_note",  # expects {PERSON, DATE_TIME, MRN}
        "phi_spans": [round2_span],                       # last pass only -- what the bug used to read
        "all_detected_spans": round1_spans + [round2_span],  # the fix -- whole run
        "audit_log": [],
        "remaining_phi_spans": [],
        "validation_status": "PASS",
        "retry_count": 1,
        "max_retries": 2,
        "human_decisions": [],
        "node_timings": [],
        "llm_usage_log": [],
        "filename": "test.txt",
    }

    validated = phi_validation_agent(state)
    # Before the fix: missing_expected_identifier_types would wrongly list
    # PERSON and MRN here, since phi_spans (last pass) only has DATE_TIME.
    assert validated["missing_expected_identifier_types"] == []
    assert validated["schema_completeness_score"] == 1.0

    report_state = {**validated, "compliance_checks": [], "compliance_score": 1.0}
    reported = audit_report_agent(report_state)
    compliance_report = reported["compliance_report"]

    # Before the fix: total_phi_spans_detected would be 1 (just the
    # residual DATE_TIME), not 3.
    assert compliance_report["total_phi_spans_detected"] == 3
    assert compliance_report["phi_spans_by_type"] == {"PERSON": 1, "MRN": 1, "DATE_TIME": 1}
    assert compliance_report["missing_expected_identifier_types"] == []
    assert compliance_report["requires_manual_followup"] is False
