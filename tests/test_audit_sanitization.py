"""
tests/test_audit_sanitization.py

Covers the PHI-in-audit-log persistence leak fix in storage/audit_store.py:
_sanitize_audit_log() and _sanitize_compliance_report() strip raw PHI
text before AuditStore.store_audit() writes a row, so the long-lived,
queryable audit_records table (and GET /records/{id}/audit, which reads
straight from it) never holds literal patient identifiers -- only
length-preserving placeholders.

This is deliberately scoped to the persisted store only. The in-flight
GraphState / LangGraph checkpointer and the immediate API response are
NOT touched by this fix (and shouldn't be -- see docstrings in
storage/audit_store.py for why).

Run with: pytest -v tests/test_audit_sanitization.py
"""
from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from storage.audit_store import (
    AuditStore,
    _sanitize_audit_log,
    _sanitize_compliance_report,
)


def _sample_audit_log():
    return [
        {
            "timestamp": "2026-07-26T00:00:00+00:00",
            "agent": "RedactionAgent",
            "action": "redacted",
            "phi_type": "PERSON",
            "span_text": "Ramya Annamraju",
            "confidence": 0.9,
            "reviewer_action": None,
            "notes": None,
        },
        {
            "timestamp": "2026-07-26T00:00:01+00:00",
            "agent": "AuditReportAgent",
            "action": "report_generated",
            "phi_type": None,
            "span_text": None,
            "confidence": None,
            "reviewer_action": None,
            "notes": "status=PASS",
        },
    ]


def _sample_compliance_report():
    return {
        "filename": "note.txt",
        "doc_type": "clinical_note",
        "validation_status": "FAIL",
        "remaining_phi_spans_after_validation": [
            {"phi_type": "MRN", "text": "MRN-00482913", "confidence": 0.55},
        ],
        "requires_manual_followup": True,
    }


def test_sanitize_audit_log_strips_span_text_but_keeps_shape():
    sanitized = _sanitize_audit_log(_sample_audit_log())
    assert sanitized[0]["span_text"] == "[REDACTED:15chars]"
    assert "Ramya Annamraju" not in str(sanitized)
    # Non-PHI-bearing fields pass through untouched.
    assert sanitized[0]["phi_type"] == "PERSON"
    assert sanitized[0]["confidence"] == 0.9
    # Entries with no span_text (None) are left alone, not stringified.
    assert sanitized[1]["span_text"] is None


def test_sanitize_audit_log_does_not_mutate_input():
    original = _sample_audit_log()
    _sanitize_audit_log(original)
    assert original[0]["span_text"] == "Ramya Annamraju"


def test_sanitize_compliance_report_strips_remaining_span_text():
    sanitized = _sanitize_compliance_report(_sample_compliance_report())
    remaining = sanitized["remaining_phi_spans_after_validation"]
    assert remaining[0]["text"] == "[REDACTED:12chars]"
    assert "MRN-00482913" not in str(sanitized)
    # Scalar summary fields (what compliance reporting actually needs)
    # are untouched.
    assert sanitized["validation_status"] == "FAIL"
    assert sanitized["requires_manual_followup"] is True


def test_sanitize_compliance_report_handles_no_remaining_spans():
    report = {"validation_status": "PASS", "remaining_phi_spans_after_validation": []}
    sanitized = _sanitize_compliance_report(report)
    assert sanitized["remaining_phi_spans_after_validation"] == []


def test_store_audit_persists_sanitized_data_end_to_end():
    store = AuditStore("sqlite:///:memory:")
    store.store_audit(
        record_id="rec-1",
        compliance_report=_sample_compliance_report(),
        audit_log=_sample_audit_log(),
    )

    fetched = store.get_audit("rec-1")
    assert fetched is not None

    # Raw PHI values must not appear anywhere in what got persisted.
    assert "Ramya Annamraju" not in str(fetched)
    assert "MRN-00482913" not in str(fetched)

    # But the audit trail is still structurally useful: type, confidence,
    # pass/fail, and counts all survive.
    assert fetched["audit_log"][0]["phi_type"] == "PERSON"
    assert fetched["audit_log"][0]["span_text"] == "[REDACTED:15chars]"
    assert fetched["validation_status"] == "FAIL"
    assert fetched["compliance_report"]["remaining_phi_spans_after_validation"][0]["phi_type"] == "MRN"


def test_store_audit_list_and_range_queries_also_stay_sanitized():
    store = AuditStore("sqlite:///:memory:")
    store.store_audit(
        record_id="rec-2",
        compliance_report=_sample_compliance_report(),
        audit_log=_sample_audit_log(),
    )
    listed = store.list_audits()
    assert len(listed) == 1
    assert "Ramya Annamraju" not in str(listed)
    assert "MRN-00482913" not in str(listed)


if __name__ == "__main__":
    import pytest
    raise SystemExit(pytest.main([__file__, "-v"]))
