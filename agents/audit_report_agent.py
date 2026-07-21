"""
AuditReportAgent (Member 6)

Terminal node. Assembles the two compliance deliverables from GraphState:
  - audit_log.json         full per-event trail (every detection, redaction,
                            human decision, retry) with source agent + confidence
  - compliance_report.json validation summary: pass/fail, retry count,
                            counts by PHI type, whether human review was
                            invoked, and (if retries were exhausted while
                            still failing) an explicit flag for follow-up.

Both are also attached onto GraphState so the API/UI layer can hand them
back to the caller without a second read from disk.
"""
from __future__ import annotations

from collections import Counter
from datetime import datetime, timezone

from graph.state import GraphState


def audit_report_agent(state: GraphState) -> GraphState:
    validation_status = state.get("validation_status", "PENDING")
    retry_count = state.get("retry_count", 0)
    max_retries = state.get("max_retries", 2)
    remaining = state.get("remaining_phi_spans", [])
    all_spans = state.get("phi_spans", [])
    human_decisions = state.get("human_decisions", [])

    phi_type_counts = Counter(s["phi_type"] for s in all_spans)
    retries_exhausted = validation_status == "FAIL" and retry_count >= max_retries

    compliance_report = {
        "filename": state.get("filename", "unknown"),
        "doc_type": state.get("doc_type", "unknown"),
        "doc_type_confidence": state.get("doc_type_confidence"),
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "validation_status": validation_status,
        "retries_used": retry_count,
        "retries_exhausted_while_failing": retries_exhausted,
        "requires_manual_followup": retries_exhausted or bool(remaining),
        "total_phi_spans_detected": len(all_spans),
        "phi_spans_by_type": dict(phi_type_counts),
        "human_review_invoked": bool(human_decisions),
        "human_review_decisions_count": len(human_decisions),
        "remaining_phi_spans_after_validation": [
            {"phi_type": s["phi_type"], "text": s["text"], "confidence": s["confidence"]}
            for s in remaining
        ],
    }

    audit_log = list(state.get("audit_log", []))
    audit_log.append({
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "agent": "AuditReportAgent",
        "action": "report_generated",
        "phi_type": None,
        "span_text": None,
        "confidence": None,
        "reviewer_action": None,
        "notes": f"status={validation_status}, requires_manual_followup={compliance_report['requires_manual_followup']}",
    })

    return {
        **state,
        "compliance_report": compliance_report,
        "audit_log": audit_log,
    }
