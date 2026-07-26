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
from observability.llm_metrics import summarize_usage


def audit_report_agent(state: GraphState) -> GraphState:
    validation_status = state.get("validation_status", "PENDING")
    retry_count = state.get("retry_count", 0)
    max_retries = state.get("max_retries", 2)
    remaining = state.get("remaining_phi_spans", [])
    # Cumulative across every detection pass this run has done
    # (GraphState.all_detected_spans), not phi_spans (current pass only) --
    # otherwise a document that triggered even one retry would report only
    # that final pass's small residual-scan count here, dramatically
    # understating total_phi_spans_detected/phi_spans_by_type relative to
    # what was actually found and redacted across the whole run.
    all_spans = state.get("all_detected_spans") or state.get("phi_spans", [])
    human_decisions = state.get("human_decisions", [])

    phi_type_counts = Counter(s["phi_type"] for s in all_spans)
    retries_exhausted = validation_status == "FAIL" and retry_count >= max_retries
    missing_expected_identifier_types = state.get("missing_expected_identifier_types", [])

    # Observability summary -- graph/workflow.py's _timed() wrapper records
    # one entry per node execution (a node hit twice via the retry loop
    # gets two entries); this aggregates that into per-node totals plus a
    # grand total so the UI/report can show pipeline latency without the
    # caller having to parse the raw node_timings list themselves.
    node_timings = state.get("node_timings", [])
    node_timing_totals: dict[str, float] = {}
    for entry in node_timings:
        node_timing_totals[entry["node"]] = node_timing_totals.get(entry["node"], 0.0) + entry["elapsed_ms"]
    node_timing_totals = {k: round(v, 2) for k, v in node_timing_totals.items()}

    # Real per-call LLM token/cost usage (observability/llm_metrics.py) --
    # empty by_agent/zero totals whenever the default heuristic
    # classification backend (or any other LLM-free run) was used, since
    # llm_usage_log itself stays empty in that case.
    llm_usage_summary = summarize_usage(state.get("llm_usage_log", []))

    compliance_report = {
        "filename": state.get("filename", "unknown"),
        "doc_type": state.get("doc_type", "unknown"),
        "doc_type_confidence": state.get("doc_type_confidence"),
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "validation_status": validation_status,
        "retries_used": retry_count,
        "retries_exhausted_while_failing": retries_exhausted,
        "escalated_to_human_review": state.get("escalation_attempted", False),
        "compliance_checks": state.get("compliance_checks", []),
        "compliance_score": state.get("compliance_score", 1.0),
        "schema_completeness_score": state.get("schema_completeness_score"),
        "missing_expected_identifier_types": missing_expected_identifier_types,
        "requires_manual_followup": retries_exhausted or bool(remaining) or bool(missing_expected_identifier_types),
        "total_phi_spans_detected": len(all_spans),
        "phi_spans_by_type": dict(phi_type_counts),
        "human_review_invoked": bool(human_decisions),
        "human_review_decisions_count": len(human_decisions),
        "remaining_phi_spans_after_validation": [
            {"phi_type": s["phi_type"], "text": s["text"], "confidence": s["confidence"]}
            for s in remaining
        ],
        "node_timings_ms": node_timing_totals,
        "total_pipeline_ms": round(sum(node_timing_totals.values()), 2),
        "llm_usage_summary": llm_usage_summary,
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
