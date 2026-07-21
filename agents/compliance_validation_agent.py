"""
ComplianceValidationAgent (Member 4)

Re-scans the redacted text with the same detection logic PHIDetectionAgent
uses. If any PHI still turns up, validation_status = FAIL and the graph
loops back to PHIDetectionAgent (graph/workflow.py wires the conditional
edge) — capped at max_retries (default 2) so a stubborn note can't loop
forever. If the cap is hit while still FAIL, we pass through to
AuditReportAgent anyway, flagged clearly as "exhausted retries" so a human
sees it rather than the pipeline silently dropping the document.
"""
from __future__ import annotations

import re
from datetime import datetime, timezone

from agents import phi_detection_agent as _phi_detection_module
from graph.state import GraphState

DEFAULT_MAX_RETRIES = 2
_PLACEHOLDER_PATTERN = re.compile(r"\[[A-Z_]+\]")


def compliance_validation_agent(state: GraphState) -> GraphState:
    redacted_text = state.get("redacted_text", "")

    if _phi_detection_module._PRESIDIO_AVAILABLE and not _phi_detection_module.USE_FALLBACK_ONLY:
        try:
            remaining = _phi_detection_module._detect_presidio(redacted_text)
        except Exception:
            remaining = _phi_detection_module._detect_fallback(redacted_text)
    else:
        remaining = _phi_detection_module._detect_fallback(redacted_text)

    # Placeholder tags like "[PERSON]" can themselves trip a naive detector
    # (e.g. "Dr. [PERSON]" still matches the Mr/Mrs/Dr fallback pattern).
    # Filter out spans that are wholly inside an already-placed [TAG].
    remaining = [s for s in remaining if not _is_inside_placeholder(redacted_text, s)]

    # Defense-in-depth: also drop anything a human already rejected as
    # "not PHI" in an earlier retry round (PHIDetectionAgent filters these
    # too -- see phi_detection_agent._filter_rejected -- but validation
    # re-scans independently, so it needs the same guard or a rejected
    # span would still register as a FAIL here).
    rejected_spans = state.get("rejected_spans", [])
    if rejected_spans:
        rejected_keys = {(r["phi_type"], r["text"]) for r in rejected_spans}
        remaining = [s for s in remaining if (s["phi_type"], s["text"]) not in rejected_keys]

    status = "FAIL" if remaining else "PASS"
    retry_count = state.get("retry_count", 0)
    max_retries = state.get("max_retries", DEFAULT_MAX_RETRIES)

    audit_log = list(state.get("audit_log", []))
    audit_log.append({
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "agent": "ComplianceValidationAgent",
        "action": "validated",
        "phi_type": None,
        "span_text": None,
        "confidence": None,
        "reviewer_action": None,
        "notes": f"status={status}, remaining_spans={len(remaining)}, retry_count={retry_count}",
    })

    return {
        **state,
        "validation_status": status,
        "remaining_phi_spans": remaining,
        "max_retries": max_retries,
        "audit_log": audit_log,
    }


def _is_inside_placeholder(text: str, span) -> bool:
    """True if this detected span sits entirely inside an already-placed
    [TYPE] redaction tag (avoids false FAILs from e.g. 'Dr. [PERSON]')."""
    for m in _PLACEHOLDER_PATTERN.finditer(text):
        if span["start"] >= m.start() and span["end"] <= m.end():
            return True
    return False


def route_after_validation(state: GraphState) -> str:
    """Conditional edge used by graph/workflow.py."""
    status = state.get("validation_status", "FAIL")
    retry_count = state.get("retry_count", 0)
    max_retries = state.get("max_retries", DEFAULT_MAX_RETRIES)

    if status == "PASS":
        return "audit_report"
    if retry_count < max_retries:
        return "retry"
    return "audit_report"  # retries exhausted — still produce output, flagged FAIL in the report


def increment_retry(state: GraphState) -> GraphState:
    """Node run on the retry edge before looping back to PHIDetectionAgent."""
    audit_log = list(state.get("audit_log", []))
    new_count = state.get("retry_count", 0) + 1
    audit_log.append({
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "agent": "ComplianceValidationAgent",
        "action": "retry",
        "phi_type": None,
        "span_text": None,
        "confidence": None,
        "reviewer_action": None,
        "notes": f"retry_count now {new_count}",
    })
    return {**state, "retry_count": new_count, "audit_log": audit_log}
