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

from langgraph.types import interrupt

from agents import phi_detection_agent as _phi_detection_module
from graph.state import GraphState

DEFAULT_MAX_RETRIES = 2
_PLACEHOLDER_PATTERN = re.compile(r"\[[A-Z_]+\]")

# Escalation auto-redact: PHI types PHIDetectionAgent's own CONFIDENCE_THRESHOLDS
# already treats as near-deterministic once matched (regex/pattern-backed,
# ~1.0 precision per eval/evaluate.py -- see that dict's comment for the
# actual numbers). Derived from CONFIDENCE_THRESHOLDS rather than a separate
# hardcoded list so recalibrating thresholds there can't silently drift out
# of sync with what escalation treats as "safe to auto-redact." A residual
# span of one of these types reaching escalation means redaction missed it
# for some structural reason (not that the type is ambiguous), so there's no
# real reviewer judgment call to make -- auto-redacting it is strictly
# faster than making a human rubber-stamp something this deterministic.
AUTO_REDACT_ESCALATION_TYPES = frozenset(
    phi_type for phi_type, threshold in _phi_detection_module.CONFIDENCE_THRESHOLDS.items()
    if threshold <= 0.6
)


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

    checks, compliance_score = _build_compliance_checks(state.get("phi_spans", []), remaining)

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
        "compliance_checks": checks,
        "compliance_score": compliance_score,
        "audit_log": audit_log,
    }


def _build_compliance_checks(
    all_detected_spans: list, remaining_spans: list
) -> tuple[list[dict], float]:
    """One named check per PHI category this document actually triggered
    (not per Safe-Harbor category in the abstract -- a category never
    detected at all isn't this check's job, that's
    PHIValidationAgent.missing_expected_identifier_types, a separate
    schema-completeness concern). Each check reports whether every span
    of that category is now clear of the redacted text, giving a
    per-category audit trail rather than one opaque pass/fail boolean --
    e.g. "MRN: clear" / "PERSON: 1 residual span" instead of just
    validation_status=FAIL with no indication of which category failed.

    all_detected_spans reflects the most recent PHIDetectionAgent pass
    (phi_spans is overwritten, not accumulated, each retry) -- consistent
    with what redaction/compliance_validation were actually acting on for
    this pass, not a cross-run history.

    compliance_score is the fraction of triggered categories fully clear
    (1.0 if the document triggered no PHI categories at all -- vacuously
    compliant, nothing to redact).
    """
    detected_types = sorted({s["phi_type"] for s in all_detected_spans})
    if not detected_types:
        return [], 1.0

    residual_counts: dict[str, int] = {}
    for s in remaining_spans:
        residual_counts[s["phi_type"]] = residual_counts.get(s["phi_type"], 0) + 1

    checks = []
    for phi_type in detected_types:
        count = residual_counts.get(phi_type, 0)
        checks.append({
            "category": phi_type,
            "passed": count == 0,
            "residual_count": count,
            "details": "clear" if count == 0 else f"{count} residual span(s) after redaction",
        })

    passed_count = sum(1 for c in checks if c["passed"])
    score = round(passed_count / len(checks), 3)
    return checks, score


def _is_inside_placeholder(text: str, span) -> bool:
    """True if this detected span sits entirely inside an already-placed
    [TYPE] redaction tag (avoids false FAILs from e.g. 'Dr. [PERSON]')."""
    for m in _PLACEHOLDER_PATTERN.finditer(text):
        if span["start"] >= m.start() and span["end"] <= m.end():
            return True
    return False


def route_after_validation(state: GraphState) -> str:
    """Conditional edge used by graph/workflow.py.

    Retries-exhausted-while-FAIL used to fall straight through to
    audit_report, flagged for manual follow-up and nothing more. It now
    gets one additional, capped chance to actually resolve: escalate to a
    dedicated human review pass (escalate_to_review -> escalation_review
    -> escalation_redaction -> back here) before giving up. The
    escalation_attempted guard means this can only happen once per run --
    if the escalation redaction still leaves the document failing
    validation, this function falls through to audit_report on the next
    call instead of escalating again.
    """
    status = state.get("validation_status", "FAIL")
    retry_count = state.get("retry_count", 0)
    max_retries = state.get("max_retries", DEFAULT_MAX_RETRIES)

    if status == "PASS":
        return "audit_report"
    if retry_count < max_retries:
        return "retry"
    if not state.get("escalation_attempted", False):
        return "escalate_to_review"
    return "audit_report"  # already escalated once and still failing — flagged FAIL in the report


def escalate_to_review(state: GraphState) -> GraphState:
    """Runs exactly once per pipeline run, only on the transition into the
    retry-cap escalation path (see route_after_validation above).

    Splits the still-detected PHI spans in two, rather than sending
    everything to a human:
      - AUTO_REDACT_ESCALATION_TYPES spans (regex-backed, near-deterministic
        per PHIDetectionAgent's own calibration) are auto-approved right
        here and stashed in human_reviewed_spans -- no reviewer judgment
        call actually exists for these; a residual span of one of these
        types means redaction missed it structurally, not that the type
        itself is ambiguous.
      - Everything else (noisy NER types: PERSON, DATE_TIME, LOCATION, NRP)
        goes into escalation_spans for escalation_review_agent to present.

    route_after_escalate (below) skips the human pause entirely if nothing
    noisy remains. escalation_attempted is set here either way, so
    route_after_validation can't send a still-failing document through
    this escalation path a second time.
    """
    remaining = state.get("remaining_phi_spans", [])
    auto_spans = [s for s in remaining if s["phi_type"] in AUTO_REDACT_ESCALATION_TYPES]
    needs_review_spans = [s for s in remaining if s["phi_type"] not in AUTO_REDACT_ESCALATION_TYPES]

    audit_log = list(state.get("audit_log", []))
    audit_log.append({
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "agent": "ComplianceValidationAgent",
        "action": "escalated_to_human_review",
        "phi_type": None,
        "span_text": None,
        "confidence": None,
        "reviewer_action": None,
        "notes": (
            f"retries exhausted while FAIL — {len(auto_spans)} span(s) auto-redacted "
            f"(near-deterministic type), {len(needs_review_spans)} span(s) escalated to human review"
        ),
    })
    for s in auto_spans:
        audit_log.append({
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "agent": "ComplianceValidationAgent",
            "action": "auto_redacted",
            "phi_type": s["phi_type"],
            "span_text": s["text"],
            "confidence": s["confidence"],
            "reviewer_action": None,
            "notes": "escalation_autofix: near-deterministic type, no reviewer needed",
        })

    return {
        **state,
        "escalation_spans": needs_review_spans,
        "human_reviewed_spans": [{**s, "source_agent": "escalation_autofix"} for s in auto_spans],
        "escalation_attempted": True,
        "audit_log": audit_log,
    }


def route_after_escalate(state: GraphState) -> str:
    """Conditional edge used by graph/workflow.py. Skips the human pause
    entirely if escalate_to_review's auto-redact split cleared everything
    that needed a reviewer's judgment call."""
    return "escalation_review" if state.get("escalation_spans") else "escalation_redaction"


def escalation_review_agent(state: GraphState) -> GraphState:
    """Dedicated interrupt() node for the retry-cap escalation path —
    deliberately separate from agents/human_review_agent.py's
    HumanReviewAgent, which reviews PRE-redaction low_confidence_spans.
    This node reviews POST-redaction leftover spans (escalation_spans,
    set by escalate_to_review above) once normal retries are exhausted
    and ComplianceValidationAgent still fails.

    Kept as its own named graph node rather than a branch inside
    HumanReviewAgent for a concrete reason, not just naming taste:
    HumanReviewAgent has one fixed outgoing edge, to redaction_agent,
    whose _spans_to_redact() merges high_confidence_spans +
    human_reviewed_spans. At the point escalation happens,
    high_confidence_spans reflects the most recent PHIDetectionAgent
    pass — spans that were ALREADY redacted earlier in this same retry
    iteration, before compliance_validation ever ran. Re-feeding them
    through redaction_agent a second time would apply their old
    start/end offsets against text that has since shifted (every prior
    redaction changes downstream offsets), corrupting the output. Routing
    escalation to its own node with its own fixed edge — to
    escalation_redaction_agent (agents/redaction_agent.py), which redacts
    ONLY the freshly-approved escalation spans — sidesteps that failure
    mode entirely instead of teaching one shared node two incompatible
    behaviors.
    """
    spans = state.get("escalation_spans", [])
    if not spans:
        # Shouldn't normally be reached at all -- route_after_escalate
        # routes straight to escalation_redaction when there's nothing
        # noisy left after escalate_to_review's auto-redact split. Keep
        # this node safe to call directly anyway (e.g. in tests), and
        # preserve whatever escalate_to_review already auto-approved
        # rather than clobbering it.
        return {**state, "human_decisions": []}

    # --- Pause here. Execution resumes when the caller sends decisions,
    # via the same /redact/resume contract HumanReviewAgent uses (same
    # decision shape: span_index/approved/reviewer) — the caller doesn't
    # need special-case handling just because this is the escalation path.
    decisions = interrupt({
        "reason": "compliance_validation_escalation",
        "instructions": (
            "These spans still matched PHI patterns after redaction, and "
            f"{state.get('max_retries', DEFAULT_MAX_RETRIES)} retry pass(es) failed to clear "
            "them. Approve to redact now, or reject if this is a false "
            "positive (e.g. a placeholder tag or non-PHI pattern match)."
        ),
        "spans": [
            {
                "span_index": i,
                "text": s["text"],
                "phi_type": s["phi_type"],
                "confidence": s["confidence"],
            }
            for i, s in enumerate(spans)
        ],
    })

    approved_spans = []
    rejected_spans = list(state.get("rejected_spans", []))
    audit_log = list(state.get("audit_log", []))
    now = datetime.now(timezone.utc).isoformat()

    decision_by_index = {d["span_index"]: d for d in decisions}
    for i, span in enumerate(spans):
        decision = decision_by_index.get(i, {"approved": True})  # fail-safe: default approve
        approved = bool(decision.get("approved", True))
        reviewer = decision.get("reviewer", "unknown_reviewer")

        if approved:
            approved_spans.append({**span, "source_agent": "human_review_escalation"})
        else:
            key = {"phi_type": span["phi_type"], "text": span["text"]}
            if key not in rejected_spans:
                rejected_spans.append(key)

        audit_log.append({
            "timestamp": now,
            "agent": "EscalationReviewAgent",
            "action": "human_approved" if approved else "human_rejected",
            "phi_type": span["phi_type"],
            "span_text": span["text"],
            "confidence": span["confidence"],
            "reviewer_action": reviewer,
            "notes": "retry_cap_escalation",
        })

    return {
        **state,
        # Append to whatever escalate_to_review already auto-approved
        # (AUTO_REDACT_ESCALATION_TYPES spans) rather than overwrite it --
        # escalation_redaction_agent needs both sets redacted together.
        "human_reviewed_spans": list(state.get("human_reviewed_spans", [])) + approved_spans,
        "human_decisions": decisions,
        "rejected_spans": rejected_spans,
        "audit_log": audit_log,
        "escalation_spans": [],  # consumed
    }


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
