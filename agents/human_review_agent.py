"""
HumanReviewAgent (Member 5)

A genuine LangGraph interrupt() node: when PHIDetectionAgent flags
low-confidence spans, graph execution pauses here and control returns to
whatever is driving the graph (Streamlit UI, FastAPI caller, test
harness). The caller inspects `low_confidence_spans`, collects
approve/reject decisions from a human reviewer, and resumes the graph
with `Command(resume=<decisions>)`.

This requires the compiled graph to have a checkpointer (see
graph/workflow.py — MemorySaver by default, swap for a persistent one in
production) since interrupt/resume relies on checkpointed state.

Decision payload the caller must send back on resume:
    [
        {"span_index": 0, "approved": True},
        {"span_index": 1, "approved": False},
        ...
    ]
one entry per item in state["low_confidence_spans"], indexed positionally.
"""
from __future__ import annotations

from datetime import datetime, timezone

from langgraph.types import interrupt

from graph.state import GraphState


def human_review_agent(state: GraphState) -> GraphState:
    low_conf_spans = state.get("low_confidence_spans", [])

    if not low_conf_spans:
        # Nothing to review — shouldn't normally happen given the routing,
        # but keep the node safe to call directly (e.g. in tests).
        return {**state, "human_reviewed_spans": [], "human_decisions": []}

    # --- Pause here. Execution resumes when the caller sends decisions. ---
    decisions = interrupt({
        "reason": "low_confidence_phi_spans",
        "instructions": "Approve or reject each span as genuine PHI.",
        "spans": [
            {
                "span_index": i,
                "text": s["text"],
                "phi_type": s["phi_type"],
                "confidence": s["confidence"],
            }
            for i, s in enumerate(low_conf_spans)
        ],
    })
    # decisions: list[{"span_index": int, "approved": bool, "reviewer": str | None}]

    approved_spans = []
    # Carry forward everything rejected on earlier retry rounds too --
    # a retry loop calls this node fresh each time, so without accumulating
    # here a prior round's rejection would be forgotten the moment a new
    # round starts.
    rejected_spans = list(state.get("rejected_spans", []))
    audit_log = list(state.get("audit_log", []))
    now = datetime.now(timezone.utc).isoformat()

    decision_by_index = {d["span_index"]: d for d in decisions}
    for i, span in enumerate(low_conf_spans):
        decision = decision_by_index.get(i, {"approved": True})  # fail-safe: default approve (favor redaction over PHI leak)
        approved = bool(decision.get("approved", True))
        reviewer = decision.get("reviewer", "unknown_reviewer")

        if approved:
            reviewed_span = {**span, "source_agent": "human_review"}
            approved_spans.append(reviewed_span)
        else:
            # Remember this as a standing "not PHI" determination so
            # PHIDetectionAgent/ComplianceValidationAgent stop re-flagging
            # the identical span on every retry pass -- see
            # phi_detection_agent._filter_rejected.
            key = {"phi_type": span["phi_type"], "text": span["text"]}
            if key not in rejected_spans:
                rejected_spans.append(key)

        audit_log.append({
            "timestamp": now,
            "agent": "HumanReviewAgent",
            "action": "human_approved" if approved else "human_rejected",
            "phi_type": span["phi_type"],
            "span_text": span["text"],
            "confidence": span["confidence"],
            "reviewer_action": reviewer,
            "notes": None,
        })

    return {
        **state,
        "human_reviewed_spans": approved_spans,
        "human_decisions": decisions,
        "rejected_spans": rejected_spans,
        "audit_log": audit_log,
    }
