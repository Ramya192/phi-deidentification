"""
graph/workflow.py (Member 1)

Wires the full PHI de-identification pipeline as a LangGraph StateGraph:

    START
      -> classification            (Member 2)
      -> [route: not_applicable]   -> END
      -> [route: clinical doc]     -> phi_detection      (Member 3)
      -> phi_detection              -> phi_validation      (schema/completeness check)
      -> phi_validation              -> llm_adjudication    (agentic tier -- no-op unless
                                                              PHI_DEID_ADJUDICATION_BACKEND=llm)
      -> [route: has low-confidence spans] -> human_review (Member 5, interrupt())
      -> [route: all high-confidence]      -> redaction    (Member 4)
      -> redaction                 -> compliance_validation (Member 4)
      -> [route: PASS]             -> audit_report        (Member 6) -> END
      -> [route: FAIL, retries left]  -> retry_bump -> phi_detection  (loop, re-runs phi_validation too)
      -> [route: FAIL, retries exhausted, not yet escalated]
            -> escalate_to_review (auto-redacts near-deterministic types,
               splits off noisy types)
            -> [route: noisy types remain]  -> escalation_review (interrupt()) -> escalation_redaction
            -> [route: nothing noisy left]  -> escalation_redaction directly (no human pause)
            -> compliance_validation  (one more check, then always audit_report either way)
      -> [route: FAIL, retries exhausted, already escalated] -> audit_report -> END

A checkpointer is compiled in because HumanReviewAgent uses interrupt() --
LangGraph needs to persist state across the pause/resume boundary.
Defaults to a SQLite-backed checkpointer (single file at
data/checkpoints.sqlite, no external infra) so an in-flight human-review
session survives a process restart, which the previous in-memory
MemorySaver did not -- falls back to MemorySaver automatically if the
optional `langgraph-checkpoint-sqlite` package isn't installed, so the
graph still runs in a bare-bones dev environment (just without
restart-survival). For multi-process/multi-replica deployment, swap for a
Postgres or Redis checkpointer instead -- SQLite's single-writer model
doesn't scale past one process.
"""
from __future__ import annotations

import functools
import os
import sqlite3
import time
from datetime import datetime, timezone

from langgraph.graph import END, START, StateGraph

from agents.audit_report_agent import audit_report_agent
from agents.classification_agent import classification_agent, route_after_classification
from agents.compliance_validation_agent import (
    compliance_validation_agent,
    escalate_to_review,
    escalation_review_agent,
    increment_retry,
    route_after_escalate,
    route_after_validation,
)
from agents.human_review_agent import human_review_agent
from agents.llm_adjudication_agent import llm_adjudication_agent
from agents.phi_detection_agent import phi_detection_agent, route_after_detection
from agents.phi_validation_agent import phi_validation_agent
from agents.redaction_agent import escalation_redaction_agent, redaction_agent
from graph.state import GraphState
from tracing import tracing_extras

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(THIS_DIR)
CHECKPOINT_DB_PATH = os.path.join(PROJECT_ROOT, "data", "checkpoints.sqlite")


# Set by _build_checkpointer() below -- exposed via get_checkpointer_backend()
# so api/main.py's /health can report real readiness (restart-survivable
# "sqlite" vs. degraded "memory") instead of a static "ok" that's true
# whether or not the persistent backend actually loaded.
_checkpointer_backend = "unknown"


def get_checkpointer_backend() -> str:
    """"sqlite" (restart-survivable) or "memory" (degraded -- human-review
    sessions and audit-adjacent pipeline state won't survive a restart)."""
    return _checkpointer_backend


def _build_checkpointer():
    """SQLite by default (survives a process restart); falls back to
    MemorySaver if the optional sqlite checkpoint package isn't installed."""
    global _checkpointer_backend
    try:
        from langgraph.checkpoint.sqlite import SqliteSaver

        os.makedirs(os.path.dirname(CHECKPOINT_DB_PATH), exist_ok=True)
        # check_same_thread=False: FastAPI/uvicorn can serve requests on
        # different threads, and this single connection is shared across
        # the whole process's request lifetime (not scoped per-request).
        conn = sqlite3.connect(CHECKPOINT_DB_PATH, check_same_thread=False)
        _checkpointer_backend = "sqlite"
        return SqliteSaver(conn)
    except ImportError:
        from langgraph.checkpoint.memory import MemorySaver

        print(
            "WARNING: langgraph-checkpoint-sqlite not installed -- falling back to "
            "in-memory checkpointing. Human-review sessions will NOT survive a "
            "process restart. Run: pip install langgraph-checkpoint-sqlite"
        )
        _checkpointer_backend = "memory"
        return MemorySaver()


# ---------------------------------------------------------------------------
# Observability: wraps each node with wall-clock timing, recorded into
# state["node_timings"] rather than each agent instrumenting itself. This
# is deliberately a thin wrapper at the graph-wiring layer (not inside the
# agents) so timing is a cross-cutting concern any node gets "for free" by
# being registered here, and adding/removing it never touches agent logic.
# human_review and escalation_review are intentionally left un-timed
# below: both call interrupt(), genuinely pausing execution waiting on a
# person, so wall-clock time there measures reviewer response time, not
# agent performance -- mixing that into the same latency metrics would be
# misleading.
# ---------------------------------------------------------------------------
def _timed(node_name: str, fn):
    @functools.wraps(fn)
    def wrapper(state: GraphState) -> GraphState:
        start = time.perf_counter()
        result = fn(state)
        elapsed_ms = round((time.perf_counter() - start) * 1000, 2)

        timings = list(state.get("node_timings", []))
        timings.append({
            "node": node_name,
            "elapsed_ms": elapsed_ms,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        })
        # result may or may not already carry node_timings depending on
        # whether the wrapped agent does `**state` passthrough -- all of
        # ours do, but overwrite defensively either way so timing is never
        # silently dropped.
        return {**result, "node_timings": timings}

    return wrapper


def build_graph():
    graph = StateGraph(GraphState)

    graph.add_node("classification", _timed("classification", classification_agent))
    graph.add_node("phi_detection", _timed("phi_detection", phi_detection_agent))
    graph.add_node("phi_validation", _timed("phi_validation", phi_validation_agent))
    graph.add_node("llm_adjudication", _timed("llm_adjudication", llm_adjudication_agent))
    graph.add_node("human_review", human_review_agent)  # see _timed() docstring: excluded on purpose
    graph.add_node("redaction", _timed("redaction", redaction_agent))
    graph.add_node("compliance_validation", _timed("compliance_validation", compliance_validation_agent))
    graph.add_node("retry_bump", increment_retry)
    graph.add_node("escalate_to_review", _timed("escalate_to_review", escalate_to_review))
    graph.add_node("escalation_review", escalation_review_agent)  # see _timed() docstring: excluded on purpose
    graph.add_node("escalation_redaction", _timed("escalation_redaction", escalation_redaction_agent))
    graph.add_node("audit_report", _timed("audit_report", audit_report_agent))

    graph.add_edge(START, "classification")

    graph.add_conditional_edges(
        "classification",
        route_after_classification,
        {"phi_detection": "phi_detection", "end_not_applicable": END},
    )

    # PHIValidationAgent runs on every detection pass (including retries,
    # since retry_bump loops back to phi_detection) -- it's a read-only
    # schema/completeness check over phi_spans, so re-running it costs
    # nothing and keeps the audit trail's schema_completeness_score current
    # against whatever PHIDetectionAgent just found.
    graph.add_edge("phi_detection", "phi_validation")

    # LLMAdjudicationAgent: the project's genuinely agentic tier -- an LLM
    # reasons over PHIDetectionAgent's low-confidence spans (tool-calling +
    # structured Pydantic output) and resolves the clear cases itself, so
    # only genuinely ambiguous spans still reach a human. A true no-op
    # unless PHI_DEID_ADJUDICATION_BACKEND=llm is set (see that module's
    # docstring) -- route_after_detection below is unchanged and just
    # reads whatever low_confidence_spans this node left behind.
    graph.add_edge("phi_validation", "llm_adjudication")

    graph.add_conditional_edges(
        "llm_adjudication",
        route_after_detection,
        {"human_review": "human_review", "redaction": "redaction"},
    )

    graph.add_edge("human_review", "redaction")
    graph.add_edge("redaction", "compliance_validation")

    graph.add_conditional_edges(
        "compliance_validation",
        route_after_validation,
        {
            "audit_report": "audit_report",
            "retry": "retry_bump",
            "escalate_to_review": "escalate_to_review",
        },
    )

    graph.add_edge("retry_bump", "phi_detection")

    # Retry-cap escalation: one extra, capped human-review pass on
    # whatever PHI survived redaction + all normal retries, before giving
    # up. Loops back into compliance_validation for one final check --
    # route_after_validation's escalation_attempted guard means a
    # still-failing result here falls through to audit_report instead of
    # escalating again. See agents/compliance_validation_agent.py's
    # escalation_review_agent docstring for why this uses its own nodes
    # rather than reusing human_review/redaction.
    graph.add_conditional_edges(
        "escalate_to_review",
        route_after_escalate,
        {"escalation_review": "escalation_review", "escalation_redaction": "escalation_redaction"},
    )
    graph.add_edge("escalation_review", "escalation_redaction")
    graph.add_edge("escalation_redaction", "compliance_validation")

    graph.add_edge("audit_report", END)

    checkpointer = _build_checkpointer()
    return graph.compile(checkpointer=checkpointer)


# Module-level compiled graph, ready to import elsewhere (api/main.py, tests, etc.)
compiled_graph = build_graph()


def run_sync(raw_text: str, filename: str = "uploaded_document", thread_id: str = "default") -> dict:
    """
    Convenience runner for scripts/tests. Runs until END or until an
    interrupt() is hit (human review needed). If interrupted, returns the
    interrupt payload instead of a final state -- caller should collect
    decisions and call `resume_sync(...)` with the same thread_id.
    """
    config = {"configurable": {"thread_id": thread_id}, **tracing_extras(thread_id, filename)}
    initial_state: GraphState = {
        "raw_text": raw_text,
        "filename": filename,
        "retry_count": 0,
        "max_retries": 2,
        "audit_log": [],
        "llm_usage_log": [],
    }
    result = compiled_graph.invoke(initial_state, config=config)

    if "__interrupt__" in result:
        return {"status": "interrupted", "interrupt": result["__interrupt__"][0].value, "thread_id": thread_id}
    return {"status": "completed", "state": result}


def resume_sync(decisions: list[dict], thread_id: str = "default") -> dict:
    """Resume a graph paused at HumanReviewAgent with reviewer decisions."""
    from langgraph.types import Command

    config = {"configurable": {"thread_id": thread_id}, **tracing_extras(thread_id)}
    result = compiled_graph.invoke(Command(resume=decisions), config=config)

    if "__interrupt__" in result:
        return {"status": "interrupted", "interrupt": result["__interrupt__"][0].value, "thread_id": thread_id}
    return {"status": "completed", "state": result}


# ---------------------------------------------------------------------------
# Streaming variants -- same graph, same checkpointer, same final result
# shape as run_sync()/resume_sync() above, but yield a small event after
# EACH node finishes instead of only returning once the whole run (or the
# next interrupt) completes. This is what powers the live "which agent is
# running right now" progress display in the Streamlit UI -- a genuine
# node-by-node trace of the actual graph execution (LangGraph's own
# .stream(..., stream_mode="updates") API), not a simulated/fake progress
# bar timed to guess at how long things take.
#
# Every one of our node functions returns a full merged state dict (via
# `**state, ...` passthrough), and graph/workflow.py's _timed() wrapper
# preserves that -- so each "updates" event's payload IS the full current
# state after that node ran, not just a delta. That's what lets this
# yield the exact same final state run_sync() would have returned, just
# observed incrementally instead of all at once.
# ---------------------------------------------------------------------------
def _stream_graph(input_or_command, thread_id: str, filename: str | None = None):
    """Shared streaming core for run_stream()/resume_stream() below.
    Yields ("node", node_name, elapsed_state) while running, then exactly
    one final ("done", status, payload) event matching run_sync()'s return
    shape (status "interrupted" -> payload is the interrupt dict; status
    "completed" -> payload is the final state dict)."""
    config = {"configurable": {"thread_id": thread_id}, **tracing_extras(thread_id, filename)}
    final_state: dict | None = None
    interrupt_payload = None

    for event in compiled_graph.stream(input_or_command, config=config, stream_mode="updates"):
        for node_name, node_output in event.items():
            if node_name == "__interrupt__":
                interrupt_payload = node_output[0].value
                continue
            final_state = node_output
            yield ("node", node_name, node_output)

    if interrupt_payload is not None:
        yield ("done", "interrupted", interrupt_payload)
    else:
        yield ("done", "completed", final_state or {})


def run_stream(raw_text: str, filename: str = "uploaded_document", thread_id: str = "default"):
    """Streaming counterpart to run_sync() -- see _stream_graph() above."""
    initial_state: GraphState = {
        "raw_text": raw_text,
        "filename": filename,
        "retry_count": 0,
        "max_retries": 2,
        "audit_log": [],
        "llm_usage_log": [],
    }
    yield from _stream_graph(initial_state, thread_id, filename)


def resume_stream(decisions: list[dict], thread_id: str = "default"):
    """Streaming counterpart to resume_sync() -- see _stream_graph() above."""
    from langgraph.types import Command

    yield from _stream_graph(Command(resume=decisions), thread_id)
