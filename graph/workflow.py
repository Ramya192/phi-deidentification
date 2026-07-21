"""
graph/workflow.py (Member 1)

Wires the full PHI de-identification pipeline as a LangGraph StateGraph:

    START
      -> classification            (Member 2)
      -> [route: not_applicable]   -> END
      -> [route: clinical doc]     -> phi_detection      (Member 3)
      -> [route: has low-confidence spans] -> human_review (Member 5, interrupt())
      -> [route: all high-confidence]      -> redaction    (Member 4)
      -> redaction                 -> compliance_validation (Member 4)
      -> [route: PASS]             -> audit_report        (Member 6) -> END
      -> [route: FAIL, retries left]  -> retry_bump -> phi_detection  (loop)
      -> [route: FAIL, retries exhausted] -> audit_report -> END

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

import os
import sqlite3

from langgraph.graph import END, START, StateGraph

from agents.audit_report_agent import audit_report_agent
from agents.classification_agent import classification_agent, route_after_classification
from agents.compliance_validation_agent import (
    compliance_validation_agent,
    increment_retry,
    route_after_validation,
)
from agents.human_review_agent import human_review_agent
from agents.phi_detection_agent import phi_detection_agent, route_after_detection
from agents.redaction_agent import redaction_agent
from graph.state import GraphState

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(THIS_DIR)
CHECKPOINT_DB_PATH = os.path.join(PROJECT_ROOT, "data", "checkpoints.sqlite")


def _build_checkpointer():
    """SQLite by default (survives a process restart); falls back to
    MemorySaver if the optional sqlite checkpoint package isn't installed."""
    try:
        from langgraph.checkpoint.sqlite import SqliteSaver

        os.makedirs(os.path.dirname(CHECKPOINT_DB_PATH), exist_ok=True)
        # check_same_thread=False: FastAPI/uvicorn can serve requests on
        # different threads, and this single connection is shared across
        # the whole process's request lifetime (not scoped per-request).
        conn = sqlite3.connect(CHECKPOINT_DB_PATH, check_same_thread=False)
        return SqliteSaver(conn)
    except ImportError:
        from langgraph.checkpoint.memory import MemorySaver

        print(
            "WARNING: langgraph-checkpoint-sqlite not installed -- falling back to "
            "in-memory checkpointing. Human-review sessions will NOT survive a "
            "process restart. Run: pip install langgraph-checkpoint-sqlite"
        )
        return MemorySaver()


def build_graph():
    graph = StateGraph(GraphState)

    graph.add_node("classification", classification_agent)
    graph.add_node("phi_detection", phi_detection_agent)
    graph.add_node("human_review", human_review_agent)
    graph.add_node("redaction", redaction_agent)
    graph.add_node("compliance_validation", compliance_validation_agent)
    graph.add_node("retry_bump", increment_retry)
    graph.add_node("audit_report", audit_report_agent)

    graph.add_edge(START, "classification")

    graph.add_conditional_edges(
        "classification",
        route_after_classification,
        {"phi_detection": "phi_detection", "end_not_applicable": END},
    )

    graph.add_conditional_edges(
        "phi_detection",
        route_after_detection,
        {"human_review": "human_review", "redaction": "redaction"},
    )

    graph.add_edge("human_review", "redaction")
    graph.add_edge("redaction", "compliance_validation")

    graph.add_conditional_edges(
        "compliance_validation",
        route_after_validation,
        {"audit_report": "audit_report", "retry": "retry_bump"},
    )

    graph.add_edge("retry_bump", "phi_detection")
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
    config = {"configurable": {"thread_id": thread_id}}
    initial_state: GraphState = {
        "raw_text": raw_text,
        "filename": filename,
        "retry_count": 0,
        "max_retries": 2,
        "audit_log": [],
    }
    result = compiled_graph.invoke(initial_state, config=config)

    if "__interrupt__" in result:
        return {"status": "interrupted", "interrupt": result["__interrupt__"][0].value, "thread_id": thread_id}
    return {"status": "completed", "state": result}


def resume_sync(decisions: list[dict], thread_id: str = "default") -> dict:
    """Resume a graph paused at HumanReviewAgent with reviewer decisions."""
    from langgraph.types import Command

    config = {"configurable": {"thread_id": thread_id}}
    result = compiled_graph.invoke(Command(resume=decisions), config=config)

    if "__interrupt__" in result:
        return {"status": "interrupted", "interrupt": result["__interrupt__"][0].value, "thread_id": thread_id}
    return {"status": "completed", "state": result}
