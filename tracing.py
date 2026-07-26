"""
tracing.py

Optional LangSmith observability for the LangGraph pipeline.

LangGraph has built-in LangSmith integration: once LANGCHAIN_TRACING_V2=true
and LANGCHAIN_API_KEY are set as environment variables, every
compiled_graph.invoke()/.stream() call is automatically traced -- no
per-node instrumentation needed, unlike observability/llm_metrics.py
(which tracks real token/cost usage and has to be threaded through
classify_with_llm() by hand, since that's a raw OpenAI SDK call LangSmith
has no visibility into on its own).

This module's only job is to build the `tags`/`metadata` extras LangGraph
accepts in its `config` dict, so each run shows up in the LangSmith UI
labeled with its thread_id/filename instead of as an anonymous trace --
matches the "per-run tags/metadata keyed by record_id" idea from
Sandeep's tracing.py, scoped to what's actually known at invoke() time in
this project's architecture (doc_type isn't known yet at that point --
ClassificationAgent is the first node to run).

Usage (see graph/workflow.py run_sync/resume_sync):
    config = {"configurable": {"thread_id": thread_id}, **tracing_extras(thread_id, filename)}

If LANGCHAIN_API_KEY isn't set, tracing_extras() returns {} -- a true
no-op, not just relying on LangGraph silently ignoring unused fields.

Setup:
    pip install langsmith   (already in requirements.txt)
    export LANGCHAIN_TRACING_V2=true
    export LANGCHAIN_API_KEY=ls__...
    export LANGCHAIN_PROJECT=phi-deid-pipeline   # optional, defaults to "default" in LangSmith
"""
from __future__ import annotations

import os
from typing import Any, Optional


def tracing_enabled() -> bool:
    return bool(os.environ.get("LANGCHAIN_API_KEY"))


def tracing_extras(thread_id: str, filename: Optional[str] = None) -> dict[str, Any]:
    """Extra `config` keys for compiled_graph.invoke()/.stream() -- merge
    into the existing {"configurable": {"thread_id": ...}} dict. Returns
    {} (nothing added) when tracing isn't configured, so callers can
    unconditionally splat this in without an if/else at every call site."""
    if not tracing_enabled():
        return {}
    return {
        "tags": ["phi-deid-pipeline"],
        "metadata": {
            "thread_id": thread_id,
            "filename": filename or "unknown",
        },
    }
