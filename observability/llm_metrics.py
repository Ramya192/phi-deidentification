"""
observability/llm_metrics.py

Per-call token/cost tracking for LLM-backed agents, complementing
graph/workflow.py's _timed() wrapper (which measures wall-clock latency
for every node, LLM-backed or not).

Scope note: as of this writing, exactly one code path in this project
calls an external LLM API at all -- classify_with_llm() in
agents/classification_agent.py, and only when
PHI_DEID_CLASSIFICATION_BACKEND=llm is explicitly set (the default
"heuristic" backend and PHIDetectionAgent's Presidio/regex backends make
zero LLM calls). This module exists so that path -- and any future
LLM-backed agent, e.g. if classify_with_llm's sibling detection backend
or a summarization step is added later -- has a single, consistent place
to record real usage rather than each call site inventing its own ad hoc
counter. It is deliberately NOT wired into PHIDetectionAgent or
RedactionAgent, which have no LLM calls to measure.

Uses actual token counts reported by the OpenAI API response's `.usage`
field -- not an approximate `len(text) / 4` estimate -- since real counts
are already available for free on every response and are exact rather
than a rough guess.
"""
from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Optional

# Rough public per-1K-token USD pricing, current as of when this file was
# written. OpenAI (and any other provider added later) can and does change
# pricing -- treat these as approximate/directional, not billing-accurate.
# Override or extend via PHI_DEID_LLM_PRICING_OVERRIDES if you need exact
# figures for a cost report, or just edit this table when pricing changes.
_PRICING_USD_PER_1K_TOKENS: dict[str, dict[str, float]] = {
    "gpt-4o-mini": {"prompt": 0.00015, "completion": 0.0006},
    "text-embedding-3-small": {"prompt": 0.00002, "completion": 0.0},
}
_DEFAULT_PRICE = {"prompt": 0.0, "completion": 0.0}


def estimate_cost_usd(model: str, prompt_tokens: int, completion_tokens: int = 0) -> float:
    price = _PRICING_USD_PER_1K_TOKENS.get(model, _DEFAULT_PRICE)
    cost = (prompt_tokens / 1000) * price["prompt"] + (completion_tokens / 1000) * price["completion"]
    return round(cost, 8)


def make_usage_entry(
    agent: str,
    model: str,
    call_type: str,
    prompt_tokens: int,
    completion_tokens: int = 0,
    notes: Optional[str] = None,
) -> dict[str, Any]:
    """Builds one llm_usage_log entry. Call sites append the result to
    state["llm_usage_log"] themselves (same manual-list-append convention
    every other GraphState list field in this project already uses --
    see audit_log/node_timings in graph/workflow.py and the agents)."""
    total_tokens = prompt_tokens + completion_tokens
    return {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "agent": agent,
        "model": model,
        "call_type": call_type,  # "chat_completion" | "embedding"
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "total_tokens": total_tokens,
        "approx_cost_usd": estimate_cost_usd(model, prompt_tokens, completion_tokens),
        "notes": notes,
    }


def summarize_usage(llm_usage_log: list[dict[str, Any]]) -> dict[str, Any]:
    """Aggregates the raw per-call log into per-agent totals plus a grand
    total -- same shape convention as agents/audit_report_agent.py's
    node_timings_ms aggregation, so the compliance report gets a
    ready-to-display summary instead of making every consumer re-derive it."""
    by_agent: dict[str, dict[str, Any]] = {}
    for entry in llm_usage_log:
        bucket = by_agent.setdefault(entry["agent"], {
            "calls": 0,
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "total_tokens": 0,
            "approx_cost_usd": 0.0,
        })
        bucket["calls"] += 1
        bucket["prompt_tokens"] += entry["prompt_tokens"]
        bucket["completion_tokens"] += entry["completion_tokens"]
        bucket["total_tokens"] += entry["total_tokens"]
        bucket["approx_cost_usd"] = round(bucket["approx_cost_usd"] + entry["approx_cost_usd"], 8)

    total_cost = round(sum(v["approx_cost_usd"] for v in by_agent.values()), 8)
    total_tokens = sum(v["total_tokens"] for v in by_agent.values())
    return {
        "by_agent": by_agent,
        "total_tokens": total_tokens,
        "total_approx_cost_usd": total_cost,
    }
