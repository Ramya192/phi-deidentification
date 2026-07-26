"""
LLMAdjudicationAgent

The project's one genuinely agentic tier. Everywhere else in this
pipeline, "agent" means "a deterministic or ML-based processing stage" --
PHIDetectionAgent is Presidio/regex pattern matching, not reasoning;
HumanReviewAgent is a person, not AI. This node is where an LLM actually
examines evidence, optionally calls a tool to gather more of it, and
produces a structured, reasoned judgment call, via LangChain's
bind_tools() + with_structured_output() -- the standard agentic pattern.

It sits between PHIValidationAgent and the human-review/redaction fork,
reviewing PHIDetectionAgent's low-confidence spans and resolving the
clear cases itself so only genuinely ambiguous spans still reach a human
reviewer.

DELIBERATELY GATED, OFF BY DEFAULT (PHI_DEID_ADJUDICATION_BACKEND=llm to
enable) -- same "local ML, prod LLM" split as ClassificationAgent's
classify_with_llm. Local development, tests, and eval/evaluate.py all
need a free, deterministic, reproducible pipeline; that's still
PHIDetectionAgent's job, completely unchanged by this file. This agent is
for a production deployment that can afford (and wants) an LLM call on
the smaller volume of low-confidence spans specifically, to cut down how
often a human reviewer gets bothered with cases an LLM can resolve on its
own.

Fails safe: any error (missing API key, network failure, malformed
response, low adjudicator self-confidence) defers that span to human
review rather than silently guessing -- the same defensive contract
classify_with_llm already follows for its own failure modes.
"""

from __future__ import annotations

import os
import re
from datetime import datetime, timezone
from typing import Any, Optional

from pydantic import BaseModel, Field

from graph.state import GraphState, PHISpan
from observability.llm_metrics import make_usage_entry

ADJUDICATION_MODEL = "gpt-4o-mini"
_CONTEXT_WINDOW = 60  # chars of surrounding text on each side of a span

# Below this, the adjudicator itself isn't confident enough to resolve the
# span either -- defer to a human rather than let a low-confidence LLM
# guess silently override PHIDetectionAgent's own low-confidence guess.
ADJUDICATION_CONFIDENCE_FLOOR = 0.75


class AdjudicationDecision(BaseModel):
    """Structured output contract for the LLM's judgment call -- the
    Pydantic model LangChain's with_structured_output() parses the LLM's
    response into. This is the one place in the project Pydantic is used
    internally (agent-to-agent), not just at the FastAPI request boundary --
    directly because this is the one agent actually parsing an LLM's
    output, which is exactly the situation Pydantic-as-schema is for."""

    is_phi: bool = Field(
        ...,
        description="True if this span is genuinely PHI, False if it's a false positive",
    )
    confidence: float = Field(
        ..., ge=0.0, le=1.0, description="Adjudicator's own confidence in this decision"
    )
    reasoning: str = Field(
        ..., description="One or two sentence explanation, recorded in the audit trail"
    )


def _get_context(full_text: str, span: PHISpan, window: int = _CONTEXT_WINDOW) -> str:
    start = max(0, span["start"] - window)
    end = min(len(full_text), span["end"] + window)
    return full_text[start:end]


def _make_search_tool(full_text: str):
    """Builds a fresh, closure-bound tool per adjudication call -- scoped
    to this one document's text, not a shared global tool. Lets the LLM
    check whether a candidate pattern is an isolated occurrence or
    recurs elsewhere in the document: a recurring structural pattern
    (e.g. the same ID format appearing three times) is stronger PHI
    evidence than one coincidental match, mirroring the kind of
    corroborating-evidence check a human reviewer would informally do by
    skimming the rest of the document -- see Sandeep's
    HumanReviewAgent._recheck_entity for the same underlying idea,
    implemented here as a genuine LLM-callable tool instead of a
    UI-triggered helper.
    """
    from langchain_core.tools import tool

    @tool
    def search_document(pattern: str) -> list[str]:
        """Search the rest of the document for other text matching this
        regex pattern, to check whether the candidate span is an isolated
        occurrence or part of a recurring structural pattern. Returns up
        to 5 matches with a few characters of surrounding context each.
        Pass a real Python regex (e.g. r'\\b[A-Z]{2}\\d{6}\\b'), not plain
        literal text."""
        try:
            compiled = re.compile(pattern)
        except re.error as exc:
            return [f"invalid regex, try again: {exc}"]
        matches = []
        for m in compiled.finditer(full_text):
            snippet_start = max(0, m.start() - 15)
            snippet_end = min(len(full_text), m.end() + 15)
            matches.append(full_text[snippet_start:snippet_end])
            if len(matches) >= 5:
                break
        return matches or ["no matches found"]

    return search_document


def _usage_from_response(message: Any, call_type: str) -> dict:
    usage = getattr(message, "usage_metadata", None) or {}
    return make_usage_entry(
        agent="LLMAdjudicationAgent",
        model=ADJUDICATION_MODEL,
        call_type=call_type,
        prompt_tokens=usage.get("input_tokens", 0),
        completion_tokens=usage.get("output_tokens", 0),
    )


def adjudicate_span_with_llm(
    span: PHISpan, full_text: str
) -> tuple[Optional[AdjudicationDecision], list[dict]]:
    """Two-step agentic call for one span:
      1. Give the LLM the span, its context, and an optional
         search_document tool; let it decide whether to use the tool
         (tool_choice="auto" -- it isn't forced to).
      2. Always finalize with a structured, Pydantic-validated decision,
         including whatever the tool returned if it called one.

    Returns (decision, usage_entries). decision is None if the call
    failed outright -- callers must treat that as "defer to human," not
    as a negative adjudication. Raises are allowed to propagate; the
    caller (llm_adjudication_agent) is responsible for the fail-safe
    try/except, same layering as classification_agent()/classify_with_llm.
    """
    from langchain_core.messages import (
        AIMessage,
        HumanMessage,
        SystemMessage,
        ToolMessage,
    )
    from langchain_openai import ChatOpenAI

    usage_entries: list[dict] = []
    context = _get_context(full_text, span)
    search_tool = _make_search_tool(full_text)

    system_prompt = (
        "You are adjudicating a candidate PHI (Protected Health Information) span "
        "that a pattern-based detector flagged with LOW confidence. Decide whether "
        "it is genuinely PHI or a false positive, using the surrounding context. "
        "You may call search_document once to check whether a similar pattern "
        "recurs elsewhere in the document -- a recurring structural pattern is "
        "stronger evidence of a real identifier than an isolated coincidental "
        "match. Only call it if it would actually change your judgment; "
        "otherwise decide directly without calling it."
    )
    user_prompt = (
        f"Candidate span: {span['text']!r}\n"
        f"Detector's guess at entity type: {span['phi_type']}\n"
        f"Detector's own confidence: {span['confidence']}\n"
        f"Surrounding context: ...{context}...\n\n"
        "Is this genuinely PHI?"
    )
    messages: list[Any] = [SystemMessage(system_prompt), HumanMessage(user_prompt)]

    llm_with_tool = ChatOpenAI(model=ADJUDICATION_MODEL, temperature=0).bind_tools(
        [search_tool], tool_choice="auto"
    )
    first_response: AIMessage = llm_with_tool.invoke(messages)
    usage_entries.append(
        _usage_from_response(first_response, call_type="chat_completion")
    )
    messages.append(first_response)

    if getattr(first_response, "tool_calls", None):
        # Bounded to the single tool call the model made this pass -- no
        # loop, no risk of the model chaining unbounded tool calls.
        call = first_response.tool_calls[0]
        result = search_tool.invoke(call["args"])
        messages.append(ToolMessage(content=str(result), tool_call_id=call["id"]))

    llm_struct = ChatOpenAI(
        model=ADJUDICATION_MODEL, temperature=0
    ).with_structured_output(AdjudicationDecision, include_raw=True)
    final = llm_struct.invoke(messages)
    usage_entries.append(
        _usage_from_response(final["raw"], call_type="chat_completion")
    )
    decision: AdjudicationDecision = final["parsed"]

    return decision, usage_entries


def llm_adjudication_agent(state: GraphState) -> GraphState:
    """
    LangGraph node. Runs between PHIValidationAgent and the
    human-review/redaction fork -- route_after_detection
    (agents/phi_detection_agent.py, unchanged) still just checks whether
    low_confidence_spans is non-empty afterward, so this agent's only
    contract with the rest of the graph is: shrink low_confidence_spans
    for whatever it resolves, and record the resolved spans elsewhere.

    Default (PHI_DEID_ADJUDICATION_BACKEND unset or != "llm"): a true
    no-op -- state passes through completely unchanged, so local dev,
    every existing test, and eval/evaluate.py all see identical behavior
    to before this agent existed.
    """
    backend = os.environ.get("PHI_DEID_ADJUDICATION_BACKEND", "heuristic")
    low_conf_spans = state.get("low_confidence_spans", [])

    if backend != "llm" or not low_conf_spans:
        return state

    full_text = state.get("redacted_text") or state.get("raw_text", "")
    still_low_confidence: list[PHISpan] = []
    llm_reviewed_spans = list(state.get("llm_reviewed_spans", []))
    rejected_spans = list(state.get("rejected_spans", []))
    llm_usage_log = list(state.get("llm_usage_log", []))
    audit_log = list(state.get("audit_log", []))

    for span in low_conf_spans:
        try:
            decision, usage_entries = adjudicate_span_with_llm(span, full_text)
        except Exception as exc:
            still_low_confidence.append(span)
            audit_log.append(
                {
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                    "agent": "LLMAdjudicationAgent",
                    "action": "deferred_to_human",
                    "phi_type": span["phi_type"],
                    "span_text": span["text"],
                    "confidence": span["confidence"],
                    "reviewer_action": None,
                    "notes": f"adjudication call failed, deferring to human: {exc}",
                }
            )
            continue

        llm_usage_log.extend(usage_entries)

        if decision is None or decision.confidence < ADJUDICATION_CONFIDENCE_FLOOR:
            still_low_confidence.append(span)
            action = "deferred_to_human"
            notes = (
                f"adjudicator confidence {decision.confidence} below floor "
                f"{ADJUDICATION_CONFIDENCE_FLOOR}: {decision.reasoning}"
                if decision
                else "no decision returned"
            )
        elif decision.is_phi:
            llm_reviewed_spans.append({**span, "source_agent": "llm_adjudication"})
            action = "llm_confirmed_phi"
            notes = decision.reasoning
        else:
            key = {"phi_type": span["phi_type"], "text": span["text"]}
            if key not in rejected_spans:
                rejected_spans.append(key)
            action = "llm_rejected_not_phi"
            notes = decision.reasoning

        audit_log.append(
            {
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "agent": "LLMAdjudicationAgent",
                "action": action,
                "phi_type": span["phi_type"],
                "span_text": span["text"],
                "confidence": decision.confidence if decision else None,
                "reviewer_action": None,
                "notes": notes,
            }
        )

    return {
        **state,
        "low_confidence_spans": still_low_confidence,
        "llm_reviewed_spans": llm_reviewed_spans,
        "rejected_spans": rejected_spans,
        "llm_usage_log": llm_usage_log,
        "audit_log": audit_log,
    }
