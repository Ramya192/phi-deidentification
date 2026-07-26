"""
tests/test_llm_adjudication.py

Covers agents/llm_adjudication_agent.py:
  1. Default backend (heuristic/off) is a true no-op -- state passes
     through completely unchanged, regardless of what's in
     low_confidence_spans.
  2. adjudicate_span_with_llm()'s two-step call (optional tool use, then
     always a structured decision) with ChatOpenAI mocked -- no real
     network/API key needed.
  3. llm_adjudication_agent() node-level control flow: confirmed spans
     move to llm_reviewed_spans, rejected spans go to rejected_spans,
     low-confidence-adjudicator-decisions and failures both defer to
     human review, all using observability/llm_metrics.py for real usage
     tracking.

Run with: pytest -v tests/test_llm_adjudication.py
"""
from __future__ import annotations

import os
import sys
from unittest.mock import MagicMock

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pytest

from agents import llm_adjudication_agent as adjudication_module
from agents.llm_adjudication_agent import AdjudicationDecision, llm_adjudication_agent

_LOW_CONF_SPAN = {
    "start": 0, "end": 5, "text": "Smith", "phi_type": "PERSON",
    "confidence": 0.6, "source_agent": "regex_fallback",
}


def test_default_backend_is_true_noop(monkeypatch):
    monkeypatch.delenv("PHI_DEID_ADJUDICATION_BACKEND", raising=False)
    state = {
        "raw_text": "Dr. Smith saw the patient.",
        "low_confidence_spans": [dict(_LOW_CONF_SPAN)],
        "high_confidence_spans": [],
        "audit_log": [],
    }
    result = llm_adjudication_agent(state)
    # Literally the same object back -- not even a shallow copy -- proving
    # no processing happened at all.
    assert result is state
    assert result["low_confidence_spans"] == [_LOW_CONF_SPAN]


def test_noop_when_no_low_confidence_spans(monkeypatch):
    monkeypatch.setenv("PHI_DEID_ADJUDICATION_BACKEND", "llm")
    state = {"raw_text": "clean note", "low_confidence_spans": [], "audit_log": []}
    result = llm_adjudication_agent(state)
    assert result is state


def _fake_ai_message(tool_calls=None, usage=None):
    msg = MagicMock()
    msg.tool_calls = tool_calls or []
    msg.usage_metadata = usage or {"input_tokens": 100, "output_tokens": 20}
    return msg


def test_adjudicate_span_confirms_phi_with_tool_call(monkeypatch):
    pytest.importorskip("langchain_openai")

    first_response = _fake_ai_message(tool_calls=[{"name": "search_document", "args": {"pattern": r"Smith"}, "id": "call_1"}])
    fake_with_tool = MagicMock()
    fake_with_tool.invoke.return_value = first_response

    final_raw = _fake_ai_message(usage={"input_tokens": 150, "output_tokens": 30})
    decision = AdjudicationDecision(is_phi=True, confidence=0.9, reasoning="Recurs as a proper name near 'Dr.'")
    fake_struct = MagicMock()
    fake_struct.invoke.return_value = {"raw": final_raw, "parsed": decision}

    fake_chat = MagicMock()
    fake_chat.bind_tools.return_value = fake_with_tool
    fake_chat.with_structured_output.return_value = fake_struct

    monkeypatch.setattr(adjudication_module, "ChatOpenAI", lambda **kwargs: fake_chat, raising=False)
    import langchain_openai
    monkeypatch.setattr(langchain_openai, "ChatOpenAI", lambda **kwargs: fake_chat)

    result_decision, usage_entries = adjudication_module.adjudicate_span_with_llm(
        dict(_LOW_CONF_SPAN), "Dr. Smith saw the patient. Later, Smith returned."
    )

    assert result_decision.is_phi is True
    assert result_decision.confidence == 0.9
    assert len(usage_entries) == 2  # tool-decision call + final structured call
    assert all(e["agent"] == "LLMAdjudicationAgent" for e in usage_entries)
    fake_with_tool.invoke.assert_called_once()
    fake_struct.invoke.assert_called_once()


def test_llm_adjudication_agent_sorts_spans_by_decision(monkeypatch):
    monkeypatch.setenv("PHI_DEID_ADJUDICATION_BACKEND", "llm")

    confirm_span = {"start": 0, "end": 5, "text": "Jordan", "phi_type": "PERSON", "confidence": 0.6, "source_agent": "regex_fallback"}
    reject_span = {"start": 10, "end": 15, "text": "12345", "phi_type": "DATE_TIME", "confidence": 0.5, "source_agent": "regex_fallback"}
    unsure_span = {"start": 20, "end": 25, "text": "maybe", "phi_type": "LOCATION", "confidence": 0.4, "source_agent": "regex_fallback"}
    error_span = {"start": 30, "end": 35, "text": "crash", "phi_type": "NRP", "confidence": 0.3, "source_agent": "regex_fallback"}

    def fake_adjudicate(span, full_text):
        if span["text"] == "Jordan":
            return AdjudicationDecision(is_phi=True, confidence=0.95, reasoning="clear name"), [
                {"agent": "LLMAdjudicationAgent", "model": "gpt-4o-mini", "call_type": "chat_completion",
                 "prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15, "approx_cost_usd": 0.0, "notes": None}
            ]
        if span["text"] == "12345":
            return AdjudicationDecision(is_phi=False, confidence=0.9, reasoning="just a number, not a date"), []
        if span["text"] == "maybe":
            # Adjudicator itself isn't confident -- must defer to human.
            return AdjudicationDecision(is_phi=True, confidence=0.5, reasoning="unclear"), []
        raise RuntimeError("simulated adjudication failure")

    monkeypatch.setattr(adjudication_module, "adjudicate_span_with_llm", fake_adjudicate)

    state = {
        "raw_text": "irrelevant for this test",
        "low_confidence_spans": [confirm_span, reject_span, unsure_span, error_span],
        "rejected_spans": [],
        "audit_log": [],
    }
    result = llm_adjudication_agent(state)

    # Only the genuinely unresolved spans remain for human review.
    remaining_texts = {s["text"] for s in result["low_confidence_spans"]}
    assert remaining_texts == {"maybe", "crash"}

    assert len(result["llm_reviewed_spans"]) == 1
    assert result["llm_reviewed_spans"][0]["text"] == "Jordan"
    assert result["llm_reviewed_spans"][0]["source_agent"] == "llm_adjudication"

    assert {"phi_type": "DATE_TIME", "text": "12345"} in result["rejected_spans"]

    actions = [e["action"] for e in result["audit_log"] if e["agent"] == "LLMAdjudicationAgent"]
    assert actions.count("llm_confirmed_phi") == 1
    assert actions.count("llm_rejected_not_phi") == 1
    assert actions.count("deferred_to_human") == 2  # low-confidence decision + call failure

    # llm_usage_log only gets entries from the one span that actually
    # returned usage data.
    assert len(result["llm_usage_log"]) == 1


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
