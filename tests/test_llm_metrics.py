"""
tests/test_llm_metrics.py

Covers observability/llm_metrics.py and its wiring into
agents/classification_agent.py's classify_with_llm() path:
  1. estimate_cost_usd()/summarize_usage() pure math.
  2. classify_with_llm() records real usage entries when the OpenAI client
     succeeds (mocked -- no real network/API key needed).
  3. classification_agent(backend="llm") end-to-end: llm_usage_log ends up
     in state and gets aggregated into compliance_report via
     AuditReportAgent.

Run with: pytest -v tests/test_llm_metrics.py
"""
from __future__ import annotations

import os
import sys
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from agents.audit_report_agent import audit_report_agent
from agents.classification_agent import classification_agent
from observability.llm_metrics import estimate_cost_usd, summarize_usage
import agents.classification_agent as classification_module


def test_estimate_cost_usd_known_model():
    cost = estimate_cost_usd("gpt-4o-mini", prompt_tokens=1000, completion_tokens=1000)
    assert cost == round(0.00015 + 0.0006, 8)


def test_estimate_cost_usd_unknown_model_is_zero():
    assert estimate_cost_usd("some-future-model", prompt_tokens=1000) == 0.0


def test_summarize_usage_aggregates_by_agent():
    log = [
        {"agent": "ClassificationAgent", "prompt_tokens": 10, "completion_tokens": 0, "total_tokens": 10, "approx_cost_usd": 0.001},
        {"agent": "ClassificationAgent", "prompt_tokens": 5, "completion_tokens": 20, "total_tokens": 25, "approx_cost_usd": 0.002},
    ]
    summary = summarize_usage(log)
    assert summary["by_agent"]["ClassificationAgent"]["calls"] == 2
    assert summary["by_agent"]["ClassificationAgent"]["total_tokens"] == 35
    assert summary["total_tokens"] == 35
    assert summary["total_approx_cost_usd"] == round(0.003, 8)


def _fake_openai_client():
    """Mocks the two OpenAI call shapes classify_with_llm() makes:
    embeddings.create() (used twice: example-cache population + query) and
    chat.completions.create(). Real classify_with_llm() code is exercised
    unchanged -- only the network boundary (OpenAI() client) is faked."""
    client = MagicMock()

    def fake_embeddings_create(model, input):
        n = len(input)
        return SimpleNamespace(
            data=[SimpleNamespace(embedding=[0.1, 0.2, 0.3]) for _ in range(n)],
            usage=SimpleNamespace(prompt_tokens=8 * n, total_tokens=8 * n),
        )

    client.embeddings.create.side_effect = fake_embeddings_create

    def fake_chat_create(**kwargs):
        message = SimpleNamespace(content='{"doc_type": "clinical_note", "confidence": 0.92}')
        return SimpleNamespace(
            choices=[SimpleNamespace(message=message)],
            usage=SimpleNamespace(prompt_tokens=250, completion_tokens=12),
        )

    client.chat.completions.create.side_effect = fake_chat_create
    return client


def test_classify_with_llm_records_real_usage(monkeypatch):
    # Reset the module-level example-embedding cache so this test always
    # exercises the "first call in the process" path (cache-population +
    # query embedding + chat completion = 3 usage entries), regardless of
    # what other tests in the suite ran first.
    monkeypatch.setattr(classification_module, "_example_embeddings", None)

    openai = pytest.importorskip("openai")  # optional dep -- see requirements.txt
    fake_client = _fake_openai_client()
    # classify_with_llm() does `from openai import OpenAI` inline (a fresh
    # lookup each call), so patching the attribute on the real openai
    # module is what actually takes effect.
    monkeypatch.setattr(openai, "OpenAI", lambda: fake_client)

    doc_type, confidence, usage_entries = classification_module.classify_with_llm(
        "Chief Complaint: cough. Vital Signs: T 99.1F."
    )

    assert doc_type == "clinical_note"
    assert confidence == 0.92
    # cache-population embedding + query embedding + chat completion
    assert len(usage_entries) == 3
    assert usage_entries[-1]["call_type"] == "chat_completion"
    assert usage_entries[-1]["prompt_tokens"] == 250
    assert usage_entries[-1]["completion_tokens"] == 12
    assert all(e["agent"] == "ClassificationAgent" for e in usage_entries)
    assert all(e["approx_cost_usd"] >= 0.0 for e in usage_entries)


def test_classification_agent_llm_backend_populates_state_and_report(monkeypatch):
    monkeypatch.setattr(classification_module, "_example_embeddings", None)
    openai = pytest.importorskip("openai")
    fake_client = _fake_openai_client()
    monkeypatch.setattr(openai, "OpenAI", lambda: fake_client)

    state = {"raw_text": "Chief Complaint: cough. Vital Signs: T 99.1F.", "retry_count": 0, "audit_log": []}
    result = classification_agent(state, backend="llm")

    assert result["doc_type"] == "clinical_note"
    assert len(result["llm_usage_log"]) == 3

    # Feed straight into AuditReportAgent, same as the real graph does at
    # the end of a run, and confirm the summary surfaces correctly.
    report_state = {
        **result,
        "audit_log": result["audit_log"],
        "node_timings": [],
    }
    final = audit_report_agent(report_state)
    summary = final["compliance_report"]["llm_usage_summary"]
    assert summary["by_agent"]["ClassificationAgent"]["calls"] == 3
    assert summary["total_approx_cost_usd"] > 0.0


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
