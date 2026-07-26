"""
tests/test_api_batch.py

Covers api/main.py's POST /redact/batch and the storage/audit_store.py
persistence it triggers via _persist_audit_record().

Uses FastAPI's TestClient (no real server) and points
PHI_DEID_AUDIT_DB_URL at a pytest tmp_path SQLite file so this test never
touches the project's real data/audit.sqlite.

Run with: pytest -v tests/test_api_batch.py
"""
from __future__ import annotations

import importlib
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pytest
from fastapi.testclient import TestClient

from agents import phi_detection_agent as phi_detection_module


@pytest.fixture
def force_fallback_detector(monkeypatch):
    """Also pins ClassificationAgent to the heuristic backend -- see the
    matching fixture docstring in tests/test_integration.py for why: with
    PHI_DEID_CLASSIFICATION_BACKEND=llm set in a real .env, the thin
    synthetic text below can get classified as not_applicable by the real
    LLM, which skips redaction entirely and leaves redacted_text unset
    (None) in the response -- the TypeError this test file hit live."""
    original = phi_detection_module.USE_FALLBACK_ONLY
    phi_detection_module.USE_FALLBACK_ONLY = True
    monkeypatch.setenv("PHI_DEID_CLASSIFICATION_BACKEND", "heuristic")
    yield
    phi_detection_module.USE_FALLBACK_ONLY = original


@pytest.fixture
def api_client(tmp_path, monkeypatch, force_fallback_detector):
    """Fresh TestClient with a scratch audit DB and a stable API key.
    Reimports storage.audit_store and api.main so the module-level
    _default_store singleton picks up PHI_DEID_AUDIT_DB_URL for *this*
    test rather than whatever a previous test/process already cached."""
    db_path = tmp_path / "audit_test.sqlite"
    monkeypatch.setenv("PHI_DEID_AUDIT_DB_URL", f"sqlite:///{db_path}")
    monkeypatch.setenv("PHI_DEID_API_KEY", "test-key")

    import storage.audit_store as audit_store_module
    importlib.reload(audit_store_module)
    import api.main as api_main_module
    importlib.reload(api_main_module)

    return TestClient(api_main_module.app), api_main_module


def test_batch_endpoint_processes_all_documents(api_client):
    client, _ = api_client
    headers = {"X-API-Key": "test-key"}
    payload = {
        "documents": [
            {"text": "Patient contact: john.carter@example.com or (555) 234-9981.", "filename": "note1.txt"},
            {"text": "Patient contact: jane.doe@example.com or (555) 111-2222.", "filename": "note2.txt"},
        ]
    }
    resp = client.post("/redact/batch", json=payload, headers=headers)
    assert resp.status_code == 200
    body = resp.json()
    assert body["count"] == 2
    assert body["completed"] == 2
    assert body["errors"] == 0
    assert len(body["results"]) == 2
    assert body["results"][0]["filename"] == "note1.txt"
    assert body["results"][0]["index"] == 0
    assert "redacted_text" in body["results"][0]
    assert "john.carter@example.com" not in body["results"][0]["redacted_text"]

    # Each item should have persisted its own audit record.
    thread_id = body["results"][0]["thread_id"]
    audit = client.get(f"/records/{thread_id}/audit", headers=headers)
    assert audit.status_code == 200
    assert audit.json()["filename"] == "note1.txt"


def test_batch_endpoint_rejects_empty_and_oversized(api_client):
    client, api_main_module = api_client
    headers = {"X-API-Key": "test-key"}

    empty = client.post("/redact/batch", json={"documents": []}, headers=headers)
    assert empty.status_code == 422

    too_many = client.post(
        "/redact/batch",
        json={"documents": [{"text": "x", "filename": f"f{i}.txt"} for i in range(api_main_module.MAX_BATCH_SIZE + 1)]},
        headers=headers,
    )
    assert too_many.status_code == 422


def test_batch_endpoint_requires_api_key(api_client):
    client, _ = api_client
    resp = client.post("/redact/batch", json={"documents": [{"text": "hello", "filename": "a.txt"}]})
    assert resp.status_code == 401


def test_health_reports_real_readiness(api_client):
    """No API key needed for /health -- load balancers/orchestrators must
    be able to probe it without one."""
    client, _ = api_client
    resp = client.get("/health")
    assert resp.status_code == 200
    body = resp.json()
    assert body["status"] == "ok"
    assert body["graph_compiled"] is True
    assert body["checkpointer_backend"] in ("sqlite", "memory")
    assert body["checkpointer_restart_survivable"] == (body["checkpointer_backend"] == "sqlite")


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
