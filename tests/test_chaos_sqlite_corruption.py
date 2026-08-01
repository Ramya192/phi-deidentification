"""
tests/test_chaos_sqlite_corruption.py

Chaos scenario: a corrupted audit.sqlite. Not hypothetical -- README.md's
"Troubleshooting: database disk image is malformed" section documents
this happening live during development (an interrupted mid-write left
the SQLite file corrupted). This file characterizes what actually
happens today rather than asserting an idealized recovery that doesn't
exist: there is no auto-repair, and this test suite doesn't add one
(README's documented fix is still "stop the process, delete the file,
restart" -- a real, accepted gap, not silently pretended away).

What IS asserted:
  - Reopening a corrupted audit.sqlite raises immediately (locks in the
    actual failure mode, so a future change to storage/audit_store.py
    that silently swallows corruption instead of surfacing it would fail
    this test).
  - POST /redact still succeeds even with a corrupted audit store --
    _persist_audit_record()'s best-effort try/except means a caller gets
    their redacted document back regardless of whether the audit trail
    could be written, which is the actual guardrail worth protecting.
  - GET /records/{id}/audit and GET /records fail *cleanly* -- a generic
    500 with no raw exception detail, same PHI-safe posture
    _internal_error_response already gives the /redact* endpoints -- not
    an unhandled exception leaking a traceback to the caller.

Run with: pytest -v tests/test_chaos_sqlite_corruption.py
"""
from __future__ import annotations

import importlib
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pytest
from sqlalchemy.exc import DatabaseError

from agents import phi_detection_agent as phi_detection_module
from storage.audit_store import AuditStore


def _corrupt_sqlite_file(path: str) -> None:
    """Simulates the real incident README.md documents: an interrupted
    mid-write (there, an uvicorn --reload restart) leaving a truncated,
    unreadable SQLite file. Building a valid DB first and then truncating
    it (rather than writing random bytes) reproduces the actual
    'database disk image is malformed' error SQLite raises for a
    partially-written file, not just a generic 'not a database' error for
    a file that was never valid."""
    store = AuditStore(f"sqlite:///{path}")
    store.store_audit(record_id="pre-corruption-rec", compliance_report={"filename": "a.txt"}, audit_log=[])
    size = os.path.getsize(path)
    with open(path, "r+b") as f:
        f.truncate(size // 2)


def test_corrupted_audit_db_raises_on_reopen(tmp_path):
    """Locks in the actual failure mode -- documents that AuditStore does
    NOT silently recover from corruption today."""
    db_path = str(tmp_path / "corrupt.sqlite")
    _corrupt_sqlite_file(db_path)

    with pytest.raises(DatabaseError):
        AuditStore(f"sqlite:///{db_path}")


@pytest.fixture
def force_fallback_detector(monkeypatch):
    original = phi_detection_module.USE_FALLBACK_ONLY
    phi_detection_module.USE_FALLBACK_ONLY = True
    monkeypatch.setenv("PHI_DEID_CLASSIFICATION_BACKEND", "heuristic")
    yield
    phi_detection_module.USE_FALLBACK_ONLY = original


@pytest.fixture
def corrupted_audit_api_client(tmp_path, monkeypatch, force_fallback_detector):
    """Same api_client pattern tests/test_api_batch.py uses, except the
    audit DB it points PHI_DEID_AUDIT_DB_URL at is pre-corrupted before
    api.main ever touches it -- simulating a process starting up against
    an audit.sqlite a previous crash already left broken, not corruption
    that happens mid-test."""
    db_path = str(tmp_path / "corrupt_audit.sqlite")
    _corrupt_sqlite_file(db_path)

    monkeypatch.setenv("PHI_DEID_AUDIT_DB_URL", f"sqlite:///{db_path}")
    monkeypatch.setenv("PHI_DEID_API_KEY", "test-key")

    import storage.audit_store as audit_store_module
    importlib.reload(audit_store_module)
    import api.main as api_main_module
    importlib.reload(api_main_module)

    from fastapi.testclient import TestClient

    return TestClient(api_main_module.app), api_main_module


def test_redact_still_succeeds_with_corrupted_audit_store(corrupted_audit_api_client):
    """The primary redaction flow must not depend on the audit trail
    persisting -- a caller gets their document back either way."""
    client, _ = corrupted_audit_api_client
    headers = {"X-API-Key": "test-key"}
    resp = client.post(
        "/redact",
        json={"text": "Patient contact: john.carter@example.com", "filename": "note.txt"},
        headers=headers,
    )
    assert resp.status_code == 200
    body = resp.json()
    assert body["status"] == "completed"
    assert "john.carter@example.com" not in body["redacted_text"]


def test_get_audit_record_fails_clean_on_corrupted_store(corrupted_audit_api_client):
    """Not a 200, not an unhandled exception -- a generic 500 with no
    exception detail leaked to the caller."""
    client, _ = corrupted_audit_api_client
    headers = {"X-API-Key": "test-key"}
    resp = client.get("/records/any-id/audit", headers=headers)
    assert resp.status_code == 500
    body = resp.json()
    assert "database" not in body["detail"].lower()
    assert "traceback" not in body["detail"].lower()


def test_list_audit_records_fails_clean_on_corrupted_store(corrupted_audit_api_client):
    client, _ = corrupted_audit_api_client
    headers = {"X-API-Key": "test-key"}
    resp = client.get("/records", headers=headers)
    assert resp.status_code == 500
    body = resp.json()
    assert "database" not in body["detail"].lower()
    assert "traceback" not in body["detail"].lower()


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
