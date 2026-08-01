"""
tests/test_audit_record_auth.py

Covers the per-record audit authorization gap that was previously open:
GET /records/{id}/audit and GET /records used to only check the single
shared X-API-Key, so any caller holding that key could read any other
caller's audit record. api/main.py now also scopes on an X-Client-Id
header (storage/audit_store.py's client_id column), and honors an
optional PHI_DEID_ADMIN_API_KEY that bypasses the scoping entirely.

Uses FastAPI's TestClient (no real server) and points
PHI_DEID_AUDIT_DB_URL at a pytest tmp_path SQLite file, same pattern as
tests/test_api_batch.py.

Run with: pytest -v tests/test_audit_record_auth.py
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
    original = phi_detection_module.USE_FALLBACK_ONLY
    phi_detection_module.USE_FALLBACK_ONLY = True
    monkeypatch.setenv("PHI_DEID_CLASSIFICATION_BACKEND", "heuristic")
    yield
    phi_detection_module.USE_FALLBACK_ONLY = original


@pytest.fixture
def api_client(tmp_path, monkeypatch, force_fallback_detector):
    db_path = tmp_path / "audit_test.sqlite"
    monkeypatch.setenv("PHI_DEID_AUDIT_DB_URL", f"sqlite:///{db_path}")
    monkeypatch.setenv("PHI_DEID_API_KEY", "test-key")
    monkeypatch.setenv("PHI_DEID_ADMIN_API_KEY", "admin-key")

    import storage.audit_store as audit_store_module
    importlib.reload(audit_store_module)
    import api.main as api_main_module
    importlib.reload(api_main_module)

    return TestClient(api_main_module.app), api_main_module


def _redact(client, headers, filename="note.txt"):
    resp = client.post(
        "/redact",
        json={"text": "Patient contact: john.carter@example.com", "filename": filename},
        headers=headers,
    )
    assert resp.status_code == 200
    return resp.json()["thread_id"]


def test_records_created_without_client_id_share_default_scope(api_client):
    """Single-tenant deployments (nobody sets X-Client-Id) see no behavior
    change: any caller with the shared key can still read any record."""
    client, _ = api_client
    headers = {"X-API-Key": "test-key"}
    thread_id = _redact(client, headers)

    audit = client.get(f"/records/{thread_id}/audit", headers=headers)
    assert audit.status_code == 200
    assert audit.json()["client_id"] == "default"


def test_caller_cannot_read_another_clients_record(api_client):
    client, _ = api_client
    alice = {"X-API-Key": "test-key", "X-Client-Id": "alice"}
    bob = {"X-API-Key": "test-key", "X-Client-Id": "bob"}

    thread_id = _redact(client, alice)

    own = client.get(f"/records/{thread_id}/audit", headers=alice)
    assert own.status_code == 200

    other = client.get(f"/records/{thread_id}/audit", headers=bob)
    assert other.status_code == 404


def test_list_records_is_scoped_per_client(api_client):
    client, _ = api_client
    alice = {"X-API-Key": "test-key", "X-Client-Id": "alice"}
    bob = {"X-API-Key": "test-key", "X-Client-Id": "bob"}

    _redact(client, alice, filename="alice1.txt")
    _redact(client, alice, filename="alice2.txt")
    _redact(client, bob, filename="bob1.txt")

    alice_list = client.get("/records", headers=alice).json()["records"]
    bob_list = client.get("/records", headers=bob).json()["records"]

    assert len(alice_list) == 2
    assert len(bob_list) == 1
    assert all(r["client_id"] == "alice" for r in alice_list)
    assert all(r["client_id"] == "bob" for r in bob_list)


def test_admin_key_reads_any_clients_record_and_lists_all(api_client):
    client, _ = api_client
    alice = {"X-API-Key": "test-key", "X-Client-Id": "alice"}
    admin = {"X-API-Key": "admin-key"}

    thread_id = _redact(client, alice)

    audit = client.get(f"/records/{thread_id}/audit", headers=admin)
    assert audit.status_code == 200
    assert audit.json()["client_id"] == "alice"

    listed = client.get("/records", headers=admin).json()["records"]
    assert len(listed) == 1


def test_admin_key_alone_authenticates_like_a_normal_api_key(api_client):
    """PHI_DEID_ADMIN_API_KEY is a second valid key for every endpoint,
    not just the /records ones -- verify_api_key accepts either."""
    client, _ = api_client
    admin = {"X-API-Key": "admin-key"}
    resp = client.post(
        "/redact", json={"text": "hello", "filename": "a.txt"}, headers=admin
    )
    assert resp.status_code == 200


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
