"""
api/main.py (Member 6)

FastAPI wrapper around the LangGraph workflow.

    POST /redact             submit raw text (JSON), get back either the
                              final package or an interrupt payload needing review
    POST /redact/upload       submit a file (.txt, .pdf, .docx) -- same
                              pipeline, with text extraction (ingestion/
                              document_loader.py) run first
    POST /redact/resume       submit human-review decisions for a paused run
    GET  /health              liveness check (no auth -- load balancers/
                              orchestrators need to probe this without a key)

Every endpoint that touches document text or PHI requires an API key via
the `X-API-Key` header -- see the `verify_api_key` dependency below. Set
PHI_DEID_API_KEY as an environment variable before starting the server;
if it's unset, a random key is generated for that process only and
printed to the console (fine for a quick local test, not for anything
you'd call twice or deploy).

Run locally:
    export PHI_DEID_API_KEY=dev-local-key   # or let it auto-generate and print one
    uvicorn api.main:app --reload --port 8000

Docs at http://localhost:8000/docs once running.
"""
from __future__ import annotations

import logging
import os
import secrets
import uuid
from typing import Any

from fastapi import Depends, FastAPI, File, HTTPException, UploadFile
from fastapi.security import APIKeyHeader
from pydantic import BaseModel, Field

from graph.workflow import resume_sync, run_sync
from ingestion.document_loader import (
    EmptyDocumentError,
    UnsupportedFileTypeError,
    load_text_from_bytes,
)

# ---------------------------------------------------------------------------
# Logging -- PHI-safe by construction. Every log call below is built from a
# fixed format string with only metadata as arguments (thread_id, doc_type,
# span counts, filename) -- never request.text, extracted document text, or
# redacted_text. logger.exception() in the generic error handlers below is
# the one deliberate exception: it captures whatever the underlying
# library/exception put in its message, which could in rare cases include a
# text fragment. That's a documented, accepted risk (see README's
# production roadmap) rather than something silently ignored -- and
# crucially, it goes to access-controlled server logs, not back to
# whoever called the endpoint, which is what the old `detail=str(exc)`
# responses did.
# ---------------------------------------------------------------------------
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
logger = logging.getLogger("phi_deid_api")

app = FastAPI(
    title="PHI De-identification API",
    description="LangGraph multi-agent PHI detection, redaction, and compliance validation.",
    version="0.1.0",
)

# ---------------------------------------------------------------------------
# API key auth -- every PHI-handling endpoint requires X-API-Key.
# Right now an unauthenticated caller could POST arbitrary text/files
# containing PHI to this service and get results back; that's not
# acceptable outside a closed local dev loop.
# ---------------------------------------------------------------------------
_api_key_header = APIKeyHeader(name="X-API-Key", auto_error=False)

_configured_api_key = os.environ.get("PHI_DEID_API_KEY")
if not _configured_api_key:
    _configured_api_key = secrets.token_urlsafe(32)
    print(
        "WARNING: PHI_DEID_API_KEY not set -- generated a random API key for "
        f"THIS PROCESS ONLY (it changes on every restart): {_configured_api_key}\n"
        "Set PHI_DEID_API_KEY as an environment variable for a stable key that "
        "survives restarts (required before any real deployment)."
    )


def verify_api_key(provided_key: str | None = Depends(_api_key_header)) -> None:
    if provided_key is None or not secrets.compare_digest(provided_key, _configured_api_key):
        raise HTTPException(status_code=401, detail="Missing or invalid API key (X-API-Key header).")


class RedactRequest(BaseModel):
    text: str = Field(..., description="Raw document text to de-identify.")
    filename: str = Field(default="uploaded_document", description="Original filename, for the audit trail.")


class ReviewDecision(BaseModel):
    span_index: int
    approved: bool
    reviewer: str | None = None


class ResumeRequest(BaseModel):
    thread_id: str
    decisions: list[ReviewDecision]


def _format_result(result: dict, thread_id: str) -> dict[str, Any]:
    """Shared response shaping for /redact, /redact/upload, and /redact/resume
    -- all three ultimately produce the same run_sync()/resume_sync() result
    shape, so they format it the same way."""
    if result["status"] == "interrupted":
        logger.info(
            "human review required thread_id=%s pending_spans=%d",
            thread_id, len(result["interrupt"].get("spans", []) or []),
        )
        return {
            "status": "human_review_required",
            "thread_id": thread_id,
            "review_payload": result["interrupt"],
        }

    state = result["state"]
    logger.info(
        "run completed thread_id=%s doc_type=%s validation_status=%s phi_spans=%d",
        thread_id,
        state.get("doc_type"),
        state.get("validation_status"),
        len(state.get("phi_spans", []) or []),
    )
    return {
        "status": "completed",
        "thread_id": thread_id,
        "doc_type": state.get("doc_type"),
        "redacted_text": state.get("redacted_text"),
        "audit_log": state.get("audit_log"),
        "compliance_report": state.get("compliance_report"),
    }


def _internal_error_response(thread_id: str, endpoint: str) -> HTTPException:
    """Logs the full exception server-side (logger.exception grabs the
    active exception + traceback automatically) but returns only a
    correlation ID to the caller -- never the raw exception text, which
    could echo a fragment of the document being processed back through
    the HTTP response."""
    logger.exception("unhandled error in %s (thread_id=%s)", endpoint, thread_id)
    return HTTPException(
        status_code=500,
        detail=f"Internal error processing this document. Reference ID: {thread_id}",
    )


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/redact", dependencies=[Depends(verify_api_key)])
def redact(request: RedactRequest):
    thread_id = str(uuid.uuid4())
    try:
        result = run_sync(request.text, filename=request.filename, thread_id=thread_id)
    except Exception:
        raise _internal_error_response(thread_id, "/redact") from None

    return _format_result(result, thread_id)


@app.post("/redact/upload", dependencies=[Depends(verify_api_key)])
async def redact_upload(file: UploadFile = File(...)):
    """Same pipeline as /redact, but takes an uploaded file (.txt, .pdf, or
    .docx) instead of raw JSON text. Text is extracted in-memory -- the file
    is never written to disk -- via ingestion/document_loader.py, then
    handed to the same run_sync() the JSON endpoint uses.
    """
    file_bytes = await file.read()
    filename = file.filename or "uploaded_document"

    try:
        text = load_text_from_bytes(file_bytes, filename=filename)
    except UnsupportedFileTypeError as exc:
        # Safe to return directly: this message only ever contains the
        # filename/extension, never document content.
        raise HTTPException(status_code=415, detail=str(exc)) from exc
    except EmptyDocumentError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc

    thread_id = str(uuid.uuid4())
    try:
        result = run_sync(text, filename=filename, thread_id=thread_id)
    except Exception:
        raise _internal_error_response(thread_id, "/redact/upload") from None

    return _format_result(result, thread_id)


@app.post("/redact/resume", dependencies=[Depends(verify_api_key)])
def redact_resume(request: ResumeRequest):
    decisions = [d.model_dump() for d in request.decisions]
    try:
        result = resume_sync(decisions, thread_id=request.thread_id)
    except Exception:
        raise _internal_error_response(request.thread_id, "/redact/resume") from None

    return _format_result(result, request.thread_id)
