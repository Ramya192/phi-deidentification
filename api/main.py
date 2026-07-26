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

import json
import logging
import os
import secrets
import uuid
from typing import Any

from fastapi import Depends, FastAPI, File, HTTPException, UploadFile
from fastapi.responses import StreamingResponse
from fastapi.security import APIKeyHeader
from pydantic import BaseModel, Field

try:
    # Loads variables from a local .env file into os.environ, if one
    # exists (see .env.example) -- purely a local-dev convenience so you
    # don't have to `export` everything by hand each terminal session.
    # Deployed environments (Streamlit Cloud secrets, HF Space secrets)
    # set real environment variables directly and don't need this; a
    # missing .env file or missing python-dotenv package is silently a
    # no-op either way, never an error.
    from dotenv import load_dotenv

    load_dotenv()
except ImportError:
    pass

# PHI_DEID_ENV=local|prod is a single switch that decides
# PHI_DEID_EMBED_API and PHI_DEID_SPACY_MODEL automatically, so switching
# contexts means changing one line instead of two. When PHI_DEID_ENV is
# set to a recognized value, it is authoritative -- it overrides whatever
# those two variables happen to say in .env, so there is exactly one
# source of truth. Backward compatible on purpose: if PHI_DEID_ENV is left
# unset/blank (e.g. an existing deployment's secrets set
# PHI_DEID_EMBED_API/PHI_DEID_SPACY_MODEL directly and don't know about
# this switch), this block does nothing and those two variables are read
# exactly as before.
_env_mode = os.environ.get("PHI_DEID_ENV", "").strip().lower()
if _env_mode == "prod":
    os.environ["PHI_DEID_EMBED_API"] = "1"
    os.environ["PHI_DEID_SPACY_MODEL"] = "en_core_web_sm"
elif _env_mode == "local":
    os.environ["PHI_DEID_EMBED_API"] = "0"
    os.environ["PHI_DEID_SPACY_MODEL"] = "en_core_web_lg"

from graph.workflow import compiled_graph, get_checkpointer_backend, resume_stream, resume_sync, run_stream, run_sync
from ingestion.document_loader import (
    EmptyDocumentError,
    UnsupportedFileTypeError,
    load_text_from_bytes,
)
from storage.audit_store import get_audit_store

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


MAX_BATCH_SIZE = 50


class BatchRedactRequest(BaseModel):
    documents: list[RedactRequest] = Field(
        ..., min_length=1, max_length=MAX_BATCH_SIZE,
        description=f"Documents to de-identify, 1-{MAX_BATCH_SIZE} per request.",
    )


class ReviewDecision(BaseModel):
    span_index: int
    approved: bool
    reviewer: str | None = None


class ResumeRequest(BaseModel):
    thread_id: str
    decisions: list[ReviewDecision]


def _persist_audit_record(thread_id: str, state: dict) -> None:
    """Writes the immutable audit record to storage/audit_store.py once a
    run reaches AuditReportAgent. Best-effort: a persistence failure (e.g.
    duplicate thread_id from a caller replaying a request, or a transient
    DB error) is logged server-side but never turned into a 500 for what
    was otherwise a successful de-identification -- the caller already has
    their redacted_text/compliance_report back in the response either way,
    and this table is a queryable historical record, not something the
    response itself depends on.
    """
    compliance_report = state.get("compliance_report")
    if not compliance_report:
        return
    try:
        get_audit_store().store_audit(
            record_id=thread_id,
            compliance_report=compliance_report,
            audit_log=state.get("audit_log", []),
        )
    except Exception:
        logger.exception("failed to persist audit record thread_id=%s", thread_id)


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
    _persist_audit_record(thread_id, state)
    return {
        "status": "completed",
        "thread_id": thread_id,
        "doc_type": state.get("doc_type"),
        "redacted_text": state.get("redacted_text"),
        "audit_log": state.get("audit_log"),
        "compliance_report": state.get("compliance_report"),
    }


def _sse(event: str, data: dict) -> str:
    """Formats one Server-Sent Events message. `event:` lets the client
    tell node-progress events apart from the final result without parsing
    `data` first; `data:` must be a single line, hence json.dumps with no
    indent."""
    return f"event: {event}\ndata: {json.dumps(data)}\n\n"


def _stream_response(generator, thread_id: str, endpoint: str) -> StreamingResponse:
    """Shared SSE wrapper for the three /stream endpoints below: turns
    run_stream()/resume_stream()'s ("node", ...)/("done", ...) tuples into
    SSE messages, formats the final "done" event with the exact same
    _format_result() shape the non-streaming endpoints return (so a
    client can treat the last event identically to a plain POST /redact
    response), and converts any mid-stream exception into the same
    PHI-safe 500 the non-streaming endpoints give -- logged server-side
    with full detail, only a thread_id sent to the caller.
    """
    def _generate():
        try:
            for kind, *rest in generator:
                if kind == "node":
                    node_name, _state = rest
                    yield _sse("node", {"node": node_name, "thread_id": thread_id})
                else:  # kind == "done"
                    status, payload = rest
                    if status == "interrupted":
                        result = {"status": "interrupted", "interrupt": payload}
                    else:
                        result = {"status": "completed", "state": payload}
                    yield _sse("done", _format_result(result, thread_id))
        except Exception:
            logger.exception("unhandled error in %s (thread_id=%s)", endpoint, thread_id)
            yield _sse("error", {
                "thread_id": thread_id,
                "detail": f"Internal error processing this document. Reference ID: {thread_id}",
            })

    return StreamingResponse(_generate(), media_type="text/event-stream")


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
    """Reports real readiness, not just a static "ok": whether the
    LangGraph pipeline actually compiled, and which checkpointer backend
    is active. "memory" means human-review sessions (and any in-flight
    pipeline state) will NOT survive a process restart -- worth knowing
    before relying on this deployment for anything long-running, since it
    silently falls back to that if langgraph-checkpoint-sqlite isn't
    installed (see graph/workflow.py's _build_checkpointer)."""
    checkpointer_backend = get_checkpointer_backend()
    return {
        "status": "ok",
        "graph_compiled": compiled_graph is not None,
        "checkpointer_backend": checkpointer_backend,
        "checkpointer_restart_survivable": checkpointer_backend == "sqlite",
    }


@app.get("/records/{record_id}/audit", dependencies=[Depends(verify_api_key)])
def get_audit_record(record_id: str):
    """Retrieve the persisted audit record for a completed run (storage/audit_store.py),
    keyed by the same thread_id returned in every /redact* response. Distinct
    from the LangGraph checkpointer -- see storage/audit_store.py's module
    docstring for why these are two separate stores with different lifetimes."""
    record = get_audit_store().get_audit(record_id)
    if record is None:
        raise HTTPException(status_code=404, detail="No audit record found for this record_id.")
    return record


@app.get("/records", dependencies=[Depends(verify_api_key)])
def list_audit_records(limit: int = 100, offset: int = 0):
    """Paginated list of persisted audit records, most recent first."""
    limit = max(1, min(limit, 500))
    offset = max(0, offset)
    return {"records": get_audit_store().list_audits(limit=limit, offset=offset)}


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


@app.post("/redact/batch", dependencies=[Depends(verify_api_key)])
def redact_batch(request: BatchRedactRequest):
    """Runs each document in `documents` through the same pipeline /redact
    uses, independently (its own thread_id, its own audit record). Not a
    single atomic transaction -- one document's failure or human-review
    interrupt doesn't affect any other document in the batch. A document
    that needs human review comes back with its own thread_id, resumed
    individually via the existing /redact/resume -- same per-document
    contract as calling /redact one at a time, just fanned out into one
    request/response round-trip instead of N.

    Capped at MAX_BATCH_SIZE documents per request (BatchRedactRequest
    enforces this) so one call can't tie up the process indefinitely --
    the underlying SQLite checkpointer is single-writer (see
    graph/workflow.py's _build_checkpointer docstring), so a very large
    batch would serialize badly anyway.
    """
    results: list[dict[str, Any]] = []
    for index, document in enumerate(request.documents):
        thread_id = str(uuid.uuid4())
        try:
            result = run_sync(document.text, filename=document.filename, thread_id=thread_id)
            formatted = _format_result(result, thread_id)
        except Exception:
            logger.exception(
                "unhandled error in /redact/batch item index=%d thread_id=%s", index, thread_id,
            )
            formatted = {
                "status": "error",
                "thread_id": thread_id,
                "detail": f"Internal error processing this document. Reference ID: {thread_id}",
            }
        results.append({"index": index, "filename": document.filename, **formatted})

    return {
        "count": len(results),
        "completed": sum(1 for r in results if r["status"] == "completed"),
        "human_review_required": sum(1 for r in results if r["status"] == "human_review_required"),
        "errors": sum(1 for r in results if r["status"] == "error"),
        "results": results,
    }


@app.post("/redact/resume", dependencies=[Depends(verify_api_key)])
def redact_resume(request: ResumeRequest):
    decisions = [d.model_dump() for d in request.decisions]
    try:
        result = resume_sync(decisions, thread_id=request.thread_id)
    except Exception:
        raise _internal_error_response(request.thread_id, "/redact/resume") from None

    return _format_result(result, request.thread_id)


# ---------------------------------------------------------------------------
# Streaming (SSE) variants -- same auth, same graph, same final response
# shape as the three endpoints above, but emit an `event: node` message
# after each LangGraph node finishes so a client can show live pipeline
# progress instead of a single opaque "processing..." spinner. See
# graph/workflow.run_stream/resume_stream and _stream_response() above.
# The final message is always `event: done` with a payload identical to
# what the non-streaming endpoint would have returned outright.
# ---------------------------------------------------------------------------
@app.post("/redact/stream", dependencies=[Depends(verify_api_key)])
def redact_stream(request: RedactRequest):
    thread_id = str(uuid.uuid4())
    return _stream_response(
        run_stream(request.text, filename=request.filename, thread_id=thread_id),
        thread_id, "/redact/stream",
    )


@app.post("/redact/upload/stream", dependencies=[Depends(verify_api_key)])
async def redact_upload_stream(file: UploadFile = File(...)):
    file_bytes = await file.read()
    filename = file.filename or "uploaded_document"

    try:
        text = load_text_from_bytes(file_bytes, filename=filename)
    except UnsupportedFileTypeError as exc:
        raise HTTPException(status_code=415, detail=str(exc)) from exc
    except EmptyDocumentError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc

    thread_id = str(uuid.uuid4())
    return _stream_response(
        run_stream(text, filename=filename, thread_id=thread_id),
        thread_id, "/redact/upload/stream",
    )


@app.post("/redact/resume/stream", dependencies=[Depends(verify_api_key)])
def redact_resume_stream(request: ResumeRequest):
    decisions = [d.model_dump() for d in request.decisions]
    return _stream_response(
        resume_stream(decisions, thread_id=request.thread_id),
        request.thread_id, "/redact/resume/stream",
    )
