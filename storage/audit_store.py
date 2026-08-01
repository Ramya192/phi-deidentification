"""
storage/audit_store.py

Persistent, queryable audit trail -- separate from graph/workflow.py's
LangGraph checkpointer.

The checkpointer (data/checkpoints.sqlite) persists GraphState across the
interrupt()/resume boundary so an in-flight human-review session survives
a process restart -- that's pipeline *state*, scoped to one in-progress
run, and its schema is whatever GraphState happens to contain. It is not
designed to be queried across runs ("show me every FAIL in March"), and
LangGraph makes no promise about its internal row shape being stable.

This module is the other half: once a run reaches AuditReportAgent and
produces a compliance_report + audit_log, store_audit() persists an
immutable row keyed by thread_id (the same id api/main.py already uses
per run) into its own table with a fixed, versioned schema. That's what
"was this document HIPAA-compliant" and "give me every audit record from
last week" should be queried against, and it must survive a restart the
same way the checkpointer does -- SQLite by default, swap the
database_url for Postgres/MySQL for a multi-replica deployment (same
single-writer caveat as the checkpointer applies to the SQLite default;
see graph/workflow.py's _build_checkpointer docstring).

Rows are append-only: no update_audit()/delete_audit() is exposed. A
HIPAA audit trail that could be edited after the fact isn't one.
"""
from __future__ import annotations

import os
from datetime import datetime, timezone
from typing import Any, Optional

from sqlalchemy import (
    JSON,
    Boolean,
    Column,
    DateTime,
    Float,
    Integer,
    String,
    create_engine,
    inspect,
    text,
)
from sqlalchemy.orm import DeclarativeBase, Session, sessionmaker

DEFAULT_CLIENT_ID = "default"

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(THIS_DIR)
DEFAULT_DB_PATH = os.path.join(PROJECT_ROOT, "data", "audit.sqlite")
DEFAULT_DATABASE_URL = f"sqlite:///{DEFAULT_DB_PATH}"


class Base(DeclarativeBase):
    pass


class AuditRecordDB(Base):
    """One immutable row per completed (or exhausted-retry-flagged) run.

    Mirrors agents/audit_report_agent.py's compliance_report shape --
    the summary fields are promoted to real columns so they're indexable/
    queryable; the full compliance_report and audit_log are also kept
    verbatim as JSON so nothing is lost to the summarization.
    """

    __tablename__ = "audit_records"

    record_id = Column(String(64), primary_key=True)  # thread_id
    persisted_at = Column(DateTime, default=lambda: datetime.now(timezone.utc), index=True)

    # Caller-scoping identity (api/main.py's X-Client-Id header, or
    # DEFAULT_CLIENT_ID if the caller never set one) -- lets
    # GET /records/{id}/audit and GET /records refuse to hand back a
    # different caller's record. Not itself an authentication mechanism;
    # every caller still needs a valid X-API-Key first. See "Persistent
    # audit trail" in README.md.
    client_id = Column(String(255), nullable=False, default=DEFAULT_CLIENT_ID, index=True)

    filename = Column(String(255), nullable=True)
    doc_type = Column(String(64), nullable=True, index=True)
    doc_type_confidence = Column(Float, nullable=True)

    validation_status = Column(String(16), nullable=True, index=True)
    retries_used = Column(Integer, default=0)
    retries_exhausted_while_failing = Column(Boolean, default=False)
    requires_manual_followup = Column(Boolean, default=False, index=True)

    schema_completeness_score = Column(Float, nullable=True)
    compliance_score = Column(Float, nullable=True)
    total_phi_spans_detected = Column(Integer, default=0)
    human_review_invoked = Column(Boolean, default=False)
    human_review_decisions_count = Column(Integer, default=0)
    total_pipeline_ms = Column(Float, nullable=True)
    total_llm_cost_usd = Column(Float, nullable=True)  # 0.0 for LLM-free runs (default heuristic backend)

    # Full detail, verbatim -- compliance_report already contains most of
    # the scalar fields above too; kept redundant on purpose so a schema
    # migration to the summary columns never has to backfill from a
    # lossy source.
    compliance_report = Column(JSON, nullable=False)
    audit_log = Column(JSON, nullable=False)


def _redact_marker(value: str) -> str:
    return f"[REDACTED:{len(value)}chars]"


def _sanitize_audit_log(audit_log: list[dict]) -> list[dict]:
    """Strip literal PHI text out of audit_log entries before they're
    persisted long-term.

    Every agent (PHIDetectionAgent, RedactionAgent, HumanReviewAgent,
    ComplianceValidationAgent, LLMAdjudicationAgent, ...) writes the raw
    span_text into its audit_log entries -- see each agent's
    audit_log.append() call. That's correct and necessary for the
    in-flight GraphState: a human reviewer mid-run needs to see the
    actual text to approve/reject it, and the LangGraph checkpointer that
    holds that state exists specifically to survive a restart during that
    review. It's also fine in the API's immediate response, since the
    caller already submitted that exact text themselves.

    It stops being fine the moment it lands in AuditStore: this table is
    long-lived, queryable across every run ("show me every FAIL in
    March"), and reachable through GET /records/{id}/audit -- a persisted
    store of raw patient identifiers is exactly the kind of secondary PHI
    exposure a de-identification pipeline is supposed to prevent, and
    nothing about the audit trail's actual purpose (which agent acted,
    what type, what confidence, pass/fail) requires keeping the literal
    value around. So this function -- called only from
    AuditStore.store_audit(), not from the agents themselves -- replaces
    span_text with a length-preserving placeholder right before the
    permanent write.

    Note: this is the data-exposure half of GET /records/{id}/audit's
    hardening -- per-record access control (client_id scoping, checked in
    api/main.py's get_audit_record) is the separate, now-also-closed
    authorization half.
    """
    sanitized = []
    for entry in audit_log:
        entry = dict(entry)
        if entry.get("span_text"):
            entry["span_text"] = _redact_marker(entry["span_text"])
        sanitized.append(entry)
    return sanitized


def _sanitize_compliance_report(compliance_report: dict[str, Any]) -> dict[str, Any]:
    """Same rationale as _sanitize_audit_log(), applied to the one other
    place raw PHI text ends up in a persisted record:
    compliance_report["remaining_phi_spans_after_validation"] (see
    agents/audit_report_agent.py) -- populated whenever validation
    failed and PHI is still present in the final redacted_text.
    """
    report = dict(compliance_report)
    remaining = report.get("remaining_phi_spans_after_validation")
    if remaining:
        report["remaining_phi_spans_after_validation"] = [
            {**span, "text": _redact_marker(span["text"])} if span.get("text") else span
            for span in remaining
        ]
    return report


def _row_to_dict(row: AuditRecordDB) -> dict[str, Any]:
    return {
        "record_id": row.record_id,
        "client_id": row.client_id or DEFAULT_CLIENT_ID,
        "persisted_at": row.persisted_at.isoformat() if row.persisted_at else None,
        "filename": row.filename,
        "doc_type": row.doc_type,
        "doc_type_confidence": row.doc_type_confidence,
        "validation_status": row.validation_status,
        "retries_used": row.retries_used,
        "retries_exhausted_while_failing": row.retries_exhausted_while_failing,
        "requires_manual_followup": row.requires_manual_followup,
        "schema_completeness_score": row.schema_completeness_score,
        "compliance_score": row.compliance_score,
        "total_phi_spans_detected": row.total_phi_spans_detected,
        "human_review_invoked": row.human_review_invoked,
        "human_review_decisions_count": row.human_review_decisions_count,
        "total_pipeline_ms": row.total_pipeline_ms,
        "total_llm_cost_usd": row.total_llm_cost_usd,
        "compliance_report": row.compliance_report,
        "audit_log": row.audit_log,
    }


class AuditStore:
    """Production-grade audit store. Defaults to a local SQLite file
    (data/audit.sqlite, same directory convention as checkpoints.sqlite)
    so it works out of the box with zero extra infra; pass a Postgres/
    MySQL database_url for a real multi-replica deployment.
    """

    def __init__(self, database_url: str = DEFAULT_DATABASE_URL):
        self.database_url = database_url
        connect_args = {"check_same_thread": False} if database_url.startswith("sqlite") else {}
        if database_url.startswith("sqlite:///") and database_url != "sqlite:///:memory:":
            db_path = database_url.replace("sqlite:///", "", 1)
            os.makedirs(os.path.dirname(db_path) or ".", exist_ok=True)

        self.engine = create_engine(database_url, echo=False, connect_args=connect_args)
        Base.metadata.create_all(self.engine)
        self._ensure_client_id_column()
        self._SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=self.engine)

    def _ensure_client_id_column(self) -> None:
        """create_all() only creates tables that don't exist yet -- it
        never alters an existing one. Any audit.sqlite created before
        per-record scoping was added would be missing this column, so
        best-effort ALTER TABLE it in. SQLite backfills existing rows
        with the DEFAULT value when a column is added this way, so old
        records land in DEFAULT_CLIENT_ID rather than NULL. Same
        never-let-a-persistence-hiccup-break-the-run posture as the rest
        of this module: any failure here is swallowed, not raised."""
        try:
            inspector = inspect(self.engine)
            columns = {c["name"] for c in inspector.get_columns("audit_records")}
            if "client_id" not in columns:
                with self.engine.begin() as conn:
                    conn.execute(
                        text(
                            f"ALTER TABLE audit_records ADD COLUMN client_id VARCHAR(255) "
                            f"DEFAULT '{DEFAULT_CLIENT_ID}'"
                        )
                    )
        except Exception:
            pass

    def store_audit(
        self,
        record_id: str,
        compliance_report: dict[str, Any],
        audit_log: list[dict],
        client_id: str = DEFAULT_CLIENT_ID,
    ) -> None:
        """Insert one immutable audit record. Calling this twice with the
        same record_id raises an IntegrityError -- audit rows don't get
        overwritten, including by a caller retrying the same run.

        compliance_report and audit_log are sanitized (raw PHI span text
        replaced with length-preserving placeholders -- see
        _sanitize_audit_log / _sanitize_compliance_report docstrings)
        before anything is written, so this is the single enforcement
        point for every caller, not just api/main.py's one call site.
        """
        compliance_report = _sanitize_compliance_report(compliance_report)
        audit_log = _sanitize_audit_log(audit_log)
        session: Session = self._SessionLocal()
        try:
            row = AuditRecordDB(
                record_id=record_id,
                client_id=client_id or DEFAULT_CLIENT_ID,
                filename=compliance_report.get("filename"),
                doc_type=compliance_report.get("doc_type"),
                doc_type_confidence=compliance_report.get("doc_type_confidence"),
                validation_status=compliance_report.get("validation_status"),
                retries_used=compliance_report.get("retries_used", 0),
                retries_exhausted_while_failing=bool(compliance_report.get("retries_exhausted_while_failing", False)),
                requires_manual_followup=bool(compliance_report.get("requires_manual_followup", False)),
                schema_completeness_score=compliance_report.get("schema_completeness_score"),
                compliance_score=compliance_report.get("compliance_score"),
                total_phi_spans_detected=compliance_report.get("total_phi_spans_detected", 0),
                human_review_invoked=bool(compliance_report.get("human_review_invoked", False)),
                human_review_decisions_count=compliance_report.get("human_review_decisions_count", 0),
                total_pipeline_ms=compliance_report.get("total_pipeline_ms"),
                total_llm_cost_usd=(compliance_report.get("llm_usage_summary") or {}).get("total_approx_cost_usd", 0.0),
                compliance_report=compliance_report,
                audit_log=audit_log,
            )
            session.add(row)
            session.commit()
        finally:
            session.close()

    def get_audit(self, record_id: str) -> Optional[dict[str, Any]]:
        session: Session = self._SessionLocal()
        try:
            row = session.get(AuditRecordDB, record_id)
            return _row_to_dict(row) if row else None
        finally:
            session.close()

    def list_audits(
        self, limit: int = 100, offset: int = 0, client_id: Optional[str] = None
    ) -> list[dict[str, Any]]:
        """client_id=None (the admin-key path in api/main.py) returns
        records across every caller; a real client_id scopes the query
        to that caller's own records only, filtered in SQL so limit/
        offset paginate correctly rather than over-fetching and
        filtering in Python."""
        session: Session = self._SessionLocal()
        try:
            query = session.query(AuditRecordDB)
            if client_id is not None:
                query = query.filter(AuditRecordDB.client_id == client_id)
            rows = (
                query.order_by(AuditRecordDB.persisted_at.desc())
                .limit(limit)
                .offset(offset)
                .all()
            )
            return [_row_to_dict(r) for r in rows]
        finally:
            session.close()

    def list_by_date_range(self, start: datetime, end: datetime) -> list[dict[str, Any]]:
        session: Session = self._SessionLocal()
        try:
            rows = (
                session.query(AuditRecordDB)
                .filter(AuditRecordDB.persisted_at >= start, AuditRecordDB.persisted_at <= end)
                .order_by(AuditRecordDB.persisted_at)
                .all()
            )
            return [_row_to_dict(r) for r in rows]
        finally:
            session.close()


_default_store: Optional[AuditStore] = None


def get_audit_store() -> AuditStore:
    """Module-level singleton, matching the pattern graph/workflow.py uses
    for compiled_graph -- one store per process, database_url overridable
    via PHI_DEID_AUDIT_DB_URL for pointing at Postgres in production."""
    global _default_store
    if _default_store is None:
        database_url = os.environ.get("PHI_DEID_AUDIT_DB_URL", DEFAULT_DATABASE_URL)
        _default_store = AuditStore(database_url)
    return _default_store
