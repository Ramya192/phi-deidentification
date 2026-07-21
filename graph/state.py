"""
GraphState — the shared state object that flows through every node in the
LangGraph PHI de-identification workflow.

Every agent reads from and writes to this TypedDict. Keep new fields
optional (Not-required / default-producing) so older nodes don't break
when new fields are added.
"""
from __future__ import annotations

from typing import Any, Literal, TypedDict

# ---------------------------------------------------------------------------
# Document types the ClassificationAgent routes between.
# "not_applicable" is not a document type per se — it's the signal that the
# uploaded file isn't a clinical/health document at all and should be
# routed straight to END without running PHI detection on it.
# ---------------------------------------------------------------------------
DocType = Literal[
    "clinical_note",
    "discharge_summary",
    "radiology_report",
    "pathology_report",
    "lab_report",
    "referral_letter",
    "insurance_document",
    "not_applicable",
]

CLINICAL_DOC_TYPES: tuple[DocType, ...] = (
    "clinical_note",
    "discharge_summary",
    "radiology_report",
    "pathology_report",
    "lab_report",
    "referral_letter",
    "insurance_document",
)

ValidationStatus = Literal["PENDING", "PASS", "FAIL"]


class PHISpan(TypedDict):
    """A single detected (or redacted) PHI entity span."""
    start: int
    end: int
    text: str
    phi_type: str          # e.g. PERSON, DATE_TIME, MRN, PHONE_NUMBER, LOCATION
    confidence: float      # 0.0 - 1.0
    source_agent: str      # which detector found it (presidio, regex, human_review, ...)


class AuditEntry(TypedDict):
    """One row of the audit trail — attached to every redaction / decision."""
    timestamp: str
    agent: str
    action: str             # e.g. "detected", "redacted", "human_approved", "human_rejected", "retry"
    phi_type: str | None
    span_text: str | None
    confidence: float | None
    reviewer_action: str | None
    notes: str | None


class GraphState(TypedDict, total=False):
    # --- input ---
    raw_text: str
    filename: str

    # --- ClassificationAgent output ---
    doc_type: DocType
    doc_type_confidence: float

    # --- PHIDetectionAgent output ---
    phi_spans: list[PHISpan]
    confidence_scores: dict[str, float]     # span_id -> confidence (mirror, convenient for eval)
    low_confidence_spans: list[PHISpan]
    high_confidence_spans: list[PHISpan]

    # --- HumanReviewAgent output ---
    human_reviewed_spans: list[PHISpan]
    human_decisions: list[dict[str, Any]]
    # Accumulates across retry loops: every (phi_type, text) a human has
    # explicitly rejected as "not PHI". PHIDetectionAgent and
    # ComplianceValidationAgent both consult this so a rejected span isn't
    # re-detected and re-sent to human review forever on each retry pass --
    # see phi_detection_agent._filter_rejected for why this exists.
    rejected_spans: list[dict[str, str]]

    # --- RedactionAgent output ---
    redacted_text: str

    # --- ComplianceValidationAgent output ---
    validation_status: ValidationStatus
    remaining_phi_spans: list[PHISpan]
    retry_count: int
    max_retries: int

    # --- AuditReportAgent output ---
    audit_log: list[AuditEntry]
    compliance_report: dict[str, Any]

    # --- control / bookkeeping ---
    errors: list[str]
