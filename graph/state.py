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
    # phi_spans is scoped to the CURRENT detection pass only -- on a retry,
    # PHIDetectionAgent deliberately re-scans just the current redacted_text
    # for residual PHI (see that module's docstring), so this list is small
    # and round-specific by design; it drives the immediate confidence
    # split/redaction for *this* pass, nothing more.
    phi_spans: list[PHISpan]
    # all_detected_spans accumulates every span found across every pass of
    # this run (first pass + every retry), so anything that asks "what PHI
    # did this document ever contain" -- PHIValidationAgent's completeness
    # check, AuditReportAgent's final totals -- reflects the whole run, not
    # just whatever the most recent retry pass happened to re-scan. Without
    # this, a document that triggers even one retry would under-report
    # (missing_expected_identifier_types would wrongly list types that were
    # in fact found and redacted in an earlier pass).
    all_detected_spans: list[PHISpan]
    confidence_scores: dict[str, float]     # span_id -> confidence (mirror, convenient for eval)
    low_confidence_spans: list[PHISpan]
    high_confidence_spans: list[PHISpan]

    # --- PHIValidationAgent output (schema/completeness check, runs right
    # after PHIDetectionAgent, before the confidence-based redaction/
    # human-review split) ---
    expected_identifier_types: list[str]
    missing_expected_identifier_types: list[str]
    schema_completeness_score: float

    # --- LLMAdjudicationAgent output (agents/llm_adjudication_agent.py) ---
    # Only populated when PHI_DEID_ADJUDICATION_BACKEND=llm; otherwise this
    # agent is a no-op and the field stays empty/absent. Spans the LLM
    # confirmed as genuine PHI, tagged source_agent="llm_adjudication" --
    # redaction_agent._spans_to_redact() includes these alongside
    # high_confidence_spans and human_reviewed_spans.
    llm_reviewed_spans: list[PHISpan]

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

    # --- ComplianceValidationAgent output: per-category audit breakdown.
    # One check per PHI category this document actually triggered (not
    # per abstract Safe Harbor category -- see
    # compliance_validation_agent._build_compliance_checks), each with
    # passed/residual_count/details, plus an aggregate 0-1 score. A more
    # granular complement to the single validation_status PASS/FAIL.
    compliance_checks: list[dict[str, Any]]
    compliance_score: float

    # --- Retry-cap escalation (compliance_validation_agent.escalate_to_review
    # / escalation_review_agent / escalation_redaction_agent) ---
    # Populated only when retries are exhausted while validation_status is
    # still FAIL: escalate_to_review copies remaining_phi_spans here for a
    # dedicated one-shot human review pass (distinct from HumanReviewAgent's
    # pre-redaction low_confidence_spans review -- see
    # agents/human_review_agent.py vs escalation_review_agent's docstring
    # in agents/compliance_validation_agent.py for why these are separate
    # nodes rather than one branching node). escalation_attempted gates
    # this to run at most once per pipeline run, so a still-failing
    # escalation redaction routes to AuditReportAgent (flagged) instead of
    # looping again.
    escalation_spans: list[PHISpan]
    escalation_attempted: bool

    # --- AuditReportAgent output ---
    audit_log: list[AuditEntry]
    compliance_report: dict[str, Any]

    # --- Observability: per-node wall-clock timing, one entry per node
    # execution (a node hit twice via the retry loop gets two entries).
    # Populated by graph/workflow.py's _timed() wrapper, not by the agents
    # themselves -- this is a cross-cutting infra concern, not business
    # logic any individual agent should own. See AuditReportAgent /
    # compliance_report["node_timings_summary"] for the aggregated view.
    node_timings: list[dict[str, Any]]

    # --- Observability: real per-call LLM token/cost usage (see
    # observability/llm_metrics.py). Empty for the default heuristic
    # classification backend and for PHIDetectionAgent/RedactionAgent,
    # none of which call an LLM -- only populated when
    # PHI_DEID_CLASSIFICATION_BACKEND=llm actually invokes
    # classify_with_llm(). Aggregated into
    # compliance_report["llm_usage_summary"] by AuditReportAgent.
    llm_usage_log: list[dict[str, Any]]

    # --- control / bookkeeping ---
    errors: list[str]
