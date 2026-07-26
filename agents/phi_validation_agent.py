"""
PHIValidationAgent (new — closes a gap between the capstone spec and what
was actually built)

The capstone architecture (claude-prompt-capstone.txt) describes this as a
standalone node performing two distinct jobs:
  1. Confidence check — split detected spans into high/low confidence so
     the graph can route to RedactionAgent vs. HumanReviewAgent.
  2. Completeness/schema check — does the detected entity list cover the
     identifier types this document type is expected to carry at all?
     e.g. a pathology report with zero MRN/accession-number hits is a red
     flag ("possible detection miss or misclassification"), not a clean
     document.

Job (1) was already implemented, but inside PHIDetectionAgent itself
(CONFIDENCE_THRESHOLDS + the high/low split in phi_detection_agent()) —
that stays there deliberately: the thresholds are backend-specific and
calibrated against eval/ numbers for that exact detector, so keeping them
next to the detector avoids the two drifting out of sync. Splitting that
into a second node would just add a hop with no new information.

Job (2) — the schema completeness check — was designed in the original
conversation but never built. This module is that missing piece: a real,
separate node (so it gets its own audit-trail identity, "PHIValidationAgent",
matching the spec) that runs immediately after PHIDetectionAgent and before
the confidence-based redaction/human-review split.

Deliberately advisory, not blocking: a missing expected identifier does NOT
force a retry or human review by itself (that would risk a second,
uncapped loop alongside ComplianceValidationAgent's existing retry
mechanism, and would flag almost every short/terse document as suspicious).
Instead it's surfaced as a flag on GraphState and rolled into
AuditReportAgent's compliance_report (requires_manual_followup), which is
exactly the "route_to_reprocessing" / "flagged for review" spirit of the
spec's failure-case examples, applied as a soft signal rather than a hard
gate.
"""
from __future__ import annotations

from datetime import datetime, timezone

from graph.state import DocType, GraphState

# ---------------------------------------------------------------------------
# Minimum identifier types each document type is expected to carry at least
# one instance of. Grounded in the same field sets
# scripts/build_sample_dataset.py injects as ground truth per doc type
# (_FIELD_SETS) — not invented — so this check is validated against the
# same vocabulary PHIDetectionAgent actually emits and eval/ actually
# scores. Deliberately a MINIMUM set (e.g. not every doc type's optional
# fields), so this stays a real completeness signal and not noise.
# ---------------------------------------------------------------------------
EXPECTED_IDENTIFIER_TYPES_BY_DOC_TYPE: dict[DocType, frozenset[str]] = {
    "clinical_note": frozenset({"PERSON", "DATE_TIME", "MRN"}),
    "discharge_summary": frozenset({"PERSON", "DATE_TIME", "MRN", "ACCOUNT_NUMBER"}),
    "radiology_report": frozenset({"PERSON", "DATE_TIME", "MRN"}),
    "pathology_report": frozenset({"PERSON", "DATE_TIME", "MRN", "ACCESSION_NUMBER"}),
    "lab_report": frozenset({"PERSON", "DATE_TIME", "MRN"}),
    "referral_letter": frozenset({"PERSON", "DATE_TIME"}),
    "insurance_document": frozenset({"PERSON", "HEALTH_PLAN_ID", "CLAIM_NUMBER"}),
}


def phi_validation_agent(state: GraphState) -> GraphState:
    doc_type = state.get("doc_type")
    # Cumulative across every detection pass this run has done so far
    # (GraphState.all_detected_spans), not just phi_spans (the current
    # pass only). On a retry, PHIDetectionAgent re-scans just the current
    # redacted_text for residual PHI, so phi_spans alone would make a type
    # found and redacted in an earlier pass look "missing" here -- a real
    # false alarm this bug used to produce.
    all_spans = state.get("all_detected_spans") or state.get("phi_spans", [])
    # Presence, not confidence: a low-confidence hit still counts as "this
    # category was seen at all" for schema-completeness purposes. Whether
    # it's trustworthy enough to auto-redact is a separate question,
    # already decided by route_after_detection's confidence split.
    detected_types = {s["phi_type"] for s in all_spans}

    expected_types = EXPECTED_IDENTIFIER_TYPES_BY_DOC_TYPE.get(doc_type, frozenset())
    missing_types = sorted(expected_types - detected_types)
    matched_count = len(expected_types) - len(missing_types)
    schema_completeness_score = 1.0 if not expected_types else round(matched_count / len(expected_types), 3)

    audit_log = list(state.get("audit_log", []))
    audit_log.append({
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "agent": "PHIValidationAgent",
        "action": "schema_check",
        "phi_type": None,
        "span_text": None,
        "confidence": None,
        "reviewer_action": None,
        "notes": (
            f"doc_type={doc_type}, expected={sorted(expected_types)}, "
            f"missing={missing_types}, schema_completeness_score={schema_completeness_score}"
        ),
    })

    return {
        **state,
        "expected_identifier_types": sorted(expected_types),
        "missing_expected_identifier_types": missing_types,
        "schema_completeness_score": schema_completeness_score,
        "audit_log": audit_log,
    }
