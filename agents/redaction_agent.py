"""
RedactionAgent (Member 4)

Takes the confirmed PHI spans (high-confidence spans straight from
PHIDetectionAgent, plus whatever HumanReviewAgent approved) and replaces
them with [TYPE] placeholder tags, preserving the rest of the text so the
note stays clinically readable.

Runs after BOTH branches of the confidence split (high-confidence ->
directly here; low-confidence -> HumanReviewAgent -> here), so nothing
that was flagged for human review skips redaction — the human review
step decides *whether* a span is real PHI, this agent decides how it
gets masked once confirmed.

Primary path: Presidio AnonymizerEngine (keeps parity with the
Analyzer's span format). Fallback: plain string splicing, used
automatically when Presidio isn't installed.
"""
from __future__ import annotations

from datetime import datetime, timezone

from graph.state import GraphState, PHISpan

_PRESIDIO_AVAILABLE = False
try:
    from presidio_anonymizer import AnonymizerEngine
    from presidio_anonymizer.entities import OperatorConfig, RecognizerResult

    _PRESIDIO_AVAILABLE = True
except Exception:
    _PRESIDIO_AVAILABLE = False


def _spans_to_redact(state: GraphState) -> list[PHISpan]:
    """
    Confirmed spans = high-confidence spans from detection, plus any
    low-confidence spans a human reviewer approved, plus any low-confidence
    spans agents/llm_adjudication_agent.py's optional LLM adjudicator
    confirmed (empty unless PHI_DEID_ADJUDICATION_BACKEND=llm). Rejected
    spans ("not PHI", from either a human or the LLM adjudicator) are
    excluded.
    """
    confirmed = list(state.get("high_confidence_spans", []))
    for span in state.get("human_reviewed_spans", []):
        confirmed.append(span)
    for span in state.get("llm_reviewed_spans", []):
        confirmed.append(span)
    # de-dup by (start, end) in case a span shows up in both lists
    seen = set()
    unique: list[PHISpan] = []
    for span in confirmed:
        key = (span["start"], span["end"])
        if key not in seen:
            seen.add(key)
            unique.append(span)
    return sorted(unique, key=lambda s: s["start"])


def _redact_presidio(text: str, spans: list[PHISpan]) -> str:
    engine = AnonymizerEngine()
    results = [
        RecognizerResult(entity_type=s["phi_type"], start=s["start"], end=s["end"], score=s["confidence"])
        for s in spans
    ]
    operators = {
        span.entity_type: OperatorConfig("replace", {"new_value": f"[{span.entity_type}]"})
        for span in results
    }
    outcome = engine.anonymize(text=text, analyzer_results=results, operators=operators)
    return outcome.text


def _redact_fallback(text: str, spans: list[PHISpan]) -> str:
    """Splice [TYPE] tags in from the end of the string backwards so earlier
    offsets stay valid as we mutate."""
    result = text
    for span in sorted(spans, key=lambda s: s["start"], reverse=True):
        tag = f"[{span['phi_type']}]"
        result = result[: span["start"]] + tag + result[span["end"]:]
    return result


def redaction_agent(state: GraphState) -> GraphState:
    # First pass redacts the original text. On a retry loop (ComplianceValidationAgent
    # sent us back through PHIDetectionAgent), we redact against the text as it
    # stood after the previous redaction round, since that's what the new spans'
    # offsets refer to.
    text = state.get("redacted_text") if state.get("retry_count", 0) > 0 else None
    if text is None:
        text = state.get("raw_text", "")
    spans = _spans_to_redact(state)

    if _PRESIDIO_AVAILABLE and spans:
        try:
            redacted = _redact_presidio(text, spans)
        except Exception:
            redacted = _redact_fallback(text, spans)
    else:
        redacted = _redact_fallback(text, spans)

    audit_log = list(state.get("audit_log", []))
    for s in spans:
        audit_log.append({
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "agent": "RedactionAgent",
            "action": "redacted",
            "phi_type": s["phi_type"],
            "span_text": s["text"],
            "confidence": s["confidence"],
            "reviewer_action": None,
            "notes": None,
        })

    return {
        **state,
        "redacted_text": redacted,
        "audit_log": audit_log,
    }


def escalation_redaction_agent(state: GraphState) -> GraphState:
    """Redacts ONLY the spans approved during the retry-cap escalation
    review (agents/compliance_validation_agent.py's escalation_review_agent
    stashes them in state["human_reviewed_spans"], same field the normal
    path uses). Deliberately separate from redaction_agent() above: that
    node's _spans_to_redact() merges high_confidence_spans +
    human_reviewed_spans, which is correct for the first-pass confidence
    split but wrong here — at the point escalation runs,
    high_confidence_spans reflects the most recent PHIDetectionAgent pass,
    whose spans were already redacted earlier in this same retry
    iteration, before compliance_validation ever ran. Re-applying their
    old start/end offsets against text that has since shifted (every
    redaction changes downstream offsets) would corrupt the output. This
    node only ever touches the freshly-approved escalation spans against
    the CURRENT redacted_text, so that failure mode can't happen here.
    See escalation_review_agent's docstring for the full reasoning.
    """
    text = state.get("redacted_text", "")
    spans = list(state.get("human_reviewed_spans", []))

    if spans:
        if _PRESIDIO_AVAILABLE:
            try:
                redacted = _redact_presidio(text, spans)
            except Exception:
                redacted = _redact_fallback(text, spans)
        else:
            redacted = _redact_fallback(text, spans)
    else:
        redacted = text

    audit_log = list(state.get("audit_log", []))
    for s in spans:
        audit_log.append({
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "agent": "RedactionAgent",
            "action": "redacted",
            "phi_type": s["phi_type"],
            "span_text": s["text"],
            "confidence": s["confidence"],
            "reviewer_action": None,
            "notes": "retry_cap_escalation_redaction",
        })

    return {
        **state,
        "redacted_text": redacted,
        "audit_log": audit_log,
    }
