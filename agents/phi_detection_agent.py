"""
PHIDetectionAgent (Member 3)

Primary path: Microsoft Presidio (AnalyzerEngine, spaCy NLP engine) plus a
few clinical-specific custom recognizers (MRN, clinical-format dates,
age-over-89 per Safe Harbor) layered on top, mirroring the
"scispaCy + Presidio -> confidence scores" design in the workflow diagram.

Fallback path: if presidio/spacy aren't installed (e.g. offline dev,
CI without model downloads), a dependency-free regex detector kicks in
automatically so the graph is still runnable end-to-end. Swap
USE_FALLBACK_ONLY = True to force it for fast local testing.

Both paths produce the same PHISpan shape defined in graph/state.py, so
everything downstream (Redaction, HumanReview, ComplianceValidation) is
agnostic to which backend actually ran.
"""
from __future__ import annotations

import re
from datetime import datetime, timezone

from graph.state import GraphState, PHISpan

HIGH_CONFIDENCE_THRESHOLD = 0.7  # default bar for any phi_type not listed in CONFIDENCE_THRESHOLDS below

# Per-entity-type override for the "auto-redact without human review" bar.
# A single global threshold treats every detector as equally trustworthy,
# which isn't true: calibrated against real numbers from
# eval/evaluate.py --backend presidio (see eval/final_results.csv).
#
#   - Regex/pattern-backed types (MRN, HEALTH_PLAN_ID, ACCOUNT_NUMBER,
#     ACCESSION_NUMBER, CLAIM_NUMBER, EMAIL_ADDRESS, SSN, IP_ADDRESS, URL,
#     FAX_NUMBER) are near-deterministic once matched -- our eval measured
#     ~1.0 precision on these. Gating them behind human review at the same
#     bar as noisy NER types would just be reviewer fatigue with no real
#     safety benefit, so they get a LOWER auto-redact threshold.
#   - NER-driven types are noisier. PERSON measured only 0.58 precision
#     (OVERLAP) on our eval set -- Presidio's own confidence score doesn't
#     cleanly separate its true and false positives for this type. Raising
#     its threshold pushes more of those calls into human review instead of
#     silently auto-redacting (or silently trusting) a mislabeled span.
#     DATE_TIME had a lot of false-positive noise too (0.35-0.75 precision
#     depending on backend), largely from real dates in body text that
#     aren't part of the injected ground truth -- still worth a higher bar
#     since we can't cleanly separate "real PHI date" from "clinical
#     narrative date" on confidence alone.
CONFIDENCE_THRESHOLDS: dict[str, float] = {
    "MRN": 0.6,
    "HEALTH_PLAN_ID": 0.6,
    "ACCOUNT_NUMBER": 0.6,
    "ACCESSION_NUMBER": 0.6,
    "CLAIM_NUMBER": 0.6,
    "EMAIL_ADDRESS": 0.6,
    "SSN": 0.6,
    "IP_ADDRESS": 0.6,
    "URL": 0.6,
    "FAX_NUMBER": 0.6,
    "PHONE_NUMBER": 0.65,
    "PERSON": 0.85,
    "DATE_TIME": 0.8,
    "LOCATION": 0.85,
    "NRP": 0.85,
}


def _threshold_for(phi_type: str) -> float:
    return CONFIDENCE_THRESHOLDS.get(phi_type, HIGH_CONFIDENCE_THRESHOLD)

# ---------------------------------------------------------------------------
# Try to load the Presidio + spaCy backend. Fall back gracefully if the
# heavy deps aren't installed in this environment.
# ---------------------------------------------------------------------------
_PRESIDIO_AVAILABLE = False
try:
    from presidio_analyzer import AnalyzerEngine, Pattern, PatternRecognizer

    _PRESIDIO_AVAILABLE = True
except Exception:  # ImportError or spacy model missing
    _PRESIDIO_AVAILABLE = False

USE_FALLBACK_ONLY = False  # flip to True to force the regex-only backend


# ---------------------------------------------------------------------------
# Fallback backend: dependency-free regex PHI detector.
# Covers the 18 HIPAA Safe Harbor identifier categories at a basic level:
# names (weak — needs NER for real coverage, flagged low-confidence),
# dates, phone, fax, email, SSN, MRN, health plan #, account #,
# certificate/license #, vehicle ID, device ID, URL, IP, biometric refs.
# ---------------------------------------------------------------------------
# Both phone patterns end with an optional extension suffix
# (`x462`, `ext. 462`, ` x 462`, ...) -- found live via the Streamlit UI:
# "+1-834-576-8701x462" was passing through completely unredacted because
# the old pattern's trailing `\b` requires a non-word character right
# after the last digit, and "1x" (digit directly followed by a letter)
# has no such boundary, so the whole match failed rather than just
# stopping short of the extension.
_PHONE_EXTENSION = r"(?:\s?(?:ext\.?|x)\.?\s?\d{2,6})?"
_FALLBACK_PATTERNS: list[tuple[str, str, float]] = [
    ("EMAIL_ADDRESS", r"\b[\w.+-]+@[\w-]+\.[\w.-]+\b", 0.95),
    ("FAX_NUMBER", r"\bFax[:\s#]*(?:\+?1[-.\s]?)?\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}" + _PHONE_EXTENSION + r"\b", 0.85),
    ("PHONE_NUMBER", r"(?:\+?1[-.\s]?)?\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}" + _PHONE_EXTENSION + r"\b", 0.85),
    ("SSN", r"\b\d{3}-\d{2}-\d{4}\b", 0.95),
    ("MRN", r"\b(?:MRN|Medical Record Number)[:\s#]*[A-Z0-9-]{5,15}\b", 0.9),
    # NOTE: "Number"/"No."/"#" is optional and can appear *before* the
    # colon (e.g. "Account Number: ACCT-123456"), which the original
    # `(?:Account|Acct)[:\s#]*...` pattern missed entirely -- it required
    # the label to be followed immediately by punctuation/whitespace, so
    # every real "<Label> Number: <value>" header line fell through and
    # this type scored 0% recall even on the deterministic fallback.
    ("ACCOUNT_NUMBER", r"\b(?:Account|Acct)\s*(?:Number|No\.?|#)?[:\s#]*[A-Z0-9-]{5,15}\b", 0.8),
    ("ACCESSION_NUMBER", r"\bAccession\s*(?:Number|No\.?|#)?[:\s#]*[A-Z0-9-]{5,15}\b", 0.85),
    ("CLAIM_NUMBER", r"\bClaim\s*(?:Number|No\.?|#)?[:\s#]*[A-Z0-9-]{5,15}\b", 0.85),
    ("HEALTH_PLAN_ID", r"\b(?:Member|Policy|Plan)\s*(?:ID|#|Number)[:\s#]*[A-Z0-9-]{5,15}\b", 0.8),
    ("DATE_TIME", r"\b(?:0?[1-9]|1[0-2])[/-](?:0?[1-9]|[12]\d|3[01])[/-](?:\d{2}|\d{4})\b", 0.75),
    ("DATE_TIME", r"\b(?:January|February|March|April|May|June|July|August|September|"
                   r"October|November|December)\s+\d{1,2},?\s+\d{4}\b", 0.8),
    ("URL", r"\bhttps?://[^\s]+\b", 0.9),
    ("IP_ADDRESS", r"\b(?:\d{1,3}\.){3}\d{1,3}\b", 0.85),
    ("ZIP_CODE", r"\b\d{5}(?:-\d{4})?\b", 0.4),  # weak — lots of false positives (e.g. lab values)
    ("AGE_OVER_89", r"\bage(?:d)?\s*(?:9[0-9]|[1-9]\d{2,})\b", 0.7),
    # Names are the hardest without NER. Heuristic: "Mr./Mrs./Ms./Dr. <Capitalized word(s)>"
    ("PERSON", r"\b(?:Mr|Mrs|Ms|Dr|Pt)\.?\s+[A-Z][a-z]+(?:\s+[A-Z][a-z]+)?\b", 0.6),
]


def _detect_fallback(text: str) -> list[PHISpan]:
    spans: list[PHISpan] = []
    for phi_type, pattern, confidence in _FALLBACK_PATTERNS:
        for m in re.finditer(pattern, text):
            spans.append({
                "start": m.start(),
                "end": m.end(),
                "text": m.group(),
                "phi_type": phi_type,
                "confidence": confidence,
                "source_agent": "regex_fallback",
            })
    return _dedupe_overlaps(spans)


def _clip_at_newlines(spans: list[PHISpan]) -> list[PHISpan]:
    """Defensive guard found live via the Streamlit UI: a Presidio PERSON
    match extended across a line break in a structured header ("Patient
    Name: James Simmons MD\\r\\nDOB:"), swallowing the next line's "DOB:"
    label and the newline itself. The redacted output visually merged two
    header lines into one and silently deleted a field label -- not a PHI
    leak in that specific case (the date value still got redacted as its
    own span), but a real correctness bug and a plausible path to a worse
    one (an over-extended span consuming adjacent structure instead of
    just the intended value).

    No legitimate single PHI value in our structured documents should
    ever contain a line break -- header fields are one label:value per
    line, and body prose doesn't contain injected PHI. So: any span whose
    matched text contains \\r or \\n gets truncated to end right before the
    first line-break character, rather than trusting the detector's
    boundary. This only ever shrinks a span, never grows one, so it can't
    introduce a new leak -- worst case it clips a legitimately multi-line
    match down to its first line, which doesn't currently happen anywhere
    in this codebase's PHI types.
    """
    clipped: list[PHISpan] = []
    for span in spans:
        text_value = span["text"]
        break_pos = next((i for i, ch in enumerate(text_value) if ch in "\r\n"), None)
        if break_pos is None:
            clipped.append(span)
        elif break_pos > 0:
            clipped.append({**span, "end": span["start"] + break_pos, "text": text_value[:break_pos]})
        # break_pos == 0: span starts with a line break -- nothing
        # meaningful survives, drop it.
    return clipped


def _dedupe_overlaps(spans: list[PHISpan]) -> list[PHISpan]:
    """If two spans overlap, keep the higher-confidence one -- regardless of
    which one starts first in the text.

    BUG FIX (found via a real eval run + diagnose_accession.py): the
    previous version sorted by `(start, -confidence)`, which processes
    spans in POSITION order and greedily keeps the first non-overlapping
    one it sees. That means an earlier-starting span wins purely by
    position even if its confidence is much lower than a later-starting
    span it overlaps -- the later, better span never gets a chance,
    because by the time it's considered, its slot is already "taken".
    Confirmed against real data: Presidio's raw analyzer.analyze() output
    had a correct, high-confidence ACCESSION_NUMBER match for every one of
    50 pathology_report documents, but 15 were silently dropped here
    because some unrelated, earlier-starting, lower-confidence entity
    (e.g. a stray NER hit on the preceding "Pathologist:" line bleeding
    into this one) claimed the slot first.

    Fixed by sorting on confidence descending instead: process the
    highest-confidence span first, always let it claim its slot, and only
    let a lower-confidence span in if it doesn't collide with anything
    already kept. This is standard weighted-interval-scheduling-by-score,
    and is what "keep the higher-confidence one" actually requires.
    """
    spans = _clip_at_newlines(spans)
    spans_sorted = sorted(spans, key=lambda s: (-s["confidence"], s["start"]))
    kept: list[PHISpan] = []
    for span in spans_sorted:
        overlaps = False
        for existing in kept:
            if span["start"] < existing["end"] and span["end"] > existing["start"]:
                overlaps = True
                break
        if not overlaps:
            kept.append(span)
    return sorted(kept, key=lambda s: s["start"])


# ---------------------------------------------------------------------------
# Presidio backend
# ---------------------------------------------------------------------------
_analyzer = None


def _get_analyzer():
    global _analyzer
    if _analyzer is None:
        _analyzer = AnalyzerEngine()

        # Clinical-specific custom recognizers on top of Presidio's built-ins
        # (PERSON, DATE_TIME, PHONE_NUMBER, EMAIL_ADDRESS, US_SSN, LOCATION, ...)
        #
        # `context` words boost the match score when they appear near the
        # pattern hit (Presidio's built-in context-enhancement pass) -- this
        # is what lets a bare "12345678"-shaped ID score higher when it's
        # actually next to "Accession" vs. floating in body text, rather
        # than relying on the regex alone to carry all the precision.
        mrn_recognizer = PatternRecognizer(
            supported_entity="MRN",
            patterns=[Pattern(
                name="mrn_pattern",
                regex=r"\b(?:MRN|Medical Record Number)[:\s#]*[A-Z0-9-]{5,15}\b",
                score=0.9,
            )],
            context=["mrn", "medical record"],
        )
        health_plan_recognizer = PatternRecognizer(
            supported_entity="HEALTH_PLAN_ID",
            patterns=[Pattern(
                name="health_plan_pattern",
                regex=r"\b(?:Member|Policy|Plan)\s*(?:ID|#|Number)[:\s#]*[A-Z0-9-]{5,15}\b",
                score=0.85,
            )],
            context=["member", "policy", "plan"],
        )
        # Added to close a real recall gap: these three types previously had
        # NO Presidio recognizer at all (Presidio has no built-in concept of
        # a clinical accession number or insurance claim number), so every
        # single injected span of these types was a guaranteed miss --
        # confirmed via eval/evaluate.py --backend presidio: 150 FN, 100%
        # of ACCESSION_NUMBER/ACCOUNT_NUMBER/CLAIM_NUMBER ground truth.
        # Scores set to 0.9-0.95 (matching/exceeding MRN's 0.9 precedent)
        # since these are deterministic label+value patterns, not
        # probabilistic NER guesses -- they should outrank anything else
        # they conflict with in our own _dedupe_overlaps (see that
        # function's docstring for a real bug this surfaced: it used to
        # let position, not confidence, decide overlap conflicts).
        account_recognizer = PatternRecognizer(
            supported_entity="ACCOUNT_NUMBER",
            patterns=[Pattern(
                name="account_pattern",
                regex=r"\b(?:Account|Acct)\s*(?:Number|No\.?|#)?[:\s#]*[A-Z0-9-]{5,15}\b",
                score=0.9,
            )],
            context=["account", "acct"],
        )
        accession_recognizer = PatternRecognizer(
            supported_entity="ACCESSION_NUMBER",
            patterns=[Pattern(
                name="accession_pattern",
                regex=r"\bAccession\s*(?:Number|No\.?|#)?[:\s#]*[A-Z0-9-]{5,15}\b",
                score=0.95,
            )],
            context=["accession", "specimen"],
        )
        claim_recognizer = PatternRecognizer(
            supported_entity="CLAIM_NUMBER",
            patterns=[Pattern(
                name="claim_pattern",
                regex=r"\bClaim\s*(?:Number|No\.?|#)?[:\s#]*[A-Z0-9-]{5,15}\b",
                score=0.95,
            )],
            context=["claim"],
        )
        # FAX_NUMBER is its own Safe Harbor identifier category (distinct
        # from PHONE_NUMBER) -- not currently present in the synthetic
        # ground truth (no fax fields are injected by
        # scripts/build_sample_dataset.py), so it won't show TP/FP/FN in
        # eval output yet, but real fax numbers in uploaded documents will
        # now actually be caught and redacted instead of silently passing
        # through as plain unlabeled text.
        fax_recognizer = PatternRecognizer(
            supported_entity="FAX_NUMBER",
            patterns=[Pattern(
                name="fax_pattern",
                regex=r"\bFax[:\s#]*(?:\+?1[-.\s]?)?\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}" + _PHONE_EXTENSION + r"\b",
                score=0.9,
            )],
            context=["fax"],
        )
        # Supplements Presidio's built-in PhoneRecognizer (phonenumbers
        # library), which also missed an extension-suffixed number live in
        # testing ("+1-834-576-8701x462" passed through unredacted). Same
        # extension-aware pattern as the fallback backend. Deliberately
        # lower score (0.75, below the built-in recognizer's typical
        # range) so on the common case where Presidio's own recognizer
        # already gets it right, this doesn't fight for the slot --
        # _dedupe_overlaps only needs this one to win when nothing else
        # caught the span at all.
        phone_ext_recognizer = PatternRecognizer(
            supported_entity="PHONE_NUMBER",
            patterns=[Pattern(
                name="phone_with_extension_pattern",
                regex=r"(?:\+?1[-.\s]?)?\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}" + _PHONE_EXTENSION + r"\b",
                score=0.75,
            )],
            context=["phone", "tel", "call", "fax"],
        )
        for recognizer in (
            mrn_recognizer, health_plan_recognizer, account_recognizer,
            accession_recognizer, claim_recognizer, fax_recognizer,
            phone_ext_recognizer,
        ):
            _analyzer.registry.add_recognizer(recognizer)
    return _analyzer


def _detect_presidio(text: str) -> list[PHISpan]:
    analyzer = _get_analyzer()
    results = analyzer.analyze(text=text, language="en")
    spans: list[PHISpan] = []
    for r in results:
        spans.append({
            "start": r.start,
            "end": r.end,
            "text": text[r.start:r.end],
            "phi_type": r.entity_type,
            "confidence": round(float(r.score), 3),
            "source_agent": "presidio",
        })
    return _dedupe_overlaps(spans)


def _filter_rejected(spans: list[PHISpan], rejected_spans: list[dict]) -> list[PHISpan]:
    """Drop spans a human has already explicitly rejected as "not PHI" in an
    earlier round of this same retry loop.

    BUG FIX: found live via the Streamlit UI. ComplianceValidationAgent
    re-scans redacted_text on every retry pass using this same detector.
    Since a rejected span is (correctly) left un-redacted, its text is
    still sitting right there in redacted_text -- so without this filter,
    the very next pass re-detects the identical span, ComplianceValidation
    calls that a FAIL, and the graph loops back to human_review asking the
    *same* question again. A reviewer's "reject, this isn't PHI" decision
    could never actually stick: it either wore the reviewer down into
    clicking Approve, or burned through max_retries and came back FAIL.
    That defeats the purpose of human review as an authoritative call.

    Matched on (phi_type, text) rather than position, since positions
    shift as earlier spans get redacted into [TAG] placeholders on each
    pass, but the rejected span's literal text is stable (it was never
    redacted).
    """
    if not rejected_spans:
        return spans
    rejected_keys = {(r["phi_type"], r["text"]) for r in rejected_spans}
    return [s for s in spans if (s["phi_type"], s["text"]) not in rejected_keys]


# ---------------------------------------------------------------------------
# LangGraph node
# ---------------------------------------------------------------------------
def phi_detection_agent(state: GraphState) -> GraphState:
    text = state.get("redacted_text") or state.get("raw_text", "")
    # NOTE: on a retry loop, ComplianceValidationAgent routes back here with
    # the *current* redacted_text so we re-scan what's left, not the
    # original raw_text. First pass always uses raw_text.
    if state.get("retry_count", 0) == 0:
        text = state.get("raw_text", "")

    backend_used = "regex_fallback"
    if _PRESIDIO_AVAILABLE and not USE_FALLBACK_ONLY:
        try:
            spans = _detect_presidio(text)
            backend_used = "presidio"
        except Exception:
            spans = _detect_fallback(text)
    else:
        spans = _detect_fallback(text)

    spans = _filter_rejected(spans, state.get("rejected_spans", []))

    high = [s for s in spans if s["confidence"] >= _threshold_for(s["phi_type"])]
    low = [s for s in spans if s["confidence"] < _threshold_for(s["phi_type"])]
    confidence_scores = {f"{s['start']}:{s['end']}": s["confidence"] for s in spans}

    audit_log = list(state.get("audit_log", []))
    for s in spans:
        audit_log.append({
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "agent": "PHIDetectionAgent",
            "action": "detected",
            "phi_type": s["phi_type"],
            "span_text": s["text"],
            "confidence": s["confidence"],
            "reviewer_action": None,
            "notes": f"backend={backend_used}",
        })

    return {
        **state,
        "phi_spans": spans,
        "confidence_scores": confidence_scores,
        "high_confidence_spans": high,
        "low_confidence_spans": low,
        "audit_log": audit_log,
    }


def route_after_detection(state: GraphState) -> str:
    """Conditional edge: any low-confidence spans -> human review, else straight to redaction."""
    return "human_review" if state.get("low_confidence_spans") else "redaction"
