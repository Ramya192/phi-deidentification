"""
ClassificationAgent (Member 2)

Routes an uploaded document into one of 7 clinical document types, or
flags it as not_applicable (not a health document -> graph routes to END
without running PHI detection on it).

Two backends are supported:
  - "heuristic" (default, no API key needed): weighted keyword/regex
    scoring per document type. Deterministic, fast, good enough to unblock
    the rest of the team while the graph is being wired up.
  - "llm": swap in GPT-4o-mini (or any chat model) for higher accuracy.
    Stubbed via `classify_with_llm` — wire up your own LangChain chat
    model client there when you have API access.

Both backends return the same shape: (doc_type, confidence).
"""
from __future__ import annotations

import re
from datetime import datetime, timezone

from graph.state import CLINICAL_DOC_TYPES, DocType, GraphState

# ---------------------------------------------------------------------------
# Keyword / phrase signals per document type. Order matters only in that
# ties are broken by the order CLINICAL_DOC_TYPES lists them (first wins).
# Weights let a strong, unambiguous phrase (e.g. "discharge summary") beat
# a bunch of weak generic hits.
# ---------------------------------------------------------------------------
_SIGNALS: dict[DocType, list[tuple[str, float]]] = {
    "discharge_summary": [
        (r"\bdischarge summary\b", 5.0),
        (r"\bdischarge diagnosis\b", 3.0),
        (r"\bdischarge instructions\b", 2.5),
        (r"\bhospital course\b", 2.0),
        (r"\bdate of discharge\b", 2.0),
        (r"\bcondition on discharge\b", 1.5),
    ],
    "radiology_report": [
        (r"\bradiology report\b", 5.0),
        (r"\b(x-?ray|ct scan|mri|ultrasound|mammogram|pet scan)\b", 2.5),
        (r"\bimpression\s*:", 2.0),
        (r"\bfindings\s*:", 1.5),
        (r"\bradiologist\b", 1.5),
        (r"\bcontrast\b", 1.0),
    ],
    "pathology_report": [
        (r"\bpathology report\b", 5.0),
        (r"\bspecimen\b", 2.0),
        (r"\bgross description\b", 2.5),
        (r"\bmicroscopic (description|examination)\b", 2.5),
        (r"\bbiopsy\b", 1.5),
        (r"\bhistolog(y|ical)\b", 1.5),
        (r"\bmalignant|benign\b", 1.0),
    ],
    "lab_report": [
        (r"\blab(oratory)? report\b", 5.0),
        (r"\breference range\b", 2.5),
        (r"\b(cbc|complete blood count|metabolic panel|lipid panel)\b", 2.0),
        (r"\bspecimen collected\b", 1.5),
        (r"\bresult(s)?\s*:", 1.0),
        (r"\bunits?\s*:", 0.5),
    ],
    "referral_letter": [
        (r"\breferral letter\b", 5.0),
        (r"\bdear (dr|doctor)\b", 2.5),
        (r"\bi am referring\b", 3.0),
        (r"\bplease (see|evaluate|assess)\b", 1.5),
        (r"\bthank you for (seeing|evaluating)\b", 1.5),
    ],
    "insurance_document": [
        (r"\b(insurance|payer|policy) (claim|number|id)\b", 3.0),
        (r"\bclaim (number|form)\b", 3.0),
        (r"\bprior authorization\b", 2.5),
        (r"\bexplanation of benefits\b", 3.0),
        (r"\bcopay|deductible|coinsurance\b", 2.0),
        (r"\bmember id\b", 1.5),
    ],
    "clinical_note": [
        (r"\b(chief complaint|hpi|history of present illness)\b", 3.0),
        (r"\bsubjective\s*:.*\bobjective\s*:", 2.0),
        (r"\b(assessment and plan|a/p)\s*:", 2.0),
        (r"\bvital signs\b", 1.0),
        (r"\bprogress note\b", 2.5),
    ],
}

# Minimum total weighted score before we trust the classification at all.
# Below this, doc_type still gets set (best guess) but confidence is capped
# low so the UI/human can sanity check it.
_MIN_CONFIDENT_SCORE = 2.0


def _score_document(text: str) -> dict[DocType, float]:
    lowered = text.lower()
    scores: dict[DocType, float] = {}
    for doc_type, patterns in _SIGNALS.items():
        total = 0.0
        for pattern, weight in patterns:
            hits = len(re.findall(pattern, lowered, flags=re.IGNORECASE))
            if hits:
                # diminishing returns for repeated hits of the same phrase
                total += weight + min(hits - 1, 3) * (weight * 0.15)
        scores[doc_type] = total
    return scores


def _looks_like_health_document(text: str, scores: dict[DocType, float]) -> bool:
    if any(score > 0 for score in scores.values()):
        return True
    # fallback generic health-domain terms, in case none of the specific
    # per-type signals fired but this is clearly still clinical content
    generic = re.search(
        r"\b(patient|diagnosis|physician|medication|dosage|symptom|treatment)\b",
        text,
        flags=re.IGNORECASE,
    )
    return bool(generic)


def classify_with_heuristic(text: str) -> tuple[DocType, float]:
    """Deterministic keyword-scoring classifier. No external dependencies."""
    scores = _score_document(text)
    best_type = max(scores, key=lambda k: scores[k]) if scores else "clinical_note"
    best_score = scores.get(best_type, 0.0)

    if not _looks_like_health_document(text, scores):
        return "not_applicable", 0.9

    if best_score <= 0:
        # No specific signal fired but generic health terms are present ->
        # default bucket is clinical_note with modest confidence.
        return "clinical_note", 0.5

    # Normalize a rough confidence: scale the winning score against a
    # ceiling, cap at 0.97 (heuristics should never claim near-certainty).
    confidence = min(0.97, 0.4 + best_score / 10.0)
    if best_score < _MIN_CONFIDENT_SCORE:
        confidence = min(confidence, 0.55)
    return best_type, round(confidence, 3)


def classify_with_llm(text: str) -> tuple[DocType, float]:
    """
    Placeholder for an LLM-backed classifier (e.g. GPT-4o-mini via
    LangChain). Wire this up when API access is available:

        from langchain_openai import ChatOpenAI
        llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
        ...

    Must return (doc_type, confidence) with doc_type being one of the
    values in graph.state.DocType.
    """
    raise NotImplementedError(
        "LLM classification backend not configured. Set backend='heuristic' "
        "or implement classify_with_llm()."
    )


def classification_agent(state: GraphState, backend: str = "heuristic") -> GraphState:
    """
    LangGraph node. Reads state['raw_text'], writes doc_type,
    doc_type_confidence, and appends an audit_log entry.
    """
    text = state.get("raw_text", "")
    if backend == "llm":
        doc_type, confidence = classify_with_llm(text)
    else:
        doc_type, confidence = classify_with_heuristic(text)

    audit_log = list(state.get("audit_log", []))
    audit_log.append({
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "agent": "ClassificationAgent",
        "action": "classified",
        "phi_type": None,
        "span_text": None,
        "confidence": confidence,
        "reviewer_action": None,
        "notes": f"doc_type={doc_type}",
    })

    return {
        **state,
        "doc_type": doc_type,
        "doc_type_confidence": confidence,
        "audit_log": audit_log,
    }


def route_after_classification(state: GraphState) -> str:
    """Conditional edge function used by graph/workflow.py."""
    doc_type = state.get("doc_type", "not_applicable")
    return "phi_detection" if doc_type in CLINICAL_DOC_TYPES else "end_not_applicable"
