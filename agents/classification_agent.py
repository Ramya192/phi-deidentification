"""
ClassificationAgent (Member 2)

Routes an uploaded document into one of 7 clinical document types, or
flags it as not_applicable (not a health document -> graph routes to END
without running PHI detection on it).

Two backends are supported:
  - "heuristic" (default, no API key needed): weighted keyword/regex
    scoring per document type. Deterministic, fast, good enough to unblock
    the rest of the team while the graph is being wired up.
  - "llm": real GPT-4o-mini call via the OpenAI API, with a small
    retrieval step (few-shot examples picked by embedding similarity --
    functionally a vector-store lookup, see _retrieve_examples below) for
    higher accuracy than the bare heuristic. Set PHI_DEID_CLASSIFICATION_BACKEND=llm
    and OPENAI_API_KEY to enable; falls back to the heuristic automatically
    on any failure (missing key, network error, malformed response) so the
    graph never crashes because of this node -- same defensive pattern as
    PHIDetectionAgent's presidio-with-regex-fallback.

Both backends return the same shape: (doc_type, confidence).

PRIVACY NOTE: classification runs BEFORE PHIDetectionAgent/RedactionAgent,
so at this point in the pipeline nothing has been redacted yet. Sending
the raw document straight to a third-party LLM API would ship real PHI
off-machine before this tool has done its one job -- a real problem for a
PHI de-identification project specifically, not a generic LLM-cost
concern. classify_with_llm() therefore runs the text through
_prescrub_for_llm() (the same class of fast deterministic regex patterns
PHIDetectionAgent's fallback backend uses) before it's ever included in a
prompt sent to OpenAI. This is a coarser, faster pass than the full
detection pipeline -- it's not a substitute for RedactionAgent -- but it
meaningfully cuts what leaves the process for a call that only needs to
guess a document *type*, not read every value in it.
"""
from __future__ import annotations

import json
import math
import os
import re
from datetime import datetime, timezone
from typing import Optional

from graph.state import CLINICAL_DOC_TYPES, DocType, GraphState
from observability.llm_metrics import make_usage_entry

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


# ---------------------------------------------------------------------------
# Privacy pre-scrub: strip the highest-signal PHI patterns before any text
# is sent to a third-party LLM API. Deliberately narrower/faster than
# PHIDetectionAgent's full fallback regex set (see that module for the
# complete, tested version) -- this only needs to knock out the most
# obvious identifiers ahead of a classification call, not serve as the
# system's actual redaction guarantee.
# ---------------------------------------------------------------------------
_PRESCRUB_PATTERNS: list[tuple[str, str]] = [
    (r"\b[\w.+-]+@[\w-]+\.[\w.-]+\b", "[EMAIL]"),
    (r"\b\d{3}-\d{2}-\d{4}\b", "[SSN]"),
    (r"(?:\+?1[-.\s]?)?\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}\b", "[PHONE]"),
    (r"\b(?:MRN|Medical Record Number)[:\s#]*[A-Z0-9-]{5,15}\b", "[MRN]"),
    (r"\b(?:Account|Acct|Claim|Accession)\s*(?:Number|No\.?|#)?[:\s#]*[A-Z0-9-]{5,15}\b", "[ID]"),
    (r"\b(?:0?[1-9]|1[0-2])[/-](?:0?[1-9]|[12]\d|3[01])[/-](?:\d{2}|\d{4})\b", "[DATE]"),
    (r"\b(?:Mr|Mrs|Ms|Dr|Pt)\.?\s+[A-Z][a-z]+(?:\s+[A-Z][a-z]+)?\b", "[NAME]"),
]


def _prescrub_for_llm(text: str) -> str:
    """Coarse, fast PHI masking applied before text leaves the process for
    an external LLM call. See module docstring's PRIVACY NOTE."""
    scrubbed = text
    for pattern, tag in _PRESCRUB_PATTERNS:
        scrubbed = re.sub(pattern, tag, scrubbed)
    return scrubbed


# ---------------------------------------------------------------------------
# Few-shot retrieval: a handful of short, hand-written (no real/synthetic
# PHI -- these are examples we authored ourselves, not sampled documents)
# illustrative snippets per document type, embedded once and cached. At
# classification time we embed the (pre-scrubbed) input and pick the
# most-similar few-shot examples by cosine similarity -- this is a small,
# in-memory instance of the same "vector store retrieval" pattern a real
# system would run through FAISS/Chroma/Pinecone: for a static set of ~8
# examples a full vector DB is unnecessary overhead, but the retrieval
# concept (embed -> nearest-neighbor lookup -> inject into prompt) is
# identical. Swap _EXAMPLES for a larger corpus + Chroma/FAISS if this
# ever needs to scale past a few dozen examples.
# ---------------------------------------------------------------------------
_EXAMPLES: list[tuple[DocType, str]] = [
    ("discharge_summary", "DISCHARGE SUMMARY\nAdmission Diagnosis: ...\nHospital Course: patient was treated with...\nDischarge Diagnosis: ...\nDischarge Instructions: follow up with primary care in 2 weeks. Condition on discharge: stable."),
    ("radiology_report", "RADIOLOGY REPORT\nExam: CT Chest with contrast\nFindings: no acute cardiopulmonary process identified...\nImpression: no evidence of pulmonary embolism. Radiologist: [NAME]."),
    ("pathology_report", "PATHOLOGY REPORT\nSpecimen: skin, left forearm, biopsy\nGross Description: received in formalin, a 0.5 cm punch biopsy...\nMicroscopic Examination: epidermis with mild spongiosis...\nDiagnosis: benign, no malignancy identified."),
    ("lab_report", "LABORATORY REPORT\nSpecimen Collected: 08:15\nTest: Complete Blood Count (CBC)\nResult: WBC 6.2, reference range 4.0-11.0, units K/uL. All values within reference range."),
    ("referral_letter", "Dear Doctor,\nI am referring this patient for further evaluation of persistent abdominal pain. Please see and assess at your earliest convenience. Thank you for evaluating this patient."),
    ("insurance_document", "EXPLANATION OF BENEFITS\nClaim Number: [ID]\nMember ID: [ID]\nPrior Authorization required for this service. Copay: $25.00. Deductible remaining: $500.00."),
    ("clinical_note", "Chief Complaint: cough and fever x3 days\nHistory of Present Illness: patient reports...\nVital Signs: T 100.4F, HR 88, BP 120/80\nAssessment and Plan: likely viral URI, supportive care advised."),
    ("not_applicable", "Meeting Agenda\n1. Review Q3 budget\n2. Discuss marketing campaign timeline\n3. Vendor contract renewal\nAction items assigned to team leads for follow-up next week."),
]

_example_embeddings: list[tuple[DocType, str, list[float]]] | None = None


def _cosine_similarity(a: list[float], b: list[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    norm_a = math.sqrt(sum(x * x for x in a))
    norm_b = math.sqrt(sum(y * y for y in b))
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return dot / (norm_a * norm_b)


def _get_example_embeddings(client) -> tuple[list[tuple[DocType, str, list[float]]], Optional[dict]]:
    """Returns (embeddings, usage_entry). usage_entry is None on every call
    after the first in this process -- the few-shot examples are embedded
    once and cached in _example_embeddings, so only the cache-populating
    call actually costs tokens; subsequent classify_with_llm() calls reuse
    the cached vectors for free."""
    global _example_embeddings
    usage_entry = None
    if _example_embeddings is None:
        texts = [snippet for _, snippet in _EXAMPLES]
        resp = client.embeddings.create(model="text-embedding-3-small", input=texts)
        _example_embeddings = [
            (doc_type, snippet, item.embedding)
            for (doc_type, snippet), item in zip(_EXAMPLES, resp.data)
        ]
        usage_entry = make_usage_entry(
            agent="ClassificationAgent",
            model="text-embedding-3-small",
            call_type="embedding",
            prompt_tokens=resp.usage.prompt_tokens,
            notes="one-time few-shot example embedding cache population",
        )
    return _example_embeddings, usage_entry


def _retrieve_examples(client, text: str, k: int = 3) -> tuple[list[tuple[DocType, str]], list[dict]]:
    """Embed `text` and return the k most similar few-shot examples,
    ranked by cosine similarity -- the retrieval step of this node's
    small retrieval-augmented classification prompt. Also returns any
    llm_usage_log entries incurred (0-2: the one-time cache-population
    embedding call if this is the first call in the process, plus the
    per-call query embedding)."""
    usage_entries: list[dict] = []
    examples, cache_usage = _get_example_embeddings(client)
    if cache_usage is not None:
        usage_entries.append(cache_usage)

    query_resp = client.embeddings.create(model="text-embedding-3-small", input=[text[:2000]])
    usage_entries.append(make_usage_entry(
        agent="ClassificationAgent",
        model="text-embedding-3-small",
        call_type="embedding",
        prompt_tokens=query_resp.usage.prompt_tokens,
        notes="query embedding for few-shot example retrieval",
    ))
    query_vec = query_resp.data[0].embedding

    scored = [
        (doc_type, snippet, _cosine_similarity(query_vec, emb))
        for doc_type, snippet, emb in examples
    ]
    scored.sort(key=lambda item: item[2], reverse=True)
    return [(doc_type, snippet) for doc_type, snippet, _score in scored[:k]], usage_entries


_ALL_TYPES = list(CLINICAL_DOC_TYPES) + ["not_applicable"]


def classify_with_llm(text: str) -> tuple[DocType, float, list[dict]]:
    """
    Real LLM-backed classifier: OpenAI gpt-4o-mini, with a small
    retrieval-augmented few-shot prompt (see _retrieve_examples above).
    Raises on any failure -- classification_agent() catches and falls
    back to the heuristic backend, mirroring PHIDetectionAgent's
    presidio-with-regex-fallback pattern.

    Returns (doc_type, confidence, usage_entries) -- usage_entries is a
    list of 2-3 observability/llm_metrics.py entries (embedding call(s)
    plus the chat completion), for classification_agent() to append to
    state["llm_usage_log"]. Populated even when this function is about to
    raise partway through is NOT attempted -- a failed call's partial
    entries are discarded along with everything else, since a caller that
    fell back to the heuristic backend didn't get a usable classification
    from this path and shouldn't be billed-and-reported for it either.
    """
    from openai import OpenAI

    client = OpenAI()  # reads OPENAI_API_KEY from the environment

    scrubbed = _prescrub_for_llm(text)
    examples, usage_entries = _retrieve_examples(client, scrubbed, k=3)
    examples_block = "\n\n".join(
        f"Example ({doc_type}):\n{snippet}" for doc_type, snippet in examples
    )

    prompt = (
        "You are classifying a document into exactly one of these types: "
        f"{', '.join(_ALL_TYPES)}.\n\n"
        "Use 'not_applicable' if the document is not a clinical/health document at all.\n\n"
        f"Reference examples (for calibration only, not the document to classify):\n{examples_block}\n\n"
        f"Document to classify (PHI has already been coarsely masked with tags like [NAME]/[DATE]):\n{scrubbed[:4000]}\n\n"
        'Respond with strict JSON only: {"doc_type": "<one of the types above>", "confidence": <float 0-1>}'
    )

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        temperature=0,
        response_format={"type": "json_object"},
        messages=[
            {"role": "system", "content": "You are a precise clinical document classifier. Always respond with valid JSON only."},
            {"role": "user", "content": prompt},
        ],
    )
    usage_entries.append(make_usage_entry(
        agent="ClassificationAgent",
        model="gpt-4o-mini",
        call_type="chat_completion",
        prompt_tokens=response.usage.prompt_tokens,
        completion_tokens=response.usage.completion_tokens,
    ))

    parsed = json.loads(response.choices[0].message.content)
    doc_type = parsed["doc_type"]
    confidence = float(parsed["confidence"])

    if doc_type not in _ALL_TYPES:
        raise ValueError(f"LLM returned an unrecognized doc_type: {doc_type!r}")
    return doc_type, round(min(max(confidence, 0.0), 1.0), 3), usage_entries


def classification_agent(state: GraphState, backend: str | None = None) -> GraphState:
    """
    LangGraph node. Reads state['raw_text'], writes doc_type,
    doc_type_confidence, and appends an audit_log entry.

    `backend` defaults to the PHI_DEID_CLASSIFICATION_BACKEND env var (and
    that in turn defaults to "heuristic") rather than a hardcoded literal,
    since this node is wired into the graph as a plain callable (see
    graph/workflow.py) with no per-call arguments -- an env var is what
    lets a deployment opt into the LLM backend without touching the graph
    wiring.
    """
    backend = backend or os.environ.get("PHI_DEID_CLASSIFICATION_BACKEND", "heuristic")
    text = state.get("raw_text", "")

    backend_used = "heuristic"
    usage_entries: list[dict] = []
    if backend == "llm":
        try:
            doc_type, confidence, usage_entries = classify_with_llm(text)
            backend_used = "llm"
        except Exception as exc:
            doc_type, confidence = classify_with_heuristic(text)
            backend_used = f"heuristic_fallback (llm error: {exc})"
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
        "notes": f"doc_type={doc_type}, backend={backend_used}",
    })

    llm_usage_log = list(state.get("llm_usage_log", []))
    llm_usage_log.extend(usage_entries)

    return {
        **state,
        "doc_type": doc_type,
        "doc_type_confidence": confidence,
        "audit_log": audit_log,
        "llm_usage_log": llm_usage_log,
    }


def route_after_classification(state: GraphState) -> str:
    """Conditional edge function used by graph/workflow.py."""
    doc_type = state.get("doc_type", "not_applicable")
    return "phi_detection" if doc_type in CLINICAL_DOC_TYPES else "end_not_applicable"
