---
title: PHI De-identification & Compliance Workflow
emoji: 🏥
colorFrom: blue
colorTo: green
sdk: docker
app_port: 7860
pinned: false
---

# PHI De-identification & Compliance Workflow

LangGraph multi-agent pipeline that classifies uploaded health documents,
detects PHI, redacts it, validates the redaction, and produces an audit
trail + compliance report.

**GitHub repository:** `[FILL IN — paste the repo URL after pushing]`
**Live demo (Hugging Face Space):** `[FILL IN — paste the Space URL after it finishes building]`

The live demo runs the exact same FastAPI + Streamlit architecture
described below in a single Hugging Face Space container — see
`Dockerfile` and `start.sh`. No account or API key entry is required to
try it: the API key is pre-configured as a Space secret and the Streamlit
sidebar picks it up automatically.

## Architecture

```
Upload (.txt / .pdf / .docx)
  -> Document ingestion         (ingestion/document_loader.py — extracts
                                  raw text; NOT a graph node, see below)
  -> ClassificationAgent          (routes into 7 document types, or "not
                                    applicable" -> END)
  -> PHIDetectionAgent             (Presidio + custom clinical recognizers;
                                    regex fallback if Presidio/spaCy aren't
                                    installed)
  -> [confidence >= per-type threshold]  -> RedactionAgent
  -> [confidence <  per-type threshold]  -> HumanReviewAgent (LangGraph interrupt()) -> RedactionAgent
  -> ComplianceValidationAgent     (re-scans redacted text)
  -> [PASS]                -> AuditReportAgent -> END
  -> [FAIL, retries left]  -> loop back to PHIDetectionAgent (max 2 retries)
  -> [FAIL, retries exhausted] -> AuditReportAgent -> END (flagged for manual follow-up)
```

Both confidence branches converge at `RedactionAgent` before validation —
human review decides *whether* a low-confidence span is real PHI;
redaction is what actually masks it. This is deliberate: skipping
redaction after human review would leave approved-as-PHI spans
un-redacted going into compliance validation.

Document ingestion (file -> raw text) happens **before** the graph, not
as a graph node. `GraphState` (`graph/state.py`) is checkpointed (SQLite
by default, see `graph/workflow.py`) to support `HumanReviewAgent`'s
`interrupt()`/resume flow, and keeping that state a plain
JSON-serializable string (not binary file content) keeps checkpointing
simple. See "File upload / ingestion" below.

### Document types

`ClassificationAgent` routes into one of 7 clinical document types, or
`not_applicable` for anything that isn't a health document:

- clinical_note
- discharge_summary
- radiology_report
- pathology_report
- lab_report
- referral_letter
- insurance_document

Routing uses weighted keyword/phrase heuristics (`classify_with_heuristic`
in `agents/classification_agent.py`) so it runs with zero API keys. Swap
in `classify_with_llm` (stubbed, ready for GPT-4o-mini via LangChain) for
higher accuracy once you have API access.

## Project layout

```
ingestion/
  document_loader.py  Extracts text from .txt/.pdf/.docx before the graph runs
graph/
  state.py        GraphState TypedDict — shared state every node reads/writes
  workflow.py      StateGraph wiring: nodes, conditional edges, retry loop,
                   checkpointer, run_sync()/resume_sync() helpers
agents/
  classification_agent.py
  phi_detection_agent.py
  redaction_agent.py
  human_review_agent.py
  compliance_validation_agent.py
  audit_report_agent.py
api/
  main.py          FastAPI: POST /redact, POST /redact/upload,
                    POST /redact/resume, GET /health
app/
  streamlit_app.py  Streamlit UI — HTTP client of api/main.py (see
                     "Run the Streamlit UI")
scripts/
  build_sample_dataset.py  Generates the labeled dataset (see below)
  generate_demo_files.py   Converts a few samples to .pdf/.docx for manual
                            upload testing (see "Demo files")
eval/
  evaluate.py      Scores PHIDetectionAgent against ground_truth.json —
                    precision/recall/F1, overall and per PHI type
data/
  raw/mtsamples.csv   Kaggle "Medical Transcriptions" source (you provide)
  txt_format/         Labeled sample dataset, .txt + ground_truth.json
                       (see "Sample data generation")
  pdf_format/          .pdf versions of a few samples (see "Demo files")
  docx_format/         .docx versions of a few samples (see "Demo files")
tests/
  test_integration.py
  test_document_ingestion.py
```

## File upload / ingestion

`ingestion/document_loader.py` extracts text from an uploaded file before
it reaches `ClassificationAgent`. Supported formats:

- `.txt` — read directly (tries utf-8, utf-8-sig, then latin-1)
- `.pdf` — text layer extracted via `pdfplumber`, page by page
- `.docx` — paragraphs and table cells extracted via `python-docx`

Not supported: legacy `.doc` (convert to `.docx` or `.pdf` first), and
scanned/image-only PDFs with no embedded text layer — those need OCR.
`extract_pdf_with_ocr()` in the same module is an opt-in fallback
(`pytesseract` + `pdf2image`, plus the `tesseract-ocr` and `poppler`
system binaries — not pip-installable, see `requirements.txt` comments
for platform-specific install commands). It's not wired into the default
path automatically, since OCR is slow and shouldn't silently fire on
every PDF — call it explicitly when you know the source is a scan.

Extraction happens entirely in memory (`io.BytesIO`) — uploaded files are
never written to disk, which matters for a PHI pipeline. Unsupported file
types and empty/unreadable documents raise clear, typed exceptions
(`UnsupportedFileTypeError`, `EmptyDocumentError`) that `api/main.py`
turns into proper HTTP 415/422 responses rather than a generic 500.

### Demo files

`scripts/generate_demo_files.py` converts a few samples per doc type from
`data/txt_format/` into both `.pdf` and `.docx`, written to
`data/pdf_format/` and `data/docx_format/` respectively, purely so you can
manually exercise `POST /redact/upload` (or the Streamlit UI's file
upload mode — see "Run the Streamlit UI") against real files instead of
only `.txt`.

```bash
pip install fpdf2 python-docx
python -m scripts.generate_demo_files          # 1 per type (default)
python -m scripts.generate_demo_files --per-type 2
```

These are deliberately **not** added to `ground_truth.json` and never
touched by `eval/evaluate.py` — round-tripping through PDF/DOCX and back
introduces small whitespace/line-break differences from the source
`.txt`, which would silently invalidate the exact character-offset
ground truth. They exist only to test the ingestion *code path*, not to
score detection accuracy.

## Sample data generation

`n2c2 2014` (the benchmark named in the project overview) is currently
listed as **"Temporarily Unavailable"** on the DBMI Data Portal —
registration is closed, not just gated behind a DUA. Rather than block
on that, `scripts/build_sample_dataset.py` builds a labeled test set from
two sources instead:

1. **Kaggle "Medical Transcriptions"** (`tboyle10/medicaltranscriptions`,
   CC0) for realistic document structure — download the CSV manually
   (no Kaggle API token needed) from
   https://www.kaggle.com/datasets/tboyle10/medicaltranscriptions and
   place it at `data/raw/mtsamples.csv`.
2. A **synthetic PHI injector** (Faker) that prepends a labeled
   identifying-info header (name, DOB, MRN, phone, dates, provider
   names, etc.) to each document and records the exact character span
   of every value it inserted.

`insurance_document` has no mtsamples equivalent, so those are generated
fully synthetically. If `data/raw/mtsamples.csv` isn't present yet, the
script still runs using a small set of built-in fallback templates per
doc type, so you always have *some* labeled data.

All generated `.txt` samples live in `data/txt_format/`, alongside
`data/pdf_format/` and `data/docx_format/` (the demo files, see below) —
the three folders group every sample document by file format rather than
by dataset version. If you regenerate later with different parameters and
want to keep the current batch around for comparison, pass
`--output-dir data/txt_format_v2` (or similar) instead of overwriting it.

```bash
pip install faker
python -m scripts.build_sample_dataset --per-type 50
```

`pathology_report` and `lab_report` only have 7 and 1 real mtsamples rows
respectively (mtsamples' "Lab Medicine - Pathology" specialty is thin
overall), so at `--per-type 50` most of those two categories come from
the built-in fallback templates rather than real transcription text —
each has 3 template variants to avoid excessive repetition, but they're
still less varied than the mtsamples-backed types. Everything else
(clinical_note, discharge_summary, radiology_report, referral_letter)
has ample real source rows.

This writes documents to `data/txt_format/` and a
`data/txt_format/ground_truth.json` manifest (filename ->
doc_type + exact PHI spans). Current batch: **354 documents, 50 per type
(51 for clinical_note/discharge_summary/insurance_document, which also
have one original hand-written fixture each), 2,150+ labeled PHI spans.**

## Evaluation

`eval/evaluate.py` runs `PHIDetectionAgent` against
`data/txt_format/ground_truth.json` and reports real
precision/recall/F1 — the actual number your n2c2 benchmark comparison
needs, not a self-reported one.

```bash
python -m eval.evaluate                    # uses whatever backend is installed
python -m eval.evaluate --backend fallback # force the regex fallback
python -m eval.evaluate --backend presidio # force Presidio (errors if unavailable)
```

Writes `eval/final_results.csv` and `eval/final_results.json` (overall +
per-PHI-type precision/recall/F1), and prints the same to console.

Two matching strategies are reported, per standard i2b2/n2c2 practice:
**STRICT** requires an exact `(start, end, phi_type)` match; **OVERLAP**
counts any character overlap with the same type, matched greedily so a
span can't be double-counted. OVERLAP exists because our injected ground
truth spans cover just the value (e.g. `MR-969634`), while a detector
might reasonably flag the label too (`MRN: MR-969634`) — that's a
boundary convention difference, not a genuine miss, and STRICT alone
would understate recall for exactly that reason.

I ran this for real against all 350 generated documents with
`--backend fallback` (Presidio isn't installed in my environment) and
verified the numbers make sense against the code:

| PHI type | Strict F1 | Overlap F1 | Why |
|---|---|---|---|
| DATE_TIME | 0.86 | 0.86 | Solid on both — no label-boundary issue for dates |
| HEALTH_PLAN_ID | 0.00 | 1.00 | Detector includes the label in the span; pure boundary difference |
| MRN | 0.00 | 1.00 | Same — fallback pattern matches `MRN: MR-xxxxxx` as one span |
| ACCESSION_NUMBER, ACCOUNT_NUMBER, CLAIM_NUMBER | 0.00 | 1.00 | Same boundary story — **fixed**; these three were 0.00/0.00 (no fallback pattern existed at all) until the label-optional regex fix in `agents/phi_detection_agent.py` (see Production roadmap) |
| PHONE_NUMBER | 0.52 | 0.56 | Faker sometimes generates formats (e.g. extensions) the regex misses |
| PERSON | 0.01 | 0.02 | **Expected, not a bug** — the fallback's PERSON pattern requires a title (`Mr./Dr./Pt.`); injected names like "Patient Name: Jordan Ellis" have none |

OVERLAP OVERALL (fallback backend) went from **F1 0.6329 → 0.6960**
(recall 0.5377 → 0.6074) after that fix — a real, measured improvement:
1,306 TP / 297 FP / 844 FN overall (see `eval/final_results.csv` for the
full per-type breakdown).

The remaining PERSON gap is a real fallback-detector coverage limitation,
not an eval bug — this is exactly why Presidio/spaCy is the production
path, not the fallback.

**Real `--backend presidio` run** (project owner's environment, all 350
docs, OVERLAP mode): OVERALL went from **F1 0.6023 → 0.6440** (recall
0.9023 → 0.9651, FN 210 → 75) after adding the ACCESSION_NUMBER/
ACCOUNT_NUMBER/CLAIM_NUMBER recognizers — HEALTH_PLAN_ID, MRN,
ACCOUNT_NUMBER, and CLAIM_NUMBER are all now at ~100% precision/recall,
and PERSON recall jumped to 0.99 (vs. 0.01 on the fallback), confirming
the whole reason for having Presidio as the production backend.

One type initially came back partial: ACCESSION_NUMBER landed at 70%
recall (35/50) despite 100% precision, even though the identical regex
hit 100% under the fallback. Diagnosed with a throwaway script that
called Presidio's raw `analyzer.analyze()` directly and compared against
ground truth: **Presidio's own output was correct for all 50** — the
miss was a real bug in our own `_dedupe_overlaps()` post-processing (in
`phi_detection_agent.py`), not in Presidio. It sorted candidate spans by
`(start, -confidence)` and greedily kept the first non-overlapping one
per position — which means an earlier-starting span could permanently
block a later-starting, much-higher-confidence span from ever being
kept, regardless of how much better it was. In practice: some unrelated,
earlier-starting, lower-confidence entity (plausibly a stray NER hit
bleeding over from the preceding "Pathologist:" line) was winning by
position alone. Fixed by sorting on confidence descending instead — the
highest-confidence candidate always gets first pick now.

**Confirmed fixed** — re-ran `--backend presidio` after the dedupe fix:
ACCESSION_NUMBER is now **50/50, 100% precision and recall.**
ACCOUNT_NUMBER stayed 100/100, CLAIM_NUMBER 50/50 (1 FP). OVERALL landed
at **F1 0.6433** (P 0.4833, R 0.9619, 2068 TP / 2211 FP / 82 FN) —
essentially flat vs. the pre-dedupe-fix run (F1 0.6440), which makes
sense: the fix doesn't add new detections, it just arbitrates existing
overlap conflicts correctly, so a handful of PERSON/PHONE_NUMBER spans
that previously won by lucky position-ordering now correctly lose to a
higher-confidence competing span elsewhere (PERSON recall 0.99 → 0.97,
PHONE_NUMBER 0.79 → 0.76) — a small, expected, and *more correct*
trade-off, not a regression. Verified the fallback backend separately
had zero change from this fix (still F1 0.6960 exactly), since the
fallback path never happened to hit this particular conflict.

**Confirmed after the phone-extension fix below** — re-ran
`--backend presidio` once more: PHONE_NUMBER landed at **0.9960
precision / 0.9960 recall** (249 TP / 1 FP / 1 FN), up from leaking
extension-suffixed numbers entirely unredacted. OVERALL ticked up to
**F1 0.6602** (P 0.4937, R 0.9963). The still-low overall precision is
the same labeled-taxonomy caveat below, not a regression — DATE_TIME,
LOCATION, NRP, and US_DRIVER_LICENSE false positives are Presidio
correctly flagging real entities in the borrowed mtsamples body text
that `ground_truth.json` was never told to label.

One more honest caveat: some fallback "false positives" for DATE_TIME —
and a good chunk of the LOCATION/NRP/US_DRIVER_LICENSE/etc. false
positives you'll see under `--backend presidio` — are likely real
entities present in the borrowed mtsamples body text (not part of the
injected header) that `ground_truth.json` never labeled at all, since
the synthetic ground truth only covers the fields we deliberately
injected. So true precision is probably understated for types Presidio
detects outside that labeled taxonomy — not inflated. Restricting
Presidio's `entities=[...]` list to just the labeled taxonomy would give
a cleaner apples-to-apples precision number if you want one for the
write-up (see Production roadmap).

## Setup

```bash
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
python -m spacy download en_core_web_lg   # or en_core_web_sm if that's too slow/large
```

**Note:** every agent that needs Presidio/spaCy degrades gracefully to a
dependency-free regex detector if those packages (or the spaCy model)
aren't available — see `_PRESIDIO_AVAILABLE` in
`agents/phi_detection_agent.py`. This means the graph is runnable and
testable immediately after `pip install langgraph langchain-core fastapi
uvicorn pydantic pytest`, even before Presidio/spaCy finish installing.
The regex fallback is meant for dev/offline use — swap
`USE_FALLBACK_ONLY = True` in `phi_detection_agent.py` to force it during
local testing, and don't rely on it for the n2c2 evaluation numbers
(Presidio + a real NLP backend is what should be benchmarked).

## Run tests

```bash
pytest -v
```

**Verification status:** `pytest -v` has been run for real, twice, in the
project owner's own environment (Python 3.13, real langgraph/presidio/
spaCy/fastapi/pdfplumber/python-docx installed) — **16/16 passing**,
including the full compiled graph, the `interrupt()`/resume flow, the
retry loop, and real `.pdf`/`.docx` round-trip extraction. Two tests
failed on the first real run because they assumed the deterministic
regex fallback but the real environment had Presidio installed and
active; fixed by pinning those two tests to the fallback explicitly (see
`force_fallback_detector` fixture in `tests/test_integration.py`) since
they're testing graph control flow, not backend-specific PHI-detection
accuracy. That fix also caught a real bug in
`compliance_validation_agent.py`, which imported `USE_FALLBACK_ONLY` by
name (copying the value at import time) instead of referencing the
`phi_detection_agent` module live — meaning a runtime flag change would
never have reached the compliance re-scan step. Both are fixed and
re-verified.

`eval/evaluate.py` and `scripts/generate_demo_files.py` have also been
run for real against the actual generated dataset (the latter caught and
fixed a real `fpdf2` API bug — `multi_cell` doesn't reset cursor position
by default, so a second call raises `FPDFException`; fixed with explicit
`new_x=XPos.LMARGIN, new_y=YPos.NEXT`).

## Run the API

Every endpoint except `/health` requires an `X-API-Key` header — set
`PHI_DEID_API_KEY` yourself for a stable key, or let the server generate
and print a random one for that process only (fine for a quick local
test, annoying if you restart often, not usable for anyone else to call).

```bash
export PHI_DEID_API_KEY=dev-local-key
uvicorn api.main:app --reload --port 8000
```

Submit raw text directly:

```bash
curl -X POST http://localhost:8000/redact \
  -H "Content-Type: application/json" \
  -H "X-API-Key: dev-local-key" \
  -d '{"text": "Patient Mr. John Carter, MRN: MR-88213, seen 07/10/2026.", "filename": "note.txt"}'
```

Or upload a file (.txt, .pdf, or .docx — try one of the generated demo files):

```bash
curl -X POST http://localhost:8000/redact/upload \
  -H "X-API-Key: dev-local-key" \
  -F "file=@data/pdf_format/clinical_note_001.pdf"
```

Both endpoints return the same response shape. If it comes back
`"status": "human_review_required"`, collect approve/reject decisions for
each span in `review_payload.spans` and post them to `/redact/resume`
(also requires `X-API-Key`) with the same `thread_id`.

## Run the Streamlit UI

`app/streamlit_app.py` is a thin HTTP client of `api/main.py` — it talks
to the API over REST using the `requests` library, the same as the curl
examples above, rather than importing `graph`/`agents` directly. That's
deliberate: the API is the single enforcement point for auth, logging,
and error handling, and a real deployment might have other clients
(mobile app, batch jobs, another team's service) hitting the same
endpoints — the UI shouldn't be a second, unaudited path into the
pipeline. See "Production roadmap" if you want the fuller reasoning.

Run both processes (API first, then UI, in separate terminals):

```bash
export PHI_DEID_API_KEY=dev-local-key
uvicorn api.main:app --reload --port 8000
```

```bash
streamlit run app/streamlit_app.py
```

Streamlit opens at `http://localhost:8501`. Paste the same
`PHI_DEID_API_KEY` value into the "API key" field in the sidebar (it's
never hardcoded into the UI itself) — until you do, every request gets a
401. From there: paste text or upload a `.txt`/`.pdf`/`.docx` file, run
de-identification, and the UI walks you through the same states the API
returns: a human-review approve/reject form if `PHIDetectionAgent`
flagged low-confidence spans, otherwise straight to the redacted text,
compliance report, and full audit log.

## Production roadmap

This started as a list of known simplifications; the items below marked
**Implemented** were closed out in a later hardening pass (see git
history / conversation log), the rest are documented as real gaps rather
than fixed, given the capstone deadline — they'd be the next things to
build for an actual production deployment, not things that got
overlooked.

### Implemented

- **Per-entity-type confidence thresholds** (`CONFIDENCE_THRESHOLDS` in
  `phi_detection_agent.py`) replaced the single flat `0.7`. Regex-backed
  types (MRN, HEALTH_PLAN_ID, ACCOUNT_NUMBER, ACCESSION_NUMBER,
  CLAIM_NUMBER, EMAIL_ADDRESS, SSN, IP_ADDRESS, URL, FAX_NUMBER) auto-redact
  at a lower bar (~0.6) since they're near-deterministic once matched;
  NER-driven types (PERSON 0.85, DATE_TIME 0.8) require a higher bar
  since their confidence scores don't cleanly separate true/false
  positives — see the comment block above `CONFIDENCE_THRESHOLDS` for the
  eval numbers behind each choice.
- **ACCESSION_NUMBER / ACCOUNT_NUMBER / CLAIM_NUMBER recall gap closed.**
  These were at 0% recall on *both* the fallback (a regex bug — the
  pattern required the label to be followed immediately by punctuation,
  so `"Account Number: ..."` never matched) and Presidio (no recognizer
  existed for them at all — Presidio has no built-in concept of a
  clinical accession number or insurance claim number). Fixed in both
  places; also added a `FAX_NUMBER` type (regex + Presidio recognizer)
  for Safe Harbor completeness, though no fax fields are currently
  injected into the synthetic dataset so it won't show up in eval numbers
  yet.
- **Persistent checkpointer.** `graph/workflow.py` now defaults to a
  SQLite-backed checkpointer (`data/checkpoints.sqlite`) instead of
  in-memory `MemorySaver`, so an in-flight human-review session survives
  a process restart. Falls back to `MemorySaver` automatically if
  `langgraph-checkpoint-sqlite` isn't installed.
- **API key auth.** Every PHI-handling endpoint (`/redact`,
  `/redact/upload`, `/redact/resume`) now requires an `X-API-Key` header,
  checked in constant time via `secrets.compare_digest`. Previously wide
  open — anyone who could reach the port could submit arbitrary PHI and
  get results back.
- **PHI-safe logging.** Unhandled exceptions used to be returned to the
  caller as `detail=str(exc)`, which could echo a fragment of the
  document being processed back through the HTTP response. Now logged
  server-side via `logger.exception(...)` (access-controlled logs only)
  and the caller gets a generic message plus a `thread_id` correlation
  ID. Request-level logging (`doc_type`, span counts, validation status)
  never includes `raw_text`/`redacted_text` content.

- **Three bugs found via live Streamlit UI testing** (real uploaded
  `.docx`/`.pdf` files, not synthetic eval data — the kind of thing an
  offline eval run against clean ground truth can't surface):
  - **Phone number with an extension leaked completely unredacted**
    (`+1-834-576-8701x462` passed straight through as plain text). Root
    cause: both the fallback regex and Presidio's built-in phone
    recognizer end in a `\b` word boundary, which doesn't exist between
    a digit and a directly-adjacent letter — so the whole match failed
    rather than just stopping short of the extension. Fixed with a
    shared `_PHONE_EXTENSION` regex suffix (`agents/phi_detection_agent.py`)
    used in the fallback PHONE_NUMBER/FAX_NUMBER patterns and a new
    supplementary Presidio `phone_ext_recognizer`. Confirmed fixed live
    (redacts to `[PHONE_NUMBER]`) and via eval (PHONE_NUMBER 0.996
    precision/recall, see Evaluation above).
  - **A PERSON match extended across a line break**, swallowing the next
    line's field label and the newline itself (`"Patient Name: James
    Simmons MD\r\nDOB:"` collapsed into a single corrupted line instead
    of two clean ones). Fixed with `_clip_at_newlines()`, called at the
    start of `_dedupe_overlaps()`: any span whose matched text contains
    `\r`/`\n` is truncated to end right before the line break, since no
    legitimate single PHI value in these structured documents should
    ever span two lines. Confirmed fixed live — header fields now render
    as clean, separate lines.
  - **A human reviewer's "reject, not PHI" decision couldn't actually
    stick.** `ComplianceValidationAgent` re-scans `redacted_text` on
    every retry pass using the same detector; since a rejected span is
    correctly left un-redacted, its text was still sitting right there
    to be re-detected, which counted as a validation FAIL and looped
    back to human review asking the *identical* question again — a
    reject could only ever end in the reviewer eventually caving to
    Approve, or in an exhausted-retries FAIL. Fixed by adding
    `rejected_spans` to `GraphState`, accumulated across retries in
    `human_review_agent.py`, and filtered out of detection results in
    both `phi_detection_agent.py` (`_filter_rejected`) and
    `compliance_validation_agent.py` (defense-in-depth, since it
    re-scans independently). Confirmed fixed live: rejecting a span now
    reaches `PASS` on `retry_count=0`, no repeated review round.

### Documented, not built (deadline trade-off)

- **Clinical-domain NER fine-tuning.** Presidio's default `en_core_web_lg`
  spaCy model is general-purpose; a scispaCy model (`en_core_sci_lg`) or
  a fine-tuned clinical NER model would likely improve PERSON/LOCATION
  accuracy meaningfully on real clinical text. This needs a model
  download, an eval cycle to confirm it's actually better, and probably
  a held-out validation set beyond what we have — real work, not a
  config flip.
- **Confidence calibration.** The per-type thresholds above are
  hand-picked from one eval run's precision numbers, not statistically
  calibrated (e.g. Platt scaling / isotonic regression against a
  held-out set). Fine for a capstone demo; not rigorous enough to defend
  as "the threshold is correct" in a real deployment.
- **Precision restricted to the labeled taxonomy for eval.** Presidio
  detects entity types (LOCATION, NRP, US_BANK_NUMBER,
  US_DRIVER_LICENSE, UK_NHS) that the synthetic ground truth never
  labels at all, which inflates the measured false-positive count
  without those detections necessarily being wrong. Passing an explicit
  `entities=[...]` list to `analyzer.analyze()` matching just the
  labeled taxonomy would give a cleaner precision number for reporting —
  not done here since it would also *suppress* real Safe-Harbor-relevant
  detections (a real production redactor arguably *should* catch
  LOCATION) rather than fix a measurement problem.
- **Full HIPAA Safe Harbor coverage for rare identifier types.** Vehicle
  identifiers and device/serial numbers have no recognizer at all
  (neither regex nor Presidio); certificate/license numbers rely on
  Presidio's generic `US_DRIVER_LICENSE`, unvalidated against this
  dataset since none are injected. Low incidence in outpatient clinical
  notes, but a real gap if this ever processes documents where they
  appear (e.g. workers' comp, DMV-adjacent records).
- **Monitoring / drift detection.** No periodic re-evaluation against a
  held-out labeled set, no alerting if precision/recall degrades (e.g.
  after a spaCy model update). `eval/evaluate.py` is a manual, on-demand
  tool right now, not a scheduled job.
- **Rate limiting** on the API — none. A single caller could currently
  submit unlimited requests.
- **Single-process deployment ceiling.** The SQLite checkpointer doesn't
  scale past one process (single-writer). A multi-replica deployment
  needs a Postgres or Redis checkpointer instead — same `_build_checkpointer()`
  seam in `graph/workflow.py`, different backend.
- **Ground truth doesn't cover PHI in narrative body text**, only the
  synthetic header fields we deliberately inject — real dates, names, or
  locations that happen to appear in the borrowed mtsamples transcription
  body are neither labeled nor scored, which is why some measured
  "false positives" (especially DATE_TIME, LOCATION) are likely real,
  correct detections the eval methodology just can't see.
- **LLM classification backend** (`classify_with_llm`) is still stubbed —
  wire up an LLM call if the heuristic classifier isn't accurate enough
  on real (non-synthetic) documents.
- **Scanned PDFs** need `extract_pdf_with_ocr()` called explicitly — not
  wired into the default `/redact/upload` path, since OCR is slow and
  shouldn't silently fire on every PDF.
- **Deployment infra** (Docker, secrets management for `PHI_DEID_API_KEY`,
  HTTPS termination, HuggingFace Spaces or equivalent config) — not set
  up. `PHI_DEID_API_KEY` currently must be set as a plain environment
  variable by whoever runs the process.
