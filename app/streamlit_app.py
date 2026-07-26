"""
app/streamlit_app.py (Member 7)

Streamlit UI -- a thin HTTP client of api/main.py. Talks to the FastAPI
server over REST (it never imports graph/ or agents/ directly), matching
the documented UI -> API -> LangGraph architecture and exercising the
same auth, logging, and error handling the API enforces for every other
caller. See README's "Run the Streamlit UI" section.

Run (two processes, in order):
    export PHI_DEID_API_KEY=dev-local-key
    uvicorn api.main:app --reload --port 8000
    streamlit run app/streamlit_app.py

Single-process deployment (e.g. Streamlit Community Cloud, which only runs
one entrypoint): set PHI_DEID_EMBED_API=1 as well, and this file starts the
FastAPI backend itself in a background thread -- see _start_embedded_api()
below. Also set PHI_DEID_SPACY_MODEL=en_core_web_sm in that environment;
the platform's 1GB RAM cap is too tight for en_core_web_lg (the model every
number in the report was produced with) running alongside two web servers
in one process -- see agents/phi_detection_agent.py._get_analyzer().

This UI submits to the API's /stream endpoints (Server-Sent Events), not
the plain POST ones -- see _run_streaming() below -- so the "pipeline
progress" stepper shown while a document is processing reflects the
actual LangGraph node execution in real time (graph/workflow.py's
run_stream()/resume_stream()), not a simulated/timed animation.
"""
from __future__ import annotations

import csv
import json
import os
import pathlib
import socket
import sys
import threading
import time

import requests
import streamlit as st

try:
    # Same local-dev convenience as api/main.py -- loads .env into
    # os.environ if present. No-op if python-dotenv isn't installed or
    # there's no .env file (e.g. on Streamlit Cloud, which uses its own
    # secrets mechanism instead).
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

st.set_page_config(page_title="PHI De-identification", page_icon=":material/health_and_safety:", layout="wide")

PROJECT_ROOT = pathlib.Path(__file__).resolve().parent.parent
EVAL_DIR = PROJECT_ROOT / "eval"

# Streamlit Community Cloud (and `streamlit run` in general) sets sys.path[0]
# to this script's own directory (app/), not the repo root -- so top-level
# packages like api/, agents/, graph/, storage/ are NOT importable by
# default. This only bites _start_embedded_api() below (the single-process
# deployment path), which is exactly why it went uncaught locally: local dev
# always runs the two-process mode (`uvicorn api.main:app` as a separate
# process), so `from api.main import app` was never actually exercised
# in-process until the first real Streamlit Cloud deploy surfaced
# "ModuleNotFoundError: No module named 'api'".
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# One pre-generated sample per supported upload format -- lets an evaluator
# exercise the full .txt/.pdf/.docx ingestion path (ingestion/) without
# needing to source or write their own test document. Same underlying
# document (clinical_note_001) in all three formats, so the only variable
# being demonstrated is the file-format handling itself.
#
# discharge_summary_01 is included separately as the "ambiguous" sample:
# its header PHI (Patient/Date of Admission/Date of Discharge/Account
# Number) is clean, structured, high-confidence text like clinical_note_001,
# but the body's "Internal case reference: 048261953" line was added
# specifically to force a low-confidence hit. An earlier version of this
# file relied on "Contact: (555) 812-3345" for this instead -- real testing
# showed that DIDN'T trigger review after all (this project's own
# phone_ext_recognizer, added for a different bug, scores any 10-digit
# phone-shaped match at 0.75 regardless of context, well above the 0.65
# threshold). The bare 9-digit reference number is a better fit: it's too
# short to match any 10-digit phone pattern (built-in or custom), but
# Presidio's *built-in* PhoneRecognizer (`phonenumbers`-library-backed)
# still catches it at a flat, unboosted 0.4 -- confirmed directly against
# the installed presidio_analyzer package, not assumed:
#   PhoneRecognizer().analyze(text="...048261953...", entities=["PHONE_NUMBER"], nlp_artifacts=None)
#   -> [type: PHONE_NUMBER, start: 26, end: 35, score: 0.4]
# 0.4 < PHONE_NUMBER's 0.65 threshold (CONFIDENCE_THRESHOLDS in
# agents/phi_detection_agent.py), so it should route to HumanReviewAgent.
# See tests/test_ambiguous_identifier_routing.py for the routing-logic
# regression test. Still worth confirming once against the real
# Presidio+spaCy backend (this sandbox has no working spaCy model, so the
# NER-driven parts of the pipeline can't be exercised end-to-end here).
SAMPLE_DOCS = {
    "Clinical note -- plain text (.txt)": PROJECT_ROOT / "data" / "txt_format" / "clinical_note_001.txt",
    "Clinical note -- Word (.docx)": PROJECT_ROOT / "data" / "docx_format" / "clinical_note_001.docx",
    "Clinical note -- PDF (.pdf)": PROJECT_ROOT / "data" / "pdf_format" / "clinical_note_001.pdf",
    "Discharge summary -- ambiguous identifier, likely triggers human review (.txt)": PROJECT_ROOT / "data" / "txt_format" / "discharge_summary_01.txt",
    "Discharge summary -- ambiguous identifier, likely triggers human review (.docx)": PROJECT_ROOT / "data" / "docx_format" / "discharge_summary_01.docx",
    "Discharge summary -- ambiguous identifier, likely triggers human review (.pdf)": PROJECT_ROOT / "data" / "pdf_format" / "discharge_summary_01.pdf",
}


class _SampleFile:
    """Minimal stand-in for Streamlit's UploadedFile (just .name and
    .getvalue()) so a pre-loaded sample can go through _submit_file() --
    and therefore POST /redact/upload/stream -- exactly like a real
    upload, with no separate code path to keep in sync."""

    def __init__(self, path: pathlib.Path) -> None:
        self.name = path.name
        self._data = path.read_bytes()

    def getvalue(self) -> bytes:
        return self._data

# ---------------------------------------------------------------------------
# Visual theme -- a plain, unstyled Streamlit form is fine for a personal
# script but reads as a prototype, not a compliance tool a lecturer/reviewer
# would trust with clinical documents. This is intentionally restrained
# (no illustrations, no marketing copy) -- a healthcare/compliance UI
# should look calm and precise, not flashy. Palette matches the report/deck
# (NAVY/TEAL) for a consistent "this is one project" feel across
# deliverables.
# ---------------------------------------------------------------------------
st.markdown(
    """
    <style>
    :root {
        --phi-navy: #1F4E79;
        --phi-navy-dark: #163A5C;
        --phi-teal: #0F6E56;
        --phi-bg: #F6F8FA;
        --phi-border: #E2E6EA;
    }
    .stApp { background-color: var(--phi-bg); }
    .phi-banner {
        background: linear-gradient(135deg, var(--phi-navy) 0%, var(--phi-navy-dark) 100%);
        color: white; padding: 1.6rem 2rem; border-radius: 12px; margin-bottom: 1.4rem;
    }
    .phi-banner h1 { color: white; margin: 0; font-size: 1.5rem; font-weight: 600; }
    .phi-banner p { color: #d3e2f0; margin: 0.4rem 0 0; font-size: 0.92rem; }
    .phi-badge {
        display: inline-block; background: rgba(255,255,255,0.14); border: 1px solid rgba(255,255,255,0.25);
        border-radius: 20px; padding: 2px 12px; font-size: 12px; margin-top: 10px; margin-right: 6px;
    }
    div[data-testid="stMetric"] {
        background: white; border: 1px solid var(--phi-border); border-radius: 10px; padding: 0.7rem 1rem;
    }
    section[data-testid="stSidebar"] { background-color: #FFFFFF; border-right: 1px solid var(--phi-border); }
    </style>
    """,
    unsafe_allow_html=True,
)


# ---------------------------------------------------------------------------
# Optional in-process API launch, for single-process hosts (e.g. Streamlit
# Community Cloud) that only run one entrypoint and can't start a second
# uvicorn process alongside it the way local dev / Docker do. Local dev and
# the Docker/HF Spaces path are unaffected -- they keep starting the API
# separately (see the two-process instructions in this file's docstring and
# in start.sh) since that's a faster iteration loop and doesn't need this.
#
# Gated behind an explicit env var rather than always-on so this file's
# behavior doesn't silently change for the two-process deployments.
# st.cache_resource (not a plain module global) is what makes this safe
# under Streamlit's rerun model: Streamlit re-executes this whole script on
# every interaction, but a cache_resource-wrapped function's body only runs
# once per process -- later reruns just return the cached thread object
# instead of spawning a second uvicorn server on an already-bound port.
# ---------------------------------------------------------------------------
@st.cache_resource
def _start_embedded_api() -> threading.Thread:
    import uvicorn

    from api.main import app as fastapi_app

    def _run() -> None:
        uvicorn.run(fastapi_app, host="127.0.0.1", port=8000, log_level="warning")

    thread = threading.Thread(target=_run, daemon=True, name="phi-deid-embedded-api")
    thread.start()

    for _ in range(30):
        try:
            with socket.create_connection(("127.0.0.1", 8000), timeout=1):
                break
        except OSError:
            time.sleep(1)
    return thread


if os.environ.get("PHI_DEID_EMBED_API", "").lower() in ("1", "true", "yes"):
    _start_embedded_api()

# ---------------------------------------------------------------------------
# Session state -- Streamlit reruns the whole script on every interaction,
# so anything that needs to survive a click (the pending review payload,
# the thread_id linking a review back to its paused graph run, the final
# result) has to live here instead of a local variable.
# ---------------------------------------------------------------------------
for key, default in (
    ("result", None),
    ("review_payload", None),
    ("thread_id", None),
    ("error", None),
):
    if key not in st.session_state:
        st.session_state[key] = default


def _reset():
    st.session_state.result = None
    st.session_state.review_payload = None
    st.session_state.thread_id = None
    st.session_state.error = None


# ---------------------------------------------------------------------------
# Sidebar: page selector + API connection
# ---------------------------------------------------------------------------
st.sidebar.markdown("### PHI De-identification")
page = st.sidebar.radio("View", ["De-identify a document", "Evaluation dashboard"], label_visibility="collapsed")

st.sidebar.divider()

# About: the thing a first-time visitor actually wants to see before
# diving into a form -- what this is built with and what it does --
# rather than API plumbing, which is a real requirement but not the most
# useful use of the sidebar's most visible space.
st.sidebar.markdown("#### About this project")
st.sidebar.caption(
    "A multi-agent PHI de-identification pipeline: 8 LangGraph agents "
    "detect, validate, redact, and audit Safe Harbor identifiers in "
    "clinical documents, with a human (or optional LLM) reviewing any "
    "low-confidence call before it's finalized."
)
with st.sidebar.expander("Tech stack"):
    st.markdown(
        "- **Orchestration:** LangGraph (StateGraph, checkpointed, human-in-the-loop `interrupt()`)\n"
        "- **PHI detection:** Microsoft Presidio + spaCy NER, with a dependency-free regex fallback\n"
        "- **Optional LLM tier:** LangChain + OpenAI (classification and/or span adjudication, off by default)\n"
        "- **API:** FastAPI\n"
        "- **UI:** Streamlit (this app)\n"
        "- **Persistence:** SQLAlchemy (audit trail), SQLite (checkpointer)\n"
        "- **Ingestion:** .txt / .docx / .pdf"
    )
with st.sidebar.expander("The 8 agents"):
    st.markdown(
        "1. **ClassificationAgent** -- routes to a document type\n"
        "2. **PHIDetectionAgent** -- finds candidate PHI spans\n"
        "3. **PHIValidationAgent** -- schema/completeness check\n"
        "4. **LLMAdjudicationAgent** -- optional, reviews low-confidence spans\n"
        "5. **HumanReviewAgent** -- pauses for a human on anything still ambiguous\n"
        "6. **RedactionAgent** -- masks every approved span\n"
        "7. **ComplianceValidationAgent** -- re-scans the redacted output\n"
        "8. **AuditReportAgent** -- assembles the audit log + compliance report"
    )

st.sidebar.divider()
with st.sidebar.expander("API connection", expanded=False):
    api_base = st.text_input("API base URL", value="http://localhost:8000").rstrip("/")
    # Pre-fill from the same PHI_DEID_API_KEY env var the API server reads,
    # so a deployed instance (e.g. a Hugging Face Space, where both
    # processes share one container's environment) works out of the box
    # for a reviewer -- no secret to copy-paste. Local dev users can still
    # override it in the field.
    api_key = st.text_input(
        "API key (X-API-Key)",
        value=os.environ.get("PHI_DEID_API_KEY", ""),
        type="password",
        help=(
            "Set PHI_DEID_API_KEY before starting the API server for a stable key, "
            "or copy the auto-generated one printed in the server's console on startup. "
            "Pre-filled automatically when both processes share an environment."
        ),
    )
    if not api_key:
        st.warning("No API key set -- every request will get a 401 Unauthorized.")

    if st.button("Check API health"):
        try:
            r = requests.get(f"{api_base}/health", timeout=5)
            if r.status_code == 200:
                st.success("API is reachable.")
            else:
                st.error(f"API responded with status {r.status_code}.")
        except requests.RequestException as exc:
            st.error(f"Could not reach the API at {api_base}: {exc}")


def _headers() -> dict:
    return {"X-API-Key": api_key} if api_key else {}


st.sidebar.divider()
if st.sidebar.button("Reset session"):
    _reset()
    st.rerun()


# ---------------------------------------------------------------------------
# Server-Sent Events parsing -- api/main.py's /stream endpoints emit
# `event: <type>\ndata: <json>\n\n` messages. requests' iter_lines() gives
# us raw lines; this reassembles them into (event_type, data_str) pairs on
# each blank-line message boundary, per the SSE spec.
# ---------------------------------------------------------------------------
def _iter_sse_events(response: requests.Response):
    event_type = None
    data_lines: list[str] = []
    for raw_line in response.iter_lines(decode_unicode=True):
        if raw_line is None:
            continue
        line = raw_line.rstrip("\r")
        if line == "":
            if event_type is not None and data_lines:
                yield event_type, "\n".join(data_lines)
            event_type, data_lines = None, []
            continue
        if line.startswith("event:"):
            event_type = line[len("event:"):].strip()
        elif line.startswith("data:"):
            data_lines.append(line[len("data:"):].strip())
    if event_type is not None and data_lines:
        yield event_type, "\n".join(data_lines)


# ---------------------------------------------------------------------------
# Live pipeline stepper. Node names/order mirror graph/workflow.py's
# build_graph() exactly. retry_bump is deliberately not a stepper row of
# its own -- it's internal bookkeeping (bumps retry_count by one), not a
# stage a reviewer needs to watch separately; a retry just re-lights the
# "Detecting PHI" step.
# ---------------------------------------------------------------------------
_STAGE_LABELS = {
    "classification": "Classifying document type",
    "phi_detection": "Detecting PHI",
    "human_review": "Human review",
    "redaction": "Redacting PHI",
    "compliance_validation": "Validating redaction",
    "audit_report": "Generating audit report",
}
_STEPPER_STAGES = ["classification", "phi_detection", "human_review", "redaction", "compliance_validation", "audit_report"]


def _run_streaming(url: str, **request_kwargs) -> None:
    """POST to one of api/main.py's /stream endpoints and render a live,
    node-by-node progress stepper as LangGraph actually executes -- each
    checkmark below reflects a real `event: node` message received over
    the wire, not a fixed-duration animation. Populates st.session_state
    exactly like the old plain-JSON _handle_response() did, so everything
    downstream (the review form, the results tabs) is unchanged."""
    st.session_state.error = None
    final_payload: dict | None = None
    seen_nodes: set[str] = set()
    retry_count = 0

    with st.status("Running the pipeline...", expanded=True) as status:
        stepper_placeholder = st.empty()

        def _render(current: str | None = None, paused: str | None = None) -> None:
            lines = []
            for stage in _STEPPER_STAGES:
                if stage == paused:
                    icon = "⏸️"
                elif stage in seen_nodes:
                    icon = "✅"
                elif stage == current:
                    icon = "🔵"
                else:
                    icon = "⚪"
                suffix = f"  _(retry {retry_count})_" if stage == "phi_detection" and retry_count and stage == current else ""
                lines.append(f"{icon} {_STAGE_LABELS[stage]}{suffix}")
            stepper_placeholder.markdown("  \n".join(lines))

        _render()
        try:
            with requests.post(url, stream=True, timeout=180, **request_kwargs) as r:
                if r.status_code == 401:
                    status.update(label="Unauthorized", state="error")
                    st.session_state.error = "Unauthorized -- check the API key in the sidebar."
                    return
                if r.status_code >= 400:
                    try:
                        detail = r.json().get("detail", r.text)
                    except Exception:
                        detail = r.text
                    status.update(label="Request failed", state="error")
                    st.session_state.error = f"API error ({r.status_code}): {detail}"
                    return

                for event_type, data_raw in _iter_sse_events(r):
                    data = json.loads(data_raw)
                    if event_type == "node":
                        node = data["node"]
                        if node == "retry_bump":
                            retry_count += 1
                            seen_nodes.discard("compliance_validation")  # looping back -- no longer "done"
                            _render(current="phi_detection")
                            continue
                        seen_nodes.add(node)
                        _render(current=node)
                    elif event_type == "error":
                        status.update(label="Error", state="error")
                        st.session_state.error = data.get("detail", "Unknown error.")
                        return
                    elif event_type == "done":
                        final_payload = data
        except requests.RequestException as exc:
            status.update(label="Connection error", state="error")
            st.session_state.error = f"Connection error -- is the API running at {url}? ({exc})"
            return

        if final_payload is None:
            status.update(label="No result received", state="error")
            st.session_state.error = "Stream ended without a final result."
            return

        if final_payload.get("status") == "human_review_required":
            seen_nodes.discard("human_review")
            _render(paused="human_review")
            status.update(label="Paused for human review", state="complete", expanded=True)
        else:
            status.update(label="Pipeline complete", state="complete", expanded=False)

    st.session_state.thread_id = final_payload.get("thread_id")
    if final_payload.get("status") == "human_review_required":
        st.session_state.review_payload = final_payload["review_payload"]
        st.session_state.result = None
    else:
        st.session_state.result = final_payload
        st.session_state.review_payload = None


def _submit_text(text: str, filename: str) -> None:
    _run_streaming(f"{api_base}/redact/stream", headers=_headers(), json={"text": text, "filename": filename})


def _submit_file(uploaded_file) -> None:
    _run_streaming(
        f"{api_base}/redact/upload/stream",
        headers=_headers(),
        files={"file": (uploaded_file.name, uploaded_file.getvalue())},
    )


def _submit_resume(decisions: list[dict]) -> None:
    _run_streaming(
        f"{api_base}/redact/resume/stream",
        headers=_headers(),
        json={"thread_id": st.session_state.thread_id, "decisions": decisions},
    )


# ---------------------------------------------------------------------------
# Evaluation dashboard -- reads eval/*.py's own output files directly off
# disk (final_results.csv, ablation_results.md, error_analysis.md). Doesn't
# re-run anything itself (those scripts need Presidio/spaCy and take a
# while against 350+ documents) -- this is a viewer for results already
# produced, so a lecturer can see the ablation study and error analysis
# without leaving the UI or opening the repo separately.
# ---------------------------------------------------------------------------
def _render_evaluation_dashboard() -> None:
    st.markdown(
        '<div class="phi-banner"><h1>Evaluation Dashboard</h1>'
        "<p>Real numbers from eval/evaluate.py, eval/ablation_study.py, and eval/error_analysis.py -- "
        "read directly from this repo's eval/ output files, not hand-entered.</p></div>",
        unsafe_allow_html=True,
    )

    final_csv = EVAL_DIR / "final_results.csv"
    if final_csv.exists():
        with open(final_csv, newline="", encoding="utf-8") as f:
            rows = list(csv.DictReader(f))
        overall = next((r for r in rows if r["mode"] == "overlap" and r["phi_type"] == "OVERALL"), None)
        if overall:
            c1, c2, c3 = st.columns(3)
            c1.metric("Overall Precision (OVERLAP)", f"{float(overall['precision']):.4f}")
            c2.metric("Overall Recall (OVERLAP)", f"{float(overall['recall']):.4f}")
            c3.metric("Overall F1 (OVERLAP)", f"{float(overall['f1']):.4f}")
        with st.expander("Full per-type results (STRICT + OVERLAP)", expanded=False):
            st.dataframe(rows, width='stretch', hide_index=True)
    else:
        st.info("No `eval/final_results.csv` yet -- run `python -m eval.evaluate` first.")

    st.divider()
    st.subheader("Ablation study: Human Review ON vs OFF")
    ablation_md = EVAL_DIR / "ablation_results.md"
    if ablation_md.exists():
        st.markdown(ablation_md.read_text(encoding="utf-8"))
    else:
        st.info("No `eval/ablation_results.md` yet -- run `python -m eval.ablation_study` first.")

    st.divider()
    st.subheader("Error analysis: real false positives and false negatives")
    error_md = EVAL_DIR / "error_analysis.md"
    if error_md.exists():
        st.markdown(error_md.read_text(encoding="utf-8"))
    else:
        st.info("No `eval/error_analysis.md` yet -- run `python -m eval.error_analysis` first.")


# ---------------------------------------------------------------------------
# Page routing
# ---------------------------------------------------------------------------
if page == "Evaluation dashboard":
    _render_evaluation_dashboard()
    st.stop()

st.markdown(
    '<div class="phi-banner"><h1>PHI De-identification &amp; Compliance Workflow</h1>'
    "<p>LangGraph multi-agent pipeline: classify document type &rarr; detect PHI &rarr; human review "
    "&rarr; redact &rarr; validate &rarr; audit report. Talks to the FastAPI server over HTTP, never "
    "imports the pipeline directly.</p>"
    '<span class="phi-badge">LangGraph</span><span class="phi-badge">Presidio + spaCy</span>'
    '<span class="phi-badge">Human-in-the-loop</span><span class="phi-badge">HIPAA Safe Harbor</span>'
    "</div>",
    unsafe_allow_html=True,
)

if st.session_state.error:
    st.error(st.session_state.error)


# ---------------------------------------------------------------------------
# 1. Input -- only shown when there's nothing pending and no result yet
# ---------------------------------------------------------------------------
if st.session_state.review_payload is None and st.session_state.result is None:
    input_mode = st.radio(
        "Input", ["Paste text", "Upload file", "Try a sample document"], horizontal=True
    )

    if input_mode == "Paste text":
        filename = st.text_input("Filename (for the audit trail)", value="note.txt")
        text = st.text_area(
            "Document text",
            height=250,
            placeholder="Paste clinical note, discharge summary, lab report, etc. here...",
        )
        if st.button("Run de-identification", type="primary", disabled=not text.strip()):
            _submit_text(text, filename)
            st.rerun()
    elif input_mode == "Upload file":
        uploaded = st.file_uploader("Upload a document", type=["txt", "pdf", "docx"])
        if uploaded is not None and st.button("Run de-identification", type="primary"):
            _submit_file(uploaded)
            st.rerun()
    else:
        st.caption(
            "Pre-loaded from data/ -- the same document in all three supported "
            "formats, so you can test end-to-end without sourcing your own file."
        )
        sample_label = st.selectbox("Sample document", list(SAMPLE_DOCS.keys()))
        if st.button("Run de-identification", type="primary"):
            _submit_file(_SampleFile(SAMPLE_DOCS[sample_label]))
            st.rerun()

# ---------------------------------------------------------------------------
# 2. Human review -- shown when PHIDetectionAgent flagged low-confidence
#    spans and the graph is paused at HumanReviewAgent's interrupt()
# ---------------------------------------------------------------------------
elif st.session_state.review_payload is not None:
    payload = st.session_state.review_payload
    st.subheader("Human review required")
    st.info(payload.get("instructions", "Approve or reject each span as genuine PHI."))
    reviewer = st.text_input("Reviewer name", value="reviewer")

    with st.form("review_form"):
        choices = {}
        for span in payload["spans"]:
            idx = span["span_index"]
            cols = st.columns([1, 2, 3, 2])
            cols[0].markdown(f"**#{idx}**")
            cols[1].markdown(f"`{span['phi_type']}`")
            cols[2].code(span["text"])
            cols[3].markdown(f"confidence: **{span['confidence']:.2f}**")
            choices[idx] = st.radio(
                f"Decision for span #{idx}",
                ["Approve (treat as PHI, redact it)", "Reject (not PHI, leave as-is)"],
                key=f"decision_{idx}",
                horizontal=True,
                label_visibility="collapsed",
            )
            st.divider()
        submitted = st.form_submit_button("Submit review decisions", type="primary")

    if submitted:
        decisions = [
            {
                "span_index": idx,
                "approved": choice.startswith("Approve"),
                "reviewer": reviewer,
            }
            for idx, choice in choices.items()
        ]
        _submit_resume(decisions)
        st.rerun()

# ---------------------------------------------------------------------------
# 3. Results
# ---------------------------------------------------------------------------
if st.session_state.result is not None:
    result = st.session_state.result
    report = result.get("compliance_report") or {}
    doc_type = result.get("doc_type", "unknown")

    st.subheader("Result")

    if doc_type == "not_applicable":
        st.info(
            "This document was classified as **not_applicable** (not a "
            "recognized clinical document type) and routed straight through "
            "without PHI detection or redaction."
        )
    else:
        status = report.get("validation_status", "UNKNOWN")
        status_display = {"PASS": ":green[PASS]", "FAIL": ":red[FAIL]"}.get(status, status)
        st.markdown(f"**Document type:** `{doc_type}`  |  **Validation status:** {status_display}")

        m1, m2, m3, m4, m5 = st.columns(5)
        m1.metric("PHI spans detected", report.get("total_phi_spans_detected", 0))
        m2.metric("Retries used", report.get("retries_used", 0))
        m3.metric("Human review invoked", "Yes" if report.get("human_review_invoked") else "No")
        m4.metric("Manual follow-up needed", "Yes" if report.get("requires_manual_followup") else "No")
        m5.metric("Pipeline time", f"{report.get('total_pipeline_ms', 0):.0f} ms")

        tab1, tab2, tab3, tab4 = st.tabs(["Redacted text", "Compliance report", "Audit log", "Observability"])
        with tab1:
            st.text_area("Redacted document", value=result.get("redacted_text", ""), height=350)
        with tab2:
            summary_cols = st.columns(3)
            summary_cols[0].markdown(
                f"**Filename:** `{report.get('filename', 'unknown')}`  \n"
                f"**Doc type confidence:** {report.get('doc_type_confidence')}  \n"
                f"**Generated at:** {report.get('generated_at', '')[:19].replace('T', ' ')}"
            )
            summary_cols[1].markdown(
                f"**Compliance score:** {report.get('compliance_score')}  \n"
                f"**Schema completeness:** {report.get('schema_completeness_score')}  \n"
                f"**Retries used:** {report.get('retries_used', 0)}"
            )
            summary_cols[2].markdown(
                f"**Retries exhausted while failing:** {report.get('retries_exhausted_while_failing')}  \n"
                f"**Escalated to human review:** {report.get('escalated_to_human_review')}  \n"
                f"**Requires manual follow-up:** {report.get('requires_manual_followup')}"
            )

            missing_types = report.get("missing_expected_identifier_types", [])
            if missing_types:
                st.warning(f"Missing expected identifier types: {', '.join(missing_types)}")

            st.markdown("**Compliance checks** (per PHI category, after redaction)")
            compliance_checks = report.get("compliance_checks", [])
            if compliance_checks:
                st.dataframe(compliance_checks, width='stretch', hide_index=True)
            else:
                st.caption("No compliance checks recorded (no PHI categories to validate).")

            st.markdown("**PHI spans detected, by type**")
            phi_by_type = report.get("phi_spans_by_type", {})
            if phi_by_type:
                st.dataframe(
                    [{"phi_type": k, "count": v} for k, v in phi_by_type.items()],
                    width='stretch',
                    hide_index=True,
                )
            else:
                st.caption("No PHI spans detected.")

            remaining = report.get("remaining_phi_spans_after_validation", [])
            if remaining:
                st.markdown("**Residual PHI spans after validation** (should be empty on PASS)")
                st.dataframe(remaining, width='stretch', hide_index=True)

            with st.expander("Raw JSON (for copying)"):
                # st.code, not st.json: st.json's tree viewer puts a copy
                # icon on every expandable object/array (no parameter to
                # turn that off). A plain code block is static text, so it
                # gets exactly one copy icon, top-right of the block.
                st.code(json.dumps(report, indent=2, default=str), language="json")
        with tab3:
            audit_log = result.get("audit_log", [])
            if audit_log:
                st.dataframe(audit_log, width='stretch', hide_index=True)
            else:
                st.write("No audit log entries.")
        with tab4:
            # Per-node wall-clock timing, recorded by graph/workflow.py's
            # _timed() wrapper -- see that module for why human_review is
            # excluded (it measures reviewer response time, not agent
            # performance, if included at all).
            node_timings = report.get("node_timings_ms", {})
            if node_timings:
                st.bar_chart(node_timings)
                st.caption(
                    "Wall-clock time per LangGraph node (ms). human_review is excluded -- "
                    "its duration is reviewer response time, not agent compute time."
                )
            else:
                st.write("No timing data recorded for this run.")

            st.divider()
            st.markdown("**LLM usage** (this is the real proof an LLM call actually "
                        "happened, not just that a backend flag was set)")
            llm_usage = report.get("llm_usage_summary", {})
            total_tokens = llm_usage.get("total_tokens", 0)
            if total_tokens:
                lm1, lm2 = st.columns(2)
                lm1.metric("Total tokens", total_tokens)
                lm2.metric("Approx. cost (USD)", f"${llm_usage.get('total_approx_cost_usd', 0):.6f}")
                by_agent = llm_usage.get("by_agent", {})
                if by_agent:
                    st.dataframe(
                        [{"agent": k, **v} for k, v in by_agent.items()],
                        width='stretch',
                        hide_index=True,
                    )
            else:
                st.caption(
                    "Zero tokens -- no LLM call was made in this run. Expected when "
                    "PHI_DEID_CLASSIFICATION_BACKEND and PHI_DEID_ADJUDICATION_BACKEND "
                    "are both left at their default (heuristic), since neither backend "
                    "calls an LLM at all in that case. If you expected a real call here, "
                    "check those two .env variables, not PHI_DEID_ENV -- PHI_DEID_ENV only "
                    "controls deployment topology (embedded API + spaCy model size), not "
                    "which detection/adjudication backend runs."
                )

    st.caption(f"thread_id: `{result.get('thread_id')}`")
    if st.button("Process another document"):
        _reset()
        st.rerun()
