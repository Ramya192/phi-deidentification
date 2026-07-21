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
"""
from __future__ import annotations

import os

import requests
import streamlit as st

st.set_page_config(page_title="PHI De-identification", page_icon=":material/health_and_safety:", layout="wide")

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
# Sidebar: API connection
# ---------------------------------------------------------------------------
st.sidebar.header("API connection")
api_base = st.sidebar.text_input("API base URL", value="http://localhost:8000").rstrip("/")
# Pre-fill from the same PHI_DEID_API_KEY env var the API server reads, so a
# deployed instance (e.g. a Hugging Face Space, where both processes share
# one container's environment) works out of the box for a reviewer -- no
# secret to copy-paste. Local dev users can still override it in the field.
api_key = st.sidebar.text_input(
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
    st.sidebar.warning("No API key set -- every request will get a 401 Unauthorized.")


def _headers() -> dict:
    return {"X-API-Key": api_key} if api_key else {}


if st.sidebar.button("Check API health"):
    try:
        r = requests.get(f"{api_base}/health", timeout=5)
        if r.status_code == 200:
            st.sidebar.success("API is reachable.")
        else:
            st.sidebar.error(f"API responded with status {r.status_code}.")
    except requests.RequestException as exc:
        st.sidebar.error(f"Could not reach the API at {api_base}: {exc}")

st.sidebar.divider()
if st.sidebar.button("Reset session"):
    _reset()
    st.rerun()

st.title("PHI De-identification")
st.caption(
    "LangGraph multi-agent pipeline: classify document type -> detect PHI "
    "-> redact -> validate -> audit report. Talks to the FastAPI server, "
    "not the pipeline directly."
)

if st.session_state.error:
    st.error(st.session_state.error)


# ---------------------------------------------------------------------------
# API calls
# ---------------------------------------------------------------------------
def _handle_response(r: requests.Response) -> None:
    st.session_state.error = None

    if r.status_code == 401:
        st.session_state.error = "Unauthorized -- check the API key in the sidebar."
        return
    if r.status_code >= 400:
        try:
            detail = r.json().get("detail", r.text)
        except Exception:
            detail = r.text
        st.session_state.error = f"API error ({r.status_code}): {detail}"
        return

    data = r.json()
    st.session_state.thread_id = data.get("thread_id")
    if data.get("status") == "human_review_required":
        st.session_state.review_payload = data["review_payload"]
        st.session_state.result = None
    else:
        st.session_state.result = data
        st.session_state.review_payload = None


def _submit_text(text: str, filename: str) -> None:
    try:
        r = requests.post(
            f"{api_base}/redact",
            headers=_headers(),
            json={"text": text, "filename": filename},
            timeout=120,
        )
    except requests.RequestException as exc:
        st.session_state.error = f"Connection error -- is the API running at {api_base}? ({exc})"
        return
    _handle_response(r)


def _submit_file(uploaded_file) -> None:
    try:
        r = requests.post(
            f"{api_base}/redact/upload",
            headers=_headers(),
            files={"file": (uploaded_file.name, uploaded_file.getvalue())},
            timeout=120,
        )
    except requests.RequestException as exc:
        st.session_state.error = f"Connection error -- is the API running at {api_base}? ({exc})"
        return
    _handle_response(r)


def _submit_resume(decisions: list[dict]) -> None:
    try:
        r = requests.post(
            f"{api_base}/redact/resume",
            headers=_headers(),
            json={"thread_id": st.session_state.thread_id, "decisions": decisions},
            timeout=120,
        )
    except requests.RequestException as exc:
        st.session_state.error = f"Connection error -- is the API running at {api_base}? ({exc})"
        return
    _handle_response(r)


# ---------------------------------------------------------------------------
# 1. Input -- only shown when there's nothing pending and no result yet
# ---------------------------------------------------------------------------
if st.session_state.review_payload is None and st.session_state.result is None:
    input_mode = st.radio("Input", ["Paste text", "Upload file"], horizontal=True)

    if input_mode == "Paste text":
        filename = st.text_input("Filename (for the audit trail)", value="note.txt")
        text = st.text_area(
            "Document text",
            height=250,
            placeholder="Paste clinical note, discharge summary, lab report, etc. here...",
        )
        if st.button("Run de-identification", type="primary", disabled=not text.strip()):
            with st.spinner("Running the pipeline..."):
                _submit_text(text, filename)
            st.rerun()
    else:
        uploaded = st.file_uploader("Upload a document", type=["txt", "pdf", "docx"])
        if uploaded is not None and st.button("Run de-identification", type="primary"):
            with st.spinner("Running the pipeline..."):
                _submit_file(uploaded)
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
        with st.spinner("Resuming pipeline with your decisions..."):
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

        m1, m2, m3, m4 = st.columns(4)
        m1.metric("PHI spans detected", report.get("total_phi_spans_detected", 0))
        m2.metric("Retries used", report.get("retries_used", 0))
        m3.metric("Human review invoked", "Yes" if report.get("human_review_invoked") else "No")
        m4.metric("Manual follow-up needed", "Yes" if report.get("requires_manual_followup") else "No")

        tab1, tab2, tab3 = st.tabs(["Redacted text", "Compliance report", "Audit log"])
        with tab1:
            st.text_area("Redacted document", value=result.get("redacted_text", ""), height=350)
        with tab2:
            st.json(report)
        with tab3:
            audit_log = result.get("audit_log", [])
            if audit_log:
                st.dataframe(audit_log, use_container_width=True, hide_index=True)
            else:
                st.write("No audit log entries.")

    st.caption(f"thread_id: `{result.get('thread_id')}`")
    if st.button("Process another document"):
        _reset()
        st.rerun()
