"""
scripts/build_sample_dataset.py

Builds a realistic, labeled test dataset for the PHI de-identification
pipeline by combining two sources, per the team's decision (n2c2 2014 is
currently "Temporarily Unavailable" on the DBMI Data Portal, so this is
the active plan, not just a fallback):

  1. Kaggle "Medical Transcriptions" (tboyle10/medicaltranscriptions,
     scraped from mtsamples.com, CC0) — for realistic document
     *structure* per specialty. This dataset is NOT bundled here; you
     download it once (see SETUP below).

  2. A synthetic PHI injector (stdlib + Faker) — for real, LABELED PHI.
     mtsamples text is teaching material and mostly already has little
     to no real PHI in it, so we prepend a realistic identifying header
     (patient name, DOB, MRN, phone, dates, provider names, etc.) built
     from Faker, and record the exact character span of every injected
     value. That gives us ground truth to score PHIDetectionAgent
     against (precision/recall/F1), the same way the n2c2 benchmark
     would — see OUTPUT below.

`insurance_document` has no mtsamples equivalent at all (it's billing
data, not clinical transcription), so those are generated fully
synthetically from a template.

If you haven't downloaded mtsamples yet, this script still runs — it
falls back to a small set of built-in generic templates per doc type so
you always get *some* labeled data to test against, just less varied
than the real thing.

SETUP
-----
    pip install faker

    # Get the CSV (no Kaggle API token needed):
    #   1. https://www.kaggle.com/datasets/tboyle10/medicaltranscriptions
    #   2. Click Download, unzip, and place mtsamples.csv at:
    #        data/raw/mtsamples.csv
    #
    # If you do have a Kaggle API token (~/.kaggle/kaggle.json) set up,
    # you can instead run:
    #   pip install kagglehub
    #   python -c "import kagglehub; print(kagglehub.dataset_download('tboyle10/medicaltranscriptions'))"
    # and copy the CSV it downloads to data/raw/mtsamples.csv.

USAGE
-----
    python -m scripts.build_sample_dataset --per-type 50

OUTPUT
------
    data/txt_format/<doc_type>_<n>.txt   generated documents
    data/txt_format/ground_truth.json    {filename: {doc_type, phi_spans: [...]}}
                                          — feed this to eval/evaluate.py
                                          (Member 3/1) to score PHIDetectionAgent.
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import random
import sys
from dataclasses import dataclass, field

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    from faker import Faker
except ImportError:
    print("Missing dependency. Run: pip install faker", file=sys.stderr)
    raise

from agents.classification_agent import classify_with_heuristic
from graph.state import CLINICAL_DOC_TYPES, DocType

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(THIS_DIR)
DEFAULT_CSV_PATH = os.path.join(PROJECT_ROOT, "data", "raw", "mtsamples.csv")
# All generated .txt documents (across all 7 doc types) plus ground_truth.json
# live together in one flat folder, alongside data/pdf_format/ and
# data/docx_format/ (produced by scripts/generate_demo_files.py) — the three
# folders group every sample document by file format.
OUTPUT_DIR = os.path.join(PROJECT_ROOT, "data", "txt_format")

# ---------------------------------------------------------------------------
# mtsamples `medical_specialty` -> our DocType. Specialties not listed here
# are skipped (mtsamples has ~40 specialties; only these map cleanly onto
# the 7 document types the ClassificationAgent routes between).
# "Lab Medicine - Pathology" is ambiguous in mtsamples (it covers both), so
# we sub-classify those rows with our own heuristic classifier below rather
# than a static mapping.
# ---------------------------------------------------------------------------
SPECIALTY_TO_DOCTYPE: dict[str, DocType] = {
    "Discharge Summary": "discharge_summary",
    "Radiology": "radiology_report",
    "Letters": "referral_letter",
    "General Medicine": "clinical_note",
    "Consult - History and Phy.": "clinical_note",
    "SOAP / Chart / Progress Notes": "clinical_note",
    "Office Notes": "clinical_note",
    "Emergency Room Reports": "clinical_note",
}
AMBIGUOUS_LAB_PATH_SPECIALTY = "Lab Medicine - Pathology"


@dataclass
class GroundTruthSpan:
    start: int
    end: int
    text: str
    phi_type: str


@dataclass
class GeneratedDoc:
    filename: str
    doc_type: str
    text: str
    phi_spans: list[GroundTruthSpan] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Synthetic PHI injection
# ---------------------------------------------------------------------------
# Field sets per doc type, using the same phi_type vocabulary PHIDetectionAgent
# emits (graph/state.py / agents/phi_detection_agent.py) so ground truth can
# be compared directly against detector output.
_FIELD_SETS: dict[str, list[tuple[str, str]]] = {
    # (label, phi_type)
    "clinical_note": [
        ("Patient Name", "PERSON"), ("DOB", "DATE_TIME"), ("MRN", "MRN"),
        ("Phone", "PHONE_NUMBER"), ("Visit Date", "DATE_TIME"), ("Provider", "PERSON"),
    ],
    "discharge_summary": [
        ("Patient Name", "PERSON"), ("DOB", "DATE_TIME"), ("MRN", "MRN"),
        ("Admission Date", "DATE_TIME"), ("Discharge Date", "DATE_TIME"),
        ("Account Number", "ACCOUNT_NUMBER"), ("Phone", "PHONE_NUMBER"),
    ],
    "radiology_report": [
        ("Patient Name", "PERSON"), ("DOB", "DATE_TIME"), ("MRN", "MRN"),
        ("Exam Date", "DATE_TIME"), ("Referring Physician", "PERSON"), ("Radiologist", "PERSON"),
    ],
    "pathology_report": [
        ("Patient Name", "PERSON"), ("DOB", "DATE_TIME"), ("MRN", "MRN"),
        ("Specimen Date", "DATE_TIME"), ("Pathologist", "PERSON"), ("Accession Number", "ACCESSION_NUMBER"),
    ],
    "lab_report": [
        ("Patient Name", "PERSON"), ("DOB", "DATE_TIME"), ("MRN", "MRN"),
        ("Collection Date", "DATE_TIME"), ("Ordering Physician", "PERSON"), ("Phone", "PHONE_NUMBER"),
    ],
    "referral_letter": [
        ("Patient Name", "PERSON"), ("DOB", "DATE_TIME"), ("Referring Physician", "PERSON"),
        ("Receiving Physician", "PERSON"), ("Phone", "PHONE_NUMBER"), ("Date", "DATE_TIME"),
    ],
    "insurance_document": [
        ("Patient Name", "PERSON"), ("Member ID", "HEALTH_PLAN_ID"), ("Policy Number", "HEALTH_PLAN_ID"),
        ("Claim Number", "CLAIM_NUMBER"), ("Date of Service", "DATE_TIME"), ("Phone", "PHONE_NUMBER"),
    ],
}


def _date_ordered_overrides(doc_type: str, fake: Faker) -> dict[str, str]:
    """Some doc types have two date fields that must stay chronologically
    ordered (e.g. discharge >= admission) — independent random draws per
    field can otherwise put discharge before admission. Pre-compute those
    pairs here; build_header_and_spans() checks this before falling back
    to the generic per-field logic in _fake_value_for()."""
    if doc_type == "discharge_summary":
        admission = fake.date_between(start_date="-60d", end_date="-3d")
        discharge = fake.date_between(start_date=admission, end_date="today")
        return {
            "Admission Date": admission.strftime("%m/%d/%Y"),
            "Discharge Date": discharge.strftime("%m/%d/%Y"),
        }
    return {}


def _fake_value_for(label: str, fake: Faker) -> str:
    key = label.lower()
    if "name" in key or "physician" in key or "provider" in key or "radiologist" in key or "pathologist" in key:
        return fake.name()
    if "dob" in key:
        return fake.date_of_birth(minimum_age=1, maximum_age=95).strftime("%m/%d/%Y")
    if "date" in key:
        return fake.date_between(start_date="-60d", end_date="today").strftime("%m/%d/%Y")
    if "mrn" in key:
        return f"MR-{fake.random_number(digits=6, fix_len=True)}"
    if "phone" in key:
        return fake.phone_number()
    if "account" in key:
        return f"ACCT-{fake.random_number(digits=6, fix_len=True)}"
    if "accession" in key:
        return f"ACC-{fake.random_number(digits=8, fix_len=True)}"
    if "member id" in key:
        return f"MBR-{fake.random_number(digits=7, fix_len=True)}"
    if "policy" in key:
        return f"POL-{fake.bothify('####-?')}"
    if "claim" in key:
        return f"CLM-2026-{fake.random_number(digits=5, fix_len=True)}"
    return fake.word()


def build_header_and_spans(doc_type: str, fake: Faker) -> tuple[str, list[GroundTruthSpan]]:
    """Builds a labeled identifying-info header block and returns (header_text, spans)
    with span offsets relative to the header itself (caller must shift by the
    header's insertion offset when splicing into the full document)."""
    fields = _FIELD_SETS.get(doc_type, _FIELD_SETS["clinical_note"])
    overrides = _date_ordered_overrides(doc_type, fake)
    lines = []
    spans: list[GroundTruthSpan] = []
    cursor = 0
    for label, phi_type in fields:
        value = overrides.get(label) or _fake_value_for(label, fake)
        line = f"{label}: {value}"
        value_start_in_line = len(f"{label}: ")
        line_start = cursor
        spans.append(GroundTruthSpan(
            start=line_start + value_start_in_line,
            end=line_start + value_start_in_line + len(value),
            text=value,
            phi_type=phi_type,
        ))
        lines.append(line)
        cursor += len(line) + 1  # +1 for the newline joining lines
    header = "\n".join(lines) + "\n"
    return header, spans


def inject_header(body_text: str, doc_type: str, fake: Faker) -> tuple[str, list[GroundTruthSpan]]:
    header, header_spans = build_header_and_spans(doc_type, fake)
    full_text = header + "\n" + body_text
    shifted = [GroundTruthSpan(s.start, s.end, s.text, s.phi_type) for s in header_spans]
    # header spans are already relative to position 0 of `header`, and header
    # sits at the start of full_text, so no shift needed for header itself.
    return full_text, shifted


# ---------------------------------------------------------------------------
# mtsamples loading
# ---------------------------------------------------------------------------
def load_mtsamples(csv_path: str) -> list[dict]:
    if not os.path.exists(csv_path):
        return []
    rows = []
    with open(csv_path, encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            transcription = (row.get("transcription") or "").strip()
            specialty = (row.get("medical_specialty") or "").strip()
            if transcription and specialty:
                rows.append(row)
    return rows


def bucket_by_doctype(rows: list[dict]) -> dict[str, list[dict]]:
    buckets: dict[str, list[dict]] = {dt: [] for dt in CLINICAL_DOC_TYPES}
    for row in rows:
        specialty = row["medical_specialty"].strip()
        transcription = row["transcription"].strip()

        if specialty in SPECIALTY_TO_DOCTYPE:
            buckets[SPECIALTY_TO_DOCTYPE[specialty]].append(row)
        elif specialty == AMBIGUOUS_LAB_PATH_SPECIALTY:
            # Reuse the production classifier to split lab vs pathology,
            # keeping this sample generator consistent with what
            # ClassificationAgent would actually decide at inference time.
            doc_type, _ = classify_with_heuristic(transcription)
            if doc_type in ("lab_report", "pathology_report"):
                buckets[doc_type].append(row)
            else:
                buckets["pathology_report"].append(row)  # default for this specialty
    return buckets


# ---------------------------------------------------------------------------
# Fallback canned templates (used when mtsamples.csv isn't present, or a
# bucket comes up short) so the script always produces *some* labeled data.
# ---------------------------------------------------------------------------
_FALLBACK_BODIES: dict[str, list[str]] = {
    "clinical_note": [
        "Chief Complaint: Follow-up for hypertension management.\n\n"
        "The patient reports good adherence to medication with no side effects. "
        "Blood pressure today is well controlled. Continue current regimen and "
        "return in 3 months for reassessment.",
    ],
    "discharge_summary": [
        "Hospital Course: Patient was admitted for management of acute symptoms "
        "and responded well to treatment. Discharge Diagnosis: resolved. "
        "Discharge Instructions: continue prescribed medications and follow up "
        "with primary care within one week.",
    ],
    "radiology_report": [
        "Exam: Chest X-ray, 2 views.\nFindings: Lungs are clear bilaterally. No "
        "acute cardiopulmonary process. Impression: No acute findings.",
    ],
    "pathology_report": [
        "Specimen: Skin, left forearm, punch biopsy.\nGross Description: Single "
        "tan-pink skin ellipse. Microscopic Description: Unremarkable epidermis "
        "and dermis. Diagnosis: Benign, no evidence of malignancy.",
        "Specimen: Breast, right, core needle biopsy.\nGross Description: Three "
        "cores of tan-white fibrofatty tissue, measuring 1.2 cm in aggregate "
        "length. Microscopic Description: Fibrocystic changes with no atypia "
        "identified. Diagnosis: Benign fibrocystic change.",
        "Specimen: Colon, sigmoid, polypectomy.\nGross Description: Single "
        "polypoid fragment of tan-pink mucosal tissue, 0.6 cm in greatest "
        "dimension. Microscopic Description: Tubular adenoma with low-grade "
        "dysplasia. Diagnosis: Tubular adenoma, margins uninvolved.",
    ],
    "lab_report": [
        "Test: Complete Blood Count (CBC).\nResults: WBC 6.8 (ref 4.0-11.0), "
        "Hgb 13.9 (ref 12.0-16.0), Platelets 250 (ref 150-400). All results "
        "within reference range.",
        "Test: Comprehensive Metabolic Panel.\nResults: Glucose 94 (ref "
        "70-99), Sodium 139 (ref 135-145), Potassium 4.1 (ref 3.5-5.0), "
        "Creatinine 0.9 (ref 0.6-1.3). All results within reference range.",
        "Test: Lipid Panel.\nResults: Total Cholesterol 178 (ref <200), LDL "
        "104 (ref <130), HDL 52 (ref >40), Triglycerides 110 (ref <150). All "
        "results within reference range.",
    ],
    "referral_letter": [
        "Dear Doctor,\n\nI am referring this patient for further evaluation of "
        "persistent symptoms unresponsive to first-line treatment. Relevant "
        "history and current medications are summarized below. Thank you for "
        "seeing this patient at your earliest convenience.",
        "Dear Colleague,\n\nThank you for agreeing to see this patient for "
        "specialist evaluation. Relevant history, current medications, and "
        "recent lab results are summarized below. Please let me know if you "
        "require additional information prior to the appointment.",
        "Dear Doctor,\n\nI am writing to request a consultation for this "
        "patient regarding findings noted on recent imaging. A summary of "
        "the clinical picture and relevant history is enclosed. I would "
        "appreciate your assessment at your earliest convenience.",
    ],
    "insurance_document": [
        "Service Description: Office visit, established patient, moderate "
        "complexity.\nAmount Billed: $185.00\nCopay: $25.00\nAmount Paid by "
        "Plan: $160.00\n\nQuestions about this claim? Call member services.",
        "Service Description: Outpatient physical therapy session.\nAmount "
        "Billed: $120.00\nCoinsurance: $18.00\nAmount Paid by Plan: $102.00\n\n"
        "This is not a bill. Questions about this claim? Call member services.",
        "Service Description: Diagnostic imaging, MRI without contrast.\n"
        "Amount Billed: $950.00\nDeductible Applied: $250.00\nAmount Paid by "
        "Plan: $700.00\n\nQuestions about this claim? Call member services.",
    ],
}


# ---------------------------------------------------------------------------
# Main build
# ---------------------------------------------------------------------------
def generate_docs(per_type: int, csv_path: str, seed: int) -> list[GeneratedDoc]:
    random.seed(seed)
    fake = Faker()
    Faker.seed(seed)

    mtsamples_rows = load_mtsamples(csv_path)
    buckets = bucket_by_doctype(mtsamples_rows) if mtsamples_rows else {dt: [] for dt in CLINICAL_DOC_TYPES}

    if mtsamples_rows:
        print(f"Loaded {len(mtsamples_rows)} mtsamples rows.")
        for dt in CLINICAL_DOC_TYPES:
            print(f"  {dt}: {len(buckets[dt])} candidates from mtsamples")
    else:
        print(f"No mtsamples CSV found at {csv_path} — using built-in fallback templates only.")
        print("See the SETUP section at the top of this script to add the real dataset.")

    docs: list[GeneratedDoc] = []
    counter = 0

    for doc_type in CLINICAL_DOC_TYPES:
        candidates = list(buckets.get(doc_type, []))
        random.shuffle(candidates)

        n_from_data = min(per_type, len(candidates))
        n_fallback = per_type - n_from_data

        bodies: list[str] = [c["transcription"].strip() for c in candidates[:n_from_data]]
        if doc_type == "insurance_document":
            # always fully synthetic — no mtsamples source for this type
            bodies = []
            n_fallback = per_type

        fallback_pool = _FALLBACK_BODIES.get(doc_type, ["No sample body available."])
        for i in range(n_fallback):
            bodies.append(fallback_pool[i % len(fallback_pool)])

        for body in bodies:
            counter += 1
            full_text, spans = inject_header(body, doc_type, fake)
            filename = f"{doc_type}_{counter:03d}.txt"
            docs.append(GeneratedDoc(filename=filename, doc_type=doc_type, text=full_text, phi_spans=spans))

    return docs


def write_docs(docs: list[GeneratedDoc], output_dir: str) -> None:
    os.makedirs(output_dir, exist_ok=True)
    ground_truth = {}

    for doc in docs:
        path = os.path.join(output_dir, doc.filename)
        with open(path, "w", encoding="utf-8") as f:
            f.write(doc.text)
        ground_truth[doc.filename] = {
            "doc_type": doc.doc_type,
            "phi_spans": [
                {"start": s.start, "end": s.end, "text": s.text, "phi_type": s.phi_type}
                for s in doc.phi_spans
            ],
        }

    gt_path = os.path.join(output_dir, "ground_truth.json")
    with open(gt_path, "w", encoding="utf-8") as f:
        json.dump(ground_truth, f, indent=2)

    print(f"\nWrote {len(docs)} documents to {output_dir}")
    print(f"Wrote ground truth ({sum(len(d.phi_spans) for d in docs)} labeled spans) to {gt_path}")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--per-type", type=int, default=15, help="Target documents per doc type (default: 15)")
    parser.add_argument("--csv", type=str, default=DEFAULT_CSV_PATH, help="Path to mtsamples.csv")
    parser.add_argument("--output-dir", type=str, default=OUTPUT_DIR, help="Where to write generated docs")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")
    args = parser.parse_args()

    docs = generate_docs(per_type=args.per_type, csv_path=args.csv, seed=args.seed)
    write_docs(docs, args.output_dir)


if __name__ == "__main__":
    main()
