"""
eval/evaluate.py (Member 1/3, per capstone deliverable list)

Scores PHIDetectionAgent against the labeled ground truth produced by
scripts/build_sample_dataset.py (data/txt_format/ground_truth.json),
giving real precision/recall/F1 -- the number your n2c2 benchmark
comparison actually needs, not a self-reported one.

Two matching strategies are reported, per standard PHI de-identification /
NER evaluation practice (i2b2/n2c2 challenges report both):

  - STRICT   exact (start, end, phi_type) match required.
  - OVERLAP  any character overlap between predicted and gold span (same
             phi_type), matched greedily so each span counts at most
             once. This is the more forgiving "did we catch this PHI at
             all" metric -- our injected ground truth spans cover just the
             value (e.g. "MR-969634"), while a detector might reasonably
             flag the label too ("MRN: MR-969634"). That's a boundary
             convention difference, not a real detection miss, so STRICT
             alone would understate recall.

Both overall and per-PHI-type precision/recall/F1 are reported.

USAGE
-----
    python -m eval.evaluate
    python -m eval.evaluate --dataset-dir data/txt_format
    python -m eval.evaluate --backend fallback   # force the regex fallback detector
    python -m eval.evaluate --backend presidio   # force Presidio (errors if unavailable)

OUTPUT
------
    Console report (overall + per-type, both matching strategies)
    eval/final_results.csv   per-type precision/recall/F1 (both strategies)
    eval/final_results.json  full machine-readable results

NOTE ON BACKENDS
----------------
Results will differ substantially between the regex fallback and real
Presidio+spaCy -- that's expected, not a bug. The fallback's PERSON
pattern only fires on "Mr./Mrs./Dr./Pt. <Name>" (needs a title prefix),
so it will show poor recall on bare names like our injected
"Patient Name: Jordan Ellis" header field, which has no title. Presidio's
NER-based PERSON detection doesn't need a title and should score much
better on exactly that case. Don't tune the regex patterns to chase
fallback recall up -- the fallback exists for dependency-free dev/offline
use, not as the production detector (see README's "Known simplifications").
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from collections import defaultdict

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from agents import phi_detection_agent as phi_detection_module
from agents.phi_detection_agent import phi_detection_agent

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(THIS_DIR)
DEFAULT_DATASET_DIR = os.path.join(PROJECT_ROOT, "data", "txt_format")


# ---------------------------------------------------------------------------
# Matching
# ---------------------------------------------------------------------------
def _spans_overlap(a_start: int, a_end: int, b_start: int, b_end: int) -> bool:
    return a_start < b_end and a_end > b_start


def _match_same_type_spans(gold: list[dict], pred: list[dict], mode: str) -> tuple[int, int, int]:
    """Greedy matching between gold and predicted spans of a SINGLE phi_type
    (caller pre-filters by type). Each predicted span is used at most once,
    so a document with 3 gold PERSON spans and 5 predicted PERSON spans
    can't inflate TP past 3. Returns (tp, fp, fn)."""
    unmatched_pred_idx = list(range(len(pred)))
    tp = 0
    for g in sorted(gold, key=lambda s: s["start"]):
        chosen = None
        for idx in unmatched_pred_idx:
            p = pred[idx]
            if mode == "strict":
                ok = p["start"] == g["start"] and p["end"] == g["end"]
            else:  # overlap
                ok = _spans_overlap(p["start"], p["end"], g["start"], g["end"])
            if ok:
                chosen = idx
                break
        if chosen is not None:
            tp += 1
            unmatched_pred_idx.remove(chosen)
    fp = len(unmatched_pred_idx)
    fn = len(gold) - tp
    return tp, fp, fn


def evaluate_document(gold_spans: list[dict], pred_spans: list[dict], mode: str) -> dict[str, tuple[int, int, int]]:
    """Returns {phi_type: (tp, fp, fn)} for one document, evaluated
    per-type so a wrong-type prediction can't accidentally count as a
    match for the type it wasn't."""
    types = {s["phi_type"] for s in gold_spans} | {s["phi_type"] for s in pred_spans}
    per_type: dict[str, tuple[int, int, int]] = {}
    for t in types:
        g = [s for s in gold_spans if s["phi_type"] == t]
        p = [s for s in pred_spans if s["phi_type"] == t]
        per_type[t] = _match_same_type_spans(g, p, mode)
    return per_type


def _prf1(tp: int, fp: int, fn: int) -> tuple[float, float, float]:
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) else 0.0
    return precision, recall, f1


# ---------------------------------------------------------------------------
# Main eval run
# ---------------------------------------------------------------------------
def run_eval(dataset_dir: str, backend: str) -> dict:
    gt_path = os.path.join(dataset_dir, "ground_truth.json")
    if not os.path.exists(gt_path):
        raise FileNotFoundError(
            f"No ground_truth.json found at {gt_path}. Run "
            f"`python -m scripts.build_sample_dataset` first."
        )
    with open(gt_path, encoding="utf-8") as f:
        ground_truth = json.load(f)

    original_fallback_flag = phi_detection_module.USE_FALLBACK_ONLY
    if backend == "fallback":
        phi_detection_module.USE_FALLBACK_ONLY = True
    elif backend == "presidio":
        if not phi_detection_module._PRESIDIO_AVAILABLE:
            raise RuntimeError(
                "backend=presidio requested but Presidio/spaCy aren't available "
                "in this environment. Install presidio-analyzer, presidio-anonymizer, "
                "spacy, and run `python -m spacy download en_core_web_lg` first."
            )
        phi_detection_module.USE_FALLBACK_ONLY = False
    # backend == "auto": leave USE_FALLBACK_ONLY as whatever the module default is

    actual_backend = (
        "presidio" if (phi_detection_module._PRESIDIO_AVAILABLE and not phi_detection_module.USE_FALLBACK_ONLY)
        else "regex_fallback"
    )

    totals = {
        "strict": defaultdict(lambda: [0, 0, 0]),   # phi_type -> [tp, fp, fn]
        "overlap": defaultdict(lambda: [0, 0, 0]),
    }
    per_doc_results = []
    docs_evaluated = 0
    docs_missing = []

    try:
        for filename, meta in sorted(ground_truth.items()):
            file_path = os.path.join(dataset_dir, filename)
            if not os.path.exists(file_path):
                docs_missing.append(filename)
                continue

            with open(file_path, encoding="utf-8") as f:
                raw_text = f.read()

            state = {"raw_text": raw_text, "retry_count": 0, "audit_log": []}
            result = phi_detection_agent(state)
            pred_spans = result["phi_spans"]
            gold_spans = meta["phi_spans"]

            doc_result = {"filename": filename, "doc_type": meta["doc_type"]}
            for mode in ("strict", "overlap"):
                per_type = evaluate_document(gold_spans, pred_spans, mode)
                for phi_type, (tp, fp, fn) in per_type.items():
                    bucket = totals[mode][phi_type]
                    bucket[0] += tp
                    bucket[1] += fp
                    bucket[2] += fn
                doc_result[mode] = {t: list(v) for t, v in per_type.items()}
            per_doc_results.append(doc_result)
            docs_evaluated += 1
    finally:
        phi_detection_module.USE_FALLBACK_ONLY = original_fallback_flag

    return {
        "backend": actual_backend,
        "docs_evaluated": docs_evaluated,
        "docs_missing": docs_missing,
        "totals": totals,
        "per_doc_results": per_doc_results,
    }


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------
def _build_summary_rows(totals: dict) -> list[dict]:
    """One row per (mode, phi_type), plus an OVERALL row per mode."""
    rows = []
    for mode in ("strict", "overlap"):
        overall_tp = overall_fp = overall_fn = 0
        for phi_type, (tp, fp, fn) in sorted(totals[mode].items()):
            precision, recall, f1 = _prf1(tp, fp, fn)
            rows.append({
                "mode": mode, "phi_type": phi_type,
                "tp": tp, "fp": fp, "fn": fn,
                "precision": round(precision, 4), "recall": round(recall, 4), "f1": round(f1, 4),
            })
            overall_tp += tp
            overall_fp += fp
            overall_fn += fn
        precision, recall, f1 = _prf1(overall_tp, overall_fp, overall_fn)
        rows.append({
            "mode": mode, "phi_type": "OVERALL",
            "tp": overall_tp, "fp": overall_fp, "fn": overall_fn,
            "precision": round(precision, 4), "recall": round(recall, 4), "f1": round(f1, 4),
        })
    return rows


def print_report(eval_result: dict) -> None:
    print(f"\nBackend used: {eval_result['backend']}")
    print(f"Documents evaluated: {eval_result['docs_evaluated']}")
    if eval_result["docs_missing"]:
        print(f"WARNING: {len(eval_result['docs_missing'])} files listed in ground_truth.json "
              f"were not found on disk: {eval_result['docs_missing'][:5]}"
              f"{' ...' if len(eval_result['docs_missing']) > 5 else ''}")

    rows = _build_summary_rows(eval_result["totals"])
    for mode in ("strict", "overlap"):
        label = "STRICT (exact span match)" if mode == "strict" else "OVERLAP (any overlap, same type)"
        print(f"\n--- {label} ---")
        print(f"{'PHI TYPE':<18}{'TP':>6}{'FP':>6}{'FN':>6}{'PRECISION':>11}{'RECALL':>9}{'F1':>8}")
        for row in rows:
            if row["mode"] != mode:
                continue
            marker = "*" if row["phi_type"] == "OVERALL" else " "
            print(f"{marker}{row['phi_type']:<17}{row['tp']:>6}{row['fp']:>6}{row['fn']:>6}"
                  f"{row['precision']:>11.4f}{row['recall']:>9.4f}{row['f1']:>8.4f}")


def write_outputs(eval_result: dict, output_dir: str) -> None:
    os.makedirs(output_dir, exist_ok=True)
    rows = _build_summary_rows(eval_result["totals"])

    csv_path = os.path.join(output_dir, "final_results.csv")
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["mode", "phi_type", "tp", "fp", "fn", "precision", "recall", "f1"])
        writer.writeheader()
        writer.writerows(rows)

    json_path = os.path.join(output_dir, "final_results.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump({
            "backend": eval_result["backend"],
            "docs_evaluated": eval_result["docs_evaluated"],
            "docs_missing": eval_result["docs_missing"],
            "summary": rows,
            "per_doc_results": eval_result["per_doc_results"],
        }, f, indent=2)

    print(f"\nWrote {csv_path}")
    print(f"Wrote {json_path}")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dataset-dir", type=str, default=DEFAULT_DATASET_DIR,
                         help="Directory containing ground_truth.json and the sample docs")
    parser.add_argument("--backend", type=str, default="auto", choices=["auto", "fallback", "presidio"],
                         help="Force a specific PHIDetectionAgent backend (default: whatever's installed)")
    parser.add_argument("--output-dir", type=str, default=THIS_DIR, help="Where to write final_results.csv/.json")
    args = parser.parse_args()

    eval_result = run_eval(dataset_dir=args.dataset_dir, backend=args.backend)
    print_report(eval_result)
    write_outputs(eval_result, args.output_dir)


if __name__ == "__main__":
    main()
