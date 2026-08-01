"""
eval/evaluate_vanilla_presidio.py

One-off baseline comparison: bare, out-of-the-box Presidio (AnalyzerEngine()
with none of this project's customizations) vs. the tuned backend scored by
eval/evaluate.py --backend presidio.

IMPORTANT: this is NOT the same comparison as --backend presidio vs
--backend fallback. The regex fallback (agents/phi_detection_agent.py,
USE_FALLBACK_ONLY) is a separate dependency-free detector, not "vanilla
Presidio" -- comparing tuned-Presidio to the regex fallback doesn't tell you
what the custom recognizers/deregistrations bought you, because the
fallback was never Presidio to begin with. To measure that, this script
builds a plain AnalyzerEngine() with:
  - no header_person_recognizer, mrn_recognizer, account_recognizer,
    accession_recognizer, claim_recognizer, health_plan_recognizer,
    fax_recognizer, or phone_ext_recognizer added
  - UsLicenseRecognizer and NhsRecognizer NOT removed (both ship enabled
    by default)
and reuses the exact same span-conflict resolution (_clip_at_newlines,
_dedupe_overlaps) and scoring (eval.evaluate's matching/PRF1 code) so the
only variable is the recognizer set itself.

USAGE
-----
    python -m eval.evaluate_vanilla_presidio
"""
from __future__ import annotations

import json
import os
import sys
from collections import defaultdict

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from presidio_analyzer import AnalyzerEngine

from agents.phi_detection_agent import _clip_at_newlines, _dedupe_overlaps
from eval.evaluate import DEFAULT_DATASET_DIR, THIS_DIR, _build_summary_rows, evaluate_document, print_report, write_outputs


def _detect_vanilla(analyzer: AnalyzerEngine, text: str) -> list[dict]:
    results = analyzer.analyze(text=text, language="en")
    spans = [
        {
            "start": r.start,
            "end": r.end,
            "text": text[r.start:r.end],
            "phi_type": r.entity_type,
            "confidence": r.score,
            "source_agent": "vanilla_presidio",
        }
        for r in results
    ]
    return _dedupe_overlaps(_clip_at_newlines(spans))


def run_vanilla_eval(dataset_dir: str) -> dict:
    gt_path = os.path.join(dataset_dir, "ground_truth.json")
    with open(gt_path, encoding="utf-8") as f:
        ground_truth = json.load(f)

    analyzer = AnalyzerEngine()  # bare, default recognizer registry -- no project customizations

    totals = {"strict": defaultdict(lambda: [0, 0, 0]), "overlap": defaultdict(lambda: [0, 0, 0])}
    per_doc_results = []
    docs_evaluated = 0
    docs_missing = []

    for filename, meta in sorted(ground_truth.items()):
        file_path = os.path.join(dataset_dir, filename)
        if not os.path.exists(file_path):
            docs_missing.append(filename)
            continue
        with open(file_path, encoding="utf-8") as f:
            raw_text = f.read()

        pred_spans = _detect_vanilla(analyzer, raw_text)
        gold_spans = meta["phi_spans"]

        doc_result = {"filename": filename, "doc_type": meta["doc_type"]}
        for mode in ("strict", "overlap"):
            per_type = evaluate_document(gold_spans, pred_spans, mode)
            for phi_type, (tp, fp, fn, _fp_s, _fn_s) in per_type.items():
                bucket = totals[mode][phi_type]
                bucket[0] += tp
                bucket[1] += fp
                bucket[2] += fn
            doc_result[mode] = {t: [tp, fp, fn] for t, (tp, fp, fn, _fp_s, _fn_s) in per_type.items()}
        per_doc_results.append(doc_result)
        docs_evaluated += 1

    return {
        "backend": "vanilla_presidio",
        "docs_evaluated": docs_evaluated,
        "docs_missing": docs_missing,
        "totals": totals,
        "per_doc_results": per_doc_results,
    }


def main():
    eval_result = run_vanilla_eval(DEFAULT_DATASET_DIR)
    print_report(eval_result)
    write_outputs(eval_result, os.path.join(THIS_DIR, "vanilla_presidio_run"))


if __name__ == "__main__":
    main()
