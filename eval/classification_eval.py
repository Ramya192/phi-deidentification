"""
eval/classification_eval.py

Evaluates ClassificationAgent's two backends (agents/classification_agent.py)
against the same labeled dataset eval/evaluate.py uses for PHI detection --
data/txt_format/ground_truth.json already carries the true doc_type for
every one of the 350 documents (7 types x 50 docs), it just wasn't scored
against until now. Closes the gap flagged in LECTURER_EVALUATION.md: "No
measured improvement shown vs. heuristic" for the LLM classification tier.

Runs classification_agent() itself (not the bare classify_with_heuristic/
classify_with_llm functions) for both backends, so this measures exactly
what production sees -- including classify_with_llm's automatic fallback
to the heuristic on any failure (missing key, network error, malformed
response). A doc where the LLM path silently fell back is counted under
"LLM (production)" using whatever the fallback produced, but also
tracked separately as a fallback-rate stat, since a backend that quietly
degrades to the other backend a meaningful fraction of the time is a
reliability finding in itself, not just an accuracy number.

Reports, per backend: overall accuracy, per-type precision/recall/F1
(treating each of the 8 doc_type labels as its own one-vs-rest class,
same tp/fp/fn convention eval/evaluate.py uses for PHI spans), a
confusion matrix, and -- for the LLM backend -- token cost and latency
per document (observability/llm_metrics.py) plus the fallback rate.

Needs OPENAI_API_KEY set for the LLM backend (real calls, no mocking --
see tests/test_integration.py for the mocked unit tests).

USAGE
-----
    python -m eval.classification_eval
    python -m eval.classification_eval --limit 30
    python -m eval.classification_eval --dataset-dir data/txt_format

OUTPUT
------
    Console comparison table + confusion matrix + cost/latency summary
    eval/classification_results.md
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from collections import defaultdict

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    from dotenv import load_dotenv

    load_dotenv()
except ImportError:
    pass

from agents.classification_agent import _ALL_TYPES, classification_agent
from eval.evaluate import DEFAULT_DATASET_DIR, THIS_DIR, _prf1
from observability.llm_metrics import summarize_usage

BACKENDS = ["heuristic", "llm"]


def _new_totals() -> dict:
    return {b: defaultdict(lambda: [0, 0, 0]) for b in BACKENDS}  # doc_type -> [tp, fp, fn]


def _score(totals: dict, gold: str, pred: str) -> None:
    if pred == gold:
        totals[gold][0] += 1  # tp
    else:
        totals[pred][1] += 1  # fp for whatever it wrongly guessed
        totals[gold][2] += 1  # fn for the type it missed


def _overall_accuracy(confusion_correct: int, total: int) -> float:
    return confusion_correct / total if total else 0.0


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dataset-dir", type=str, default=DEFAULT_DATASET_DIR)
    parser.add_argument(
        "--limit", type=int, default=None,
        help="Only evaluate the first N documents that have ground truth (for a fast/cheap smoke run).",
    )
    parser.add_argument("--output-dir", type=str, default=THIS_DIR)
    args = parser.parse_args()

    if not os.environ.get("OPENAI_API_KEY"):
        print("ERROR: OPENAI_API_KEY is not set -- the LLM classification backend needs a real key to run.", file=sys.stderr)
        sys.exit(1)

    gt_path = os.path.join(args.dataset_dir, "ground_truth.json")
    if not os.path.exists(gt_path):
        raise FileNotFoundError(
            f"No ground_truth.json found at {gt_path}. Run `python -m scripts.build_sample_dataset` first."
        )
    with open(gt_path, encoding="utf-8") as f:
        ground_truth = json.load(f)

    items = sorted(ground_truth.items())
    if args.limit:
        items = items[: args.limit]

    totals = _new_totals()
    correct = {b: 0 for b in BACKENDS}
    confusion: dict[str, dict[str, int]] = {b: defaultdict(lambda: defaultdict(int)) for b in BACKENDS}
    llm_usage_log: list[dict] = []
    llm_fallback_count = 0
    total_llm_latency_s = 0.0
    docs_evaluated = 0

    for filename, meta in items:
        file_path = os.path.join(args.dataset_dir, filename)
        if not os.path.exists(file_path):
            continue
        with open(file_path, encoding="utf-8") as f:
            raw_text = f.read()
        gold = meta["doc_type"]

        heuristic_state = classification_agent(
            {"raw_text": raw_text, "audit_log": [], "llm_usage_log": []}, backend="heuristic"
        )
        heuristic_pred = heuristic_state["doc_type"]

        t0 = time.perf_counter()
        llm_state = classification_agent(
            {"raw_text": raw_text, "audit_log": [], "llm_usage_log": []}, backend="llm"
        )
        total_llm_latency_s += time.perf_counter() - t0
        llm_pred = llm_state["doc_type"]
        llm_usage_log.extend(llm_state["llm_usage_log"])
        if "fallback" in llm_state["audit_log"][-1]["notes"]:
            llm_fallback_count += 1

        for backend, pred in (("heuristic", heuristic_pred), ("llm", llm_pred)):
            _score(totals[backend], gold, pred)
            confusion[backend][gold][pred] += 1
            if pred == gold:
                correct[backend] += 1

        docs_evaluated += 1
        print(
            f"  [{docs_evaluated}/{len(items)}] {filename}: gold={gold} "
            f"heuristic={heuristic_pred} llm={llm_pred}",
            flush=True,
        )

    usage_summary = summarize_usage(llm_usage_log)
    avg_latency_ms = (total_llm_latency_s / docs_evaluated) * 1000 if docs_evaluated else 0.0
    avg_cost_per_doc = usage_summary["total_approx_cost_usd"] / docs_evaluated if docs_evaluated else 0.0
    fallback_rate = llm_fallback_count / docs_evaluated if docs_evaluated else 0.0

    lines = []
    lines.append("# ClassificationAgent Evaluation: Heuristic vs LLM\n")
    lines.append(f"Documents evaluated: {docs_evaluated}\n")
    lines.append("| Backend | Accuracy |")
    lines.append("|---|---|")
    for backend in BACKENDS:
        acc = _overall_accuracy(correct[backend], docs_evaluated)
        lines.append(f"| {backend} | {acc:.4f} |")
    heuristic_acc = _overall_accuracy(correct["heuristic"], docs_evaluated)
    llm_acc = _overall_accuracy(correct["llm"], docs_evaluated)
    lines.append(f"| **LLM minus heuristic (accuracy delta)** | **{llm_acc - heuristic_acc:+.4f}** |")

    lines.append("\n## Per-type precision / recall / F1\n")
    lines.append("| Backend | doc_type | Precision | Recall | F1 |")
    lines.append("|---|---|---|---|---|")
    for backend in BACKENDS:
        for doc_type in sorted(_ALL_TYPES):
            tp, fp, fn = totals[backend][doc_type]
            if tp + fp + fn == 0:
                continue
            p, r, f1 = _prf1(tp, fp, fn)
            lines.append(f"| {backend} | {doc_type} | {p:.4f} | {r:.4f} | {f1:.4f} |")

    lines.append("\n## Confusion matrices (rows = gold, columns = predicted)\n")
    doc_types_present = sorted({meta["doc_type"] for _f, meta in items})
    for backend in BACKENDS:
        lines.append(f"\n**{backend}**\n")
        header = "| gold \\ pred | " + " | ".join(doc_types_present) + " |"
        lines.append(header)
        lines.append("|" + "---|" * (len(doc_types_present) + 1))
        for gold_type in doc_types_present:
            row = [str(confusion[backend][gold_type].get(pred_type, 0)) for pred_type in doc_types_present]
            lines.append(f"| {gold_type} | " + " | ".join(row) + " |")

    lines.append("\n## Cost & Latency (LLM backend)\n")
    lines.append(f"- Models: `gpt-4o-mini` (chat) + `text-embedding-3-small` (few-shot retrieval), "
                  f"{usage_summary['total_tokens']} total tokens, "
                  f"${usage_summary['total_approx_cost_usd']:.6f} total approx. cost")
    lines.append(f"- Approx. cost per document: ${avg_cost_per_doc:.6f}")
    lines.append(f"- Approx. classification latency per document: {avg_latency_ms:.0f} ms")
    lines.append(f"- Fallback-to-heuristic rate: {llm_fallback_count}/{docs_evaluated} ({fallback_rate:.2%}) "
                  f"-- how often classify_with_llm() raised and classification_agent() silently used the "
                  f"heuristic result instead")
    lines.append(
        "\nPricing is the approximate public rate table in `observability/llm_metrics.py`, "
        "not a billing-accurate figure -- see that module's docstring."
    )
    lines.append(
        "\nNote: this dataset (data/txt_format/ground_truth.json) contains only the 7 clinical "
        "document types actually injected by scripts/build_sample_dataset.py -- no `not_applicable` "
        "(non-health) examples, so neither backend's ability to correctly reject a non-clinical "
        "document is measured here."
    )

    report = "\n".join(lines)
    print("\n" + report)

    out_path = os.path.join(args.output_dir, "classification_results.md")
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(report + "\n")
    print(f"\nWrote {out_path}")


if __name__ == "__main__":
    main()
