"""
eval/adjudication_eval.py

Evaluates LLMAdjudicationAgent (agents/llm_adjudication_agent.py) against
the same labeled dataset eval/evaluate.py and eval/ablation_study.py use.
Closes the gap the capstone evaluation flagged: LLMAdjudicationAgent is
"optional and unevaluated ... measure: does LLM adjudication catch more
PHI than human review would? What's the cost per document?"

Runs PHIDetectionAgent once per document (as evaluate.py does), then
scores four span sets against ground truth:

  NO_REVIEW      high_confidence_spans only -- baseline, nothing routed
                 to a human or an LLM (eval/evaluate.py --no-human-review).
  PERFECT_HUMAN  high + low confidence spans, assuming a human reviewer
                 correctly approves every genuine low-confidence span
                 (eval/ablation_study.py's "ON") -- the upper bound.
  LLM_ONLY       high_confidence_spans + whatever LLMAdjudicationAgent
                 itself confirms as PHI from the low-confidence set;
                 spans it rejects or can't resolve confidently (deferred)
                 are left un-redacted. Shows what the LLM tier achieves
                 completely standalone, with no human backstop.
  LLM_PLUS_HUMAN high_confidence_spans + LLM-confirmed spans + whatever
                 LLMAdjudicationAgent still defers, assuming (same as
                 PERFECT_HUMAN) a human downstream catches every genuine
                 deferred span. This is the actual production
                 configuration: the LLM resolves the easy low-confidence
                 cases, a human only sees what's left.

Also reports total LLM token usage/cost (observability/llm_metrics) and
wall-clock adjudication latency per document, since "what's the cost per
document" was explicitly asked for.

Needs OPENAI_API_KEY set -- this makes real calls to gpt-4o-mini, no
mocking (that's what the unit tests in tests/ are for).

USAGE
-----
    python -m eval.adjudication_eval
    python -m eval.adjudication_eval --limit 30
    python -m eval.adjudication_eval --dataset-dir data/txt_format --backend fallback

OUTPUT
------
    Console comparison table + cost/latency summary
    eval/adjudication_results.md
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
    # Same local-dev convenience as api/main.py -- picks up OPENAI_API_KEY
    # from a local .env file if present; no-op if python-dotenv isn't
    # installed or there's no .env file.
    from dotenv import load_dotenv

    load_dotenv()
except ImportError:
    pass

from agents import phi_detection_agent as phi_detection_module
from agents.llm_adjudication_agent import llm_adjudication_agent
from agents.phi_detection_agent import phi_detection_agent
from eval.evaluate import DEFAULT_DATASET_DIR, THIS_DIR, _prf1, evaluate_document
from observability.llm_metrics import summarize_usage

CONFIGS = ["no_review", "perfect_human", "llm_only", "llm_plus_human"]
CONFIG_LABELS = {
    "no_review": "NO_REVIEW (high-confidence only, nothing else redacted)",
    "perfect_human": "PERFECT_HUMAN (all low-confidence spans assumed correctly approved)",
    "llm_only": "LLM_ONLY (LLM-confirmed spans only, deferred spans left un-redacted)",
    "llm_plus_human": "LLM_PLUS_HUMAN (LLM-confirmed + deferred spans assumed caught by a human)",
}


def _new_totals() -> dict:
    return {c: {"strict": defaultdict(lambda: [0, 0, 0]), "overlap": defaultdict(lambda: [0, 0, 0])} for c in CONFIGS}


def _score(bucket: dict, gold_spans: list[dict], pred_spans: list[dict]) -> None:
    for mode in ("strict", "overlap"):
        per_type = evaluate_document(gold_spans, pred_spans, mode)
        for phi_type, (tp, fp, fn, _fp_s, _fn_s) in per_type.items():
            b = bucket[mode][phi_type]
            b[0] += tp
            b[1] += fp
            b[2] += fn


def _overall(bucket: dict, mode: str) -> tuple[float, float, float]:
    tp = fp = fn = 0
    for t_tp, t_fp, t_fn in bucket[mode].values():
        tp += t_tp
        fp += t_fp
        fn += t_fn
    return _prf1(tp, fp, fn)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dataset-dir", type=str, default=DEFAULT_DATASET_DIR)
    parser.add_argument("--backend", type=str, default="auto", choices=["auto", "fallback", "presidio"])
    parser.add_argument(
        "--limit", type=int, default=None,
        help="Only evaluate the first N documents that have ground truth (for a fast/cheap smoke run).",
    )
    parser.add_argument("--output-dir", type=str, default=THIS_DIR)
    args = parser.parse_args()

    if not os.environ.get("OPENAI_API_KEY"):
        print("ERROR: OPENAI_API_KEY is not set -- LLMAdjudicationAgent needs a real key to run.", file=sys.stderr)
        sys.exit(1)
    os.environ["PHI_DEID_ADJUDICATION_BACKEND"] = "llm"

    gt_path = os.path.join(args.dataset_dir, "ground_truth.json")
    if not os.path.exists(gt_path):
        raise FileNotFoundError(
            f"No ground_truth.json found at {gt_path}. Run `python -m scripts.build_sample_dataset` first."
        )
    with open(gt_path, encoding="utf-8") as f:
        ground_truth = json.load(f)

    original_fallback_flag = phi_detection_module.USE_FALLBACK_ONLY
    if args.backend == "fallback":
        phi_detection_module.USE_FALLBACK_ONLY = True
    elif args.backend == "presidio":
        if not phi_detection_module._PRESIDIO_AVAILABLE:
            raise RuntimeError(
                "backend=presidio requested but Presidio/spaCy aren't available in this environment."
            )
        phi_detection_module.USE_FALLBACK_ONLY = False
    actual_backend = (
        "presidio" if (phi_detection_module._PRESIDIO_AVAILABLE and not phi_detection_module.USE_FALLBACK_ONLY)
        else "regex_fallback"
    )

    items = sorted(ground_truth.items())
    if args.limit:
        items = items[: args.limit]

    totals = _new_totals()
    llm_usage_log: list[dict] = []
    total_adjudication_latency_s = 0.0
    docs_evaluated = 0
    docs_with_low_conf = 0
    spans_adjudicated = 0

    try:
        for filename, meta in items:
            file_path = os.path.join(args.dataset_dir, filename)
            if not os.path.exists(file_path):
                continue
            with open(file_path, encoding="utf-8") as f:
                raw_text = f.read()
            gold_spans = meta["phi_spans"]

            det_result = phi_detection_agent({"raw_text": raw_text, "retry_count": 0, "audit_log": []})
            high = det_result["high_confidence_spans"]
            low = det_result["low_confidence_spans"]

            _score(totals["no_review"], gold_spans, high)
            _score(totals["perfect_human"], gold_spans, high + low)

            if low:
                docs_with_low_conf += 1
                spans_adjudicated += len(low)
                adj_state = {
                    "raw_text": raw_text,
                    "retry_count": 0,
                    "audit_log": [],
                    "low_confidence_spans": low,
                    "llm_reviewed_spans": [],
                    "rejected_spans": [],
                    "llm_usage_log": [],
                }
                t0 = time.perf_counter()
                adj_result = llm_adjudication_agent(adj_state)
                total_adjudication_latency_s += time.perf_counter() - t0

                llm_confirmed = adj_result["llm_reviewed_spans"]
                still_low = adj_result["low_confidence_spans"]
                llm_usage_log.extend(adj_result["llm_usage_log"])
            else:
                llm_confirmed, still_low = [], []

            _score(totals["llm_only"], gold_spans, high + llm_confirmed)
            _score(totals["llm_plus_human"], gold_spans, high + llm_confirmed + still_low)

            docs_evaluated += 1
            print(
                f"  [{docs_evaluated}/{len(items)}] {filename}: "
                f"{len(low)} low-confidence spans -> {len(llm_confirmed)} LLM-confirmed, "
                f"{len(still_low)} deferred",
                flush=True,
            )
    finally:
        phi_detection_module.USE_FALLBACK_ONLY = original_fallback_flag

    usage_summary = summarize_usage(llm_usage_log)
    avg_latency_ms = (
        (total_adjudication_latency_s / docs_with_low_conf) * 1000 if docs_with_low_conf else 0.0
    )
    avg_cost_per_doc = (
        usage_summary["total_approx_cost_usd"] / docs_evaluated if docs_evaluated else 0.0
    )

    lines = []
    lines.append("# LLMAdjudicationAgent Evaluation: LLM vs Human Review\n")
    lines.append(
        f"Backend: `{actual_backend}`  |  Documents evaluated: {docs_evaluated}  |  "
        f"Documents with low-confidence spans: {docs_with_low_conf}  |  "
        f"Low-confidence spans adjudicated: {spans_adjudicated}\n"
    )
    for label in CONFIG_LABELS.values():
        lines.append(f"- **{label}**")
    lines.append("")
    lines.append("| Mode | Config | Precision | Recall | F1 |")
    lines.append("|---|---|---|---|---|")
    for mode_label, mode_key in (("STRICT", "strict"), ("OVERLAP", "overlap")):
        for config in CONFIGS:
            p, r, f1 = _overall(totals[config], mode_key)
            lines.append(f"| {mode_label} | {config} | {p:.4f} | {r:.4f} | {f1:.4f} |")
        llm_only_f1 = _overall(totals["llm_only"], mode_key)[2]
        no_review_f1 = _overall(totals["no_review"], mode_key)[2]
        perfect_human_f1 = _overall(totals["perfect_human"], mode_key)[2]
        llm_plus_human_f1 = _overall(totals["llm_plus_human"], mode_key)[2]
        lines.append(
            f"| {mode_label} | **LLM_ONLY minus NO_REVIEW (F1 delta)** | | | "
            f"**{llm_only_f1 - no_review_f1:+.4f}** |"
        )
        lines.append(
            f"| {mode_label} | **LLM_ONLY minus PERFECT_HUMAN (F1 gap to upper bound)** | | | "
            f"**{llm_only_f1 - perfect_human_f1:+.4f}** |"
        )
        lines.append(
            f"| {mode_label} | **LLM_PLUS_HUMAN minus PERFECT_HUMAN (F1 delta)** | | | "
            f"**{llm_plus_human_f1 - perfect_human_f1:+.4f}** |"
        )

    lines.append("\n## Cost & Latency\n")
    lines.append(f"- Model: `gpt-4o-mini`, {usage_summary['total_tokens']} total tokens, "
                  f"${usage_summary['total_approx_cost_usd']:.6f} total approx. cost")
    lines.append(f"- Approx. cost per document: ${avg_cost_per_doc:.6f}")
    lines.append(f"- Approx. adjudication latency per document (with low-confidence spans): {avg_latency_ms:.0f} ms")
    lines.append(
        "\nPricing is the approximate public rate table in `observability/llm_metrics.py`, "
        "not a billing-accurate figure -- see that module's docstring."
    )
    lines.append(
        "\nA positive LLM_ONLY-minus-NO_REVIEW F1 delta means the LLM adjudicator recovers real "
        "PHI on its own, with no human involved, that the no-review baseline would leak. The gap "
        "to PERFECT_HUMAN shows how much recall/precision the LLM tier leaves on the table versus "
        "an (unrealistic) perfect human reviewer. LLM_PLUS_HUMAN close to PERFECT_HUMAN means "
        "routing the LLM's deferred spans to a human recovers most of that gap, which is the "
        "actual production configuration this agent is designed for -- the LLM cuts reviewer "
        "workload without giving up the upper-bound accuracy a human-only pipeline gets."
    )

    report = "\n".join(lines)
    print("\n" + report)

    out_path = os.path.join(args.output_dir, "adjudication_results.md")
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(report + "\n")
    print(f"\nWrote {out_path}")


if __name__ == "__main__":
    main()
