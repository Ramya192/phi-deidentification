"""
eval/ablation_study.py

Ablation study required by the original project brief ("Member 5 must run
HumanReviewAgent ON vs OFF and report the F1 delta... without it, the
human review node is just decoration"):

Runs eval/evaluate.py's harness twice against the same labeled dataset --
once scoring every detected span (human_review=True: high + low confidence
combined, i.e. what the system produces assuming HumanReviewAgent
correctly approves every genuine low-confidence PHI span it's shown), and
once scoring ONLY high_confidence_spans (human_review=False: what the
system would catch if low-confidence spans were never routed to a human
reviewer at all and were simply left un-redacted).

The gap between the two OVERALL numbers is what the human-in-the-loop
step is actually worth, measured, not asserted.

USAGE
-----
    python -m eval.ablation_study
    python -m eval.ablation_study --backend presidio

OUTPUT
------
    Console comparison table
    eval/ablation_results.md   the same table, written up for the report
"""
from __future__ import annotations

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from eval.evaluate import DEFAULT_DATASET_DIR, THIS_DIR, _build_summary_rows, run_eval


def _overall_row(rows: list[dict], mode: str) -> dict:
    return next(r for r in rows if r["mode"] == mode and r["phi_type"] == "OVERALL")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dataset-dir", type=str, default=DEFAULT_DATASET_DIR)
    parser.add_argument("--backend", type=str, default="auto", choices=["auto", "fallback", "presidio"])
    parser.add_argument("--output-dir", type=str, default=THIS_DIR)
    args = parser.parse_args()

    print("Running WITH human review (high + low confidence spans, perfect-reviewer assumption)...")
    on_result = run_eval(dataset_dir=args.dataset_dir, backend=args.backend, human_review=True)
    on_rows = _build_summary_rows(on_result["totals"])

    print("Running WITHOUT human review (high-confidence spans only)...")
    off_result = run_eval(dataset_dir=args.dataset_dir, backend=args.backend, human_review=False)
    off_rows = _build_summary_rows(off_result["totals"])

    lines = []
    lines.append("# Ablation Study: HumanReviewAgent ON vs OFF\n")
    lines.append(
        f"Backend: `{on_result['backend']}`  |  Documents evaluated: {on_result['docs_evaluated']}\n"
    )
    lines.append(
        "**ON** = every detected span (high + low confidence), i.e. what the pipeline produces "
        "assuming HumanReviewAgent correctly approves every genuine low-confidence PHI span it's "
        "shown -- a perfect-reviewer assumption, since this eval harness calls PHIDetectionAgent "
        "directly and doesn't simulate reviewer error.\n"
    )
    lines.append(
        "**OFF** = only high-confidence spans -- what the system would redact if low-confidence "
        "spans were never routed to a human at all and were simply left un-redacted (auto-approve "
        "nothing below the per-type threshold).\n"
    )
    lines.append("\n| Mode | Overall Precision | Overall Recall | Overall F1 |")
    lines.append("|---|---|---|---|")
    for mode_label, mode_key in (("STRICT", "strict"), ("OVERLAP", "overlap")):
        on_overall = _overall_row(on_rows, mode_key)
        off_overall = _overall_row(off_rows, mode_key)
        lines.append(
            f"| {mode_label} -- Human Review ON  | {on_overall['precision']:.4f} | "
            f"{on_overall['recall']:.4f} | {on_overall['f1']:.4f} |"
        )
        lines.append(
            f"| {mode_label} -- Human Review OFF | {off_overall['precision']:.4f} | "
            f"{off_overall['recall']:.4f} | {off_overall['f1']:.4f} |"
        )
        delta = on_overall["f1"] - off_overall["f1"]
        lines.append(f"| {mode_label} -- **F1 delta (ON minus OFF)** | | | **{delta:+.4f}** |")

    lines.append(
        "\nA positive F1 delta means the low-confidence spans HumanReviewAgent routes to a "
        "reviewer are, on net, real PHI that would otherwise leak through un-redacted -- i.e. the "
        "human-in-the-loop step is measurably load-bearing, not decorative. Per-type numbers in "
        "the console output above show which specific PHI types depend most on human review "
        "(expect PERSON and DATE_TIME, given their higher confidence thresholds -- see "
        "agents/phi_detection_agent.py CONFIDENCE_THRESHOLDS)."
    )

    report = "\n".join(lines)
    print("\n" + report)

    out_path = os.path.join(args.output_dir, "ablation_results.md")
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(report + "\n")
    print(f"\nWrote {out_path}")


if __name__ == "__main__":
    main()
