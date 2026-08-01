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

Also reports a 95% confidence interval on the F1 delta itself, via
paired bootstrap resampling over documents (not spans -- resampling
spans would break the tp/fp/fn pairing within a document and inflate
apparent precision). Answers the "is a +0.0735 F1 delta significant or
just noise?" question directly, rather than reporting a single point
estimate with no sense of its variance.

USAGE
-----
    python -m eval.ablation_study
    python -m eval.ablation_study --backend presidio
    python -m eval.ablation_study --bootstrap-iterations 5000

OUTPUT
------
    Console comparison table
    eval/ablation_results.md   the same table, written up for the report
"""
from __future__ import annotations

import argparse
import os
import random
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from eval.evaluate import DEFAULT_DATASET_DIR, THIS_DIR, _build_summary_rows, _prf1, run_eval


def _overall_row(rows: list[dict], mode: str) -> dict:
    return next(r for r in rows if r["mode"] == mode and r["phi_type"] == "OVERALL")


def _doc_overall_counts(per_doc_results: list[dict], mode: str) -> dict[str, tuple[int, int, int]]:
    """filename -> (tp, fp, fn) summed across every phi_type for that one
    document, in the given mode. Per-document (not per-span) because the
    bootstrap below resamples whole documents -- a document's spans share
    whatever made that document easy or hard (doc_type, phrasing, injected
    field density), so resampling spans independently would treat
    correlated errors as independent and understate the true variance."""
    counts = {}
    for doc in per_doc_results:
        tp = fp = fn = 0
        for t_tp, t_fp, t_fn in doc[mode].values():
            tp += t_tp
            fp += t_fp
            fn += t_fn
        counts[doc["filename"]] = (tp, fp, fn)
    return counts


def _f1_of(counts: list[tuple[int, int, int]]) -> float:
    tp = sum(c[0] for c in counts)
    fp = sum(c[1] for c in counts)
    fn = sum(c[2] for c in counts)
    return _prf1(tp, fp, fn)[2]


def _bootstrap_f1_delta_ci(
    on_counts: dict[str, tuple[int, int, int]],
    off_counts: dict[str, tuple[int, int, int]],
    n_iterations: int = 2000,
    alpha: float = 0.05,
    seed: int = 42,
) -> tuple[float, float]:
    """Paired percentile bootstrap 95% CI (default alpha=0.05) for the
    ON-minus-OFF F1 delta. Paired because ON and OFF are the same
    documents scored two different ways, not two independent samples --
    each bootstrap draw resamples one set of document indices and applies
    it to both ON and OFF, preserving that pairing. If the returned
    interval excludes 0, the delta is significant at the 1-alpha
    confidence level; if it contains 0, the observed delta could be
    sampling noise. seed is fixed for a reproducible report, not tuned.
    """
    filenames = sorted(set(on_counts) & set(off_counts))
    n = len(filenames)
    on_list = [on_counts[f] for f in filenames]
    off_list = [off_counts[f] for f in filenames]

    rng = random.Random(seed)
    deltas = []
    for _ in range(n_iterations):
        idx = [rng.randrange(n) for _ in range(n)]
        resampled_on = [on_list[i] for i in idx]
        resampled_off = [off_list[i] for i in idx]
        deltas.append(_f1_of(resampled_on) - _f1_of(resampled_off))
    deltas.sort()
    lo_idx = int((alpha / 2) * n_iterations)
    hi_idx = min(n_iterations - 1, int((1 - alpha / 2) * n_iterations))
    return deltas[lo_idx], deltas[hi_idx]


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dataset-dir", type=str, default=DEFAULT_DATASET_DIR)
    parser.add_argument("--backend", type=str, default="auto", choices=["auto", "fallback", "presidio"])
    parser.add_argument("--output-dir", type=str, default=THIS_DIR)
    parser.add_argument("--bootstrap-iterations", type=int, default=2000,
                         help="Resamples for the F1 delta's 95%% confidence interval.")
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

    lines.append("\n## Statistical significance (paired bootstrap)\n")
    lines.append(
        f"{args.bootstrap_iterations} resamples, documents resampled with replacement (paired -- "
        "the same resampled document set is used for both ON and OFF each iteration), 95% "
        "percentile confidence interval, seed=42.\n"
    )
    lines.append("| Mode | F1 delta | 95% CI | Significant at p<0.05? |")
    lines.append("|---|---|---|---|")
    for mode_label, mode_key in (("STRICT", "strict"), ("OVERLAP", "overlap")):
        on_counts = _doc_overall_counts(on_result["per_doc_results"], mode_key)
        off_counts = _doc_overall_counts(off_result["per_doc_results"], mode_key)
        ci_lo, ci_hi = _bootstrap_f1_delta_ci(on_counts, off_counts, n_iterations=args.bootstrap_iterations)
        on_overall = _overall_row(on_rows, mode_key)
        off_overall = _overall_row(off_rows, mode_key)
        delta = on_overall["f1"] - off_overall["f1"]
        significant = ci_lo > 0 or ci_hi < 0
        lines.append(
            f"| {mode_label} | {delta:+.4f} | [{ci_lo:+.4f}, {ci_hi:+.4f}] | {'Yes' if significant else 'No'} |"
        )
    lines.append(
        "\nAn interval that excludes 0 means the F1 delta is unlikely to be sampling noise at the "
        "95% confidence level -- resampling which documents happened to be in the eval set still "
        "produces a delta with the same sign essentially every time. An interval that includes 0 "
        "means the observed delta could plausibly be noise from which 350 documents happened to be "
        "in this particular synthetic dataset, not a real effect."
    )

    report = "\n".join(lines)
    print("\n" + report)

    out_path = os.path.join(args.output_dir, "ablation_results.md")
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(report + "\n")
    print(f"\nWrote {out_path}")


if __name__ == "__main__":
    main()
