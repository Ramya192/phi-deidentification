# Ablation Study: HumanReviewAgent ON vs OFF

Backend: `presidio`  |  Documents evaluated: 350

**ON** = every detected span (high + low confidence), i.e. what the pipeline produces assuming HumanReviewAgent correctly approves every genuine low-confidence PHI span it's shown -- a perfect-reviewer assumption, since this eval harness calls PHIDetectionAgent directly and doesn't simulate reviewer error.

**OFF** = only high-confidence spans -- what the system would redact if low-confidence spans were never routed to a human at all and were simply left un-redacted (auto-approve nothing below the per-type threshold).


| Mode | Overall Precision | Overall Recall | Overall F1 |
|---|---|---|---|
| STRICT -- Human Review ON  | 0.2564 | 0.4963 | 0.3381 |
| STRICT -- Human Review OFF | 0.2086 | 0.3614 | 0.2646 |
| STRICT -- **F1 delta (ON minus OFF)** | | | **+0.0735** |
| OVERLAP -- Human Review ON  | 0.5167 | 1.0000 | 0.6814 |
| OVERLAP -- Human Review OFF | 0.4989 | 0.8642 | 0.6326 |
| OVERLAP -- **F1 delta (ON minus OFF)** | | | **+0.0488** |

A positive F1 delta means the low-confidence spans HumanReviewAgent routes to a reviewer are, on net, real PHI that would otherwise leak through un-redacted -- i.e. the human-in-the-loop step is measurably load-bearing, not decorative. Per-type numbers in the console output above show which specific PHI types depend most on human review (expect PERSON and DATE_TIME, given their higher confidence thresholds -- see agents/phi_detection_agent.py CONFIDENCE_THRESHOLDS).
