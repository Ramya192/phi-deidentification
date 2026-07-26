# Ablation Study: HumanReviewAgent ON vs OFF

Backend: `presidio`  |  Documents evaluated: 350

**ON** = every detected span (high + low confidence), i.e. what the pipeline produces assuming HumanReviewAgent correctly approves every genuine low-confidence PHI span it's shown -- a perfect-reviewer assumption, since this eval harness calls PHIDetectionAgent directly and doesn't simulate reviewer error.

**OFF** = only high-confidence spans -- what the system would redact if low-confidence spans were never routed to a human at all and were simply left un-redacted (auto-approve nothing below the per-type threshold).


| Mode | Overall Precision | Overall Recall | Overall F1 |
|---|---|---|---|
| STRICT -- Human Review ON  | 0.3678 | 0.7423 | 0.4919 |
| STRICT -- Human Review OFF | 0.3509 | 0.6074 | 0.4448 |
| STRICT -- **F1 delta (ON minus OFF)** | | | **+0.0471** |
| OVERLAP -- Human Review ON  | 0.4934 | 0.9958 | 0.6599 |
| OVERLAP -- Human Review OFF | 0.4973 | 0.8609 | 0.6304 |
| OVERLAP -- **F1 delta (ON minus OFF)** | | | **+0.0295** |

A positive F1 delta means the low-confidence spans HumanReviewAgent routes to a reviewer are, on net, real PHI that would otherwise leak through un-redacted -- i.e. the human-in-the-loop step is measurably load-bearing, not decorative. Per-type numbers in the console output above show which specific PHI types depend most on human review (expect PERSON and DATE_TIME, given their higher confidence thresholds -- see agents/phi_detection_agent.py CONFIDENCE_THRESHOLDS).
