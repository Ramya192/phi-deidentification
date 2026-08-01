# Ablation Study: HumanReviewAgent ON vs OFF

Backend: `presidio`  |  Documents evaluated: 350

**ON** = every detected span (high + low confidence), i.e. what the pipeline produces assuming HumanReviewAgent correctly approves every genuine low-confidence PHI span it's shown -- a perfect-reviewer assumption, since this eval harness calls PHIDetectionAgent directly and doesn't simulate reviewer error.

**OFF** = only high-confidence spans -- what the system would redact if low-confidence spans were never routed to a human at all and were simply left un-redacted (auto-approve nothing below the per-type threshold).


| Mode | Overall Precision | Overall Recall | Overall F1 |
|---|---|---|---|
| STRICT -- Human Review ON  | 0.2433 | 0.4916 | 0.3255 |
| STRICT -- Human Review OFF | 0.2032 | 0.3726 | 0.2630 |
| STRICT -- **F1 delta (ON minus OFF)** | | | **+0.0625** |
| OVERLAP -- Human Review ON  | 0.4923 | 0.9949 | 0.6587 |
| OVERLAP -- Human Review OFF | 0.4773 | 0.8749 | 0.6176 |
| OVERLAP -- **F1 delta (ON minus OFF)** | | | **+0.0411** |

A positive F1 delta means the low-confidence spans HumanReviewAgent routes to a reviewer are, on net, real PHI that would otherwise leak through un-redacted -- i.e. the human-in-the-loop step is measurably load-bearing, not decorative. Per-type numbers in the console output above show which specific PHI types depend most on human review (expect PERSON and DATE_TIME, given their higher confidence thresholds -- see agents/phi_detection_agent.py CONFIDENCE_THRESHOLDS).

## Statistical significance (paired bootstrap)

2000 resamples, documents resampled with replacement (paired -- the same resampled document set is used for both ON and OFF each iteration), 95% percentile confidence interval, seed=42.

| Mode | F1 delta | 95% CI | Significant at p<0.05? |
|---|---|---|---|
| STRICT | +0.0625 | [+0.0566, +0.0685] | Yes |
| OVERLAP | +0.0411 | [+0.0345, +0.0476] | Yes |

An interval that excludes 0 means the F1 delta is unlikely to be sampling noise at the 95% confidence level -- resampling which documents happened to be in the eval set still produces a delta with the same sign essentially every time. An interval that includes 0 means the observed delta could plausibly be noise from which 350 documents happened to be in this particular synthetic dataset, not a real effect.
