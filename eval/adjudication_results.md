# LLMAdjudicationAgent Evaluation: LLM vs Human Review

Backend: `presidio`  |  Documents evaluated: 350  |  Documents with low-confidence spans: 285  |  Low-confidence spans adjudicated: 426

- **NO_REVIEW (high-confidence only, nothing else redacted)**
- **PERFECT_HUMAN (all low-confidence spans assumed correctly approved)**
- **LLM_ONLY (LLM-confirmed spans only, deferred spans left un-redacted)**
- **LLM_PLUS_HUMAN (LLM-confirmed + deferred spans assumed caught by a human)**

| Mode | Config | Precision | Recall | F1 |
|---|---|---|---|---|
| STRICT | no_review | 0.1291 | 0.3591 | 0.1899 |
| STRICT | perfect_human | 0.1650 | 0.4916 | 0.2471 |
| STRICT | llm_only | 0.1657 | 0.4916 | 0.2478 |
| STRICT | llm_plus_human | 0.1656 | 0.4916 | 0.2477 |
| STRICT | **LLM_ONLY minus NO_REVIEW (F1 delta)** | | | **+0.0579** |
| STRICT | **LLM_ONLY minus PERFECT_HUMAN (F1 gap to upper bound)** | | | **+0.0007** |
| STRICT | **LLM_PLUS_HUMAN minus PERFECT_HUMAN (F1 delta)** | | | **+0.0006** |
| OVERLAP | no_review | 0.3101 | 0.8623 | 0.4561 |
| OVERLAP | perfect_human | 0.3343 | 0.9958 | 0.5005 |
| OVERLAP | llm_only | 0.3356 | 0.9958 | 0.5020 |
| OVERLAP | llm_plus_human | 0.3354 | 0.9958 | 0.5018 |
| OVERLAP | **LLM_ONLY minus NO_REVIEW (F1 delta)** | | | **+0.0458** |
| OVERLAP | **LLM_ONLY minus PERFECT_HUMAN (F1 gap to upper bound)** | | | **+0.0015** |
| OVERLAP | **LLM_PLUS_HUMAN minus PERFECT_HUMAN (F1 delta)** | | | **+0.0012** |

## Cost & Latency

- Model: `gpt-4o-mini`, 407464 total tokens, $0.089709 total approx. cost
- Approx. cost per document: $0.000256
- Approx. adjudication latency per document (with low-confidence spans): 4040 ms

Pricing is the approximate public rate table in `observability/llm_metrics.py`, not a billing-accurate figure -- see that module's docstring.

A positive LLM_ONLY-minus-NO_REVIEW F1 delta means the LLM adjudicator recovers real PHI on its own, with no human involved, that the no-review baseline would leak. The gap to PERFECT_HUMAN shows how much recall/precision the LLM tier leaves on the table versus an (unrealistic) perfect human reviewer. LLM_PLUS_HUMAN close to PERFECT_HUMAN means routing the LLM's deferred spans to a human recovers most of that gap, which is the actual production configuration this agent is designed for -- the LLM cuts reviewer workload without giving up the upper-bound accuracy a human-only pipeline gets.
