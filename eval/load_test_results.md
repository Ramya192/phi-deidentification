# API Load Test: POST /redact

Checkpointer backend: `sqlite`  |  Sample documents: 7  |  Classification/adjudication backends forced to `heuristic` (zero LLM calls)

| Concurrency | Requests | Throughput (docs/min) | p50 (ms) | p95 (ms) | p99 (ms) | Max (ms) | Success | Rate-limited (429) | Errors |
|---|---|---|---|---|---|---|---|---|---|
| 1 | 20 | 551.3 | 81 | 257 | 331 | 331 | 20 | 0 | 0 |
| 5 | 20 | 915.1 | 223 | 837 | 842 | 842 | 20 | 0 | 0 |
| 10 | 20 | 542.9 | 772 | 2031 | 2105 | 2105 | 20 | 0 | 0 |
| 20 | 20 | 448.0 | 2219 | 2593 | 2600 | 2600 | 20 | 0 | 0 |

Rate-limited (429) responses are the `slowapi` guardrail (100 requests/minute per caller, see "Guardrails" in README.md) engaging as designed, not a failure -- a single caller shouldn't be able to exceed that regardless of concurrency. A high error count (any status outside 200/429, or a raised exception) at a given concurrency level is the actual finding worth investigating -- e.g. `database is locked` errors would point at the SQLite checkpointer's single shared connection as the real ceiling under concurrent writes.
