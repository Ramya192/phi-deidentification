"""
eval/load_test.py

Lightweight throughput/latency load test for POST /redact, closing the
"No performance/load testing" gap in LECTURER_EVALUATION.md ("How fast is
redaction? Can it handle 100 docs/min?").

Not a locust/k6-style sustained-load harness -- this project isn't
deployed anywhere with real traffic to point one at, so standing up that
infrastructure would be effort with no real target. Instead: fire
increasing levels of concurrent POST /redact requests at the actual
FastAPI app in-process, using fastapi.testclient.TestClient plus a thread
pool (the same technique tests/test_api_batch.py already uses to
exercise the app without a live uvicorn server), and report real
p50/p95/p99 latency and throughput at each concurrency level, plus error
and rate-limited counts.

This exercises the real bottleneck the report already documents:
graph/workflow.py's SQLite checkpointer holds a single shared connection
(check_same_thread=False) across the whole process, and
storage/audit_store.py's SQLite backend has the same single-writer
shape -- both are called out in README.md as a "single-process deployment
ceiling." This script measures what that costs in practice at a handful
of concurrency levels, rather than leaving it as an assertion.

Zero LLM calls by design (classification/adjudication backends are
forced to "heuristic" below) -- concurrency numbers should reflect the
deterministic pipeline's own overhead, not OpenAI network latency, and a
load test shouldn't rack up API cost proportional to how hard it hammers
the server. PHI detection uses whatever backend is actually installed
(Presidio if available, else regex fallback) since that's what a real
deployment would run.

Writes real rows into data/checkpoints.sqlite, same as normal server
operation (tests/test_api_batch.py already does this without special
handling) -- but points the audit store at a scratch tmp file so repeated
load-test runs don't pile junk into data/audit.sqlite's real history.

USAGE
-----
    python -m eval.load_test
    python -m eval.load_test --concurrency 1,5,10,20 --requests-per-level 30

OUTPUT
------
    Console table (per concurrency level: throughput, p50/p95/p99 latency,
    success/error/rate-limited counts)
    eval/load_test_results.md
"""
from __future__ import annotations

import argparse
import importlib
import json
import math
import os
import sys
import tempfile
import time
from concurrent.futures import ThreadPoolExecutor

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from eval.evaluate import DEFAULT_DATASET_DIR, THIS_DIR


def _load_sample_texts(dataset_dir: str, per_type: int = 1) -> list[str]:
    """One (or a few) real document per doc_type, picked from
    ground_truth.json rather than a hardcoded filename list -- file
    numbering isn't consistent across types (e.g. discharge_summary
    starts at _01, radiology_report at _101), same reason
    eval/evaluate.py and eval/classification_eval.py read the manifest
    instead of guessing names."""
    gt_path = os.path.join(dataset_dir, "ground_truth.json")
    texts = []
    if os.path.exists(gt_path):
        with open(gt_path, encoding="utf-8") as f:
            ground_truth = json.load(f)
        seen_per_type: dict[str, int] = {}
        for filename, meta in sorted(ground_truth.items()):
            doc_type = meta.get("doc_type", "")
            if seen_per_type.get(doc_type, 0) >= per_type:
                continue
            path = os.path.join(dataset_dir, filename)
            if os.path.exists(path):
                with open(path, encoding="utf-8") as f:
                    texts.append(f.read())
                seen_per_type[doc_type] = seen_per_type.get(doc_type, 0) + 1
    if not texts:
        texts = ["Patient contact: john.carter@example.com or (555) 234-9981."]
    return texts


def _percentile(sorted_values: list[float], pct: float) -> float:
    if not sorted_values:
        return 0.0
    idx = min(len(sorted_values) - 1, math.ceil(pct / 100 * len(sorted_values)) - 1)
    return sorted_values[max(0, idx)]


def _run_level(client, headers, texts: list[str], concurrency: int, total_requests: int) -> dict:
    latencies_ms: list[float] = []
    status_counts: dict[int, int] = {}
    exceptions = 0

    def _one_request(i: int):
        text = texts[i % len(texts)]
        payload = {"text": text, "filename": f"load_test_{i}.txt"}
        t0 = time.perf_counter()
        try:
            resp = client.post("/redact", json=payload, headers=headers)
            latency_ms = (time.perf_counter() - t0) * 1000
            return latency_ms, resp.status_code
        except Exception:
            latency_ms = (time.perf_counter() - t0) * 1000
            return latency_ms, None

    wall_start = time.perf_counter()
    with ThreadPoolExecutor(max_workers=concurrency) as pool:
        for latency_ms, status_code in pool.map(_one_request, range(total_requests)):
            latencies_ms.append(latency_ms)
            if status_code is None:
                exceptions += 1
            else:
                status_counts[status_code] = status_counts.get(status_code, 0) + 1
    wall_s = time.perf_counter() - wall_start

    latencies_ms.sort()
    return {
        "concurrency": concurrency,
        "total_requests": total_requests,
        "wall_s": wall_s,
        "throughput_docs_per_min": (total_requests / wall_s) * 60 if wall_s > 0 else 0.0,
        "p50_ms": _percentile(latencies_ms, 50),
        "p95_ms": _percentile(latencies_ms, 95),
        "p99_ms": _percentile(latencies_ms, 99),
        "max_ms": max(latencies_ms) if latencies_ms else 0.0,
        "success_count": status_counts.get(200, 0),
        "rate_limited_count": status_counts.get(429, 0),
        "error_count": sum(c for code, c in status_counts.items() if code not in (200, 429)) + exceptions,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--concurrency", type=str, default="1,5,10,20",
                         help="Comma-separated concurrency levels to test in sequence.")
    parser.add_argument("--requests-per-level", type=int, default=20,
                         help="Total POST /redact requests fired at each concurrency level.")
    parser.add_argument("--dataset-dir", type=str, default=DEFAULT_DATASET_DIR)
    parser.add_argument("--output-dir", type=str, default=THIS_DIR)
    args = parser.parse_args()

    levels = [int(c.strip()) for c in args.concurrency.split(",") if c.strip()]
    texts = _load_sample_texts(args.dataset_dir)

    os.environ["PHI_DEID_API_KEY"] = "load-test-key"
    os.environ["PHI_DEID_CLASSIFICATION_BACKEND"] = "heuristic"
    os.environ["PHI_DEID_ADJUDICATION_BACKEND"] = "heuristic"
    scratch_audit_db = os.path.join(tempfile.gettempdir(), "phi_deid_load_test_audit.sqlite")
    os.environ["PHI_DEID_AUDIT_DB_URL"] = f"sqlite:///{scratch_audit_db}"

    import storage.audit_store as audit_store_module
    importlib.reload(audit_store_module)
    import api.main as api_main_module
    importlib.reload(api_main_module)

    from fastapi.testclient import TestClient

    client = TestClient(api_main_module.app)
    headers = {"X-API-Key": "load-test-key"}
    checkpointer_backend = api_main_module.get_checkpointer_backend()

    print(f"Checkpointer backend: {checkpointer_backend}")
    print(f"Sample documents: {len(texts)}\n")

    # Warm-up request, not timed or reported: the first call in a fresh
    # process can trigger one-time costs (Presidio/spaCy lazily
    # downloading/loading its model) that have nothing to do with steady
    # -state throughput and would otherwise swamp concurrency=1's numbers
    # with a single multi-second outlier.
    print("Warm-up request (not timed)...")
    warmup_resp = client.post("/redact", json={"text": texts[0], "filename": "warmup.txt"}, headers=headers)
    print(f"  warm-up status: {warmup_resp.status_code}\n")

    results = []
    for level in levels:
        print(f"Running concurrency={level}, requests={args.requests_per_level}...")
        result = _run_level(client, headers, texts, level, args.requests_per_level)
        results.append(result)
        print(
            f"  throughput={result['throughput_docs_per_min']:.1f} docs/min  "
            f"p50={result['p50_ms']:.0f}ms  p95={result['p95_ms']:.0f}ms  "
            f"success={result['success_count']}  rate_limited={result['rate_limited_count']}  "
            f"errors={result['error_count']}",
            flush=True,
        )

    lines = []
    lines.append("# API Load Test: POST /redact\n")
    lines.append(f"Checkpointer backend: `{checkpointer_backend}`  |  Sample documents: {len(texts)}  |  "
                  f"Classification/adjudication backends forced to `heuristic` (zero LLM calls)\n")
    lines.append("| Concurrency | Requests | Throughput (docs/min) | p50 (ms) | p95 (ms) | p99 (ms) | Max (ms) | Success | Rate-limited (429) | Errors |")
    lines.append("|---|---|---|---|---|---|---|---|---|---|")
    for r in results:
        lines.append(
            f"| {r['concurrency']} | {r['total_requests']} | {r['throughput_docs_per_min']:.1f} | "
            f"{r['p50_ms']:.0f} | {r['p95_ms']:.0f} | {r['p99_ms']:.0f} | {r['max_ms']:.0f} | "
            f"{r['success_count']} | {r['rate_limited_count']} | {r['error_count']} |"
        )
    lines.append(
        "\nRate-limited (429) responses are the `slowapi` guardrail (100 requests/minute per caller, "
        "see \"Guardrails\" in README.md) engaging as designed, not a failure -- a single caller "
        "shouldn't be able to exceed that regardless of concurrency. A high error count (any status "
        "outside 200/429, or a raised exception) at a given concurrency level is the actual finding "
        "worth investigating -- e.g. `database is locked` errors would point at the SQLite "
        "checkpointer's single shared connection as the real ceiling under concurrent writes."
    )

    report = "\n".join(lines)
    print("\n" + report)

    out_path = os.path.join(args.output_dir, "load_test_results.md")
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(report + "\n")
    print(f"\nWrote {out_path}")


if __name__ == "__main__":
    main()
