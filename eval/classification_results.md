# ClassificationAgent Evaluation: Heuristic vs LLM

Documents evaluated: 350

| Backend | Accuracy |
|---|---|
| heuristic | 0.8086 |
| llm | 0.9686 |
| **LLM minus heuristic (accuracy delta)** | **+0.1600** |

## Per-type precision / recall / F1

| Backend | doc_type | Precision | Recall | F1 |
|---|---|---|---|---|
| heuristic | clinical_note | 0.5283 | 0.5600 | 0.5437 |
| heuristic | discharge_summary | 1.0000 | 0.5400 | 0.7013 |
| heuristic | insurance_document | 0.9804 | 1.0000 | 0.9901 |
| heuristic | lab_report | 1.0000 | 1.0000 | 1.0000 |
| heuristic | pathology_report | 0.9074 | 0.9800 | 0.9423 |
| heuristic | radiology_report | 0.5814 | 1.0000 | 0.7353 |
| heuristic | referral_letter | 1.0000 | 0.5800 | 0.7342 |
| llm | clinical_note | 0.9773 | 0.8600 | 0.9149 |
| llm | discharge_summary | 0.9434 | 1.0000 | 0.9709 |
| llm | insurance_document | 0.9615 | 1.0000 | 0.9804 |
| llm | lab_report | 0.9804 | 1.0000 | 0.9901 |
| llm | pathology_report | 1.0000 | 1.0000 | 1.0000 |
| llm | radiology_report | 1.0000 | 0.9600 | 0.9796 |
| llm | referral_letter | 0.9231 | 0.9600 | 0.9412 |

## Confusion matrices (rows = gold, columns = predicted)


**heuristic**

| gold \ pred | clinical_note | discharge_summary | insurance_document | lab_report | pathology_report | radiology_report | referral_letter |
|---|---|---|---|---|---|---|---|
| clinical_note | 28 | 0 | 0 | 0 | 2 | 20 | 0 |
| discharge_summary | 11 | 27 | 0 | 0 | 2 | 10 | 0 |
| insurance_document | 0 | 0 | 50 | 0 | 0 | 0 | 0 |
| lab_report | 0 | 0 | 0 | 50 | 0 | 0 | 0 |
| pathology_report | 0 | 0 | 0 | 0 | 49 | 1 | 0 |
| radiology_report | 0 | 0 | 0 | 0 | 0 | 50 | 0 |
| referral_letter | 14 | 0 | 1 | 0 | 1 | 5 | 29 |

**llm**

| gold \ pred | clinical_note | discharge_summary | insurance_document | lab_report | pathology_report | radiology_report | referral_letter |
|---|---|---|---|---|---|---|---|
| clinical_note | 43 | 2 | 1 | 0 | 0 | 0 | 4 |
| discharge_summary | 0 | 50 | 0 | 0 | 0 | 0 | 0 |
| insurance_document | 0 | 0 | 50 | 0 | 0 | 0 | 0 |
| lab_report | 0 | 0 | 0 | 50 | 0 | 0 | 0 |
| pathology_report | 0 | 0 | 0 | 0 | 50 | 0 | 0 |
| radiology_report | 1 | 1 | 0 | 0 | 0 | 48 | 0 |
| referral_letter | 0 | 0 | 1 | 1 | 0 | 0 | 48 |

## Cost & Latency (LLM backend)

- Models: `gpt-4o-mini` (chat) + `text-embedding-3-small` (few-shot retrieval), 323359 total tokens, $0.039141 total approx. cost
- Approx. cost per document: $0.000112
- Approx. classification latency per document: 1587 ms
- Fallback-to-heuristic rate: 0/350 (0.00%) -- how often classify_with_llm() raised and classification_agent() silently used the heuristic result instead

Pricing is the approximate public rate table in `observability/llm_metrics.py`, not a billing-accurate figure -- see that module's docstring.

Note: this dataset (data/txt_format/ground_truth.json) contains only the 7 clinical document types actually injected by scripts/build_sample_dataset.py -- no `not_applicable` (non-health) examples, so neither backend's ability to correctly reject a non-clinical document is measured here.
