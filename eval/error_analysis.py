"""
eval/error_analysis.py

Pulls real false positives and false negatives out of an eval run (OVERLAP
mode -- see evaluate.py's docstring for why STRICT-mode boundary
differences aren't an interesting failure story on their own) and writes
them up with a root-cause explanation for each, rather than just reporting
aggregate precision/recall numbers.

Root-cause explanations are drafted automatically from a small set of
pattern rules based on this project's own documented, previously-confirmed
failure modes (see README "Evaluation" and "Real Bugs Found & Fixed"
sections) -- e.g. Presidio's built-in LOCATION/NRP recognizers firing on
real entities in borrowed mtsamples narrative text that the synthetic
ground truth never labels (a known, documented measurement-methodology
gap, not a detector bug). Each entry is labeled with which rule fired so
you can sanity-check or hand-edit any explanation that doesn't actually
fit once you see the real output.

USAGE
-----
    python -m eval.error_analysis
    python -m eval.error_analysis --backend presidio --n 5

OUTPUT
------
    Console summary
    eval/error_analysis.md
"""
from __future__ import annotations

import argparse
import os
import random
import re
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from eval.evaluate import DEFAULT_DATASET_DIR, THIS_DIR, run_eval

_NON_GROUND_TRUTH_TYPES = {"LOCATION", "NRP", "US_DRIVER_LICENSE", "URL", "IP_ADDRESS"}

# Rough "does this matched text actually look like its claimed type"
# sanity checks. These exist because a first pass of this script (run
# against real output) produced confidently-wrong explanations for cases
# where the type label didn't match the text at all -- e.g. PERSON
# matching "6/14/2009" (a date) and PHONE_NUMBER matching "168.4" (a lab
# value). Guessing root cause from phi_type alone, without checking
# whether the matched text is even plausible for that type, was the bug
# in the first version of this script -- not just a gap in the pattern
# rules below.
_DATE_SHAPE = re.compile(r"^\d{1,2}[/-]\d{1,2}[/-]\d{2,4}$")
_PHONE_SHAPE = re.compile(r"^\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}(?:\s?(?:ext\.?|x)\.?\s?\d{2,6})?$")
_NAME_SHAPE = re.compile(r"^[A-Z][a-z]+(?:\s+[A-Z][a-z]+)+$")


def _explain_fp(item: dict) -> str:
    phi_type = item["phi_type"]
    text = item.get("text", "")

    if phi_type == "PERSON" and _DATE_SHAPE.match(text.strip()):
        return (
            f"MISMATCH: matched text `{text}` looks like a date, not a name. This is not the "
            "\"noisy NER\" story -- something mislabeled a date-shaped span as PERSON, or an "
            "overlap/dedup artifact assigned the wrong type to this position. Worth a targeted "
            "look at this specific document/offset rather than assuming ordinary NER noise; see "
            "the context field above for the surrounding text."
        )
    if phi_type == "PHONE_NUMBER" and not _PHONE_SHAPE.match(text.strip()):
        return (
            f"MISMATCH: matched text `{text}` doesn't have phone-number shape. Check the context "
            "above -- this looks like a case where a lab value, vital sign, or other bare number "
            "sequence got picked up by the phone recognizer rather than a genuine phone-number "
            "false positive."
        )
    if phi_type in _NON_GROUND_TRUTH_TYPES:
        return (
            f"Likely a real {phi_type} entity in borrowed mtsamples narrative text that the "
            "synthetic ground truth never labels (only the injected header fields are labeled) -- "
            "a documented measurement-methodology gap (README Section 5.2 / report Section 5.2), "
            "not necessarily an incorrect detection. Confirm by reading the context above."
        )
    if phi_type == "DATE_TIME":
        return (
            "Plausible: DATE_TIME has real false-positive noise from dates appearing in clinical "
            "narrative body text (procedure dates, historical dates in free text) that aren't part "
            "of the injected ground truth. Confirm against the context above -- if the matched "
            "text isn't actually a date, this explanation doesn't apply; re-diagnose from context."
        )
    if phi_type == "PERSON":
        if _NAME_SHAPE.match(text.strip()):
            return (
                "Matched text has name shape (capitalized multi-word), consistent with PERSON's "
                "known 0.58 OVERLAP precision -- likely a non-patient name (clinician, "
                "institution, drug/eponym) picked up by NER without ground-truth coverage for "
                "that specific mention. Confirm against context above."
            )
        return f"MISMATCH: matched text `{text}` doesn't look like a typical name -- inspect the context above to determine what actually happened here."
    return f"No reliable pattern rule for {phi_type} -- read the context field above to determine root cause."


def _explain_fn(item: dict, backend: str) -> str:
    phi_type = item["phi_type"]
    text = item.get("text", "")

    if phi_type == "PHONE_NUMBER" and _PHONE_SHAPE.match(text.strip()) and "ext" not in text.lower() and "x" not in text.lower():
        return (
            f"UNEXPECTED: gold text `{text}` is a standard-format phone number with no extension "
            "-- this should have matched both the fallback regex and Presidio's phone recognizer "
            "without issue. This is NOT the extension-leak bug (README Section 6.2), which is "
            "already fixed. Worth investigating directly: check whether an overlapping "
            "higher-confidence span claimed this position in _dedupe_overlaps, or whether this "
            "specific document's context (see above) has unusual surrounding punctuation/spacing."
        )
    if backend == "regex_fallback" and phi_type == "PERSON":
        return (
            "Expected on the fallback backend: its PERSON pattern requires a title prefix "
            "(Mr./Mrs./Dr./Pt.), so a bare injected name like \"Patient Name: Jordan Ellis\" with "
            "no title is a guaranteed miss -- see README's fallback PERSON note. Presidio's "
            "NER-based PERSON detection doesn't need a title and should catch this case."
        )
    if backend == "presidio" and phi_type == "PERSON":
        return (
            f"Presidio backend missed `{text}` despite PERSON recall measuring ~0.99 overall in "
            "this project's eval history -- plausible for an uncommon/rare first name spaCy's "
            "general-purpose NER model doesn't recognize as a person entity, but check the "
            "context above for anything unusual (unexpected punctuation, all-caps, embedded in a "
            "longer label) before concluding it's just an NER model limitation."
        )
    if phi_type in ("ACCESSION_NUMBER", "ACCOUNT_NUMBER", "CLAIM_NUMBER"):
        return (
            "These three types previously had 0% recall entirely (README Section 4.2) before "
            "custom recognizers/regex fixes were added -- if this still shows up as an FN, check "
            "whether the label text in this specific document varies from the "
            "\"<Label> Number: <value>\" pattern the fix assumes (e.g. abbreviated or reordered "
            "label wording)."
        )
    if phi_type == "PHONE_NUMBER":
        return (
            "Check the context above for an extension suffix or unusual formatting -- the "
            "extension-leak bug (README Section 6.2) was fixed for the common cases, but a "
            "sufficiently unusual phone format could still slip past both the fallback regex and "
            "Presidio's built-in recognizer."
        )
    return f"No reliable pattern rule for {phi_type} -- read the context field above to determine why this span wasn't caught."


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dataset-dir", type=str, default=DEFAULT_DATASET_DIR)
    parser.add_argument("--backend", type=str, default="auto", choices=["auto", "fallback", "presidio"])
    parser.add_argument("--output-dir", type=str, default=THIS_DIR)
    parser.add_argument("--n", type=int, default=5, help="How many FPs and FNs to sample (default 5 each)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for sampling, for reproducibility")
    args = parser.parse_args()

    result = run_eval(dataset_dir=args.dataset_dir, backend=args.backend, human_review=True)
    backend = result["backend"]

    all_fp, all_fn = [], []
    for doc in result["per_doc_results"]:
        for item in doc.get("fp_spans", []):
            all_fp.append({**item, "filename": doc["filename"]})
        for item in doc.get("fn_spans", []):
            all_fn.append({**item, "filename": doc["filename"]})

    rng = random.Random(args.seed)
    # Prefer spreading across distinct phi_types rather than n random picks
    # that could all land on the single most common error type.
    def _diverse_sample(items: list[dict], n: int) -> list[dict]:
        by_type: dict[str, list[dict]] = {}
        for item in items:
            by_type.setdefault(item["phi_type"], []).append(item)
        for bucket in by_type.values():
            rng.shuffle(bucket)
        types_cycle = list(by_type.keys())
        rng.shuffle(types_cycle)
        picked, i = [], 0
        while len(picked) < n and any(by_type.values()):
            t = types_cycle[i % len(types_cycle)]
            if by_type[t]:
                picked.append(by_type[t].pop())
            i += 1
            if i > 10 * n and not any(by_type.values()):
                break
        return picked

    fp_sample = _diverse_sample(all_fp, args.n)
    fn_sample = _diverse_sample(all_fn, args.n)

    lines = []
    lines.append("# Error Analysis: False Positives and False Negatives\n")
    lines.append(f"Backend: `{backend}`  |  Documents evaluated: {result['docs_evaluated']}  |  "
                  f"Total FPs (OVERLAP): {len(all_fp)}  |  Total FNs (OVERLAP): {len(all_fn)}\n")
    lines.append(
        "Root-cause explanations below are drafted from this project's own documented failure "
        "patterns (see comments in this file). Verify against the actual backend/document before "
        "citing in the final report -- these are a starting draft, not a substitute for reading "
        "the real output.\n"
    )

    lines.append("## False Positives\n")
    if not fp_sample:
        lines.append(
            f"No false positives found -- {len(all_fp)} total across {result['docs_evaluated']} documents.\n"
        )
    for i, item in enumerate(fp_sample, 1):
        lines.append(f"**FP {i}.** `{item['phi_type']}` matched text: `{item.get('text', '(n/a)')}` "
                      f"in `{item['filename']}`\n")
        lines.append(f"- Context: `...{item.get('context', '(n/a)')}...` (matched span shown as `[[...]]`)\n")
        lines.append(f"- Root cause: {_explain_fp(item)}\n")

    lines.append("## False Negatives\n")
    if not fn_sample:
        # Not a rendering gap -- a genuinely empty section here means
        # all_fn was empty, i.e. OVERLAP recall was 1.0 across the whole
        # eval run (every gold-labeled span was matched by at least one
        # detection). Worth stating outright rather than leaving a bare
        # heading with nothing under it, which reads as broken/incomplete
        # rather than as the actual finding it is.
        lines.append(
            f"No false negatives found -- every gold-labeled span was matched by at least one "
            f"detection (OVERLAP recall = 1.0000 across {result['docs_evaluated']} documents). "
            f"See `eval/final_results.csv` / the Evaluation Dashboard's Overall Recall (OVERLAP) "
            f"metric for confirmation.\n"
        )
    for i, item in enumerate(fn_sample, 1):
        lines.append(f"**FN {i}.** `{item['phi_type']}` missed gold text: `{item.get('text', '(n/a)')}` "
                      f"in `{item['filename']}`\n")
        lines.append(f"- Context: `...{item.get('context', '(n/a)')}...` (span that should have matched shown as `[[...]]`)\n")
        lines.append(f"- Root cause: {_explain_fn(item, backend)}\n")

    report = "\n".join(lines)
    print(report)

    out_path = os.path.join(args.output_dir, "error_analysis.md")
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(report + "\n")
    print(f"\nWrote {out_path}")


if __name__ == "__main__":
    main()
