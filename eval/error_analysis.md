# Error Analysis: False Positives and False Negatives

Backend: `presidio`  |  Documents evaluated: 350  |  Total FPs (OVERLAP): 2195  |  Total FNs (OVERLAP): 2

Root-cause explanations below are drafted from this project's own documented failure patterns (see comments in this file). Verify against the actual backend/document before citing in the final report -- these are a starting draft, not a substitute for reading the real output.

## False Positives

**FP 1.** `LOCATION` matched text: `East` in `clinical_note_026.txt`

- Context: `...ms, was born.  At that time, mother had spent 2 months back [[East]] with the brother due to his feeding issues and will have to...` (matched span shown as `[[...]]`)

- Root cause: Likely a real LOCATION entity in borrowed mtsamples narrative text that the synthetic ground truth never labels (only the injected header fields are labeled) -- a documented measurement-methodology gap (README Section 5.2 / report Section 5.2), not necessarily an incorrect detection. Confirm by reading the context above.

**FP 2.** `PHONE_NUMBER` matched text: `168.4` in `clinical_note_015.txt`

- Context: `...etite.  He is a nonsmoker.,OBJECTIVE: , His weight today is [[168.4]] pounds, blood pressure 142/76, temperature 97.7, pulse 68, ...` (matched span shown as `[[...]]`)

- Root cause: MISMATCH: matched text `168.4` doesn't have phone-number shape. Check the context above -- this looks like a case where a lab value, vital sign, or other bare number sequence got picked up by the phone recognizer rather than a genuine phone-number false positive.

**FP 3.** `NRP` matched text: `Spanish` in `clinical_note_010.txt`

- Context: `...le.,ALLERGIES: , No known drug allergies.,MEDICATIONS:,  In [[Spanish]] label.  They are the diabetic medication, and also blood pr...` (matched span shown as `[[...]]`)

- Root cause: Likely a real NRP entity in borrowed mtsamples narrative text that the synthetic ground truth never labels (only the injected header fields are labeled) -- a documented measurement-methodology gap (README Section 5.2 / report Section 5.2), not necessarily an incorrect detection. Confirm by reading the context above.

**FP 4.** `PERSON` matched text: `B.` in `discharge_summary_081.txt`

- Context: `...ncouraged to follow up with her primary care physician, Dr. [[B.]]  As mentioned above, the patient will be discharged on 09/0...` (matched span shown as `[[...]]`)

- Root cause: MISMATCH: matched text `B.` doesn't look like a typical name -- inspect the context above to determine what actually happened here.

**FP 5.** `US_DRIVER_LICENSE` matched text: `S1` in `referral_letter_261.txt`

- Context: `...her side.,On palpation, he reports midline tenderness at L5-[[S1]] without additional areas of tenderness noted even to very f...` (matched span shown as `[[...]]`)

- Root cause: Likely a real US_DRIVER_LICENSE entity in borrowed mtsamples narrative text that the synthetic ground truth never labels (only the injected header fields are labeled) -- a documented measurement-methodology gap (README Section 5.2 / report Section 5.2), not necessarily an incorrect detection. Confirm by reading the context above.

## False Negatives

**FN 1.** `PHONE_NUMBER` missed gold text: `5394308705` in `lab_report_204.txt`

- Context: `...n Date: 06/28/2026
Ordering Physician: Casey Johnson
Phone: [[5394308705]]

Test: Lipid Panel.
Results: Total Cholesterol 178 (ref <20...` (span that should have matched shown as `[[...]]`)

- Root cause: UNEXPECTED: gold text `5394308705` is a standard-format phone number with no extension -- this should have matched both the fallback regex and Presidio's phone recognizer without issue. This is NOT the extension-leak bug (README Section 6.2), which is already fixed. Worth investigating directly: check whether an overlapping higher-confidence span claimed this position in _dedupe_overlaps, or whether this specific document's context (see above) has unusual surrounding punctuation/spacing.

**FN 2.** `PHONE_NUMBER` missed gold text: `847-532-1046` in `discharge_summary_089.txt`

- Context: `...scharge Date: 07/03/2026
Account Number: ACCT-589131
Phone: [[847-532-1046]]

FINAL DIAGNOSIS/REASON FOR ADMISSION:,1.  Acute right loba...` (span that should have matched shown as `[[...]]`)

- Root cause: UNEXPECTED: gold text `847-532-1046` is a standard-format phone number with no extension -- this should have matched both the fallback regex and Presidio's phone recognizer without issue. This is NOT the extension-leak bug (README Section 6.2), which is already fixed. Worth investigating directly: check whether an overlapping higher-confidence span claimed this position in _dedupe_overlaps, or whether this specific document's context (see above) has unusual surrounding punctuation/spacing.

