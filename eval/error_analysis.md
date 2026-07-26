# Error Analysis: False Positives and False Negatives

Backend: `presidio`  |  Documents evaluated: 350  |  Total FPs (OVERLAP): 2011  |  Total FNs (OVERLAP): 0

Root-cause explanations below are drafted from this project's own documented failure patterns (see comments in this file). Verify against the actual backend/document before citing in the final report -- these are a starting draft, not a substitute for reading the real output.

## False Positives

**FP 1.** `URL` matched text: `2.Bi` in `radiology_report_149.txt`

- Context: `...wup to ensure resolution given its consolidated appearance.,[[2.Bi]]lateral atelectasis versus fibrosis....` (matched span shown as `[[...]]`)

- Root cause: Likely a real URL entity in borrowed mtsamples narrative text that the synthetic ground truth never labels (only the injected header fields are labeled) -- a documented measurement-methodology gap (README Section 5.2 / report Section 5.2), not necessarily an incorrect detection. Confirm by reading the context above.

**FP 2.** `PHONE_NUMBER` matched text: `168.4` in `clinical_note_015.txt`

- Context: `...etite.  He is a nonsmoker.,OBJECTIVE: , His weight today is [[168.4]] pounds, blood pressure 142/76, temperature 97.7, pulse 68, ...` (matched span shown as `[[...]]`)

- Root cause: MISMATCH: matched text `168.4` doesn't have phone-number shape. Check the context above -- this looks like a case where a lab value, vital sign, or other bare number sequence got picked up by the phone recognizer rather than a genuine phone-number false positive.

**FP 3.** `LOCATION` matched text: `East` in `clinical_note_026.txt`

- Context: `...ms, was born.  At that time, mother had spent 2 months back [[East]] with the brother due to his feeding issues and will have to...` (matched span shown as `[[...]]`)

- Root cause: Likely a real LOCATION entity in borrowed mtsamples narrative text that the synthetic ground truth never labels (only the injected header fields are labeled) -- a documented measurement-methodology gap (README Section 5.2 / report Section 5.2), not necessarily an incorrect detection. Confirm by reading the context above.

**FP 4.** `PERSON` matched text: `Bonaparte` in `clinical_note_044.txt`

- Context: `...hich occur about once a week.  She is under the care of Dr. [[Bonaparte]] for hyperlipidemia and hypothyroidism.  She has a long hist...` (matched span shown as `[[...]]`)

- Root cause: MISMATCH: matched text `Bonaparte` doesn't look like a typical name -- inspect the context above to determine what actually happened here.

**FP 5.** `CLAIM_NUMBER` matched text: `CLAIM NUMBER : 12345-67890` in `referral_letter_261.txt`

- Context: `... 05/25/2026

P.O. Box 12345,City, State ,RE: EXAMINEE : Abc,[[CLAIM NUMBER : 12345-67890]],DATE OF INJURY : April 20, 2003,DATE OF EXAMINATION : Augus...` (matched span shown as `[[...]]`)

- Root cause: No reliable pattern rule for CLAIM_NUMBER -- read the context field above to determine root cause.

## False Negatives

