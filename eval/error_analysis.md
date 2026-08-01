# Error Analysis: False Positives and False Negatives

Backend: `presidio`  |  Documents evaluated: 350  |  Total FPs (OVERLAP): 2206  |  Total FNs (OVERLAP): 11

Root-cause explanations below are drafted from this project's own documented failure patterns (see comments in this file). Verify against the actual backend/document before citing in the final report -- these are a starting draft, not a substitute for reading the real output.

## False Positives

**FP 1.** `DATE_TIME` matched text: `daily` in `clinical_note_037.txt`

- Context: `....  Theophylline.,3.  Z-Pak.,4.  Chantix.,5.  Januvia 100 mg [[daily]].,6.  K-Lor.,7.  OxyContin.,8.  Flomax.,9.  Lasix.,10.  Adva...` (matched span shown as `[[...]]`)

- Root cause: Plausible: DATE_TIME has real false-positive noise from dates appearing in clinical narrative body text (procedure dates, historical dates in free text) that aren't part of the injected ground truth. Confirm against the context above -- if the matched text isn't actually a date, this explanation doesn't apply; re-diagnose from context.

**FP 2.** `PERSON` matched text: `Zofran` in `discharge_summary_093.txt`

- Context: `...-Dur 20 mEq p.o. daily.,9.  Prilosec 40 mg p.o. daily.,10.  [[Zofran]] 4 mg p.o. q.4-6 hourly p.r.n.,She is to follow up with her ...` (matched span shown as `[[...]]`)

- Root cause: MISMATCH: matched text `Zofran` doesn't look like a typical name -- inspect the context above to determine what actually happened here.

**FP 3.** `LOCATION` matched text: `Madera` in `discharge_summary_099.txt`

- Context: `...e.,HEMATOLOGY: , The patient is status post phototherapy at [[Madera]] and was started on iron.,OPHTHALMOLOGY: ,  Exam on 07/17/20...` (matched span shown as `[[...]]`)

- Root cause: Likely a real LOCATION entity in borrowed mtsamples narrative text that the synthetic ground truth never labels (only the injected header fields are labeled) -- a documented measurement-methodology gap (README Section 5.2 / report Section 5.2), not necessarily an incorrect detection. Confirm by reading the context above.

**FP 4.** `CLAIM_NUMBER` matched text: `CLAIM NUMBER : 12345-67890` in `referral_letter_261.txt`

- Context: `... 05/25/2026

P.O. Box 12345,City, State ,RE: EXAMINEE : Abc,[[CLAIM NUMBER : 12345-67890]],DATE OF INJURY : April 20, 2003,DATE OF EXAMINATION : Augus...` (matched span shown as `[[...]]`)

- Root cause: No reliable pattern rule for CLAIM_NUMBER -- read the context field above to determine root cause.

**FP 5.** `NRP` matched text: `L2-L3` in `referral_letter_253.txt`

- Context: `...tion at the L1-L2 level as well as a disc protrusion at the [[L2-L3]] level with disc herniations at the L3-L4 and L4-L5 level an...` (matched span shown as `[[...]]`)

- Root cause: Likely a real NRP entity in borrowed mtsamples narrative text that the synthetic ground truth never labels (only the injected header fields are labeled) -- a documented measurement-methodology gap (README Section 5.2 / report Section 5.2), not necessarily an incorrect detection. Confirm by reading the context above.

## False Negatives

**FN 1.** `DATE_TIME` missed gold text: `04/22/1991` in `referral_letter_262.txt`

- Context: `...Patient Name: Mary Adkins
DOB: [[04/22/1991]]
Referring Physician: Tiffany Garcia
Receiving Physician: Ra...` (span that should have matched shown as `[[...]]`)

- Root cause: No reliable pattern rule for DATE_TIME -- read the context field above to determine why this span wasn't caught.

**FN 2.** `DATE_TIME` missed gold text: `09/16/1975` in `clinical_note_044.txt`

- Context: `...Patient Name: Beth Sanford
DOB: [[09/16/1975]]
MRN: MR-205239
Phone: 818.370.2621x745
Visit Date: 06/12/20...` (span that should have matched shown as `[[...]]`)

- Root cause: No reliable pattern rule for DATE_TIME -- read the context field above to determine why this span wasn't caught.

**FN 3.** `DATE_TIME` missed gold text: `03/19/2024` in `referral_letter_281.txt`

- Context: `...Patient Name: Lauren Vazquez
DOB: [[03/19/2024]]
Referring Physician: Jacqueline Davis
Receiving Physician: ...` (span that should have matched shown as `[[...]]`)

- Root cause: No reliable pattern rule for DATE_TIME -- read the context field above to determine why this span wasn't caught.

**FN 4.** `DATE_TIME` missed gold text: `02/13/1941` in `discharge_summary_064.txt`

- Context: `...Patient Name: Jessica Stevens
DOB: [[02/13/1941]]
MRN: MR-428267
Admission Date: 06/19/2026
Discharge Date: 0...` (span that should have matched shown as `[[...]]`)

- Root cause: No reliable pattern rule for DATE_TIME -- read the context field above to determine why this span wasn't caught.

**FN 5.** `DATE_TIME` missed gold text: `06/25/2001` in `referral_letter_255.txt`

- Context: `...Patient Name: Matthew Garcia
DOB: [[06/25/2001]]
Referring Physician: Denise Harmon
Receiving Physician: Lau...` (span that should have matched shown as `[[...]]`)

- Root cause: No reliable pattern rule for DATE_TIME -- read the context field above to determine why this span wasn't caught.

