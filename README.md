# Machine Learning Mini-Project

> Initial project setup. The public dataset has been downloaded locally; the proposed target remains a proxy for the assigned paper and needs to be stated clearly in the final report.

## Project details

- Course: UE24CS352A — Machine Learning
- Team members: Aditya Raj (PES2UG24CS033) and Aditya Kumar (PES2UG24CS032), Batch 30
- Assigned problem serial number: **8**
- Reference problem: *Predicting a Decline in Patient Reported Outcomes for Cancer Patients on Chemotherapy* ([CS229 report](https://cs229.stanford.edu/proj2020spr/report/Ostberg_Peterson.pdf))
- Public dataset: **2023 CDC National Health Interview Survey (NHIS), Sample Adult**; locally downloaded CSV, 29,522 records
- Proposed proxy target: classify fair/poor self-rated health among adults reporting a cancer diagnosis (`PHSTAT_A` in the fair/poor categories, within `CANEV_A` cancer-history respondents)
- Evaluation metric: **TBD**

## Current status

The reference paper's 1,252-patient STARR cohort is not distributed with the report. ICPSR 39628 was also checked, but its package contained reports and documentation rather than patient-level data. The downloaded NHIS public-use data is accessible without a guestbook, and includes cancer history (`CANEV_A`) and self-rated health (`PHSTAT_A`). It is cross-sectional: it has no chemotherapy treatment variable and cannot measure within-person decline. The proxy target above changes the original problem and should be confirmed with the TA before modeling.

## Planned structure

```text
data/       Dataset notes; keep data files out of version control unless sharing is permitted.
notebooks/  Exploration and experiments.
src/        Reusable project code.
reports/    Write-up, figures, and presentation material.
```

## Setup and run

**TBD:** Choose the runtime and dependencies before analysis. The raw download and extracted files are in `data/raw/adult23csv/` and are excluded from Git.

## Work sequence

1. Record the team's assigned serial number and members (done).
2. Confirm with the TA that the proposed NHIS proxy target is acceptable.
3. Inspect variables, missingness, and class balance in the cancer-history subset.
4. Agree on the split, preprocessing, and evaluation metric before model comparison.
5. Build and evaluate a baseline, then compare a small set of suitable models.
6. Re-run from the README and complete the PDF write-up, slides, and demo.

## Assignment deliverables

- Private GitHub repository shared with faculty and assigned TAs.
- PDF project summary covering the problem, dataset, approach, implementation, and conclusions. The assignment heading says “one-page,” while its instructions request a “two-page” summary; confirm the required length.
- Presentation and live code demonstration at the scheduled review.

## Open questions

- TA approval for the revised, cross-sectional proxy target (the user has permission to use any public dataset, but this dataset cannot support chemotherapy-specific decline).
- What are the data-use and sharing constraints for the final report/repository? Raw data is currently kept out of Git.
- What are the updated review and submission dates? The dates listed in the supplied assignment have passed.
- Should the PDF summary be one page or two?
