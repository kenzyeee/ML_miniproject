# Machine Learning Mini-Project

## Project details

- Course: UE24CS352A — Machine Learning
- Team: Aditya Raj (PES2UG24CS033) and Aditya Kumar (PES2UG24CS032), Batch 30
- Assigned problem serial number: **8**
- Reference paper: [Predicting a Decline in Patient Reported Outcomes for Cancer Patients on Chemotherapy](https://cs229.stanford.edu/proj2020spr/report/Ostberg_Peterson.pdf)
- Public dataset: 2023 CDC National Health Interview Survey (NHIS), Sample Adult
- Proposed proxy task: classify fair/poor self-rated general health among adults who report ever having cancer

## Scope caveat

The reference paper predicts poor or declining PROMIS Global Physical Health (GPH) during chemotherapy. NHIS is cross-sectional and contains neither chemotherapy exposure nor before/after health scores. It cannot reproduce that task. The current proxy target is a substantial change in scope and still needs the TA's approval. Do not describe this as chemotherapy outcome or decline prediction.

## Part 1 status

Part 1 established the proxy target, checked the CDC codebook, explored the outcome distribution, and created a reusable cohort-preparation script. The source file has 29,522 adults; 3,762 reported a prior cancer diagnosis; 3,760 of those have a valid general-health response. In this analytic cohort, 1,015 (27.0%) report fair/poor health and 2,745 (73.0%) report good/very good/excellent health. These are unweighted sample counts.

See [`reports/part-1-data-exploration.md`](reports/part-1-data-exploration.md) for the findings, variable notes, source-paper methodology, and teammate handoff. See [`reports/step-1-dataset-and-scope.md`](reports/step-1-dataset-and-scope.md) for the initial setup note.

## Data and setup

The original CDC download and extracted CSV belong in `data/raw/`; raw and processed data are excluded from Git. Download the Sample Adult CSV ZIP from the [CDC 2023 NHIS documentation page](https://www.cdc.gov/nchs/nhis/documentation/2023-nhis.html) and extract it so `data/raw/adult23csv/adult23.csv` exists. The CDC `readme.txt` is also included in the archive.

From the project root, install the dependency and prepare the modeling cohort:

```powershell
python -m pip install -r requirements.txt
python src/prepare_data.py
```

The script creates `data/processed/cancer_health_modeling.csv`. It contains selected candidate predictors, `poor_health` (1=fair/poor, 0=good or better), and survey metadata. Keep `survey_weight`, `survey_stratum`, and `survey_psu` out of the predictive feature matrix unless implementing survey-aware modeling. The original outcome column and cancer cohort flag are excluded from the prepared file to prevent direct leakage.

## Handoff: Part 2

1. Confirm the proxy target with the TA before finalizing the project framing.
2. Use the prepared cohort; perform stratified train/test splitting or stratified cross-validation.
3. Fit imputation, encoding, scaling, and tuning within training folds only.
4. Establish a majority-class and regularized logistic-regression baseline, then compare a small number of models.
5. Report class-aware metrics (for example, balanced accuracy, precision/recall or F1, ROC-AUC, and PR-AUC), not accuracy alone.
6. Keep test data out of model selection, and describe findings as sample classification results rather than clinical advice or chemotherapy effects.

## Assignment deliverables

- Private GitHub repository shared with faculty and assigned TAs, including setup/run instructions.
- PDF summary describing the problem, dataset, approach, implementation, and conclusions. The supplied assignment labels it “One-Page Write-up” but describes a two-page summary; confirm the expected length.
- Presentation and live code demonstration at review.

The dates supplied in the assignment (October 5-10, 2026) have passed; check with course staff for updated submission and review arrangements.
