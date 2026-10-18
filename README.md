# Cancer health mini-project

**Team:** Aditya Raj (PES2UG24CS033) and Aditya Kumar (PES2UG24CS032), Batch 30  
**Assigned problem serial:** 8  
**Reference paper:** [Ostberg & Peterson, predicting health outcomes during chemotherapy](https://cs229.stanford.edu/proj2020spr/report/Ostberg_Peterson.pdf)

## Project status

Part 1 prepared a public-data proxy task: classify fair/poor self-rated health among adults who report ever having cancer, using the 2023 CDC NHIS Sample Adult survey. The source paper predicts PROMIS physical-health decline during chemotherapy. NHIS has no chemotherapy exposure or before/after measures, so this is a substantial reframing and needs TA approval. Do not describe it as predicting chemotherapy effects or health decline.

The NHIS file has 29,522 adults; 3,762 reported a prior cancer diagnosis, and 3,760 had a valid self-rated health response. In the analytic cohort, 1,015 (27.0%) reported fair/poor health and 2,745 (73.0%) good-or-better health. These are unweighted counts.

## Simple file layout

- `data/adult23.csv` — downloaded CDC source file (ignored by Git)
- `data/cancer_health_modeling.csv` — prepared cohort (ignored by Git)
- `prepare_data.py` — reproducible preparation script
- `PART1.md` — exploration, paper-method summary, and Part 2 handoff
- `requirements.txt` — Python dependency

## Setup and prepare data

Download the Sample Adult CSV from the [CDC 2023 NHIS documentation page](https://www.cdc.gov/nchs/nhis/documentation/2023-nhis.html) and save it as `data/adult23.csv`. Then run from this folder:

```powershell
python -m pip install -r requirements.txt
python prepare_data.py
```

The script creates `data/cancer_health_modeling.csv`. The original and prepared CSVs are excluded from Git; each teammate needs the source file locally to regenerate the cohort.

## Part 2 handoff

Confirm the proxy question with the TA, then use stratified evaluation and training-fold-only preprocessing. Compare a majority-class baseline and regularized logistic regression before trying a small number of other models. Report class-aware metrics, and keep `survey_weight`, `survey_stratum`, and `survey_psu` out of ordinary predictive features unless implementing survey-aware analysis. See [PART1.md](PART1.md).

## Assignment deliverables

Private GitHub repository shared with faculty and assigned TAs; a PDF summary of the problem, data, approach, implementation, and conclusions; and a presentation with live code demonstration. The assignment has passed dates listed as October 5–10, 2026, so check course staff for updated arrangements and expected write-up length.
