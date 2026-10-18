# Cancer health mini-project

**Team:** Aditya Raj (PES2UG24CS033) and Aditya Kumar (PES2UG24CS032), Batch 30  
**Assigned problem serial:** 8  
**Reference paper:** [Ostberg & Peterson, predicting health outcomes during chemotherapy](https://cs229.stanford.edu/proj2020spr/report/Ostberg_Peterson.pdf)

## Project status

Part 1 prepared a public-data proxy task: classify fair/poor self-rated health among adults who report ever having cancer, using the 2023 CDC NHIS Sample Adult survey. The source paper predicts PROMIS physical-health decline during chemotherapy. NHIS has no chemotherapy exposure or before/after measures, so this is a substantial reframing and needs TA approval. Do not describe it as predicting chemotherapy effects or health decline.

The NHIS file has 29,522 adults; 3,762 reported a prior cancer diagnosis, and 3,760 had a valid self-rated health response. In the analytic cohort, 1,015 (27.0%) reported fair/poor health and 2,745 (73.0%) good-or-better health.

Part 2 implements a leakage-safe machine learning pipeline, benchmarking a Dummy majority baseline, Logistic Regression, and Random Forest. Logistic Regression achieved an **ROC-AUC of 0.766** and **Balanced Accuracy of 70.0%** on the held-out test set, identifying 67.0% of patients with fair/poor health. Key risk factors identified are depression, coronary heart disease, and diabetes.

## Simple file layout

- `data/adult23.csv` — downloaded CDC source file (ignored by Git)
- `data/cancer_health_modeling.csv` — prepared cohort (ignored by Git)
- `prepare_data.py` — reproducible preparation script (Part 1)
- `explore_cohort.py` — reproducible Part 1 subgroup and missingness summaries
- `PART1.md` — exploration, paper-method summary, and Part 2 handoff
- `train_models.py` — reproducible Part 2 training, evaluation, and plotting script
- `demo.py` — simple live demonstration script for presentation
- `results_summary.png` — 3-panel visual summary (ROC curve, confusion matrix, odds ratios)
- `PART2.md` — modeling results, clinical findings, and presentation cheat sheet
- `requirements.txt` — Python dependencies

## Setup and run

1. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

2. Download `adult23.csv` from the [CDC 2023 NHIS page](https://www.cdc.gov/nchs/nhis/documentation/2023-nhis.html) and save to `data/adult23.csv`. Then run Part 1:
   ```bash
   python prepare_data.py
   python explore_cohort.py
   ```

3. Run Part 2 model training and evaluation:
   ```bash
   python train_models.py
   ```

4. Run the live demo for presentation:
   ```bash
   python demo.py
   ```

## Assignment deliverables

Private GitHub repository shared with faculty and assigned TAs; summary report of the problem, data, approach, implementation, and conclusions; and a presentation with live code demonstration.
