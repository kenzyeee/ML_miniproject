# Part 2: Machine Learning Modeling and Evaluation

**Team:** Aditya Raj (PES2UG24CS033) & Aditya Kumar (PES2UG24CS032), Batch 30  
**Assigned problem serial:** 8  
**Reference paper:** [Ostberg & Peterson, predicting health outcomes during chemotherapy](https://cs229.stanford.edu/proj2020spr/report/Ostberg_Peterson.pdf)  
**Cohort:** 2023 CDC NHIS Sample Adult cancer cohort ($N = 3,760$, 27.0% fair/poor health)

---

## 1. Overview & Goal

Part 2 completes the modeling and evaluation phase handed off from Part 1. 

Using the prepared dataset (`data/cancer_health_modeling.csv`), our objective is to classify **fair/poor self-rated health** (`poor_health = 1`) versus **good-or-better health** (`poor_health = 0`) among cancer survivors, using 11 clinical and demographic predictors.

---

## 2. Experimental Setup (Leakage-Safe Pipeline)

1. **Stratified Split:** Data is split into **80% training** ($N = 3,008$) and **20% testing** ($N = 752$) using stratified sampling (`random_state=42`), preserving the 27.0% positive prevalence in both sets.
2. **Metadata Isolation:** Survey design columns (`survey_weight`, `survey_stratum`, `survey_psu`) are excluded from predictive features $X$ to prevent target leakage.
3. **Pipeline Preprocessing:**
   - **Continuous (`AGEP_A`):** Median imputation + `StandardScaler`.
   - **Categorical & Comorbidities (10 variables):** Most-frequent imputation + `OneHotEncoder(drop='first')`.
   - All transformations are fitted strictly on the training set inside a `scikit-learn` `Pipeline`.

---

## 3. Model Benchmark & Test Results

We compared three models on the sequestered 752-patient test set:

| Model | ROC-AUC | Accuracy | Balanced Acc | Precision | Recall (Sensitivity) | $F_1$-Score |
|---|:---:|:---:|:---:|:---:|:---:|:---:|
| **Dummy Baseline** | 0.500 | 0.730 | 0.500 | 0.000 | 0.000 | 0.000 |
| **Logistic Regression** | **0.766** | **0.714** | **0.700** | **0.479** | **0.670** | **0.559** |
| **Random Forest** | 0.738 | 0.702 | 0.686 | 0.463 | 0.650 | 0.541 |

### Key Takeaways:
- **Accuracy is Deceiving:** The Dummy baseline achieves 73.0% accuracy simply by predicting "Good Health" for every patient, but misses every single at-risk patient (Recall = 0.000).
- **Strong Clinical Discrimination:** Logistic Regression achieves an **ROC-AUC of 0.766** and **Balanced Accuracy of 70.0%**, identifying **67.0% of patients with fair/poor health**.
- **Model Choice:** Regularized Logistic Regression performs on par with non-linear Random Forest while remaining fully interpretable for clinical decision-making.

---

## 4. Key Clinical Insights

From the Logistic Regression odds ratios:
1. **Depression:** Cancer survivors with depression have nearly **3 times higher odds** of fair/poor health compared to those without depression ($p < 0.001$).
2. **Heart Disease & Diabetes:** Having coronary heart disease or diabetes roughly **doubles the odds** of fair/poor health ($p < 0.001$).
3. **Education Gradient:** Higher education exhibits a strong protective association: college graduates have substantially lower odds of fair/poor health compared to those who did not complete high school.

---

## 5. Technical Discussion and Methodological Rigor

### 5.1 Leakage Prevention and Validation Architecture
Evaluation was conducted on a sequestered 20% test partition ($N = 752$) stratified by outcome prevalence. All preprocessing transformations—median imputation, standard scaling for continuous features, and one-hot encoding for categorical covariates—are encapsulated within a unified `scikit-learn` `Pipeline`. Transformation parameters are fitted exclusively on the training split ($N = 3,008$), preventing information leakage or target contamination across partitions. Survey design identifiers (`survey_weight`, `survey_stratum`, `survey_psu`) were isolated from predictor matrix $X$ to ensure strictly clinical and demographic feature inputs.

### 5.2 Evaluation Under Class Imbalance (73.0% / 27.0%)
In skewed clinical outcome distributions, unweighted classification accuracy is uninformative: a trivial baseline predicting the majority class ("Good Health") achieves 73.0% accuracy while failing to identify any at-risk patients (recall = 0.000). Consequently, model comparison was evaluated using threshold-independent discrimination (**ROC-AUC = 0.766**), **Balanced Accuracy (70.0%)**, and sensitivity (**Recall = 67.0%** at the standard decision boundary).

### 5.3 Model Selection and Interpretability
Regularized Logistic Regression with inverse class-frequency weighting performs on par with or marginally exceeds the non-linear Random Forest ensemble (ROC-AUC 0.766 vs. 0.738; $F_1$-score 0.559 vs. 0.541). Logistic Regression provides direct parametric interpretability through exponentiated coefficients (adjusted odds ratios), confirming substantial independent associations for depression, coronary heart disease, and diabetes, alongside an educational protective gradient.

### 5.4 Scope and Limitations
As the 2023 CDC NHIS is a cross-sectional observational survey rather than a longitudinal clinical cohort, findings characterize statistical associations with general health status among cancer survivors rather than longitudinal disease progression or causal chemotherapy response.

---

## 6. Execution and Verification

```bash
# Model training, test-set evaluation, and figure generation (results_summary.png)
python train_models.py

# Model inference on representative clinical patient profiles
python demo.py
```
