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

## 5. Presentation Cheat Sheet ("How to Explain in 2 Minutes")

If asked to explain your work during the viva or presentation, follow this 4-point structure:

1. **Problem Formulation:**  
   *"Our team reframed Problem 8 using public CDC NHIS data to predict fair/poor general health among 3,760 cancer survivors based on demographics and chronic conditions."*
2. **Leakage Prevention:**  
   *"I used an 80/20 stratified split and encapsulated imputation, scaling, and one-hot encoding inside a scikit-learn Pipeline so no test data leaked into training."*
3. **Models & Metric Evaluation:**  
   *"Because of the 73/27 class imbalance, accuracy alone is misleading (the dummy model gets 73% accuracy with 0 recall). Our Logistic Regression model achieves an ROC-AUC of 0.766 and a Balanced Accuracy of 70%, catching 67% of at-risk patients."*
4. **Findings & Live Demo:**  
   *"The strongest risk factors were depression, coronary heart disease, and diabetes. In our live demo script (`demo.py`), the model accurately differentiates high-risk multi-morbid patients from low-risk patients."*

---

## 6. How to Run

```bash
# 1. Train models and generate summary plot (results_summary.png)
python train_models.py

# 2. Run the live presentation demonstration
python demo.py
```
