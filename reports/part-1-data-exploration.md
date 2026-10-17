# Part 1: Data exploration and handoff

## Scope

The team's assigned paper studies prediction of poor/declining PROMIS Global Physical Health (GPH) during chemotherapy from information available before the first chemotherapy treatment. Ostberg and Peterson used a 1,252-patient Stanford STARR cohort with at least one baseline and one on-treatment PROMIS survey. Their two labels were (1) a GPH decline of at least 10 points and (2) an on-treatment GPH score at or below 40. Their features were pre-treatment EHR data, generally limited to the preceding six months (except longer-lived diagnoses); they engineered time-weighted labs/procedures and compared regularized logistic regression, tree ensembles, MLP, SVM, KNN, and voting models, tuning on training data and evaluating with AUROC.

The STARR cohort is not included with the paper. After the TA allowed a public dataset, the team selected the CDC 2023 NHIS Sample Adult public-use file. NHIS does not identify chemotherapy exposure and does not measure a before/after decline. The feasible project question is therefore only a **proxy**: classify fair/poor self-rated general health among adults who report ever having cancer. Get TA approval for this reframed question before treating it as the final assigned target. Do not call this a chemotherapy or decline prediction model.

## Dataset and outcome audit

Source: CDC/NCHS 2023 NHIS [documentation page](https://www.cdc.gov/nchs/nhis/documentation/2023-nhis.html) and its [Sample Adult codebook](https://ftp.cdc.gov/pub/Health_Statistics/NCHS/Dataset_Documentation/NHIS/2023/Adult-codebook.pdf). The CDC CSV contains 29,522 adult records and 647 variables. The CDC codebook defines `CANEV_A=1` as ever diagnosed with cancer; 3,762 sample adults have that response. `PHSTAT_A` asks respondents to rate their health as excellent, very good, good, fair, or poor (codes 1-5); codes 7, 8, and 9 are refused, not ascertained, and don't know.

| Cohort/outcome check | Count |
|---|---:|
| Full Sample Adult file | 29,522 |
| `CANEV_A=1` (ever told they had cancer) | 3,762 |
| Cancer-history records with valid `PHSTAT_A` | 3,760 |
| Cancer-history records excluded because `PHSTAT_A` was nonresponse | 2 |
| Target 0: excellent / very good / good | 2,745 (73.0%) |
| Target 1: fair / poor | 1,015 (27.0%) |

These are unweighted counts, not population estimates. The positive class is large enough for a basic classifier, but a majority-class baseline would already achieve 73% accuracy. Part 2 should report class-sensitive metrics such as balanced accuracy, precision/recall or F1, and ROC-AUC; include PR-AUC if possible.

## Prepared files

Run `python src/prepare_data.py` from the project root. It reads the original CSV using full-file type inference, subsets `CANEV_A=1`, excludes the two records with nonresponse `PHSTAT_A`, creates `poor_health` (1 for codes 4-5; 0 for codes 1-3), and maps documented feature nonresponse codes to missing values. It writes `data/processed/cancer_health_modeling.csv`, which is excluded from Git and reproducible from the raw download.

The prepared file has these candidate predictors:

| Variable | Meaning | Preparation note |
|---|---|---|
| `AGEP_A` | Age, public-use top-coded | 85 means 85 or older; 97-99 recoded missing |
| `SEX_A` | Sex of sample adult | 1 male, 2 female; 7-9 recoded missing |
| `HISPALLP_A` | Race groups with Hispanic origin | 1-7 are valid groups (7 is “other”); 97-99 are nonresponse |
| `EDUCP_A` | Education level | 00-10 are valid; 97-99 recoded missing |
| `BMICAT_A` | BMI category | 1-4 are categories; 9 (“unknown”) recoded missing |
| `DIBEV_A` | Ever diagnosed with diabetes | 1 yes, 2 no; 7-9 recoded missing |
| `HYPEV_A` | Ever diagnosed with hypertension | 1 yes, 2 no; 7-9 recoded missing |
| `CHDEV_A` | Ever diagnosed with coronary heart disease | 1 yes, 2 no; 7-9 recoded missing |
| `ANXEV_A` | Ever diagnosed with anxiety disorder | 1 yes, 2 no; 7-9 recoded missing |
| `DEPEV_A` | Ever diagnosed with depression | 1 yes, 2 no; 7-9 recoded missing |
| `ASEV_A` | Ever diagnosed with asthma | 1 yes, 2 no; 7-9 recoded missing |

`survey_weight`, `survey_stratum`, and `survey_psu` are retained for a possible survey-aware analysis; they are metadata and must not be fed in as ordinary predictive features. The outcome source `PHSTAT_A` and cohort selector `CANEV_A` are not included in the prepared file, preventing direct target leakage. The input columns remain coded as documented values; Part 2 should use categorical preprocessing for categorical fields and learn imputation/encoding on training folds only.

## Missingness among prepared rows

Feature nonresponse is retained as missing rather than dropping whole rows. Counts after codebook-based recoding (out of 3,760):

| Feature | Missing |
|---|---:|
| `AGEP_A` | 7 |
| `SEX_A` | 0 |
| `HISPALLP_A` | 0 |
| `EDUCP_A` | 8 |
| `BMICAT_A` | 63 |
| `DIBEV_A` | 2 |
| `HYPEV_A` | 4 |
| `CHDEV_A` | 12 |
| `ANXEV_A` | 5 |
| `DEPEV_A` | 5 |
| `ASEV_A` | 2 |

## Handoff to Part 2

1. Confirm the proxy target with the TA and state the cross-sectional limitation in the report.
2. Start from `data/processed/cancer_health_modeling.csv`; keep the three survey metadata columns out of `X` unless implementing survey-aware training/evaluation.
3. Use a stratified train/test split or stratified cross-validation. Fit imputers, encoders, scaling, feature selection, and tuning inside training folds only.
4. Compare a majority-class baseline and a simple regularized logistic regression before a small number of more complex models. Do not optimize on the test set.
5. Report confusion matrix and class-aware metrics. Describe findings as associations/classification in this sample, not clinical advice or chemotherapy effects.

## Reproduction

- Install the dependency listed in `requirements.txt`.
- From this directory run `python src/prepare_data.py`.
- Raw and processed data are intentionally excluded from Git; each teammate needs to download the CDC raw CSV using the source link in [`data/README.md`](../data/README.md) before running the script.
