"""Build the Part 1 cancer-history cohort and provisional binary target.

Input:  data/adult23.csv
Output: data/cancer_health_modeling.csv (ignored by Git)

This is a cross-sectional general-health proxy task, not chemotherapy outcome prediction.
"""
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parent
RAW_FILE = ROOT / "data" / "adult23.csv"
OUTPUT_FILE = ROOT / "data" / "cancer_health_modeling.csv"

# Candidate predictors for the teammate to consider. Preserve survey response codes
# and category values from the CDC codebook; only convert documented nonresponse
# or unknown codes to missing values.
PREDICTORS = [
    "AGEP_A",
    "SEX_A",
    "HISPALLP_A",
    "EDUCP_A",
    "BMICAT_A",
    "DIBEV_A",
    "HYPEV_A",
    "CHDEV_A",
    "ANXEV_A",
    "DEPEV_A",
    "ASEV_A",
]
NONRESPONSE_CODES = {
    "AGEP_A": [97, 98, 99],
    "SEX_A": [7, 8, 9],
    "HISPALLP_A": [97, 98, 99],
    "EDUCP_A": [97, 98, 99],
    "BMICAT_A": [9],
    "DIBEV_A": [7, 8, 9],
    "HYPEV_A": [7, 8, 9],
    "CHDEV_A": [7, 8, 9],
    "ANXEV_A": [7, 8, 9],
    "DEPEV_A": [7, 8, 9],
    "ASEV_A": [7, 8, 9],
}
SURVEY_COLUMNS = ["WTFA_A", "PSTRAT", "PPSU"]


def main() -> None:
    if not RAW_FILE.exists():
        raise FileNotFoundError(f"Expected CDC CSV at {RAW_FILE}")

    # CDC recommends inferring data types from the full file because sparse
    # variables can be misread when only an initial sample of rows is used.
    raw = pd.read_csv(RAW_FILE, low_memory=False)
    required = {"CANEV_A", "PHSTAT_A", *PREDICTORS, *SURVEY_COLUMNS}
    missing = required.difference(raw.columns)
    if missing:
        raise ValueError(f"Required columns missing from source: {sorted(missing)}")

    cancer_history = raw.loc[raw["CANEV_A"].eq(1)].copy()
    valid_target = cancer_history["PHSTAT_A"].isin([1, 2, 3, 4, 5])
    cohort = cancer_history.loc[valid_target].copy()

    prepared = cohort[PREDICTORS + SURVEY_COLUMNS].copy()
    for column, codes in NONRESPONSE_CODES.items():
        prepared[column] = pd.to_numeric(prepared[column], errors="coerce")
        prepared[column] = prepared[column].mask(prepared[column].isin(codes))

    prepared["poor_health"] = cohort["PHSTAT_A"].isin([4, 5]).astype("int8")
    prepared = prepared.rename(
        columns={
            "WTFA_A": "survey_weight",
            "PSTRAT": "survey_stratum",
            "PPSU": "survey_psu",
        }
    )
    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    prepared.to_csv(OUTPUT_FILE, index=False)

    print(f"Source records: {len(raw):,}")
    print(f"Ever diagnosed with cancer: {len(cancer_history):,}")
    print(f"Cancer-history records with valid PHSTAT_A: {len(prepared):,}")
    print(f"Cancer-history records excluded for missing/nonresponse PHSTAT_A: {len(cancer_history) - len(prepared):,}")
    print("poor_health counts (0=excellent/very good/good, 1=fair/poor):")
    print(prepared["poor_health"].value_counts().sort_index().to_string())
    print("Predictor missing counts after codebook-based recoding:")
    print(prepared[PREDICTORS].isna().sum().to_string())
    print(f"Prepared dataset: {OUTPUT_FILE}")
    print("Keep survey_weight, survey_stratum, and survey_psu out of the model feature matrix.")


if __name__ == "__main__":
    main()

