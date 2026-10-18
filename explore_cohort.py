"""Print unweighted exploratory summaries for the prepared cancer cohort."""
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent
COHORT_FILE = ROOT / "data" / "cancer_health_modeling.csv"

CONDITION_COLUMNS = {
    "Diabetes": "DIBEV_A",
    "Hypertension": "HYPEV_A",
    "Coronary heart disease": "CHDEV_A",
    "Anxiety": "ANXEV_A",
    "Depression": "DEPEV_A",
    "Asthma": "ASEV_A",
}


def show_rates(title: str, frame: pd.DataFrame, group_column: str) -> None:
    """Print sample size and fair/poor outcome rate for each group."""
    summary = frame.groupby(group_column, observed=True)["poor_health"].agg(
        sample_size="size", fair_poor="sum"
    )
    summary["fair_poor_percent"] = 100 * summary["fair_poor"] / summary["sample_size"]
    print(f"\n{title}")
    print(summary.to_string(float_format=lambda value: f"{value:.1f}"))


def main() -> None:
    if not COHORT_FILE.exists():
        raise FileNotFoundError(f"Expected prepared cohort at {COHORT_FILE}; run prepare_data.py first")

    cohort = pd.read_csv(COHORT_FILE)
    print(f"Prepared rows: {len(cohort):,}")
    print(f"Fair/poor health: {cohort['poor_health'].sum():,} ({100 * cohort['poor_health'].mean():.1f}%)")
    print("Rates below are unweighted and descriptive; they are not adjusted effects or model performance.")

    cohort["age_band"] = pd.cut(
        cohort["AGEP_A"],
        bins=[0, 49, 64, 74, float("inf")],
        labels=["18-49", "50-64", "65-74", "75+"],
    )
    show_rates("Fair/poor health by age band", cohort.dropna(subset=["age_band"]), "age_band")
    show_rates("Fair/poor health by sex code (1=male, 2=female)", cohort, "SEX_A")

    print("\nFair/poor health by reported condition (1=yes, 2=no)")
    rows = []
    for label, column in CONDITION_COLUMNS.items():
        valid = cohort.loc[cohort[column].isin([1, 2])]
        for code, response in [(1, "Yes"), (2, "No")]:
            group = valid.loc[valid[column].eq(code), "poor_health"]
            rows.append(
                {
                    "condition": label,
                    "response": response,
                    "sample_size": len(group),
                    "fair_poor": int(group.sum()),
                    "fair_poor_percent": 100 * group.mean(),
                }
            )
    print(
        pd.DataFrame(rows).to_string(
            index=False,
            formatters={"fair_poor_percent": lambda value: f"{value:.1f}%"},
        )
    )

    predictors = ["AGEP_A", "SEX_A", "HISPALLP_A", "EDUCP_A", "BMICAT_A", *CONDITION_COLUMNS.values()]
    missing = cohort[predictors].isna().sum().rename("missing").to_frame()
    missing["percent"] = 100 * missing["missing"] / len(cohort)
    print("\nCandidate predictor missingness")
    print(missing.to_string(formatters={"percent": lambda value: f"{value:.2f}%"}))


if __name__ == "__main__":
    main()
