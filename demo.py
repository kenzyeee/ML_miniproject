"""Clinical profile risk stratification and model inference verification.

Loads the trained scikit-learn pipeline (best_model.joblib) and evaluates
predicted risk probabilities across representative patient clinical scenarios:
- Profile 1 (High-Risk Multimorbid): Advanced age with cardiometabolic and psychiatric comorbidities.
- Profile 2 (Low-Risk Reference): Younger age, absence of diagnosed chronic conditions, higher educational attainment.
"""
from pathlib import Path
import joblib
import pandas as pd

ROOT = Path(__file__).resolve().parent
MODEL_PATH = ROOT / "best_model.joblib"


def main() -> None:
    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"Missing {MODEL_PATH}; run train_models.py first.")

    model = joblib.load(MODEL_PATH)

    # Define two representative clinical profiles
    patients = [
        {
            "Case": "Profile 1 (High-Risk Multimorbid)",
            "Description": "Age 72, prior cancer, with diabetes, coronary heart disease, and depression",
            "Data": {
                "AGEP_A": 72, "SEX_A": 1, "HISPALLP_A": 2, "EDUCP_A": 3.0, "BMICAT_A": 4.0,
                "DIBEV_A": 1.0, "HYPEV_A": 1.0, "CHDEV_A": 1.0, "ANXEV_A": 1.0, "DEPEV_A": 1.0, "ASEV_A": 1.0,
            },
        },
        {
            "Case": "Profile 2 (Low-Risk Reference)",
            "Description": "Age 48, prior cancer, college graduate, no diagnosed chronic physical or mental conditions",
            "Data": {
                "AGEP_A": 48, "SEX_A": 2, "HISPALLP_A": 2, "EDUCP_A": 7.0, "BMICAT_A": 2.0,
                "DIBEV_A": 2.0, "HYPEV_A": 2.0, "CHDEV_A": 2.0, "ANXEV_A": 2.0, "DEPEV_A": 2.0, "ASEV_A": 2.0,
            },
        },
    ]

    print("=" * 70)
    print("CLINICAL RISK STRATIFICATION INFERENCE VERIFICATION")
    print("=" * 70)

    for p in patients:
        df_patient = pd.DataFrame([p["Data"]])
        prob = model.predict_proba(df_patient)[0, 1]
        pred_label = "Fair / Poor Health" if prob >= 0.50 else "Good / Better Health"

        print(f"\n{p['Case']}")
        print(f"Context: {p['Description']}")
        print(f"  -> Predicted Probability of Fair/Poor Health: {prob:.1%}")
        print(f"  -> Model Decision: {pred_label}")

    print("\n" + "=" * 70)


if __name__ == "__main__":
    main()
