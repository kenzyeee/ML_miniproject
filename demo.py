"""Simple demonstration script for the live presentation.

Loads the trained model (best_model.joblib) and evaluates two patient cases:
- Patient A (High Risk): Multiple comorbidities (diabetes, heart disease, depression).
- Patient B (Low Risk): No chronic conditions, college degree.
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

    # Define two contrasting clinical cases
    patients = [
        {
            "Case": "Patient A (High Risk Profile)",
            "Description": "72yo, prior cancer, with Diabetes, Heart Disease, and Depression",
            "Data": {
                "AGEP_A": 72, "SEX_A": 1, "HISPALLP_A": 2, "EDUCP_A": 3.0, "BMICAT_A": 4.0,
                "DIBEV_A": 1.0, "HYPEV_A": 1.0, "CHDEV_A": 1.0, "ANXEV_A": 1.0, "DEPEV_A": 1.0, "ASEV_A": 1.0,
            },
        },
        {
            "Case": "Patient B (Low Risk Profile)",
            "Description": "48yo, prior cancer, college graduate, no chronic physical or mental conditions",
            "Data": {
                "AGEP_A": 48, "SEX_A": 2, "HISPALLP_A": 2, "EDUCP_A": 7.0, "BMICAT_A": 2.0,
                "DIBEV_A": 2.0, "HYPEV_A": 2.0, "CHDEV_A": 2.0, "ANXEV_A": 2.0, "DEPEV_A": 2.0, "ASEV_A": 2.0,
            },
        },
    ]

    print("=" * 70)
    print("LIVE DEMONSTRATION: CANCER HEALTH OUTCOME PREDICTION")
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
