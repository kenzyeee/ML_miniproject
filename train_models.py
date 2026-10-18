"""Train and evaluate baseline and machine learning models for Part 2.

Simple, leakage-free pipeline:
1. Loads cancer cohort (data/cancer_health_modeling.csv).
2. Splits into 80% train / 20% test using stratified sampling.
3. Preprocesses data inside scikit-learn Pipeline (imputation + scaling + encoding).
4. Trains and evaluates 3 models:
   - Dummy Classifier (majority-class baseline)
   - Logistic Regression (interpretable linear baseline with odds ratios)
   - Random Forest (non-linear ensemble baseline)
5. Evaluates class-aware metrics (ROC-AUC, Balanced Accuracy, F1, Precision, Recall).
6. Generates a clean 3-panel visualization (results_summary.png).
7. Saves the trained model for demo.py.
"""
from pathlib import Path
import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

ROOT = Path(__file__).resolve().parent
DATA_PATH = ROOT / "data" / "cancer_health_modeling.csv"
OUTPUT_PLOT = ROOT / "results_summary.png"
MODEL_PATH = ROOT / "best_model.joblib"

# 11 clinical and demographic features
NUMERIC_FEATURES = ["AGEP_A"]
CATEGORICAL_FEATURES = [
    "SEX_A", "HISPALLP_A", "EDUCP_A", "BMICAT_A",
    "DIBEV_A", "HYPEV_A", "CHDEV_A", "ANXEV_A", "DEPEV_A", "ASEV_A",
]


def build_pipeline(classifier) -> Pipeline:
    """Create a leakage-free preprocessing and modeling pipeline."""
    preprocessor = ColumnTransformer(
        transformers=[
            ("num", Pipeline([
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler()),
            ]), NUMERIC_FEATURES),
            ("cat", Pipeline([
                ("imputer", SimpleImputer(strategy="most_frequent")),
                ("encoder", OneHotEncoder(drop="first", handle_unknown="ignore")),
            ]), CATEGORICAL_FEATURES),
        ]
    )
    return Pipeline([("prep", preprocessor), ("clf", classifier)])


def evaluate_model(name: str, pipe: Pipeline, X_test: pd.DataFrame, y_test: np.ndarray) -> dict:
    """Compute standard classification and ranking metrics on the test set."""
    y_pred = pipe.predict(X_test)
    y_prob = pipe.predict_proba(X_test)[:, 1] if hasattr(pipe, "predict_proba") else y_pred

    return {
        "Model": name,
        "ROC-AUC": roc_auc_score(y_test, y_prob),
        "Accuracy": accuracy_score(y_test, y_pred),
        "Balanced Acc": balanced_accuracy_score(y_test, y_pred),
        "Precision": precision_score(y_test, y_pred, zero_division=0),
        "Recall": recall_score(y_test, y_pred, zero_division=0),
        "F1-Score": f1_score(y_test, y_pred, zero_division=0),
    }


def main() -> None:
    if not DATA_PATH.exists():
        raise FileNotFoundError(f"Missing {DATA_PATH}; run prepare_data.py first.")

    # 1. Load data
    df = pd.read_csv(DATA_PATH)
    X = df[NUMERIC_FEATURES + CATEGORICAL_FEATURES]
    y = df["poor_health"].values

    print(f"Total cohort: {len(df):,} patients ({y.sum():,} fair/poor health, {100 * y.mean():.1f}%)")

    # 2. Stratified Train/Test Split (80% train, 20% test)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y
    )
    print(f"Training set: {len(X_train):,} patients | Test set: {len(X_test):,} patients")

    # 3. Models to compare
    models = {
        "Dummy Baseline": DummyClassifier(strategy="most_frequent"),
        "Logistic Regression": LogisticRegression(class_weight="balanced", max_iter=1000, random_state=42),
        "Random Forest": RandomForestClassifier(n_estimators=100, max_depth=6, class_weight="balanced", random_state=42),
    }

    results = []
    fitted_pipes = {}

    print("\n--- Model Evaluation on Test Set ---")
    for name, clf in models.items():
        pipe = build_pipeline(clf)
        pipe.fit(X_train, y_train)
        fitted_pipes[name] = pipe

        metrics = evaluate_model(name, pipe, X_test, y_test)
        results.append(metrics)

    df_results = pd.DataFrame(results)
    print(df_results.to_string(index=False, float_format=lambda x: f"{x:.3f}"))

    # Save best model for demo.py
    joblib.dump(fitted_pipes["Logistic Regression"], MODEL_PATH)
    print(f"\nSaved Logistic Regression pipeline to: {MODEL_PATH}")

    # 4. Generate a clean 3-panel summary plot
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.8))

    # Panel 1: ROC Curves
    ax1 = axes[0]
    for name, pipe in fitted_pipes.items():
        if hasattr(pipe, "predict_proba"):
            probs = pipe.predict_proba(X_test)[:, 1]
            fpr, tpr, _ = roc_curve(y_test, probs)
            auc = roc_auc_score(y_test, probs)
            ax1.plot(fpr, tpr, label=f"{name} (AUC = {auc:.2f})", linewidth=2)
    ax1.plot([0, 1], [0, 1], "k--", alpha=0.5, label="Chance (AUC = 0.50)")
    ax1.set_title("A) ROC Curves", fontweight="bold")
    ax1.set_xlabel("False Positive Rate")
    ax1.set_ylabel("True Positive Rate (Recall)")
    ax1.legend(loc="lower right")

    # Panel 2: Confusion Matrix for Logistic Regression
    ax2 = axes[1]
    lr_pred = fitted_pipes["Logistic Regression"].predict(X_test)
    cm = confusion_matrix(y_test, lr_pred)
    sns.heatmap(
        cm, annot=True, fmt=",d", cmap="Blues", cbar=False, ax=ax2,
        xticklabels=["Good/Better", "Fair/Poor"],
        yticklabels=["Good/Better", "Fair/Poor"],
    )
    ax2.set_title("B) Confusion Matrix (Logistic Reg)", fontweight="bold")
    ax2.set_xlabel("Predicted Label")
    ax2.set_ylabel("True Label")

    # Panel 3: Key Risk Factors (Odds Ratios from Logistic Regression)
    ax3 = axes[2]
    lr_clf = fitted_pipes["Logistic Regression"].named_steps["clf"]
    ohe = fitted_pipes["Logistic Regression"].named_steps["prep"].named_transformers_["cat"].named_steps["encoder"]
    encoded_cols = list(NUMERIC_FEATURES) + list(ohe.get_feature_names_out(CATEGORICAL_FEATURES))
    coefs = lr_clf.coef_[0]

    # Map selected key features to friendly names
    key_features = {
        "DIBEV_A_2.0": "No Diabetes",
        "CHDEV_A_2.0": "No Heart Disease",
        "DEPEV_A_2.0": "No Depression",
        "HYPEV_A_2.0": "No Hypertension",
        "ASEV_A_2.0": "No Asthma",
        "EDUCP_A_7.0": "College Degree",
    }
    plot_names, plot_ors = [], []
    for col, label in key_features.items():
        if col in encoded_cols:
            idx = encoded_cols.index(col)
            plot_names.append(label)
            plot_ors.append(np.exp(coefs[idx]))

    y_pos = np.arange(len(plot_names))
    ax3.barh(y_pos, plot_ors, color="#2b5c8f")
    ax3.axvline(1.0, color="red", linestyle="--", alpha=0.8, label="Neutral (OR = 1.0)")
    ax3.set_yticks(y_pos)
    ax3.set_yticklabels(plot_names)
    ax3.set_title("C) Clinical Odds Ratios (<1 means Protective)", fontweight="bold")
    ax3.set_xlabel("Adjusted Odds Ratio (exp(coef))")
    ax3.legend(loc="lower right")

    plt.tight_layout()
    plt.savefig(OUTPUT_PLOT, dpi=200)
    plt.close()
    print(f"Summary visualization saved to: {OUTPUT_PLOT}")


if __name__ == "__main__":
    main()
