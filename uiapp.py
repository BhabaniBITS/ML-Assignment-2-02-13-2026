from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Optional, Tuple

import joblib
import numpy as np
import pandas as pd
import seaborn as sns
import streamlit as st
import matplotlib.pyplot as plt

from model.metrics import compute_metrics, get_confusion_and_report
from model.preprocessing import load_schema, validate_inference_columns


ARTIFACT_DIR = Path(__file__).parent / "model" / "artifacts"
SCHEMA_PATH = ARTIFACT_DIR / "feature_schema.json"
METRICS_PATH = ARTIFACT_DIR / "metrics_summary.json"


MODEL_FILES = {
    "Logistic Regression": "logistic_regression.joblib",
    "Decision Tree": "decision_tree.joblib",
    "kNN": "knn.joblib",
    "Naive Bayes": "naive_bayes.joblib",
    "Random Forest (Ensemble)": "random_forest_ensemble.joblib",  # fallback name if you rename
    "XGBoost (Ensemble)": "xgboost_ensemble.joblib",
}


def find_model_path(model_name: str) -> Optional[Path]:
    """
    Allows flexible filenames (since we generated safe_name in training script).
    We'll search artifact dir for something close if exact is not found.
    """
    # Preferred mapping (if you manually rename)
    preferred = MODEL_FILES.get(model_name)
    if preferred:
        p = ARTIFACT_DIR / preferred
        if p.exists():
            return p

    # Search by normalized pattern used in train_models.py
    safe_name = model_name.lower().replace(" ", "_").replace("(", "").replace(")", "").replace("-", "_")
    candidates = list(ARTIFACT_DIR.glob(f"{safe_name}*.joblib"))
    if candidates:
        return candidates[0]
    return None


def load_metrics_summary() -> Dict[str, Dict[str, float]]:
    if METRICS_PATH.exists():
        return json.loads(METRICS_PATH.read_text(encoding="utf-8"))
    return {}


def pretty_metric_card(label: str, value: float) -> str:
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return f"**{label}:** N/A"
    return f"**{label}:** {value:.4f}"


def plot_confusion_matrix(cm: np.ndarray, class_labels: Optional[list] = None) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(5.2, 4.2))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        cbar=False,
        ax=ax,
        xticklabels=class_labels if class_labels else "auto",
        yticklabels=class_labels if class_labels else "auto",
    )
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_title("Confusion Matrix")
    fig.tight_layout()
    return fig


def try_predict_proba(pipe, X: pd.DataFrame) -> Optional[np.ndarray]:
    if hasattr(pipe, "predict_proba"):
        try:
            return pipe.predict_proba(X)
        except Exception:
            return None
    return None


def evaluate_on_uploaded(
    pipe,
    df: pd.DataFrame,
    target_col: Optional[str],
) -> Tuple[Optional[Dict[str, float]], Optional[np.ndarray], Optional[str]]:
    """
    If target_col is provided and exists -> computes metrics + confusion matrix + report.
    Otherwise -> returns None metrics but still allows predictions outside this function.
    """
    if not target_col or target_col not in df.columns:
        return None, None, None

    X = df.drop(columns=[target_col])
    y_true = df[target_col].to_numpy()

    y_pred = pipe.predict(X)
    y_proba = try_predict_proba(pipe, X)

    m = compute_metrics(y_true=y_true, y_pred=np.asarray(y_pred), y_proba=y_proba)
    metrics_dict = {
        "Accuracy": m.accuracy,
        "AUC": m.auc,
        "Precision": m.precision,
        "Recall": m.recall,
        "F1": m.f1,
        "MCC": m.mcc,
    }
    cm, report = get_confusion_and_report(y_true, np.asarray(y_pred))
    return metrics_dict, cm, report


def main() -> None:
    st.set_page_config(page_title="ML Assignment 2 - Classification Models", layout="wide")

    st.title("Machine Learning Assignment 2 — Classification Model Demo by AA202505967")
    st.caption(
        "Upload a CSV file, choose a model, and view evaluation metrics + confusion matrix."
    )

    if not SCHEMA_PATH.exists():
        st.error(
            "Model artifacts not found. Train models first:\n\n"
            "1) Put dataset CSV in data/\n"
            "2) Run: python -m model.train_models --csv data/bank_additional_full.csv --target y\n"
            "3) Re-run this Streamlit app"
        )
        st.stop()

    schema = load_schema(SCHEMA_PATH)

    with st.sidebar:
        st.header("Controls")
        model_name = st.selectbox(
            "Select a model",
            [
                "Logistic Regression",
                "Decision Tree",
                "kNN",
                "Naive Bayes",
                "Random Forest (Ensemble)",
                "XGBoost (Ensemble)",
            ],
        )
        st.write("---")
        st.subheader("Upload test CSV")
        uploaded = st.file_uploader("Choose a CSV file", type=["csv"])

        st.write("---")
        st.subheader("Target column (optional)")
        use_target = st.checkbox("My CSV includes the target column", value=True)
        target_col = st.text_input("Target column name", value=schema.target_col) if use_target else ""

    # Show baseline metrics (from training) as a comparison table
    st.subheader("Baseline metrics (from saved test split)")
    baseline = load_metrics_summary()
    if baseline:
        baseline_df = pd.DataFrame(baseline).T.reset_index().rename(columns={"index": "ML Model Name"})
        st.dataframe(baseline_df, use_container_width=True)
    else:
        st.info("No baseline metrics found (metrics_summary.json missing). Train models again.")

    # Load selected model pipeline
    model_path = find_model_path(model_name)
    if model_path is None or not model_path.exists():
        st.error(f"Could not locate saved model for: {model_name}. Please retrain models.")
        st.stop()

    pipe = joblib.load(model_path)

    # Data input
    if uploaded is None:
        st.warning("No CSV uploaded. Using bundled sample test file if available.")
        sample_path = Path(__file__).parent / "data" / "sample_test.csv"
        if sample_path.exists():
            df = pd.read_csv(sample_path)
            st.success("Loaded data/sample_test.csv")
        else:
            st.info("Upload a CSV to proceed.")
            st.stop()
    else:
        df = pd.read_csv(uploaded)

    st.write("### Uploaded / Loaded Data Preview")
    st.dataframe(df.head(20), use_container_width=True)

    # Validate columns for inference
    # If target is present, drop it before validation
    df_for_validation = df.drop(columns=[target_col], errors="ignore") if target_col else df
    ok, missing = validate_inference_columns(df_for_validation, schema)
    if not ok:
        st.error(
            "Your CSV is missing required feature columns.\n\n"
            f"Missing ({len(missing)}): {missing}\n\n"
            "Tip: Use the generated sample_test.csv as a template."
        )
        st.stop()

    # Evaluate if possible
    metrics_dict, cm, report = evaluate_on_uploaded(pipe, df, target_col if use_target else None)

    col1, col2 = st.columns([1, 1])

    with col1:
        st.subheader(f"Selected model: {model_name}")
        if metrics_dict is None:
            st.info(
                "Target column not provided or not found. Showing predictions only.\n"
                "Enable 'My CSV includes the target column' to compute metrics."
            )
        else:
            st.write("#### Evaluation Metrics (on uploaded data)")
            st.write(pretty_metric_card("Accuracy", metrics_dict["Accuracy"]))
            st.write(pretty_metric_card("AUC", metrics_dict["AUC"]))
            st.write(pretty_metric_card("Precision", metrics_dict["Precision"]))
            st.write(pretty_metric_card("Recall", metrics_dict["Recall"]))
            st.write(pretty_metric_card("F1", metrics_dict["F1"]))
            st.write(pretty_metric_card("MCC", metrics_dict["MCC"]))

    with col2:
        if cm is not None:
            st.subheader("Confusion Matrix")
            fig = plot_confusion_matrix(cm)
            st.pyplot(fig, clear_figure=True)
        else:
            st.subheader("Classification Report")
            st.code("Upload CSV with target column to generate confusion matrix & report.")

    if report is not None:
        st.subheader("Classification Report (text)")
        st.code(report)

    # Predictions output
    st.subheader("Predictions")
    X_pred = df.drop(columns=[target_col], errors="ignore") if target_col else df
    preds = pipe.predict(X_pred)

    out = df.copy()
    out["prediction"] = preds

    proba = try_predict_proba(pipe, X_pred)
    if proba is not None and proba.ndim == 2 and proba.shape[1] >= 2:
        out["proba_positive"] = proba[:, 1]

    st.dataframe(out.head(30), use_container_width=True)

    csv_bytes = out.to_csv(index=False).encode("utf-8")
    st.download_button(
        "Download predictions as CSV",
        data=csv_bytes,
        file_name="predictions.csv",
        mime="text/csv",
    )



if __name__ == "__main__":
    main()