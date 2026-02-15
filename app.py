import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import (
    accuracy_score, roc_auc_score, precision_score, recall_score,
    f1_score, matthews_corrcoef, confusion_matrix, classification_report
)

# Import training functions from model scripts
from model import logistic_regression, decision_tree, knn, naive_bayes, random_forest, xgboost_model

# Title
st.title("📊 Bank Marketing Classification App")
st.markdown("Upload test data, select a model, and view evaluation metrics.")

# Sidebar: Model selection
model_choice = st.sidebar.selectbox(
    "Choose a classifier:",
    ["Logistic Regression", "Decision Tree", "KNN", "Naive Bayes", "Random Forest", "XGBoost"]
)

# File uploader (CSV)
uploaded_file = st.sidebar.file_uploader("Upload Test CSV", type=["csv"])

if uploaded_file is not None:
    test_data = pd.read_csv(uploaded_file)
    st.write("### Uploaded Test Data Preview")
    st.dataframe(test_data.head())

    if "y" in test_data.columns:
        X_test = test_data.drop("y", axis=1)
        y_test = test_data["y"].map({"yes":1, "no":0})
    else:
        st.error("CSV must contain target column 'y'.")
        st.stop()

    # Run selected model dynamically
    if model_choice == "Logistic Regression":
        clf = logistic_regression.get_model()
    elif model_choice == "Decision Tree":
        clf = decision_tree.get_model()
    elif model_choice == "KNN":
        clf = knn.get_model()
    elif model_choice == "Naive Bayes":
        clf = naive_bayes.get_model()
    elif model_choice == "Random Forest":
        clf = random_forest.get_model()
    elif model_choice == "XGBoost":
        clf = xgboost_model.get_model()

    # Predictions
    y_pred = clf.predict(X_test)
    y_proba = clf.predict_proba(X_test)[:,1] if hasattr(clf, "predict_proba") else None

    # Metrics
    st.markdown(f"## 🎯 Results for **{model_choice}**")
    st.success(f"Accuracy: {accuracy_score(y_test, y_pred):.4f}")
    st.info(f"Precision: {precision_score(y_test, y_pred):.4f}")
    st.warning(f"Recall: {recall_score(y_test, y_pred):.4f}")
    st.error(f"F1 Score: {f1_score(y_test, y_pred):.4f}")
    st.write(f"**MCC:** {matthews_corrcoef(y_test, y_pred):.4f}")
    if y_proba is not None:
        st.write(f"**AUC:** {roc_auc_score(y_test, y_proba):.4f}")

    # Confusion Matrix
    cm = confusion_matrix(y_test, y_pred)
    fig, ax = plt.subplots()
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=["No", "Yes"], yticklabels=["No", "Yes"], ax=ax)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    st.pyplot(fig)

    # Classification Report
    st.write("### 📑 Classification Report")
    report = classification_report(y_test, y_pred, target_names=["No", "Yes"], output_dict=True)
    report_df = pd.DataFrame(report).transpose()
    st.dataframe(report_df.style.background_gradient(cmap="coolwarm"))
else:
    st.info("⬅️ Upload a CSV file with test data to proceed.")