import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import (
    accuracy_score, roc_auc_score, precision_score,
    recall_score, f1_score, matthews_corrcoef,
    confusion_matrix, classification_report
)
import matplotlib.pyplot as plt
import seaborn as sns

# --- Streamlit UI ---

# Main Title (centered, dark blue)
st.markdown("<h1 style='text-align: center; color: darkblue;'>📊 Bank Marketing Classification App</h1>", unsafe_allow_html=True)

# Subtitle (green)
st.markdown("<h3 style='text-align: center; color: green;'>Interactive ML demo with 6 models and evaluation metrics by <b>Bhabani (2025aa05967)</b></h3>", unsafe_allow_html=True)

# Section Headings (purple)
st.markdown("<h2 style='color: purple;'>🔍 Step 2: Model Evaluation</h2>", unsafe_allow_html=True)


st.title("Bank Marketing Classification App")
st.write("Interactive ML demo with 6 models and evaluation metrics by Bhabani(2025aa05967)")

# --- Dataset Upload ---
uploaded_file = st.file_uploader("Upload CSV dataset", type=["csv"])
if uploaded_file is not None:
    data = pd.read_csv(uploaded_file, sep=';')
    st.write("### Dataset Preview")
    st.dataframe(data.head())

    # --- Preprocessing ---
    categorical_cols = data.select_dtypes(include=['object']).columns
    for col in categorical_cols:
        le = LabelEncoder()
        data[col] = le.fit_transform(data[col])

    X = data.drop('y', axis=1)
    y = data['y']

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    # --- Model Selection Dropdown ---
    model_choice = st.selectbox(
        "Choose a model",
        ["Logistic Regression", "Decision Tree", "KNN", "Naive Bayes", "Random Forest", "XGBoost"]
    )

    # --- Define Models ---
    models = {
        "Logistic Regression": LogisticRegression(max_iter=5000),
        "Decision Tree": DecisionTreeClassifier(random_state=42),
        "KNN": KNeighborsClassifier(),
        "Naive Bayes": GaussianNB(),
        "Random Forest": RandomForestClassifier(random_state=42),
        "XGBoost": XGBClassifier(eval_metric='logloss')
    }

    model = models[model_choice]
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1] if hasattr(model, "predict_proba") else None

    # --- Metrics ---
    st.write("### Evaluation Metrics")
    st.write("Accuracy:", accuracy_score(y_test, y_pred))
    if y_prob is not None:
        st.write("AUC:", roc_auc_score(y_test, y_prob))
    st.write("Precision:", precision_score(y_test, y_pred))
    st.write("Recall:", recall_score(y_test, y_pred))
    st.write("F1 Score:", f1_score(y_test, y_pred))
    st.write("MCC:", matthews_corrcoef(y_test, y_pred))

    # --- Confusion Matrix ---
    st.write("### Confusion Matrix")
    cm = confusion_matrix(y_test, y_pred)
    fig, ax = plt.subplots()
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    st.pyplot(fig)

    # --- Classification Report ---
    st.write("### Classification Report")
    st.text(classification_report(y_test, y_pred))
