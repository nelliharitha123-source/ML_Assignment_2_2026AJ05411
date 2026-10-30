"""Streamlit demonstration app for the five saved classification pipelines."""

from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import streamlit as st
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    matthews_corrcoef,
    precision_score,
    recall_score,
    roc_auc_score,
)

ROOT = Path(__file__).resolve().parent
MODEL_DIR = ROOT / "model"
TARGET_COLUMN = "target"
MODEL_FILES = {
    "Logistic Regression": "logistic_regression.joblib",
    "Decision Tree": "decision_tree.joblib",
    "KNN": "knn.joblib",
    "Naive Bayes": "naive_bayes.joblib",
    "Support Vector Machine": "support_vector_machine.joblib",
}

st.set_page_config(page_title="Classifier Comparison Lab", page_icon="ML", layout="wide")
st.title("Classifier Comparison Lab")
st.caption("Breast Cancer Wisconsin Diagnostic dataset | Five saved classifiers | No retraining during upload")

with st.sidebar:
    st.header("Evaluation controls")
    selected_model = st.selectbox("Choose a model", list(MODEL_FILES))
    uploaded_file = st.file_uploader("Upload test CSV", type="csv")
    st.info("The CSV must contain the 30 dataset feature columns and a target column named target.")
    st.download_button(
        "Download sample test_data.csv",
        (ROOT / "test_data.csv").read_bytes(),
        file_name="test_data.csv",
        mime="text/csv",
    )

@st.cache_data
def load_default_data():
    return pd.read_csv(ROOT / "test_data.csv")

@st.cache_resource
def load_model(model_name):
    return joblib.load(MODEL_DIR / MODEL_FILES[model_name])

if uploaded_file is None:
    data = load_default_data()
    st.info("Using the included test_data.csv. Upload another compatible test CSV from the sidebar to evaluate it.")
else:
    data = pd.read_csv(uploaded_file)

if TARGET_COLUMN not in data.columns:
    st.error("The uploaded CSV must contain a target column named 'target'.")
    st.stop()

x_data = data.drop(columns=[TARGET_COLUMN])
y_data = data[TARGET_COLUMN]
model = load_model(selected_model)

def score_model(fitted_model, features, labels):
    predicted = fitted_model.predict(features)
    if hasattr(fitted_model, "predict_proba"):
        probabilities = fitted_model.predict_proba(features)[:, 1]
    else:
        probabilities = fitted_model.decision_function(features)
    values = {
        "Accuracy": accuracy_score(labels, predicted),
        "AUC": roc_auc_score(labels, probabilities),
        "Precision": precision_score(labels, predicted, zero_division=0),
        "Recall": recall_score(labels, predicted, zero_division=0),
        "F1": f1_score(labels, predicted, zero_division=0),
        "MCC": matthews_corrcoef(labels, predicted),
    }
    return predicted, values


try:
    predictions, metric_values = score_model(model, x_data, y_data)
except Exception as error:
    st.error(f"The file does not match the saved model schema: {error}")
    st.stop()

st.subheader(f"Metrics: {selected_model}")
metric_columns = st.columns(6)
for column, (name, value) in zip(metric_columns, metric_values.items()):
    column.metric(name, f"{value:.4f}")

left, right = st.columns(2)
with left:
    st.subheader("Confusion matrix")
    matrix = confusion_matrix(y_data, predictions)
    figure, axis = plt.subplots(figsize=(5, 4))
    sns.heatmap(matrix, annot=True, fmt="d", cmap="Blues", cbar=False, ax=axis)
    axis.set_xlabel("Predicted label")
    axis.set_ylabel("True label")
    axis.set_title(selected_model)
    st.pyplot(figure, clear_figure=True)

with right:
    st.subheader("Classification report")
    report = classification_report(y_data, predictions, output_dict=True, zero_division=0)
    st.dataframe(pd.DataFrame(report).T.round(4))

st.subheader("All five models on this test data")
comparison = {name: score_model(load_model(name), x_data, y_data)[1] for name in MODEL_FILES}
st.dataframe(pd.DataFrame(comparison).T.round(4))

st.subheader("Uploaded test data preview")
st.dataframe(data.head(10))
st.caption(f"Evaluated {len(data)} rows using the saved {selected_model} pipeline. No model was retrained on uploaded data.")
