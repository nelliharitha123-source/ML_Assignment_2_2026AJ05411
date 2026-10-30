"""Train the five required classifiers and create reproducible submission artifacts."""

from pathlib import Path
import json

import joblib
import numpy as np
import pandas as pd
from sklearn.datasets import load_breast_cancer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    matthews_corrcoef,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

ROOT = Path(__file__).resolve().parent
MODEL_DIR = ROOT / "model"
MODEL_DIR.mkdir(exist_ok=True)
RANDOM_STATE = 2026


def evaluate_model(model, x_test, y_test):
    """Return the six metrics required by the assignment."""
    predictions = model.predict(x_test)
    if hasattr(model, "predict_proba"):
        scores = model.predict_proba(x_test)[:, 1]
    else:
        scores = model.decision_function(x_test)
    return {
        "Accuracy": accuracy_score(y_test, predictions),
        "AUC": roc_auc_score(y_test, scores),
        "Precision": precision_score(y_test, predictions, zero_division=0),
        "Recall": recall_score(y_test, predictions, zero_division=0),
        "F1": f1_score(y_test, predictions, zero_division=0),
        "MCC": matthews_corrcoef(y_test, predictions),
    }


def main():
    dataset = load_breast_cancer(as_frame=True)
    features = dataset.data.copy()
    target = dataset.target.rename("target")
    full_data = pd.concat([features, target], axis=1)

    x_train, x_test, y_train, y_test = train_test_split(
        features,
        target,
        test_size=0.20,
        stratify=target,
        random_state=RANDOM_STATE,
    )

    models = {
        "Logistic Regression": Pipeline([
            ("scaler", StandardScaler()),
            ("classifier", LogisticRegression(max_iter=3000, random_state=RANDOM_STATE)),
        ]),
        "Decision Tree": DecisionTreeClassifier(max_depth=5, random_state=RANDOM_STATE),
        "KNN": Pipeline([
            ("scaler", StandardScaler()),
            ("classifier", KNeighborsClassifier(n_neighbors=7)),
        ]),
        "Naive Bayes": Pipeline([
            ("scaler", StandardScaler()),
            ("classifier", GaussianNB()),
        ]),
        "Support Vector Machine": Pipeline([
            ("scaler", StandardScaler()),
            ("classifier", SVC(kernel="rbf", probability=True, random_state=RANDOM_STATE)),
        ]),
    }

    metrics = {}
    for name, model in models.items():
        model.fit(x_train, y_train)
        safe_name = name.lower().replace(" ", "_")
        joblib.dump(model, MODEL_DIR / f"{safe_name}.joblib")
        metrics[name] = evaluate_model(model, x_test, y_test)

    test_data = x_test.copy()
    test_data["target"] = y_test
    test_data.to_csv(ROOT / "test_data.csv", index=False)

    metrics_frame = pd.DataFrame(metrics).T
    metrics_frame.to_csv(ROOT / "metrics.csv")
    metrics_frame.round(4).to_json(ROOT / "metrics.json", orient="index", indent=2)

    metadata = {
        "dataset": "Breast Cancer Wisconsin Diagnostic",
        "dataset_source": "scikit-learn copy of the UCI Breast Cancer Wisconsin Diagnostic dataset",
        "instances": int(full_data.shape[0]),
        "features": int(features.shape[1]),
        "target_column": "target",
        "classes": {"0": "malignant", "1": "benign"},
        "test_size": 0.20,
        "random_state": RANDOM_STATE,
        "feature_columns": list(features.columns),
    }
    (ROOT / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    print("Dataset shape:", full_data.shape)
    print("Training rows:", len(x_train), "Testing rows:", len(x_test))
    print("Saved models:", ", ".join(sorted(p.name for p in MODEL_DIR.glob("*.joblib"))))
    print("\nMetrics:")
    print(metrics_frame.round(4).to_string())


if __name__ == "__main__":
    main()
