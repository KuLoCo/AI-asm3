import os
import json
import joblib
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.multiclass import OneVsRestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import (classification_report, precision_recall_curve,)

def train_models(X, y, results_dir="results", test_size=0.2, dev_size=0.25, random_state=42, class_weight="balanced"):

    os.makedirs(results_dir, exist_ok=True)

    X_train_dev, X_test, y_train_dev, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)
    X_train, X_dev, y_train, y_dev = train_test_split(X_train_dev, y_train_dev, test_size=dev_size, random_state=random_state)

    models = {
        "logistic_regression": OneVsRestClassifier(
            LogisticRegression(max_iter=2000, class_weight=class_weight)
        ),
        "linear_svc": OneVsRestClassifier(
            LinearSVC(class_weight=class_weight, dual=False, max_iter=20000)
        ),
    }

    all_metrics = {}
    all_thresholds = {}

    for name, model in models.items():
        print(f"[TRAIN] {name}")
        model.fit(X_train, y_train)

        if hasattr(model, "predict_proba"):
            dev_scores = model.predict_proba(X_dev)
            test_scores = model.predict_proba(X_test)
        else:
            dev_scores = model.decision_function(X_dev)
            test_scores = model.decision_function(X_test)

        n_labels = y.shape[1]
        thresholds = np.zeros(n_labels, dtype=float)
        for i in range(n_labels):
            p, r, t = precision_recall_curve(y_dev[:, i], dev_scores[:, i])
            f1 = 2 * p * r / (p + r + 1e-8)
            best = np.argmax(f1)
            thresholds[i] = t[best] if best < len(t) else 0.5

        y_pred = (test_scores >= thresholds).astype(int)
        report = classification_report(
            y_test, y_pred, output_dict=True, zero_division=0
        )
        all_metrics[name] = report
        all_thresholds[name] = thresholds.tolist()
        joblib.dump(model, os.path.join(results_dir, f"{name}.joblib"))

    with open(os.path.join(results_dir, "metrics.json"), "w", encoding="utf-8") as f:
        json.dump(all_metrics, f, indent=4)
    with open(os.path.join(results_dir, "thresholds.json"), "w", encoding="utf-8") as f:
        json.dump(all_thresholds, f, indent=4)

    print(f"[DONE] Models, metrics, and thresholds saved to '{results_dir}'")
    return all_metrics, all_thresholds