import os
import json
import joblib
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import (
    confusion_matrix,
    ConfusionMatrixDisplay,
)
from sklearn.model_selection import train_test_split

from src.load_data import load_all_data
from src.preprocess import preprocess_dataframe
from src.vectorize import vectorize_texts, encode_labels

def evaluate_models(
    data_dir,
    results_dir,
    test_size=0.2,
    random_state=42,
    top_n=4,
    use_thresholds=True,
):
    os.makedirs(results_dir, exist_ok=True)

    df, emotions = load_all_data(data_dir)
    df = preprocess_dataframe(df, emotions, clean=True, balance=False)

    X, _ = vectorize_texts(df['text'])
    y, _ = encode_labels(df['label_names'])

    _, X_test, _, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )

    supports = y_test.sum(axis=0)
    top_indices = sorted(range(len(supports)),
                         key=lambda i: supports[i],
                         reverse=True)[:top_n]

    thresholds_path = os.path.join(results_dir, 'thresholds.json')
    if use_thresholds and os.path.exists(thresholds_path):
        with open(thresholds_path, 'r', encoding='utf-8') as f:
            all_thresholds = json.load(f)
    else:
        all_thresholds = {}

    for fname in os.listdir(results_dir):
        if not fname.endswith('.joblib'):
            continue
        model_name = fname.replace('.joblib', '')
        model = joblib.load(os.path.join(results_dir, fname))

        if hasattr(model, 'predict_proba'):
            scores = model.predict_proba(X_test)
        else:
            scores = model.decision_function(X_test)

        if use_thresholds and model_name in all_thresholds:
            thr = np.array(all_thresholds[model_name])
            y_pred = (scores >= thr).astype(int)
        else:
            y_pred = model.predict(X_test)

        for idx in top_indices:
            label = emotions[idx]
            cm = confusion_matrix(y_test[:, idx], y_pred[:, idx], labels=[0,1])
            disp = ConfusionMatrixDisplay(cm, display_labels=[f'not {label}', label])
            fig, ax = plt.subplots(figsize=(4,4))
            disp.plot(ax=ax, colorbar=False)
            ax.set_title(f'{model_name} – {label}')
            out_file = os.path.join(
                results_dir, f'confusion_{model_name}_{label}.png'
            )
            fig.savefig(out_file, bbox_inches='tight')
            plt.close(fig)

    print(f"✅ Evaluation complete. Plots saved in '{results_dir}'")