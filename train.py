"""
train.py — Training script for the Indian Spam Detector.
Trains two models (Naive Bayes + Logistic Regression), compares them,
saves the best model to model/model.joblib and the full report to model/report.json.
"""

from pathlib import Path
import json
import numpy as np
import pandas as pd
import joblib

from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, precision_recall_fscore_support, confusion_matrix,
)

# ------------------- Config -------------------
RANDOM_STATE = 42
DATA_PATH = Path("data") / "messages.csv"
MODEL_DIR = Path("model")
MODEL_PATH = MODEL_DIR / "model.joblib"
REPORT_PATH = MODEL_DIR / "report.json"
# Fixed label order — confusion matrix and report use this order
LABELS = ["normal", "spam", "scam"]


# ------------------- Data -------------------
def load_data():
    """Load CSV, drop empty/duplicate rows, and clean the text."""
    df = pd.read_csv(DATA_PATH)
    df = df.dropna(subset=["text", "label"])
    df["text"] = (
        df["text"].astype(str)
        .str.lower()
        .str.replace(r"\s+", " ", regex=True)  # collapse extra spaces
        .str.strip()
    )
    df["label"] = df["label"].astype(str).str.lower().str.strip()
    df = df[df["text"] != ""]
    df = df[df["label"].isin(LABELS)]
    df = df.drop_duplicates(subset=["text"]).reset_index(drop=True)
    return df


# ------------------- Features -------------------
def build_vectorizer():
    """Combine word (1,2) and char (3,5) TF-IDF via FeatureUnion (handles Hinglish spelling variants)."""
    word_vec = TfidfVectorizer(
        analyzer="word", ngram_range=(1, 2), lowercase=True, sublinear_tf=True,
    )
    char_vec = TfidfVectorizer(
        analyzer="char", ngram_range=(3, 5), lowercase=True,
        sublinear_tf=True, min_df=2,
    )
    return FeatureUnion([
        ("word_tfidf", word_vec),
        ("char_tfidf", char_vec),
    ])


def build_models():
    """Return two pipelines: Naive Bayes and Logistic Regression."""
    return {
        "naive_bayes": Pipeline([
            ("features", build_vectorizer()),
            ("clf", MultinomialNB()),
        ]),
        "logistic_regression": Pipeline([
            ("features", build_vectorizer()),
            ("clf", LogisticRegression(
                max_iter=1000,
                class_weight="balanced",
                random_state=RANDOM_STATE,
            )),
        ]),
    }


# ------------------- Evaluation -------------------
def evaluate(y_true, y_pred, labels):
    """Return accuracy, per-class P/R/F1, macro F1, and confusion matrix."""
    acc = float(accuracy_score(y_true, y_pred))
    p, r, f1, support = precision_recall_fscore_support(
        y_true, y_pred, labels=labels, zero_division=0
    )
    return {
        "accuracy": acc,
        "macro_f1": float(np.mean(f1)),
        "precision": {lbl: float(v) for lbl, v in zip(labels, p)},
        "recall":    {lbl: float(v) for lbl, v in zip(labels, r)},
        "f1":        {lbl: float(v) for lbl, v in zip(labels, f1)},
        "support":   {lbl: int(v)   for lbl, v in zip(labels, support)},
        "confusion_matrix": confusion_matrix(y_true, y_pred, labels=labels).tolist(),
    }


def cross_validate(pipeline, X, y):
    """Run 5-fold stratified CV and return mean and std of macro F1."""
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    scores = cross_val_score(pipeline, X, y, cv=cv, scoring="f1_macro", n_jobs=-1)
    return float(scores.mean()), float(scores.std())


# ------------------- Pretty printing -------------------
def print_comparison(results):
    """Print a clean comparison table to the terminal."""
    rows = []
    for name, info in results.items():
        m = info["metrics"]
        rows.append({
            "model": name,
            "accuracy": f"{m['accuracy']:.4f}",
            "macro_f1": f"{m['macro_f1']:.4f}",
            "cv_macro_f1": f"{m['cv_macro_f1_mean']:.4f} ± {m['cv_macro_f1_std']:.4f}",
        })
    table = pd.DataFrame(rows)
    print("\n=== Model Comparison (Test Set + 5-fold CV) ===")
    print(table.to_string(index=False))


def print_model_details(name, metrics):
    """Print per-class metrics and confusion matrix for one model."""
    print(f"\n--- {name} ---")
    header = f"{'label':10s} {'precision':>10s} {'recall':>10s} {'f1':>10s} {'support':>8s}"
    print(header)
    print("-" * len(header))
    for lbl in LABELS:
        print(
            f"{lbl:10s} "
            f"{metrics['precision'][lbl]:10.4f} "
            f"{metrics['recall'][lbl]:10.4f} "
            f"{metrics['f1'][lbl]:10.4f} "
            f"{metrics['support'][lbl]:8d}"
        )
    print(f"Confusion matrix (rows=true, cols=pred), order={LABELS}:")
    for row in metrics["confusion_matrix"]:
        print("  ", row)


# ------------------- Main -------------------
def main():
    if not DATA_PATH.exists():
        print(f"[ERROR] Data file not found: {DATA_PATH}. Please create data/messages.csv first.")
        return

    print(f"Loading data: {DATA_PATH}")
    df = load_data()
    print(f"Rows after cleaning: {len(df)}")
    print("Label counts:")
    print(df["label"].value_counts().to_string())

    # to_numpy(dtype=object) — convert pandas 3.x pyarrow strings to numpy object arrays
    X = df["text"].astype(str).to_numpy(dtype=object)
    y = df["label"].astype(str).to_numpy(dtype=object)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=RANDOM_STATE,
    )
    print(f"\nTrain: {len(X_train)} | Test: {len(X_test)}")

    models = build_models()
    results = {}

    for name, pipe in models.items():
        print(f"\n>>> Training: {name}")
        pipe.fit(X_train, y_train)
        y_pred = pipe.predict(X_test)

        metrics = evaluate(y_test, y_pred, LABELS)
        cv_mean, cv_std = cross_validate(pipe, X, y)
        metrics["cv_macro_f1_mean"] = cv_mean
        metrics["cv_macro_f1_std"] = cv_std

        results[name] = {"pipeline": pipe, "metrics": metrics}
        print_model_details(name, metrics)
        print(f"Test Accuracy: {metrics['accuracy']:.4f} | "
              f"Macro F1: {metrics['macro_f1']:.4f} | "
              f"5-fold CV F1: {cv_mean:.4f} ± {cv_std:.4f}")

    print_comparison(results)

    # Best model = highest test macro F1
    best_name = max(results, key=lambda n: results[n]["metrics"]["macro_f1"])
    best_pipe = results[best_name]["pipeline"]
    print(f"\nBest model: {best_name} (macro F1 = "
          f"{results[best_name]['metrics']['macro_f1']:.4f})")

    # Save model + report
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(best_pipe, MODEL_PATH)
    print(f"[OK] Model saved: {MODEL_PATH}")

    report = {
        "random_state": RANDOM_STATE,
        "labels": LABELS,
        "data": {
            "total": int(len(df)),
            "train_size": int(len(X_train)),
            "test_size": int(len(X_test)),
            "label_counts": {k: int(v) for k, v in df["label"].value_counts().items()},
        },
        "models": {name: results[name]["metrics"] for name in results},
        "best_model": best_name,
    }
    with open(REPORT_PATH, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    print(f"[OK] Report saved: {REPORT_PATH}")


if __name__ == "__main__":
    main()