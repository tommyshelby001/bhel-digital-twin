"""
BHEL Digital Twin — Turbine Failure Prediction Model Training
================================================================
Generates a realistic synthetic dataset for a 500 MW steam turbo-generator
based on physical sensor behavior during normal operation vs degrading
(pre-failure) conditions, then trains + evaluates a Random Forest classifier.

Run: python train_model.py
Output: models/turbine_model.pkl, models/metrics_report.json, models/confusion_matrix.png
"""

import numpy as np
import pandas as pd
import json
import os
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report, roc_auc_score
)
import joblib

np.random.seed(42)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(BASE_DIR, "models")
os.makedirs(MODEL_DIR, exist_ok=True)

FEATURES = ["steam_temp_C", "steam_pressure_kgcm2", "vibration_mms",
            "bearing_temp_C", "rpm", "oil_pressure_kgcm2", "load_percent"]

N_SAMPLES = 5000


def generate_dataset(n=N_SAMPLES):
    """
    Simulates n turbine operating snapshots across the full severity spectrum
    (0 = healthy baseline, 1 = imminent failure), matching the physical
    degradation pattern used in the live simulator (app.py):
      - steam temp rises, pressure drops
      - vibration and bearing temp spike
      - rpm and oil pressure sag
      - load falls as the unit is throttled back
    Label is defined by domain thresholds (not a copy of the RF's own logic),
    with noise so the classes overlap somewhat — a realistic, non-trivial
    classification problem rather than a rule lookup.
    """
    severity = np.random.beta(1.8, 3.0, n)  # skewed toward healthy operation

    steam_temp = np.random.normal(540 + severity * 35, 4, n)
    steam_pressure = np.random.normal(170 - severity * 15, 3, n)
    vibration = np.clip(np.random.normal(2.5 + severity * 7, 0.6, n), 0, None)
    bearing_temp = np.random.normal(65 + severity * 30, 3.2, n)
    rpm = np.random.normal(3000 - severity * 50, 11, n)
    oil_pressure = np.clip(np.random.normal(2.2 - severity * 0.8, 0.12, n), 0, None)
    load_pct = np.clip(np.random.normal(85 - severity * 20, 5.5, n), 0, 100)

    df = pd.DataFrame({
        "steam_temp_C": steam_temp.round(1),
        "steam_pressure_kgcm2": steam_pressure.round(1),
        "vibration_mms": vibration.round(2),
        "bearing_temp_C": bearing_temp.round(1),
        "rpm": rpm.round(0),
        "oil_pressure_kgcm2": oil_pressure.round(2),
        "load_percent": load_pct.round(1),
    })

    # Domain-rule ground truth (independent of severity noise -> realistic overlap)
    risk_score = (
        (df["vibration_mms"] > 5.5).astype(int) +
        (df["bearing_temp_C"] > 82).astype(int) +
        (df["oil_pressure_kgcm2"] < 1.4).astype(int) +
        (df["steam_temp_C"] > 565).astype(int) +
        (df["rpm"] < 2920).astype(int)
    )
    label = (risk_score >= 2).astype(int)

    # small label noise to avoid a perfectly separable / unrealistic dataset
    flip_mask = np.random.rand(n) < 0.02
    label = np.where(flip_mask, 1 - label, label)

    df["failure_risk"] = label
    return df


def main():
    print("Generating synthetic training dataset...")
    df = generate_dataset()
    print(f"  {len(df)} samples | failure_risk positive rate: {df['failure_risk'].mean():.1%}")

    X = df[FEATURES]
    y = df["failure_risk"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )

    print("Training Random Forest classifier...")
    model = RandomForestClassifier(
        n_estimators=200,
        max_depth=10,
        min_samples_leaf=5,
        class_weight="balanced",
        random_state=42,
        n_jobs=-1
    )
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]

    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred)
    rec = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    auc = roc_auc_score(y_test, y_proba)
    cv_scores = cross_val_score(model, X, y, cv=5)
    cm = confusion_matrix(y_test, y_pred)

    print("\n=== Model Evaluation ===")
    print(f"Accuracy:  {acc:.4f}")
    print(f"Precision: {prec:.4f}")
    print(f"Recall:    {rec:.4f}")
    print(f"F1 Score:  {f1:.4f}")
    print(f"ROC-AUC:   {auc:.4f}")
    print(f"5-fold CV: {cv_scores.mean():.4f} (+/- {cv_scores.std():.4f})")
    print("\nConfusion Matrix:")
    print(cm)
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, target_names=["SAFE", "FAILURE_RISK"]))

    feature_importance = dict(zip(FEATURES, model.feature_importances_.round(4).tolist()))
    print("\nFeature Importances:")
    for f, imp in sorted(feature_importance.items(), key=lambda x: -x[1]):
        print(f"  {f}: {imp}")

    # Save model
    model_path = os.path.join(MODEL_DIR, "turbine_model.pkl")
    joblib.dump(model, model_path)
    print(f"\nModel saved to {model_path}")

    # Save metrics report (for admin panel / report generator to consume)
    metrics = {
        "trained_on": pd.Timestamp.now().strftime("%Y-%m-%d %H:%M:%S"),
        "n_samples": N_SAMPLES,
        "n_train": len(X_train),
        "n_test": len(X_test),
        "accuracy": round(acc, 4),
        "precision": round(prec, 4),
        "recall": round(rec, 4),
        "f1_score": round(f1, 4),
        "roc_auc": round(auc, 4),
        "cv_mean": round(float(cv_scores.mean()), 4),
        "cv_std": round(float(cv_scores.std()), 4),
        "confusion_matrix": cm.tolist(),
        "feature_importance": feature_importance,
        "model_type": "RandomForestClassifier",
        "n_estimators": 200,
        "features": FEATURES
    }
    metrics_path = os.path.join(MODEL_DIR, "metrics_report.json")
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"Metrics report saved to {metrics_path}")


if __name__ == "__main__":
    main()
