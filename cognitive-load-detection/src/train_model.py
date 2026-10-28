"""
Leak-Free Model Training, Cross-Validation & Metric Serialization Engine.

This script:
1. Loads preprocessed typing dataset.
2. Trains Scikit-Learn Pipelines (StandardScaler + Estimator):
   - Logistic Regression
   - Decision Tree Classifier
   - Random Forest Classifier
   - Gradient Boosting Classifier
3. Executes 5-Fold Stratified Cross Validation.
4. Computes feature importances and model coefficients.
5. Exports model bundle (`models/cognitive_load_model.pkl`) and metrics JSON (`models/model_metrics.json`).
6. Generates visual evaluation plots.
"""

import os
import sys
import json
import joblib
import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.model_selection import StratifiedKFold, cross_val_score

# Ensure root package import path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.data_preprocessing import prepare_data, FEATURE_COLUMNS
from src.evaluate_model import evaluate_model_performance, extract_feature_importance, generate_visualizations


def train_and_compare_models(models_dir: str = "models", data_dir: str = "data", viz_dir: str = "visualizations"):
    """Train multiple ML pipelines, evaluate, select best model, and save metrics."""
    os.makedirs(models_dir, exist_ok=True)

    # 1. Load Data
    X_train_scaled, X_test_scaled, y_train, y_test, scaler, X_train_raw, X_test_raw = prepare_data(data_dir)
    raw_df = pd.read_csv(os.path.join(data_dir, "raw", "synthetic_typing_data.csv"))

    # 2. Define Leak-Free Pipelines
    candidate_pipelines = {
        "Logistic Regression": Pipeline([
            ("scaler", StandardScaler()),
            ("classifier", LogisticRegression(max_iter=1000, random_state=42))
        ]),
        "Decision Tree": Pipeline([
            ("scaler", StandardScaler()),
            ("classifier", DecisionTreeClassifier(max_depth=5, random_state=42))
        ]),
        "Random Forest": Pipeline([
            ("scaler", StandardScaler()),
            ("classifier", RandomForestClassifier(n_estimators=100, max_depth=8, random_state=42))
        ]),
        "Gradient Boosting": Pipeline([
            ("scaler", StandardScaler()),
            ("classifier", GradientBoostingClassifier(n_estimators=100, learning_rate=0.1, random_state=42))
        ])
    }

    results = []
    trained_pipelines = {}
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    print("=" * 70)
    print("STARTING MACHINE LEARNING MODEL TRAINING & COMPARISON")
    print("=" * 70)

    best_score = -1.0
    best_model_name = ""
    best_pipeline = None

    metrics_export = {
        "models": {},
        "winning_model": "",
        "feature_names": FEATURE_COLUMNS
    }

    for name, pipeline in candidate_pipelines.items():
        # Stratified 5-Fold Cross Validation on raw X_train (pipeline handles scaling internally per fold)
        cv_scores = cross_val_score(pipeline, X_train_raw, y_train, cv=cv, scoring='f1_weighted')
        cv_mean = float(np.mean(cv_scores))
        cv_std = float(np.std(cv_scores))

        # Fit pipeline on full training data
        pipeline.fit(X_train_raw, y_train)
        trained_pipelines[name] = pipeline

        # Evaluate on test set
        perf = evaluate_model_performance(pipeline, X_test_raw, y_test)
        classifier = pipeline.named_steps["classifier"]
        feature_importance = extract_feature_importance(classifier, FEATURE_COLUMNS)

        perf["cv_f1_mean"] = round(cv_mean, 4)
        perf["cv_f1_std"] = round(cv_std, 4)
        perf["feature_importance"] = feature_importance

        metrics_export["models"][name] = perf

        results.append({
            "Model": name,
            "Accuracy": perf["accuracy"],
            "Precision": perf["precision"],
            "Recall": perf["recall"],
            "F1-Score": perf["f1_score"],
            "CV F1 Mean": perf["cv_f1_mean"]
        })

        print(f"[{name}] -> Test Acc: {perf['accuracy']:.4f} | F1: {perf['f1_score']:.4f} | 5-Fold CV F1: {cv_mean:.4f} (+/- {cv_std:.4f})")

        if perf["f1_score"] > best_score:
            best_score = perf["f1_score"]
            best_model_name = name
            best_pipeline = pipeline

    results_df = pd.DataFrame(results)
    metrics_export["winning_model"] = best_model_name

    print("=" * 70)
    print(f"WINNING MODEL SELECTED: {best_model_name} (Test F1: {best_score:.4f})")
    print("=" * 70)

    # 3. Save Best Model Bundle
    best_classifier = best_pipeline.named_steps["classifier"]
    best_scaler = best_pipeline.named_steps["scaler"]
    best_feature_importance = extract_feature_importance(best_classifier, FEATURE_COLUMNS)

    model_bundle = {
        "pipeline": best_pipeline,
        "model": best_classifier,
        "scaler": best_scaler,
        "feature_names": FEATURE_COLUMNS,
        "model_name": best_model_name,
        "classes": list(best_classifier.classes_),
        "feature_importance": best_feature_importance
    }

    model_path = os.path.join(models_dir, "cognitive_load_model.pkl")
    joblib.dump(model_bundle, model_path)
    print(f"Saved best model pipeline bundle to: {model_path}")

    # 4. Export Calculated Metrics JSON
    metrics_path = os.path.join(models_dir, "model_metrics.json")
    with open(metrics_path, "w") as f:
        json.dump(metrics_export, f, indent=4)
    print(f"Saved model metrics JSON to: {metrics_path}")

    # 5. Generate Visualizations
    generate_visualizations(
        df_raw=raw_df,
        results_df=results_df,
        best_model=best_pipeline,
        feature_importance_dict=best_feature_importance,
        X_test=X_test_raw,
        y_test=y_test,
        viz_dir=viz_dir
    )

    return results_df, best_model_name


if __name__ == "__main__":
    train_and_compare_models()
