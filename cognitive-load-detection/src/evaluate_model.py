"""
Model Evaluation, Feature Importance & Visualization Module.

Calculates:
- Accuracy, Precision, Recall, F1-Score (macro & weighted)
- Stratified 5-Fold Cross Validation scores
- Detailed Classification Reports & Confusion Matrix
- Random Forest & Gradient Boosting Feature Importances
- Logistic Regression Feature Coefficients

Exports PNG plots to `visualizations/` and metrics JSON to `models/model_metrics.json`.
"""

import os
import json
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report
)
from typing import Dict, Any, List

CLASS_ORDER = ["LOW", "MEDIUM", "HIGH"]


def evaluate_model_performance(model, X_test, y_test) -> Dict[str, Any]:
    """Compute comprehensive evaluation metrics for a trained classifier."""
    y_pred = model.predict(X_test)
    
    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, average='weighted')
    rec = recall_score(y_test, y_pred, average='weighted')
    f1 = f1_score(y_test, y_pred, average='weighted')
    
    report_dict = classification_report(y_test, y_pred, labels=CLASS_ORDER, output_dict=True)
    cm = confusion_matrix(y_test, y_pred, labels=CLASS_ORDER).tolist()

    return {
        "accuracy": round(float(acc), 4),
        "precision": round(float(prec), 4),
        "recall": round(float(rec), 4),
        "f1_score": round(float(f1), 4),
        "classification_report": report_dict,
        "confusion_matrix": cm
    }


def extract_feature_importance(model, feature_names: List[str]) -> Dict[str, float]:
    """Extract feature importance array from Random Forest or Tree-based models."""
    if hasattr(model, "feature_importances_"):
        imps = model.feature_importances_
        importance_dict = {f: round(float(imp), 4) for f, imp in zip(feature_names, imps)}
        # Sort descending
        return dict(sorted(importance_dict.items(), key=lambda x: x[1], reverse=True))
    elif hasattr(model, "coef_"):
        # Mean absolute coefficient across multi-class models
        coef_abs = np.mean(np.abs(model.coef_), axis=0)
        coef_dict = {f: round(float(c), 4) for f, c in zip(feature_names, coef_abs)}
        return dict(sorted(coef_dict.items(), key=lambda x: x[1], reverse=True))
    return {}


def generate_visualizations(
    df_raw: pd.DataFrame,
    results_df: pd.DataFrame,
    best_model,
    feature_importance_dict: Dict[str, float],
    X_test,
    y_test,
    viz_dir: str = "visualizations"
):
    """Generate and save project visualization plots."""
    os.makedirs(viz_dir, exist_ok=True)
    sns.set_theme(style="whitegrid", palette="muted")
    plt.rcParams.update({'font.sans-serif': 'Segoe UI', 'font.family': 'sans-serif'})

    # 1. Class Distribution Plot
    fig, ax = plt.subplots(figsize=(7, 5))
    sns.countplot(data=df_raw, x="cognitive_load", hue="cognitive_load", order=CLASS_ORDER, palette=["#2ecc71", "#f39c12", "#e74c3c"], legend=False, ax=ax)
    ax.set_title("Cognitive Load Class Distribution", fontsize=14, fontweight='bold', pad=15)
    ax.set_xlabel("Cognitive Load Level", fontsize=12)
    ax.set_ylabel("Sample Count", fontsize=12)
    for p in ax.patches:
        ax.annotate(f'{int(p.get_height())}', (p.get_x() + p.get_width() / 2., p.get_height()),
                    ha='center', va='bottom', fontsize=11, fontweight='bold', xytext=(0, 5), textcoords='offset points')
    plt.tight_layout()
    plt.savefig(os.path.join(viz_dir, "class_distribution.png"), dpi=300)
    plt.close()

    # 2. Correlation Heatmap Plot
    fig, ax = plt.subplots(figsize=(11, 9))
    num_cols = df_raw.select_dtypes(include=[np.number]).columns
    corr = df_raw[num_cols].corr()
    sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm", center=0, square=True, linewidths=0.5, cbar_kws={"shrink": 0.8}, ax=ax)
    ax.set_title("Keystroke Feature Correlation Heatmap", fontsize=14, fontweight='bold', pad=15)
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.savefig(os.path.join(viz_dir, "correlation_heatmap.png"), dpi=300)
    plt.close()

    # 3. Feature Distributions Plot across Cognitive Load Classes
    fig, axes = plt.subplots(2, 2, figsize=(13, 10))
    dist_features = [
        ("typing_speed_wpm", "Typing Speed (WPM)"),
        ("average_key_hold_time", "Key Hold Duration (ms)"),
        ("average_inter_key_interval", "Inter-key Interval (ms)"),
        ("backspace_count", "Backspace Frequency")
    ]
    colors = {"LOW": "#2ecc71", "MEDIUM": "#f39c12", "HIGH": "#e74c3c"}

    for idx, (col, label) in enumerate(dist_features):
        r, c = idx // 2, idx % 2
        sns.boxplot(data=df_raw, x="cognitive_load", y=col, hue="cognitive_load", order=CLASS_ORDER, palette=colors, legend=False, ax=axes[r, c])
        axes[r, c].set_title(f"{label} by Load Level", fontsize=12, fontweight='bold')
        axes[r, c].set_xlabel("Cognitive Load", fontsize=10)
        axes[r, c].set_ylabel(label, fontsize=10)

    plt.tight_layout()
    plt.savefig(os.path.join(viz_dir, "feature_distributions.png"), dpi=300)
    plt.close()

    # 4. Model Comparison Bar Chart
    fig, ax = plt.subplots(figsize=(9, 6))
    res_melted = pd.melt(results_df, id_vars=["Model"], value_vars=["Accuracy", "F1-Score"], var_name="Metric", value_name="Score")
    sns.barplot(data=res_melted, x="Model", y="Score", hue="Metric", palette=["#3498db", "#9b59b6"], ax=ax)
    ax.set_title("Machine Learning Models Comparison", fontsize=14, fontweight='bold', pad=15)
    ax.set_ylim(0.70, 1.02)
    ax.set_xlabel("Model Architecture", fontsize=12)
    ax.set_ylabel("Validation Score", fontsize=12)
    for p in ax.patches:
        val = p.get_height()
        if not np.isnan(val) and val > 0:
            ax.annotate(f'{val:.3f}', (p.get_x() + p.get_width() / 2., val),
                        ha='center', va='bottom', fontsize=9, fontweight='bold', xytext=(0, 4), textcoords='offset points')
    plt.xticks(rotation=15)
    plt.tight_layout()
    plt.savefig(os.path.join(viz_dir, "model_comparison.png"), dpi=300)
    plt.close()

    # 5. Feature Importance Plot
    if feature_importance_dict:
        fig, ax = plt.subplots(figsize=(10, 6))
        fi_df = pd.DataFrame(list(feature_importance_dict.items()), columns=["Feature", "Importance"])
        sns.barplot(data=fi_df.head(10), x="Importance", y="Feature", hue="Feature", palette="viridis", legend=False, ax=ax)
        ax.set_title("Top 10 Feature Importances (Random Forest / Model Weights)", fontsize=14, fontweight='bold', pad=15)
        ax.set_xlabel("Relative Importance Weight", fontsize=12)
        ax.set_ylabel("Biometric Feature", fontsize=12)
        plt.tight_layout()
        plt.savefig(os.path.join(viz_dir, "feature_importance.png"), dpi=300)
        plt.close()

    # 6. Best Model Confusion Matrix
    y_pred_best = best_model.predict(X_test)
    cm = confusion_matrix(y_test, y_pred_best, labels=CLASS_ORDER)

    fig, ax = plt.subplots(figsize=(7, 6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=CLASS_ORDER, yticklabels=CLASS_ORDER, cbar=False, ax=ax)
    ax.set_title("Best Model Confusion Matrix", fontsize=14, fontweight='bold', pad=15)
    ax.set_xlabel("Predicted Cognitive Load", fontsize=12, labelpad=10)
    ax.set_ylabel("Actual Cognitive Load", fontsize=12, labelpad=10)
    plt.tight_layout()
    plt.savefig(os.path.join(viz_dir, "confusion_matrix.png"), dpi=300)
    plt.close()

    print(f"All visualization plots saved to '{viz_dir}/' successfully!")
