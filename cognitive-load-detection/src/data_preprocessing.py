"""
Data Preprocessing & Dataset Management Module.

This module handles:
1. Synthetic dataset loading / creation.
2. Leak-free Stratified Train-Test Splitting.
3. Feature Scaling via StandardScaler fitted strictly on training partition.
4. Exporting raw and processed CSV data files.
"""

import os
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from typing import Tuple

from src.generate_dataset import generate_synthetic_dataset, FEATURE_COLUMNS, TARGET_COLUMN, CLASS_LABELS


def prepare_data(data_dir: str = "data") -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series, StandardScaler, pd.DataFrame, pd.DataFrame]:
    """
    Prepare dataset for machine learning without data leakage.

    :param data_dir: Target data directory
    :return: (X_train_scaled, X_test_scaled, y_train, y_test, scaler, X_train_raw, X_test_raw)
    """
    raw_dir = os.path.join(data_dir, "raw")
    processed_dir = os.path.join(data_dir, "processed")

    os.makedirs(raw_dir, exist_ok=True)
    os.makedirs(processed_dir, exist_ok=True)

    raw_csv_path = os.path.join(raw_dir, "synthetic_typing_data.csv")
    processed_csv_path = os.path.join(processed_dir, "processed_typing_features.csv")

    if not os.path.exists(raw_csv_path):
        print(f"Generating synthetic dataset at {raw_csv_path}...")
        df = generate_synthetic_dataset(n_samples_per_class=400, random_state=42)
        df.to_csv(raw_csv_path, index=False)
    else:
        df = pd.read_csv(raw_csv_path)

    # Save copy to processed directory
    df.to_csv(processed_csv_path, index=False)

    X = df[FEATURE_COLUMNS]
    y = df[TARGET_COLUMN]

    # Stratified Train-Test Split (80% Train, 20% Test)
    X_train_raw, X_test_raw, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y
    )

    # Fit scaler ONLY on training data to prevent data leakage
    scaler = StandardScaler()
    X_train_scaled_arr = scaler.fit_transform(X_train_raw)
    X_test_scaled_arr = scaler.transform(X_test_raw)

    X_train_scaled = pd.DataFrame(X_train_scaled_arr, columns=FEATURE_COLUMNS, index=X_train_raw.index)
    X_test_scaled = pd.DataFrame(X_test_scaled_arr, columns=FEATURE_COLUMNS, index=X_test_raw.index)

    return X_train_scaled, X_test_scaled, y_train, y_test, scaler, X_train_raw, X_test_raw


if __name__ == "__main__":
    X_tr, X_te, y_tr, y_te, sc, _, _ = prepare_data("data")
    print(f"Leak-free data preparation complete! Train: {X_tr.shape}, Test: {X_te.shape}")
