"""
Synthetic Dataset Generation Script for Cognitive Load Detection.

This standalone script generates realistic synthetic typing biometric data representing 
three cognitive load levels (LOW, MEDIUM, HIGH) for prototype development and machine learning testing.

README NOTE:
Synthetic data is used for prototype development and demonstration. Real-world validation 
requires ethically collected typing data with experimentally established cognitive-load labels.
"""

import os
import sys
import numpy as np
import pandas as pd

# Ensure root package import path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

FEATURE_COLUMNS = [
    "typing_speed_wpm",
    "characters_per_minute",
    "average_key_hold_time",
    "average_inter_key_interval",
    "median_inter_key_interval",
    "typing_time",
    "pause_count",
    "average_pause_duration",
    "maximum_pause_duration",
    "backspace_count",
    "correction_rate",
    "error_rate",
    "typing_rhythm",
    "keystrokes_per_second",
    "total_keystrokes"
]

TARGET_COLUMN = "cognitive_load"
CLASS_LABELS = ["LOW", "MEDIUM", "HIGH"]


def generate_synthetic_dataset(n_samples_per_class: int = 400, random_state: int = 42) -> pd.DataFrame:
    """
    Generate realistic synthetic keystroke biometric dataset.
    
    :param n_samples_per_class: Number of samples to generate per class
    :param random_state: Seed for reproducibility
    :return: Dataframe containing 15 features + target cognitive load label
    """
    np.random.seed(random_state)
    records = []

    for label in CLASS_LABELS:
        for _ in range(n_samples_per_class):
            if label == "LOW":
                # High fluency, fast typing, low error, low hold/flight times
                wpm = float(np.clip(np.random.normal(68.0, 9.0), 45.0, 95.0))
                hold_time = float(np.clip(np.random.normal(95.0, 10.0), 70.0, 125.0))
                inter_key = float(np.clip(np.random.normal(135.0, 18.0), 90.0, 185.0))
                median_inter_key = inter_key * float(np.random.uniform(0.92, 0.98))
                typing_time = float(np.clip(np.random.normal(24.0, 4.0), 12.0, 40.0))
                pause_count = int(np.clip(np.random.poisson(0.3), 0, 2))
                avg_pause = float(np.random.uniform(1500, 1900)) if pause_count > 0 else 0.0
                max_pause = avg_pause * float(np.random.uniform(1.0, 1.3)) if pause_count > 0 else 0.0
                backspaces = int(np.clip(np.random.poisson(1.2), 0, 5))
                error_rate = float(np.clip(np.random.beta(1.2, 50.0), 0.0, 0.04))
                rhythm = float(np.clip(np.random.normal(45.0, 8.0), 20.0, 75.0))

            elif label == "MEDIUM":
                # Moderate speed, occasional pauses and corrections
                wpm = float(np.clip(np.random.normal(42.0, 7.0), 28.0, 58.0))
                hold_time = float(np.clip(np.random.normal(145.0, 15.0), 115.0, 185.0))
                inter_key = float(np.clip(np.random.normal(230.0, 25.0), 170.0, 310.0))
                median_inter_key = inter_key * float(np.random.uniform(0.91, 0.97))
                typing_time = float(np.clip(np.random.normal(38.0, 6.0), 22.0, 55.0))
                pause_count = int(np.clip(np.random.poisson(2.2), 1, 5))
                avg_pause = float(np.random.normal(1950, 250))
                max_pause = avg_pause * float(np.random.uniform(1.2, 1.5))
                backspaces = int(np.clip(np.random.poisson(6.5), 3, 14))
                error_rate = float(np.clip(np.random.beta(2.5, 30.0), 0.02, 0.10))
                rhythm = float(np.clip(np.random.normal(110.0, 20.0), 60.0, 160.0))

            else:  # HIGH
                # Slow speed, long hold/inter-key times, frequent pauses & corrections
                wpm = float(np.clip(np.random.normal(22.0, 5.0), 10.0, 35.0))
                hold_time = float(np.clip(np.random.normal(210.0, 25.0), 160.0, 290.0))
                inter_key = float(np.clip(np.random.normal(380.0, 45.0), 280.0, 550.0))
                median_inter_key = inter_key * float(np.random.uniform(0.90, 0.96))
                typing_time = float(np.clip(np.random.normal(62.0, 10.0), 35.0, 95.0))
                pause_count = int(np.clip(np.random.poisson(5.8), 3, 12))
                avg_pause = float(np.random.normal(2850, 400))
                max_pause = avg_pause * float(np.random.uniform(1.3, 1.8))
                backspaces = int(np.clip(np.random.poisson(16.0), 8, 32))
                error_rate = float(np.clip(np.random.beta(4.0, 18.0), 0.07, 0.28))
                rhythm = float(np.clip(np.random.normal(210.0, 35.0), 130.0, 320.0))

            cpm = wpm * 5.0
            kps = (wpm * 5.0) / 60.0
            total_keystrokes = int(max(10, typing_time * kps + backspaces))
            correction_rate = backspaces / float(total_keystrokes)

            records.append({
                "typing_speed_wpm": round(wpm, 2),
                "characters_per_minute": round(cpm, 2),
                "average_key_hold_time": round(hold_time, 2),
                "average_inter_key_interval": round(inter_key, 2),
                "median_inter_key_interval": round(median_inter_key, 2),
                "typing_time": round(typing_time, 2),
                "pause_count": pause_count,
                "average_pause_duration": round(avg_pause, 2),
                "maximum_pause_duration": round(max_pause, 2),
                "backspace_count": backspaces,
                "correction_rate": round(correction_rate, 4),
                "error_rate": round(error_rate, 4),
                "typing_rhythm": round(rhythm, 2),
                "keystrokes_per_second": round(kps, 2),
                "total_keystrokes": total_keystrokes,
                "cognitive_load": label
            })

    df = pd.DataFrame(records)
    df = df.sample(frac=1.0, random_state=random_state).reset_index(drop=True)
    return df


def main():
    output_dir = os.path.join("data", "raw")
    os.makedirs(output_dir, exist_ok=True)
    file_path = os.path.join(output_dir, "synthetic_typing_data.csv")

    df = generate_synthetic_dataset(n_samples_per_class=400, random_state=42)
    df.to_csv(file_path, index=False)
    print(f"Synthetic dataset generated successfully! Total rows: {len(df)}")
    print(f"Saved to: {file_path}")


if __name__ == "__main__":
    main()
