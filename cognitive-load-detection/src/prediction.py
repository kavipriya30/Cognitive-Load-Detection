"""
Inference & Explainable Prediction Pipeline Engine.

This module loads the trained ML pipeline bundle and executes real-time inference,
providing confidence metrics, feature importance attribution, and behavioral driver explanations.
"""

import os
import joblib
import pandas as pd
import numpy as np
from typing import Dict, Any, List, Tuple

MODEL_PATH_DEFAULT = os.path.join(os.path.dirname(__file__), "..", "models", "cognitive_load_model.pkl")


class CognitiveLoadPredictor:
    """Predictor class for running leak-free inference & feature explanation."""

    def __init__(self, model_path: str = MODEL_PATH_DEFAULT):
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model file not found at {model_path}. Please run src/train_model.py first.")

        bundle = joblib.load(model_path)
        self.pipeline = bundle.get("pipeline")
        self.model = bundle["model"]
        self.scaler = bundle["scaler"]
        self.feature_names = bundle["feature_names"]
        self.model_name = bundle.get("model_name", "Classifier")
        self.classes = bundle.get("classes", ["LOW", "MEDIUM", "HIGH"])
        self.feature_importance = bundle.get("feature_importance", {})

    def predict(self, features_dict: Dict[str, float]) -> Dict[str, Any]:
        """
        Execute real-time inference on 15 keystroke features.

        :param features_dict: Dictionary containing extracted typing biometrics
        :return: Structured prediction result dict with confidence, feature drivers & disclaimer
        """
        # Ensure correct column ordering DataFrame
        df_input = pd.DataFrame([features_dict])[self.feature_names]

        # Use full pipeline if present, else scale & predict
        if self.pipeline:
            predicted_class = self.pipeline.predict(df_input)[0]
            if hasattr(self.pipeline, "predict_proba"):
                probs = self.pipeline.predict_proba(df_input)[0]
            else:
                probs = None
        else:
            X_scaled_arr = self.scaler.transform(df_input)
            X_scaled = pd.DataFrame(X_scaled_arr, columns=self.feature_names)
            predicted_class = self.model.predict(X_scaled)[0]
            probs = self.model.predict_proba(X_scaled)[0] if hasattr(self.model, "predict_proba") else None

        # Class probabilities
        if probs is not None:
            class_probs = {cls: round(float(prob * 100), 1) for cls, prob in zip(self.classes, probs)}
            confidence = round(float(np.max(probs) * 100), 1)
        else:
            class_probs = {cls: (100.0 if cls == predicted_class else 0.0) for cls in self.classes}
            confidence = 100.0

        # Extract Explainable Behavioral Drivers
        observed_patterns, top_contributors = self._explain_behavioral_drivers(features_dict, predicted_class)

        return {
            "cognitive_load": predicted_class,
            "confidence_pct": confidence,
            "class_probabilities": class_probs,
            "model_used": self.model_name,
            "features_used": features_dict,
            "observed_patterns": observed_patterns,
            "top_contributors": top_contributors,
            "disclaimer": (
                "NOTE: These results represent behavioral typing metrics used by the machine learning model. "
                "They are strictly for practical demonstration and educational evaluation, NOT medical diagnostic conclusions."
            )
        }

    def _explain_behavioral_drivers(self, f: Dict[str, float], load: str) -> Tuple[List[str], List[Dict[str, Any]]]:
        """Identify key feature deviations driving the prediction."""
        patterns = []
        contributors = []

        wpm = f.get("typing_speed_wpm", 0.0)
        hold_time = f.get("average_key_hold_time", 0.0)
        inter_key = f.get("average_inter_key_interval", 0.0)
        backspaces = f.get("backspace_count", 0)
        pauses = f.get("pause_count", 0)
        rhythm = f.get("typing_rhythm", 0.0)
        error_rate = f.get("error_rate", 0.0)

        if load == "HIGH":
            if wpm < 32.0:
                patterns.append(f"Lower typing speed ({wpm:.1f} WPM vs ~68 WPM nominal baseline)")
                contributors.append({"feature": "typing_speed_wpm", "value": f"{wpm:.1f} WPM", "impact": "High Negative Impact on Speed"})
            if inter_key > 300.0:
                patterns.append(f"Increased inter-key flight interval ({inter_key:.1f} ms delay between strokes)")
                contributors.append({"feature": "average_inter_key_interval", "value": f"{inter_key:.1f} ms", "impact": "High Delay Overhead"})
            if pauses >= 3:
                patterns.append(f"Longer hesitation pauses ({pauses} pauses > 1.5s)")
                contributors.append({"feature": "pause_count", "value": f"{pauses} pauses", "impact": "Increased Planning Delay"})
            if backspaces >= 8:
                patterns.append(f"Higher correction rate ({backspaces} backspaces)")
                contributors.append({"feature": "backspace_count", "value": f"{backspaces} deletes", "impact": "High Revision Frequency"})
            if hold_time > 170.0:
                patterns.append(f"Extended key press duration ({hold_time:.1f} ms hold time)")

        elif load == "MEDIUM":
            if 30.0 <= wpm <= 55.0:
                patterns.append(f"Moderate typing speed ({wpm:.1f} WPM)")
            if 180.0 <= inter_key <= 320.0:
                patterns.append(f"Moderate inter-key flight interval ({inter_key:.1f} ms)")
            if 1 <= pauses < 4:
                patterns.append(f"Occasional hesitation pauses ({pauses} pause > 1.5s)")
            if 3 <= backspaces < 10:
                patterns.append(f"Moderate correction activity ({backspaces} backspaces)")

        else:  # LOW
            if wpm > 55.0:
                patterns.append(f"High typing speed ({wpm:.1f} WPM)")
                contributors.append({"feature": "typing_speed_wpm", "value": f"{wpm:.1f} WPM", "impact": "High Fluency Indicator"})
            if inter_key < 170.0:
                patterns.append(f"Short inter-key flight intervals ({inter_key:.1f} ms)")
                contributors.append({"feature": "average_inter_key_interval", "value": f"{inter_key:.1f} ms", "impact": "Rapid Key Transitions"})
            if pauses <= 1:
                patterns.append("Minimal hesitation pauses")
            if backspaces <= 3:
                patterns.append(f"Low correction rate ({backspaces} backspaces)")

        if not patterns:
            patterns.append("Balanced nominal behavioral indicators across speed, timing, and error metrics.")

        return patterns, contributors


# Helper function for inference calls
def predict_cognitive_load(features_dict: Dict[str, float], model_path: str = MODEL_PATH_DEFAULT) -> Dict[str, Any]:
    predictor = CognitiveLoadPredictor(model_path)
    return predictor.predict(features_dict)


if __name__ == "__main__":
    dummy_feats = {
        "typing_speed_wpm": 22.0,
        "characters_per_minute": 110.0,
        "average_key_hold_time": 210.0,
        "average_inter_key_interval": 390.0,
        "median_inter_key_interval": 360.0,
        "typing_time": 60.0,
        "pause_count": 5,
        "average_pause_duration": 2800.0,
        "maximum_pause_duration": 3900.0,
        "backspace_count": 15,
        "correction_rate": 0.10,
        "error_rate": 0.08,
        "typing_rhythm": 210.0,
        "keystrokes_per_second": 2.5,
        "total_keystrokes": 150
    }
    res = predict_cognitive_load(dummy_feats)
    print("Explainable Prediction Result:")
    print("Cognitive Load:", res["cognitive_load"], f"({res['confidence_pct']}% confidence)")
    print("Observed Patterns:", res["observed_patterns"])
