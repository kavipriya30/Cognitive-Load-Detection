"""
FastAPI Web Application Server for Cognitive Load Detection using Typing Patterns.
Provides a modern, real-website user interface with live keystroke capture,
explainable ML inference, interactive charts, session history, and model analytics.
"""

import os
import sys
import json
import time
from datetime import datetime
from typing import Dict, Any, List, Optional
import pandas as pd
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

# Ensure src directory is in path
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
if BASE_DIR not in sys.path:
    sys.path.append(BASE_DIR)

from src.feature_engineering import compute_typing_features, FEATURE_NAMES
from src.prediction import predict_cognitive_load, MODEL_PATH_DEFAULT

SESSION_HISTORY_FILE = os.path.join(BASE_DIR, "data", "session_history.csv")
MODEL_METRICS_FILE = os.path.join(BASE_DIR, "models", "model_metrics.json")
RAW_DATA_FILE = os.path.join(BASE_DIR, "data", "raw", "synthetic_typing_data.csv")

app = FastAPI(
    title="Cognitive Load Detection AI",
    description="Real-Time Typing Keystroke Biometrics & Machine Learning Platform",
    version="2.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class KeyStrokePayload(BaseModel):
    keydowns: Optional[List[Dict[str, Any]]] = []
    keyups: Optional[List[Dict[str, Any]]] = []
    typed_text: Optional[str] = ""
    target_prompt: Optional[str] = ""
    manual: Optional[bool] = False
    manual_features: Optional[Dict[str, float]] = None


def save_session_record(record: dict):
    """Save an anonymized session record to CSV (no typed text stored)."""
    os.makedirs(os.path.dirname(SESSION_HISTORY_FILE), exist_ok=True)
    df_new = pd.DataFrame([record])
    if os.path.exists(SESSION_HISTORY_FILE):
        try:
            df_existing = pd.read_csv(SESSION_HISTORY_FILE)
            df_combined = pd.concat([df_existing, df_new], ignore_index=True)
        except Exception:
            df_combined = df_new
    else:
        df_combined = df_new
    df_combined.to_csv(SESSION_HISTORY_FILE, index=False)


@app.get("/api/health")
def health_check():
    return {
        "status": "online",
        "system": "Cognitive Load Detection AI",
        "model_loaded": os.path.exists(MODEL_PATH_DEFAULT),
        "timestamp": datetime.now().isoformat()
    }


@app.get("/api/metrics")
def get_metrics():
    """Return model evaluation metrics, cross validation scores, and feature importances."""
    if not os.path.exists(MODEL_METRICS_FILE):
        raise HTTPException(status_code=404, detail="Model metrics file not found. Please train models first.")
    with open(MODEL_METRICS_FILE, "r") as f:
        data = json.load(f)
    return data


@app.get("/api/history")
def get_history():
    """Return past user session records (privacy safe, no typed text stored)."""
    if not os.path.exists(SESSION_HISTORY_FILE):
        return {"history": []}
    try:
        df = pd.read_csv(SESSION_HISTORY_FILE)
        # Return most recent sessions first
        records = df.tail(50).to_dict(orient="records")
        records.reverse()
        return {"history": records}
    except Exception as e:
        return {"history": [], "error": str(e)}


@app.delete("/api/history")
@app.post("/api/history/clear")
def clear_history():
    """Clear local session history."""
    if os.path.exists(SESSION_HISTORY_FILE):
        try:
            os.remove(SESSION_HISTORY_FILE)
            return {"success": True, "message": "Session history cleared."}
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))
    return {"success": True, "message": "No history to clear."}


@app.get("/api/dataset-sample")
def get_dataset_sample():
    """Return sample data rows from raw dataset for exploration."""
    if not os.path.exists(RAW_DATA_FILE):
        return {"samples": []}
    try:
        df = pd.read_csv(RAW_DATA_FILE)
        return {
            "total_rows": len(df),
            "columns": list(df.columns),
            "samples": df.head(15).to_dict(orient="records")
        }
    except Exception as e:
        return {"samples": [], "error": str(e)}


@app.post("/api/predict")
def predict(payload: KeyStrokePayload):
    """
    Extract 15 biometric features from keystroke timing streams and predict cognitive load.
    PRIVACY SAFE: Typed text is never persisted to disk.
    """
    try:
        if payload.manual and payload.manual_features:
            features = payload.manual_features
        else:
            if not payload.keydowns or len(payload.keydowns) < 3:
                raise HTTPException(status_code=400, detail="Too few keystrokes recorded. Please type a longer passage.")
            features = compute_typing_features(
                keydown_events=payload.keydowns,
                keyup_events=payload.keyups or [],
                typed_text=payload.typed_text or "",
                target_text=payload.target_prompt or ""
            )

        # Run Inference
        result = predict_cognitive_load(features, MODEL_PATH_DEFAULT)

        # Generate unique Session ID & Timestamp
        session_id = f"SES-{int(time.time() * 1000) % 1000000}"
        timestamp_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        # Save session history (strictly behavioral metrics, NO typed content stored)
        history_record = {
            "session_id": session_id,
            "timestamp": timestamp_str,
            "typing_speed_wpm": features["typing_speed_wpm"],
            "typing_time": features["typing_time"],
            "average_key_hold_time": features["average_key_hold_time"],
            "average_inter_key_interval": features["average_inter_key_interval"],
            "pause_count": features["pause_count"],
            "backspace_count": features["backspace_count"],
            "correction_rate": features["correction_rate"],
            "typing_rhythm": features["typing_rhythm"],
            "predicted_cognitive_load": result["cognitive_load"],
            "confidence_pct": result["confidence_pct"]
        }
        save_session_record(history_record)

        return {
            "success": True,
            "session_id": session_id,
            "timestamp": timestamp_str,
            "features": features,
            "prediction": result
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction error: {str(e)}")


@app.get("/", response_class=HTMLResponse)
def serve_website():
    """Serve the modern single-page web application."""
    index_path = os.path.join(BASE_DIR, "templates", "index.html")
    if os.path.exists(index_path):
        with open(index_path, "r", encoding="utf-8") as f:
            return f.read()
    return "<h1>Cognitive Load AI Server Online</h1><p>templates/index.html is loading...</p>"


if __name__ == "__main__":
    import uvicorn
    print("Starting Cognitive Load Detection Web App on http://localhost:8000 ...")
    uvicorn.run("server:app", host="0.0.0.0", port=8000, reload=False)
