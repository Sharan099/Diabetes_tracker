# model.py — updated to load from S3 + timing measurement
import os
import time
import boto3
import joblib
import numpy as np
import io

FEATURE_ORDER = [
    "Pregnancies", "Glucose", "BloodPressure",
    "SkinThickness", "Insulin", "BMI",
    "DiabetesPedigreeFunction", "Age"
]

def load_model():

    S3_BUCKET = os.environ.get("S3_BUCKET")
    MODEL_KEY  = os.environ.get("MODEL_KEY", "models/BEST_tuned_model.pkl")

    if S3_BUCKET:
        print(f"[→] Loading model from S3: s3://{S3_BUCKET}/{MODEL_KEY}")
        try:
            s3     = boto3.client("s3")
            buffer = io.BytesIO()
            s3.download_fileobj(S3_BUCKET, MODEL_KEY, buffer)
            buffer.seek(0)
            model  = joblib.load(buffer)
            print("[✓] Model loaded from S3 successfully!")
            return model
        except Exception as e:
            print(f"[!] S3 load failed: {e} — trying local fallback...")

    # Use relative path — works inside Docker container
    LOCAL_PATH = os.environ.get("MODEL_PATH", "./models/BEST_tuned_model.pkl")
    print(f"[→] Loading model from local: {LOCAL_PATH}")

    if not os.path.exists(LOCAL_PATH):
        raise FileNotFoundError(
            f"Model not found at: {LOCAL_PATH}\n"
            f"Either set S3_BUCKET in .env or copy model into container."
        )

    model = joblib.load(LOCAL_PATH)
    print("[✓] Model loaded locally!")
    return model


def build_feature_array(data) -> np.ndarray:
    return np.array([[
        data.pregnancies,
        data.glucose,
        data.blood_pressure,
        data.skin_thickness,
        data.insulin,
        data.bmi,
        data.diabetes_pedigree,
        data.age
    ]])


def get_confidence(probability: float) -> str:
    if probability >= 0.75 or probability <= 0.25:
        return "High"
    elif probability >= 0.60 or probability <= 0.40:
        return "Medium"
    return "Low"


def run_prediction(model, data) -> dict:

    X = build_feature_array(data)

    # ── Time ONLY the model prediction ──────────
    t_start     = time.perf_counter()

    prediction  = int(model.predict(X)[0])
    probability = float(model.predict_proba(X)[0][1])

    t_end       = time.perf_counter()
    model_ms    = round((t_end - t_start) * 1000, 4)
    # ────────────────────────────────────────────

    print(f"[🧠] Model prediction time: {model_ms}ms")

    label      = "Diabetic" if prediction == 1 else "Not Diabetic"
    confidence = get_confidence(probability)
    message    = (
        f"Model predicts {round(probability*100)}% probability of diabetes. "
        f"Confidence: {confidence}. Please consult a doctor."
    )

    return {
        "prediction":    prediction,
        "probability":   round(probability, 4),
        "label":         label,
        "confidence":    confidence,
        "bmi":           round(data.bmi, 2),
        "message":       message,
        "model_time_ms": model_ms       # ← model only time
    }
