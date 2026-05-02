# main.py
# ─────────────────────────────────────────────
# FastAPI application with full timing breakdown
# Run locally:  uvicorn main:app --reload
# Run on EC2:   uvicorn main:app --host 0.0.0.0 --port 8000
# ─────────────────────────────────────────────

import time
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.gzip import GZipMiddleware
from contextlib import asynccontextmanager

from schema import PatientInput, PredictionOutput
from model  import load_model, run_prediction

# ── Global model object ───────────────────────
ml_model = None

# ── Load model once at startup ────────────────
@asynccontextmanager
async def lifespan(app: FastAPI):
    global ml_model
    print("[→] Loading model...")
    ml_model = load_model()
    print("[✓] App ready!")
    yield
    print("[→] Shutting down.")

# ── Create FastAPI app ────────────────────────
app = FastAPI(
    title       = "Diabetes Predictor API",
    description = "ML-powered diabetes risk assessment",
    version     = "1.0.0",
    lifespan    = lifespan
)

# ── CORS ──────────────────────────────────────
app.add_middleware(
    CORSMiddleware,
    allow_origins  = ["*"],
    allow_methods  = ["POST", "GET", "OPTIONS"],
    allow_headers  = ["Content-Type"],
    max_age        = 86400    # cache 24 hours
)

# ── GZip compression ──────────────────────────
app.add_middleware(GZipMiddleware, minimum_size=100)

# ══════════════════════════════════════════════
# ENDPOINTS
# ══════════════════════════════════════════════

@app.get("/")
def root():
    return {
        "status":  "running",
        "service": "Diabetes Predictor API",
        "docs":    "Visit /docs for interactive API documentation"
    }


@app.get("/health")
def health():
    if ml_model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    return {"status": "healthy", "model_loaded": True}


@app.post("/predict", response_model=PredictionOutput)
def predict(patient: PatientInput):
    """
    Main prediction endpoint with full timing breakdown.
    """
    if ml_model is None:
        raise HTTPException(status_code=503, detail="Model not loaded yet")

    try:
        # ── Time the full server processing ───────
        t_start = time.perf_counter()

        result   = run_prediction(ml_model, patient)   # model time tracked inside

        t_end    = time.perf_counter()
        total_ms = round((t_end - t_start) * 1000, 4)
        # ─────────────────────────────────────────

        model_ms = result.get("model_time_ms", 0)
        overhead_ms = round(total_ms - model_ms, 4)

        # Print full breakdown to EC2 logs
        print(f"""
┌─── Request Timing Breakdown ───────────┐
│  Model prediction : {model_ms}ms
│  FastAPI overhead : {overhead_ms}ms
│  Total server     : {total_ms}ms
└────────────────────────────────────────┘""")

        result["server_time_ms"] = total_ms
        return result

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")


# ── Run directly (testing only) ───────────────
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="127.0.0.1", port=8000, reload=True)
