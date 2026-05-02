# schema.py
# ─────────────────────────────────────────────
# Input/output data shapes for FastAPI
# ─────────────────────────────────────────────

from pydantic import BaseModel, Field
from typing import Optional

class PatientInput(BaseModel):
    pregnancies:       int   = Field(..., ge=0,   le=20)
    glucose:           float = Field(..., ge=0,   le=600)
    blood_pressure:    float = Field(..., ge=0,   le=200)
    skin_thickness:    float = Field(..., ge=0,   le=100)
    insulin:           float = Field(..., ge=0,   le=1000)
    bmi:               float = Field(..., ge=0,   le=100)
    diabetes_pedigree: float = Field(..., ge=0.0, le=2.42)
    age:               int   = Field(..., ge=1,   le=120)

    class Config:
        json_schema_extra = {
            "example": {
                "pregnancies": 2,
                "glucose": 120,
                "blood_pressure": 80,
                "skin_thickness": 20,
                "insulin": 85,
                "bmi": 28.5,
                "diabetes_pedigree": 0.35,
                "age": 35
            }
        }


class PredictionOutput(BaseModel):
    prediction:      int            # 0 or 1
    probability:     float          # 0.0 to 1.0
    label:           str            # "Diabetic" or "Not Diabetic"
    confidence:      str            # "High" / "Medium" / "Low"
    bmi:             float          # echoed back
    message:         str            # human readable summary
    model_time_ms:   Optional[float] = None   # ← model only time
    server_time_ms:  Optional[float] = None   # ← full server time
