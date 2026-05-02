# 🩺 Diabetes Risk Predictor

> An end-to-end MLOps pipeline: from raw clinical data to a live AWS-deployed prediction API with a modern web frontend.

[![Python](https://img.shields.io/badge/Python-3.11-blue?style=flat-square&logo=python)](https://python.org)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.111-009688?style=flat-square&logo=fastapi)](https://fastapi.tiangolo.com)
[![Docker](https://img.shields.io/badge/Docker-Containerised-2496ED?style=flat-square&logo=docker)](https://docker.com)
[![AWS](https://img.shields.io/badge/AWS-EC2%20%7C%20S3%20%7C%20ECR-FF9900?style=flat-square&logo=amazonaws)](https://aws.amazon.com)
[![MLflow](https://img.shields.io/badge/MLflow-Tracked-0194E2?style=flat-square&logo=mlflow)](https://mlflow.org)
[![License](https://img.shields.io/badge/License-MIT-green?style=flat-square)](LICENSE)

---

## 🎯 What This Is

A user fills in their health details — age, glucose level, BMI, family history — and gets an AI-powered diabetes risk assessment in under a second. Behind that simple interaction is a complete MLOps pipeline covering data preprocessing, model training, experiment tracking, hyperparameter tuning, containerised deployment, and cloud infrastructure.

Built from scratch as a learning project. Every piece was understood, debugged, and measured.

---

## 🏗️ Full Pipeline

```
┌─────────────────────────────────────────────────────────────┐
│                     COMPLETE PIPELINE                        │
├─────────────────────────────────────────────────────────────┤
│                                                              │
│  Raw CSV (768 records, 8 features)                          │
│         │                                                    │
│         ▼                                                    │
│  ┌─────────────────┐                                        │
│  │  PREPROCESSING  │  MICE Imputation → SMOTE              │
│  │                 │  Saved to imputed_dataset.csv         │
│  └────────┬────────┘                                        │
│           │                                                  │
│           ▼                                                  │
│  ┌─────────────────┐                                        │
│  │    TRAINING     │  5 Models × Stratified 5-Fold CV      │
│  │                 │  SMOTE inside each fold only           │
│  │  LR · RF · GB  │  MLflow tracks every run               │
│  │  XGB · CatBoost│                                         │
│  └────────┬────────┘                                        │
│           │                                                  │
│           ▼                                                  │
│  ┌─────────────────┐                                        │
│  │    TUNING       │  GridSearchCV                          │
│  │                 │  RandomizedSearchCV                    │
│  │                 │  Optuna (50 trials)                    │
│  └────────┬────────┘                                        │
│           │                                                  │
│           ▼                                                  │
│  ┌─────────────────┐                                        │
│  │  BEST MODEL     │  Logistic Regression                   │
│  │  AUC: 0.8399    │  Saved as .pkl → uploaded to S3       │
│  └────────┬────────┘                                        │
│           │                                                  │
│           ▼                                                  │
│  ┌─────────────────┐                                        │
│  │   FASTAPI       │  /predict endpoint                     │
│  │   BACKEND       │  Pydantic validation                   │
│  │                 │  Model loaded from S3                  │
│  └────────┬────────┘                                        │
│           │                                                  │
│           ▼                                                  │
│  ┌─────────────────┐                                        │
│  │    DOCKER       │  python:3.11-slim                      │
│  │   CONTAINER     │  Lean image (inference only)           │
│  └────────┬────────┘                                        │
│           │                                                  │
│           ▼                                                  │
│  ┌──────────────────────────────────────┐                   │
│  │            AWS CLOUD                 │                   │
│  │  S3 (model) → ECR (image) → EC2     │                   │
│  └────────┬─────────────────────────────┘                   │
│           │                                                  │
│           ▼                                                  │
│  ┌─────────────────┐                                        │
│  │  FRONTEND UI    │  Diabetes predictor — 5-step form             │
│  │  (Diabetes predictor)  │  Dark/light mode                       │
│  │                 │  Live latency display                  │
│  └─────────────────┘                                        │
│                                                              │
└─────────────────────────────────────────────────────────────┘
```

---

## 📊 Model Results

| Model | AUC | Accuracy | F1 | Recall |
|---|---|---|---|---|
| **Logistic Regression** | **0.8399** | **76.4%** | **0.690** | **74.2%** |
| Random Forest | 0.8277 | 75.6% | 0.671 | 70.9% |
| CatBoost | 0.8257 | 76.8% | 0.680 | 70.5% |
| Gradient Boosting | 0.8238 | 75.4% | 0.661 | 68.7% |
| XGBoost | 0.8119 | 75.1% | 0.655 | 67.5% |

Logistic Regression won. On 768 records, a linear model found the signal better than every boosted ensemble. This is consistent with the literature — complex models need more data to justify their capacity.

---

## ⚡ Latency Breakdown

```
Component                    Time        Share
─────────────────────────────────────────────
Model prediction (LR)        0.97ms       0.3%
FastAPI overhead             0.07ms       0.0%
Total server                 1.04ms       0.3%
TCP connect (Germany→US)     139ms       39.9%
Response transfer            111ms       31.9%
Browser overhead (CORS/JS)    96ms       27.6%
─────────────────────────────────────────────
Total (browser)              348ms      100.0%

curl test (no browser):      252ms
```

The server does its job in **1ms**. Everything else is the internet.

---

## 🗂️ Project Structure

```
diabetes_predictor/
│
├── data/
│   └── imputed_dataset.csv         # MICE-imputed clean data
│
├── models/
│   └── tuned/
│       └── BEST_tuned_model.pkl    # Best trained model
│
├── notebooks/
│   ├── 01_eda.ipynb                # Exploratory analysis
│   ├── 02_preprocessing.ipynb      # MICE imputation
│   └── 03_training.ipynb           # Model comparison
│
├── app/
│   ├── main.py                     # FastAPI application
│   ├── model.py                    # Model loading + prediction + timing
│   └── schema.py                   # Pydantic input/output schemas
│
├── frontend/
│   └── index.html                  # Diabetes predictor UI
│
├── training/
│   ├── train_models.py             # 5-model CV training with MLflow
│   └── hyperparameter_tuning.py   # GridSearch + Random + Optuna
│
├── Dockerfile                      # Container definition
├── requirements.txt                # Inference dependencies only
└── README.md
```

---

## 🚀 Running Locally

### 1. Clone and install

```bash
git clone https://github.com/yourusername/diabetes-predictor.git
cd diabetes-predictor
pip install -r requirements.txt
```

### 2. Start the API

```bash
# Set your S3 bucket (or leave blank to use local model)
export S3_BUCKET=your-bucket-name
export MODEL_KEY=models/BEST_tuned_model.pkl

uvicorn app.main:app --reload
```

API is live at `http://localhost:8000`  
Swagger UI at `http://localhost:8000/docs`

### 3. Open the frontend

Open `frontend/index.html` in your browser. Make sure `API_URL` in the script points to `http://localhost:8000/predict`.

---

## 🐳 Docker

```bash
# Build
docker build -t diabetes-predictor .

# Run with AWS credentials
docker run -d -p 8000:8000 \
  -e S3_BUCKET=your-bucket \
  -e MODEL_KEY=models/BEST_tuned_model.pkl \
  -e AWS_DEFAULT_REGION=us-east-1 \
  -e AWS_ACCESS_KEY_ID=your-key \
  -e AWS_SECRET_ACCESS_KEY=your-secret \
  --name diabetes-app \
  diabetes-predictor
```

---

## ☁️ AWS Deployment

The application uses three AWS services:

| Service | Role |
|---|---|
| **S3** | Stores the trained model `.pkl` file |
| **ECR** | Private registry for the Docker image |
| **EC2** | Runs the container (t2.micro, free tier) |

At startup, the EC2 container pulls the model from S3 into memory. All subsequent prediction requests are served from memory with no further S3 calls.

---

## 🔬 Techniques Used

| Technique | Why |
|---|---|
| MICE Imputation | Physiologically impossible zeros treated as missing; MICE preserves inter-feature correlations better than mean/median imputation |
| Stratified K-Fold | Each fold maintains the original 65:35 class ratio |
| SMOTE inside CV fold | Prevents data leakage — synthetic samples never touch the validation set |
| MLflow tracking | Every experiment run is reproducible; metrics and artefacts logged automatically |
| Optuna | Bayesian optimisation learns from previous trials — more efficient than random or grid search |
| Docker layered build | Dependencies cached separately from code; rebuilds take ~30s instead of 5 minutes |
| CORS max_age=86400 | Preflight OPTIONS request cached for 24 hours — eliminates extra round-trip on repeat requests |
| GZip middleware | Response compressed before transmission |

---

## 📈 What I Learnt

- **Data leakage is subtle.** Fitting SMOTE or imputation before the CV split is a common mistake that produces artificially high scores. All transformations must stay inside each fold.
- **Simple models sometimes win.** Logistic Regression beat XGBoost and CatBoost. Model complexity needs data to justify itself.
- **Latency is mostly geography.** The model runs in 1ms. The 348ms users experience is almost entirely network distance. Profiling confirmed this precisely.
- **Docker layer caching is worth understanding.** Separating pip install layers from code layers makes iterative development much faster.
- **Production readiness is a mindset.** Health check endpoints, structured logging, restart policies, and environment variable handling are not optional.

---

## 📋 API Reference

### POST `/predict`

**Request body:**
```json
{
  "pregnancies": 2,
  "glucose": 120,
  "blood_pressure": 80,
  "skin_thickness": 20,
  "insulin": 85,
  "bmi": 28.5,
  "diabetes_pedigree": 0.35,
  "age": 35
}
```

**Response:**
```json
{
  "prediction": 0,
  "probability": 0.0778,
  "label": "Not Diabetic",
  "confidence": "High",
  "bmi": 28.5,
  "message": "Model predicts 8% probability of diabetes.",
  "model_time_ms": 0.97,
  "server_time_ms": 1.04
}
```

---

## ⚠️ Disclaimer

This project was built for learning purposes to demonstrate an end-to-end MLOps workflow. It is not a medical device and should not be used for clinical diagnosis or treatment decisions. Always consult a qualified healthcare professional.

---

## 📄 Dataset

[Pima Indians Diabetes Database](https://www.kaggle.com/datasets/uciml/pima-indians-diabetes-database) — UCI Machine Learning Repository  
Originally collected by the National Institute of Diabetes and Digestive and Kidney Diseases.

---

*Built as part of an AI/ML engineering learning journey.*
