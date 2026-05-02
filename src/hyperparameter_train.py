# =============================================================
# hyperparameter_tuning.py
# Hyperparameter Tuning — Logistic Regression & Random Forest
# Methods: GridSearchCV | RandomizedSearchCV | Optuna
# Tracking: MLflow | Results saved to CSV | Best model saved
# =============================================================

# ✅ FIX 1: matplotlib backend MUST be set before ANY other import
import matplotlib
matplotlib.use("Agg")  # no tkinter — thread-safe file rendering

import os
import copy
import warnings
import pandas as pd
import numpy as np
import mlflow
import mlflow.sklearn
import matplotlib.pyplot as plt
import joblib
import optuna
import shap

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import (StratifiedKFold, GridSearchCV,
                                     RandomizedSearchCV, cross_val_score)
from sklearn.metrics import (roc_auc_score, accuracy_score, f1_score,
                              precision_score, recall_score,
                              roc_curve, auc)
from imblearn.over_sampling import SMOTE

warnings.filterwarnings("ignore")
optuna.logging.set_verbosity(optuna.logging.WARNING)

# -------------------------------------------------------------
# 1. Paths
# -------------------------------------------------------------
DATA_PATH      = r"H:\Diabetes_predictor\data\imputed_dataset.csv"
MODEL_SAVE_DIR = r"H:\Diabetes_predictor\models\tuned"
EVAL_SAVE_DIR  = r"H:\Diabetes_predictor\evaluation\tuned"
TARGET_COLUMN  = "Outcome"

os.makedirs(MODEL_SAVE_DIR, exist_ok=True)
os.makedirs(EVAL_SAVE_DIR,  exist_ok=True)

# -------------------------------------------------------------
# 2. Load Data
# -------------------------------------------------------------
df = pd.read_csv(DATA_PATH)
X  = df.drop(columns=[TARGET_COLUMN])
y  = df[TARGET_COLUMN]

print(f"Dataset loaded -> X: {X.shape} | y: {y.shape}")
print(f"Class Distribution:\n{y.value_counts()}\n")

# -------------------------------------------------------------
# 3. Setup
# -------------------------------------------------------------
smote = SMOTE(random_state=42)
cv    = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

# SMOTE on full data for final model training
X_res, y_res = smote.fit_resample(X, y)

all_results = []
mlflow.set_experiment("Diabetes_HyperparameterTuning")

# =============================================================
# HELPER: Evaluate model with 5-fold CV + log to MLflow
# =============================================================
def evaluate_and_log(model, X_data, y_data, model_label, method_label):

    y_proba_all, y_true_all = [], []
    auc_scores, acc_scores, f1_scores = [], [], []
    prec_scores, rec_scores = [], []

    for train_idx, val_idx in cv.split(X_data, y_data):
        X_train_raw = X_data.iloc[train_idx]
        X_val       = X_data.iloc[val_idx]
        y_train_raw = y_data.iloc[train_idx]
        y_val       = y_data.iloc[val_idx]

        # SMOTE only on training fold
        X_train, y_train = smote.fit_resample(X_train_raw, y_train_raw)

        fold_model = copy.deepcopy(model)
        fold_model.fit(X_train, y_train)

        y_proba = fold_model.predict_proba(X_val)[:, 1]
        y_pred  = (y_proba >= 0.5).astype(int)

        auc_scores.append(roc_auc_score(y_val, y_proba))
        acc_scores.append(accuracy_score(y_val, y_pred))
        f1_scores.append(f1_score(y_val, y_pred))
        prec_scores.append(precision_score(y_val, y_pred))
        rec_scores.append(recall_score(y_val, y_pred))

        y_proba_all.extend(y_proba)
        y_true_all.extend(y_val)

    mean_auc  = np.mean(auc_scores)
    mean_acc  = np.mean(acc_scores)
    mean_f1   = np.mean(f1_scores)
    mean_prec = np.mean(prec_scores)
    mean_rec  = np.mean(rec_scores)

    # Log metrics to MLflow
    mlflow.log_metric("mean_auc",       mean_auc)
    mlflow.log_metric("mean_accuracy",  mean_acc)
    mlflow.log_metric("mean_f1",        mean_f1)
    mlflow.log_metric("mean_precision", mean_prec)
    mlflow.log_metric("mean_recall",    mean_rec)

    print(f"  [{method_label}] AUC: {mean_auc:.4f} | Acc: {mean_acc:.4f} | "
          f"F1: {mean_f1:.4f} | Prec: {mean_prec:.4f} | Rec: {mean_rec:.4f}")

    # ROC Curve
    fpr, tpr, _ = roc_curve(y_true_all, y_proba_all)
    roc_val     = auc(fpr, tpr)
    fig, ax     = plt.subplots(figsize=(8, 6))
    ax.plot(fpr, tpr, color='blue', lw=2, label=f'AUC = {roc_val:.4f}')
    ax.plot([0, 1], [0, 1], 'r--')
    ax.set_xlabel('False Positive Rate')
    ax.set_ylabel('True Positive Rate')
    ax.set_title(f'ROC - {model_label} | {method_label}')
    ax.legend(loc='lower right')
    roc_path = os.path.join(EVAL_SAVE_DIR,
                            f"{model_label}_{method_label}_roc.png")
    plt.savefig(roc_path, bbox_inches='tight')
    plt.close(fig)
    mlflow.log_artifact(roc_path)

    return {
        "Model":          model_label,
        "Tuning_Method":  method_label,
        "Mean_AUC":       round(mean_auc,  4),
        "Mean_Accuracy":  round(mean_acc,  4),
        "Mean_F1":        round(mean_f1,   4),
        "Mean_Precision": round(mean_prec, 4),
        "Mean_Recall":    round(mean_rec,  4)
    }


# =============================================================
# A. LOGISTIC REGRESSION
# =============================================================
print("\n" + "="*60)
print("LOGISTIC REGRESSION - Hyperparameter Tuning")
print("="*60)

# ------------------------------------------------------------------
# A1. GridSearchCV
# ------------------------------------------------------------------
print("\n[1/3] GridSearchCV...")

lr_grid_params = {
    "C":       [0.01, 0.1, 1, 10, 100],
    "penalty": ["l1", "l2"],
    "solver":  ["liblinear"]
}

lr_grid_search = GridSearchCV(
    estimator  = LogisticRegression(max_iter=1000, random_state=42),
    param_grid = lr_grid_params,
    cv         = cv,
    scoring    = "roc_auc",
    n_jobs     = -1     # safe — not inside Optuna thread
)
lr_grid_search.fit(X_res, y_res)
lr_grid_best = lr_grid_search.best_estimator_
print(f"  Best Params: {lr_grid_search.best_params_}")

with mlflow.start_run(run_name="LR_GridSearchCV"):
    mlflow.log_params(lr_grid_search.best_params_)
    mlflow.log_param("model", "LogisticRegression")
    mlflow.log_param("tuning_method", "GridSearchCV")
    result = evaluate_and_log(lr_grid_best, X, y,
                              "LogisticRegression", "GridSearchCV")
    mlflow.sklearn.log_model(lr_grid_best, artifact_path="model")
    all_results.append(result)

joblib.dump(lr_grid_best,
            os.path.join(MODEL_SAVE_DIR, "LR_GridSearchCV.pkl"))

# ------------------------------------------------------------------
# A2. RandomizedSearchCV
# ------------------------------------------------------------------
print("\n[2/3] RandomizedSearchCV...")

lr_rand_params = {
    "C":       np.logspace(-3, 3, 50),
    "penalty": ["l1", "l2"],
    "solver":  ["liblinear"]
}

lr_rand_search = RandomizedSearchCV(
    estimator           = LogisticRegression(max_iter=1000, random_state=42),
    param_distributions = lr_rand_params,
    n_iter              = 20,
    cv                  = cv,
    scoring             = "roc_auc",
    random_state        = 42,
    n_jobs              = -1    # safe — not inside Optuna thread
)
lr_rand_search.fit(X_res, y_res)
lr_rand_best = lr_rand_search.best_estimator_
print(f"  Best Params: {lr_rand_search.best_params_}")

with mlflow.start_run(run_name="LR_RandomizedSearchCV"):
    mlflow.log_params(lr_rand_search.best_params_)
    mlflow.log_param("model", "LogisticRegression")
    mlflow.log_param("tuning_method", "RandomizedSearchCV")
    result = evaluate_and_log(lr_rand_best, X, y,
                              "LogisticRegression", "RandomizedSearchCV")
    mlflow.sklearn.log_model(lr_rand_best, artifact_path="model")
    all_results.append(result)

joblib.dump(lr_rand_best,
            os.path.join(MODEL_SAVE_DIR, "LR_RandomizedSearchCV.pkl"))

# ------------------------------------------------------------------
# A3. Optuna
# ------------------------------------------------------------------
print("\n[3/3] Optuna...")

def lr_objective(trial):
    C       = trial.suggest_float("C", 1e-3, 1e3, log=True)
    penalty = trial.suggest_categorical("penalty", ["l1", "l2"])
    model   = LogisticRegression(
        C=C, penalty=penalty, solver="liblinear",
        max_iter=1000, random_state=42
    )
    scores = cross_val_score(
        model, X_res, y_res,
        cv=cv, scoring="roc_auc",
        n_jobs=1    # ✅ FIX 2: n_jobs=1 inside Optuna — no nested threads
    )
    return scores.mean()

lr_study = optuna.create_study(direction="maximize")
lr_study.optimize(lr_objective, n_trials=50)
print(f"  Best Params: {lr_study.best_params}")

lr_optuna_best = LogisticRegression(
    **lr_study.best_params,
    solver="liblinear",
    max_iter=1000,
    random_state=42
)
lr_optuna_best.fit(X_res, y_res)

with mlflow.start_run(run_name="LR_Optuna"):
    mlflow.log_params(lr_study.best_params)
    mlflow.log_param("model", "LogisticRegression")
    mlflow.log_param("tuning_method", "Optuna")
    mlflow.log_param("n_trials", 50)
    result = evaluate_and_log(lr_optuna_best, X, y,
                              "LogisticRegression", "Optuna")
    mlflow.sklearn.log_model(lr_optuna_best, artifact_path="model")
    all_results.append(result)

joblib.dump(lr_optuna_best,
            os.path.join(MODEL_SAVE_DIR, "LR_Optuna.pkl"))


# =============================================================
# B. RANDOM FOREST
# =============================================================
print("\n" + "="*60)
print("RANDOM FOREST - Hyperparameter Tuning")
print("="*60)

# ------------------------------------------------------------------
# B1. GridSearchCV
# ------------------------------------------------------------------
print("\n[1/3] GridSearchCV...")

rf_grid_params = {
    "n_estimators":      [100, 200, 300],
    "max_depth":         [3, 5, 7, None],
    "min_samples_split": [2, 5]
}

rf_grid_search = GridSearchCV(
    estimator  = RandomForestClassifier(random_state=42),
    param_grid = rf_grid_params,
    cv         = cv,
    scoring    = "roc_auc",
    n_jobs     = -1    # safe — not inside Optuna thread
)
rf_grid_search.fit(X_res, y_res)
rf_grid_best = rf_grid_search.best_estimator_
print(f"  Best Params: {rf_grid_search.best_params_}")

with mlflow.start_run(run_name="RF_GridSearchCV"):
    mlflow.log_params(rf_grid_search.best_params_)
    mlflow.log_param("model", "RandomForest")
    mlflow.log_param("tuning_method", "GridSearchCV")
    result = evaluate_and_log(rf_grid_best, X, y,
                              "RandomForest", "GridSearchCV")
    mlflow.sklearn.log_model(rf_grid_best, artifact_path="model")
    all_results.append(result)

joblib.dump(rf_grid_best,
            os.path.join(MODEL_SAVE_DIR, "RF_GridSearchCV.pkl"))

# ------------------------------------------------------------------
# B2. RandomizedSearchCV
# ------------------------------------------------------------------
print("\n[2/3] RandomizedSearchCV...")

rf_rand_params = {
    "n_estimators":      [100, 200, 300, 400, 500],
    "max_depth":         [3, 4, 5, 6, 7, 8, None],
    "min_samples_split": [2, 5, 10],
    "min_samples_leaf":  [1, 2, 4],
    "max_features":      ["sqrt", "log2"]
}

rf_rand_search = RandomizedSearchCV(
    estimator           = RandomForestClassifier(random_state=42),
    param_distributions = rf_rand_params,
    n_iter              = 30,
    cv                  = cv,
    scoring             = "roc_auc",
    random_state        = 42,
    n_jobs              = -1    # safe — not inside Optuna thread
)
rf_rand_search.fit(X_res, y_res)
rf_rand_best = rf_rand_search.best_estimator_
print(f"  Best Params: {rf_rand_search.best_params_}")

with mlflow.start_run(run_name="RF_RandomizedSearchCV"):
    mlflow.log_params(rf_rand_search.best_params_)
    mlflow.log_param("model", "RandomForest")
    mlflow.log_param("tuning_method", "RandomizedSearchCV")
    result = evaluate_and_log(rf_rand_best, X, y,
                              "RandomForest", "RandomizedSearchCV")
    mlflow.sklearn.log_model(rf_rand_best, artifact_path="model")
    all_results.append(result)

joblib.dump(rf_rand_best,
            os.path.join(MODEL_SAVE_DIR, "RF_RandomizedSearchCV.pkl"))

# ------------------------------------------------------------------
# B3. Optuna
# ------------------------------------------------------------------
print("\n[3/3] Optuna...")

def rf_objective(trial):
    params = {
        "n_estimators":      trial.suggest_int("n_estimators", 100, 500),
        "max_depth":         trial.suggest_int("max_depth", 3, 10),
        "min_samples_split": trial.suggest_int("min_samples_split", 2, 10),
        "min_samples_leaf":  trial.suggest_int("min_samples_leaf", 1, 5),
        "max_features":      trial.suggest_categorical("max_features",
                                                       ["sqrt", "log2"])
    }
    model  = RandomForestClassifier(**params, random_state=42)
    scores = cross_val_score(
        model, X_res, y_res,
        cv=cv, scoring="roc_auc",
        n_jobs=1    # ✅ FIX 2: n_jobs=1 inside Optuna — no nested threads
    )
    return scores.mean()

rf_study = optuna.create_study(direction="maximize")
rf_study.optimize(rf_objective, n_trials=50)
print(f"  Best Params: {rf_study.best_params}")

rf_optuna_best = RandomForestClassifier(
    **rf_study.best_params, random_state=42
)
rf_optuna_best.fit(X_res, y_res)

with mlflow.start_run(run_name="RF_Optuna"):
    mlflow.log_params(rf_study.best_params)
    mlflow.log_param("model", "RandomForest")
    mlflow.log_param("tuning_method", "Optuna")
    mlflow.log_param("n_trials", 50)
    result = evaluate_and_log(rf_optuna_best, X, y,
                              "RandomForest", "Optuna")
    mlflow.sklearn.log_model(rf_optuna_best, artifact_path="model")
    all_results.append(result)

joblib.dump(rf_optuna_best,
            os.path.join(MODEL_SAVE_DIR, "RF_Optuna.pkl"))


# =============================================================
# C. Save All Results to CSV
# =============================================================
results_df  = pd.DataFrame(all_results).sort_values("Mean_AUC", ascending=False)
results_csv = os.path.join(EVAL_SAVE_DIR, "tuning_results.csv")
results_df.to_csv(results_csv, index=False)

print("\n" + "="*60)
print("ALL TUNING RESULTS (sorted by AUC)")
print("="*60)
print(results_df.to_string(index=False))

# =============================================================
# D. Save Best Model Overall
# =============================================================
best_row   = results_df.iloc[0]
best_label = f"{best_row['Model']}_{best_row['Tuning_Method']}"
best_auc   = best_row["Mean_AUC"]

model_map = {
    "LogisticRegression_GridSearchCV":       lr_grid_best,
    "LogisticRegression_RandomizedSearchCV": lr_rand_best,
    "LogisticRegression_Optuna":             lr_optuna_best,
    "RandomForest_GridSearchCV":             rf_grid_best,
    "RandomForest_RandomizedSearchCV":       rf_rand_best,
    "RandomForest_Optuna":                   rf_optuna_best,
}

best_model      = model_map[best_label]
best_model_path = os.path.join(MODEL_SAVE_DIR, "BEST_tuned_model.pkl")
joblib.dump(best_model, best_model_path)

print(f"\n{'='*60}")
print(f"BEST MODEL  : {best_label}")
print(f"AUC         : {best_auc:.4f}")
print(f"Saved to    : {best_model_path}")
print(f"Results CSV : {results_csv}")
print(f"{'='*60}")

# =============================================================
# E. SHAP — Feature Importance (only for Random Forest)
# =============================================================
if "RandomForest" in best_label:
    try:
        print("\nGenerating SHAP plot for best model...")
        explainer   = shap.TreeExplainer(best_model)
        shap_values = explainer.shap_values(X)
        fig = plt.figure(figsize=(10, 6))
        shap.summary_plot(shap_values[1], X, plot_type="bar", show=False)
        shap_path = os.path.join(EVAL_SAVE_DIR, "BEST_model_shap.png")
        plt.savefig(shap_path, bbox_inches='tight')
        plt.close(fig)
        print(f"SHAP plot saved -> {shap_path}")
    except Exception as e:
        print(f"SHAP skipped: {e}")