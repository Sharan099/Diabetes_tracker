import os
import copy
import pandas as pd
import numpy as np
import mlflow
import mlflow.sklearn
import mlflow.catboost
import matplotlib.pyplot as plt
import seaborn as sns
import shap
import joblib

from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
from catboost import CatBoostClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from xgboost import XGBClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import (roc_auc_score, accuracy_score,
                              f1_score, precision_score,
                              recall_score, roc_curve, auc)
from imblearn.over_sampling import SMOTE

df = pd.read_csv(r'H:\Diabetes_predictor\data\imputed_dataset.csv')

X = df.drop(columns=['Outcome'])   # ← your target column name
y = df['Outcome']

print("X shape:", X.shape)
print("y shape:", y.shape)
print("Class Distribution:\n", y.value_counts())


os.makedirs("evaluation", exist_ok=True)
os.makedirs(r"H:\Diabetes_predictor\models", exist_ok=True)

smote = SMOTE(random_state=42)
skf   = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
model_configs = {
    'LogisticRegression': {
        'model':  LogisticRegression(max_iter=1000),
        'params': {'max_iter': 1000}
    },
    'RandomForest': {
        'model':  RandomForestClassifier(n_estimators=200, random_state=42),
        'params': {'n_estimators': 200}
    },
    'GradientBoosting': {
        'model':  GradientBoostingClassifier(n_estimators=200, random_state=42),
        'params': {'n_estimators': 200}
    },
    'XGBoost': {
        'model':  XGBClassifier(
                      n_estimators=200, learning_rate=0.1,
                      max_depth=4, subsample=0.8,
                      eval_metric='logloss', random_state=42),
        'params': {'n_estimators': 200, 'learning_rate': 0.1,
                   'max_depth': 4, 'subsample': 0.8}
    },
    'CatBoost': {
        'model':  CatBoostClassifier(iterations=500, depth=4,
                                     learning_rate=0.05, verbose=0),
        'params': {'iterations': 500, 'depth': 4, 'learning_rate': 0.05}
    }
}

smote = SMOTE(random_state=42)
skf   = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

# ─────────────────────────────────────────────
# STEP 7: MLflow Tracking + CV Loop
# ─────────────────────────────────────────────
mlflow.set_experiment("Diabetes_Prediction")

best_auc        = 0
best_run_id     = None
best_model_name = None
best_model_obj  = None
all_results     = []   # collect all model results here

for name, config in model_configs.items():

    with mlflow.start_run(run_name=name):

        model  = config['model']
        params = config['params']

        # Log hyperparameters
        mlflow.log_params(params)
        mlflow.log_param("model_name", name)
        mlflow.log_param("cv_folds",   5)
        mlflow.log_param("smote",      True)
        mlflow.log_param("imputation", "MICE")

        # ── CV Loop ──────────────────────────────
        auc_scores, acc_scores = [], []
        f1_scores, prec_scores, rec_scores = [], [], []
        y_true_all, y_proba_all = [], []

        for fold, (train_idx, val_idx) in enumerate(skf.split(X, y), 1):

            X_train_raw = X.iloc[train_idx]
            X_val       = X.iloc[val_idx]
            y_train_raw = y.iloc[train_idx]
            y_val       = y.iloc[val_idx]

            # SMOTE only on training fold
            X_train, y_train = smote.fit_resample(X_train_raw, y_train_raw)

            fold_model = copy.deepcopy(model)
            fold_model.fit(X_train, y_train)

            if hasattr(fold_model, "predict_proba"):
                y_proba = fold_model.predict_proba(X_val)[:, 1]
            else:
                y_proba = fold_model.decision_function(X_val)
                y_proba = (y_proba - y_proba.min()) / (y_proba.max() - y_proba.min())

            y_pred    = (y_proba >= 0.5).astype(int)
            fold_auc  = roc_auc_score(y_val, y_proba)
            fold_acc  = accuracy_score(y_val, y_pred)
            fold_f1   = f1_score(y_val, y_pred)
            fold_prec = precision_score(y_val, y_pred)
            fold_rec  = recall_score(y_val, y_pred)

            auc_scores.append(fold_auc)
            acc_scores.append(fold_acc)
            f1_scores.append(fold_f1)
            prec_scores.append(fold_prec)
            rec_scores.append(fold_rec)
            y_true_all.extend(y_val)
            y_proba_all.extend(y_proba)

            # Log per-fold metrics
            mlflow.log_metric("fold_auc",      fold_auc,  step=fold)
            mlflow.log_metric("fold_accuracy", fold_acc,  step=fold)
            mlflow.log_metric("fold_f1",       fold_f1,   step=fold)

        # ── Mean Metrics ─────────────────────────
        mean_auc  = np.mean(auc_scores)
        mean_acc  = np.mean(acc_scores)
        mean_f1   = np.mean(f1_scores)
        mean_prec = np.mean(prec_scores)
        mean_rec  = np.mean(rec_scores)

        mlflow.log_metric("mean_auc",       mean_auc)
        mlflow.log_metric("mean_accuracy",  mean_acc)
        mlflow.log_metric("mean_f1",        mean_f1)
        mlflow.log_metric("mean_precision", mean_prec)
        mlflow.log_metric("mean_recall",    mean_rec)

        print(f"\n{name}")
        print(f"  AUC: {mean_auc:.4f} | Acc: {mean_acc:.4f} | F1: {mean_f1:.4f}")
        print(f"  Precision: {mean_prec:.4f} | Recall: {mean_rec:.4f}")

        # Collect results for comparison table
        all_results.append({
            'Model':     name,
            'AUC':       round(mean_auc,  4),
            'Accuracy':  round(mean_acc,  4),
            'F1':        round(mean_f1,   4),
            'Precision': round(mean_prec, 4),
            'Recall':    round(mean_rec,  4)
        })

        # ── ROC Curve ────────────────────────────
        fpr, tpr, _ = roc_curve(y_true_all, y_proba_all)
        roc_val     = auc(fpr, tpr)

        fig, ax = plt.subplots(figsize=(8, 6))
        ax.plot(fpr, tpr, color='blue', lw=2, label=f'AUC = {roc_val:.4f}')
        ax.plot([0, 1], [0, 1], 'r--')
        ax.set_xlabel('False Positive Rate')
        ax.set_ylabel('True Positive Rate')
        ax.set_title(f'ROC Curve — {name}')
        ax.legend(loc='lower right')
        roc_path = f"H:\\Diabetes_predictor\\evaluation\\{name}_roc.png"
        plt.savefig(roc_path)
        plt.close(fig)
        mlflow.log_artifact(roc_path)

        # ── Train Final Model on Full Data ────────
        X_final, y_final = smote.fit_resample(X, y)
        final_model      = copy.deepcopy(model)
        final_model.fit(X_final, y_final)

        # ── Log Model to MLflow ───────────────────
        if name == 'CatBoost':
            mlflow.catboost.log_model(final_model, artifact_path="model")
        else:
            mlflow.sklearn.log_model(final_model,  artifact_path="model")

        # ── Save each model locally ───────────────
        model_save_path = f"H:\\Diabetes_predictor\\models\\{name}_model.pkl"
        joblib.dump(final_model, model_save_path)
        print(f"  Model saved → {model_save_path}")

        # ── SHAP Feature Importance ───────────────
        try:
            explainer   = shap.TreeExplainer(final_model)
            shap_values = explainer.shap_values(X)
            fig = plt.figure(figsize=(10, 6))
            shap.summary_plot(shap_values, X, plot_type="bar", show=False)
            shap_path = f"H:\\Diabetes_predictor\\evaluation\\{name}_shap.png"
            plt.savefig(shap_path)
            plt.close(fig)
            mlflow.log_artifact(shap_path)
        except Exception as e:
            print(f"  SHAP skipped for {name}: {e}")

        # Track best model
        if mean_auc > best_auc:
            best_auc        = mean_auc
            best_run_id     = mlflow.active_run().info.run_id
            best_model_name = name
            best_model_obj  = final_model

# ─────────────────────────────────────────────
# STEP 8: Save All Results as CSV
# ─────────────────────────────────────────────
results_df = pd.DataFrame(all_results).sort_values('AUC', ascending=False)
results_df.to_csv(r'H:\Diabetes_predictor\results\all_model_results.csv', index=False)

print("\n━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
print("All Model Results:")
print(results_df.to_string(index=False))

# ─────────────────────────────────────────────
# STEP 9: Save Best Model Separately
# ─────────────────────────────────────────────
joblib.dump(best_model_obj, r'H:\Diabetes_predictor\models\best_model.pkl')

print("\n━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
print(f"Best Model  : {best_model_name}")
print(f"Best AUC    : {best_auc:.4f}")
print(f"MLflow RunID: {best_run_id}")
print("Best model + imputer saved successfully.")