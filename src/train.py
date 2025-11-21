import os
import pandas as pd
import numpy as np
from catboost import CatBoostClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score, accuracy_score, roc_curve, auc
from imblearn.over_sampling import SMOTE
import pickle
import matplotlib.pyplot as plt
import shap

# Create evaluation directory
os.makedirs("evaluation", exist_ok=True)

# 1. Load preprocessed data
X_preprocessed = pd.read_csv("X_preprocessed.csv")
y = pd.read_csv("y.csv")['Outcome']  # Assuming y.csv saved as single column

# 2. Handle imbalance with SMOTE
smote = SMOTE(random_state=42)
X_res, y_res = smote.fit_resample(X_preprocessed, y)

# 3. Calculate class weights dynamically
count_0 = np.sum(y_res==0)
count_1 = np.sum(y_res==1)
class_weight = [1, count_0 / count_1]

# 4. Stratified K-Fold training
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
auc_scores = []
acc_scores = []
last_model = None

fold_metrics = []

for fold, (train_idx, val_idx) in enumerate(skf.split(X_res, y_res), 1):
    X_train, X_val = X_res.iloc[train_idx], X_res.iloc[val_idx]
    y_train, y_val = y_res.iloc[train_idx], y_res.iloc[val_idx]

    model = CatBoostClassifier(
        iterations=500,
        depth=4,
        learning_rate=0.05,
        loss_function='Logloss',
        class_weights=class_weight,
        verbose=0
    )
    model.fit(X_train, y_train)

    preds_proba = model.predict_proba(X_val)[:,1]
    preds_class = model.predict(X_val)

    auc_score = roc_auc_score(y_val, preds_proba)
    acc_score = accuracy_score(y_val, preds_class)

    auc_scores.append(auc_score)
    acc_scores.append(acc_score)
    last_model = model

    fold_metrics.append({'Fold': fold, 'AUC': auc_score, 'Accuracy': acc_score})

# Save metrics as CSV
metrics_df = pd.DataFrame(fold_metrics)
metrics_df.to_csv("evaluation/fold_metrics.csv", index=False)

# Save Mean Metrics plot
fig, ax = plt.subplots(figsize=(6,4))
ax.bar(['Mean AUC','Mean Accuracy'], [np.mean(auc_scores), np.mean(acc_scores)], color=['skyblue','orange'])
ax.set_ylim(0,1)
ax.set_ylabel('Score')
ax.set_title('Mean Evaluation Metrics')
plt.savefig("evaluation/mean_metrics.png")
plt.close(fig)

# 5. Save trained model
model.save_model("catboost_diabetes.cbm")  # CatBoost native format
with open("catboost_diabetes.pkl", "wb") as f:
    pickle.dump(last_model, f)

# 6. SHAP Feature Importance
explainer = shap.TreeExplainer(last_model)
shap_values = explainer.shap_values(X_preprocessed)

# Save SHAP summary plot (bar)
fig = plt.figure(figsize=(10,6))
shap.summary_plot(shap_values, X_preprocessed, plot_type="bar", show=False)
plt.savefig("evaluation/shap_feature_importance.png")
plt.close(fig)

# 7. ROC Curve from K-Fold predictions
y_true_all = []
y_proba_all = []

for train_idx, val_idx in skf.split(X_preprocessed, y):
    X_train, X_val = X_preprocessed.iloc[train_idx], X_preprocessed.iloc[val_idx]
    y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]

    model_fold = CatBoostClassifier(
        iterations=500,
        depth=4,
        learning_rate=0.05,
        loss_function='Logloss',
        class_weights=[1, np.sum(y_train==0)/np.sum(y_train==1)],
        verbose=0
    )
    model_fold.fit(X_train, y_train)
    preds_proba = model_fold.predict_proba(X_val)[:,1]

    y_true_all.extend(y_val)
    y_proba_all.extend(preds_proba)

fpr, tpr, thresholds = roc_curve(y_true_all, y_proba_all)
roc_auc_val = auc(fpr, tpr)

# Save ROC curve
fig, ax = plt.subplots(figsize=(8,6))
ax.plot(fpr, tpr, color='blue', lw=2, label=f'ROC curve (AUC = {roc_auc_val:.4f})')
ax.plot([0,1], [0,1], color='red', lw=2, linestyle='--')
ax.set_xlabel('False Positive Rate')
ax.set_ylabel('True Positive Rate')
ax.set_title('ROC Curve (K-Fold Predictions)')
ax.legend(loc='lower right')
ax.grid(True)
plt.savefig("evaluation/roc_curve.png")
plt.close(fig)

print("Evaluation plots and metrics saved in 'evaluation/' folder")
