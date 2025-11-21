# preprocess.py
import pandas as pd
import numpy as np
from sklearn.impute import KNNImputer
from sklearn.preprocessing import StandardScaler
import pickle

# 1. Load Data
df = pd.read_csv(r"H:\Diabetes_predictor\Data\diabetes.csv")

# 2. Separate Target and Features
X = df.drop('Outcome', axis=1)
y = df['Outcome']

# 3. Handle "hidden" missing values (zeros)
cols_with_zeros = ['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI']
X[cols_with_zeros] = X[cols_with_zeros].replace(0, np.nan)

# 4. Scaling and Imputation
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

imputer = KNNImputer(n_neighbors=5)
X_imputed_scaled = imputer.fit_transform(X_scaled)

# 5. Inverse transform for interpretability
X_preprocessed = pd.DataFrame(scaler.inverse_transform(X_imputed_scaled), columns=X.columns)

# 6. Save preprocessed data and preprocessing objects
X_preprocessed.to_csv("X_preprocessed.csv", index=False)
y.to_csv("y.csv", index=False)

# Save scaler and imputer
with open("scaler.pkl", "wb") as f:
    pickle.dump(scaler, f)

with open("imputer.pkl", "wb") as f:
    pickle.dump(imputer, f)

print("Preprocessing done. Preprocessed data, scaler, and imputer saved.")
