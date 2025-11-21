# streamlit_app.py
import streamlit as st
import pandas as pd
import numpy as np
from catboost import CatBoostClassifier
import pickle
import shap
import matplotlib.pyplot as plt
import sqlite3
import os

# -------------------------
# 1️⃣ Load trained model & preprocessing
# -------------------------
model = CatBoostClassifier()
model.load_model("catboost_diabetes.cbm")   

scaler = pickle.load(open("scaler.pkl", "rb"))
imputer = pickle.load(open("imputer.pkl", "rb"))

st.title("Diabetes Prediction App 🩺")
# -------------------------
# 2️⃣ User Inputs with explanations
# -------------------------
st.markdown("**Enter your health details to predict the risk of diabetes.**")

pregnancies = st.number_input(
    "Number of Pregnancies (count of times pregnant)",
    min_value=0, max_value=20, value=0,
    help="Total number of times the person has been pregnant."
)

glucose = st.number_input(
    "Glucose level (mg/dL, fasting preferred)",
    min_value=0, max_value=300, value=120,
    help="Fasting glucose level is recommended. Normal fasting: 70-99 mg/dL, Prediabetes: 100-125 mg/dL, Diabetes: ≥126 mg/dL."
)

blood_pressure = st.number_input(
    "Blood Pressure (mmHg, Diastolic)",
    min_value=0, max_value=200, value=70,
    help="Enter average blood pressure in mmHg. Normal: <120/80 mmHg, Elevated: 120-129/<80 mmHg, High BP: ≥130/80 mmHg."
)

bmi = st.number_input(
    "BMI (Body Mass Index, kg/m²)",
    min_value=0.0, max_value=70.0, value=25.0,
    help="BMI = Iight (kg) / height² (m²). Normal: 18.5–24.9, OverIight: 25–29.9, Obese: ≥30."
)

age = st.number_input(
    "Age (years)",
    min_value=1, max_value=120, value=30,
    help="Age in years. Risk of diabetes increases with age."
)

family_history = st.multiselect(
    "Does anyone in your blood-related family have diabetes? (Select all that apply)",
    ['Mother', 'Father', 'Sister/Brother', 'Grandparent', 'Child', 'Uncle/Aunt/Cousin'],
    help="Family history affects the Diabetes Pedigree Function. Closer relatives = higher genetic risk."
)

# -------------------------
# 3️⃣ Auto-calculated features & Explanation
# -------------------------
# I calculate these because users rarely know these specific medical metrics.

# A. Estimate Skin Thickness (Triceps Skinfold) based on general body composition
skin_thickness = int(0.5 * bmi + 0.2 * age)

# B. Estimate Insulin based on Glucose and BMI (Insulin Resistance proxy)
# Logic: Higher Glucose + High BMI usually indicates the body is producing more insulin to compensate.
insulin = 80 + 2*(bmi - 25) + 0.5*(glucose - 120)
insulin = max(20, min(insulin, 900)) # Cap at realistic medical limits

# C. Calculate Diabetes Pedigree Function (DPF)
dpf_base = 0.08 
genetic_score = 0
# Flatten the list to string just in case, though multiselect returns a list
family_list = [x for x in family_history] 

if 'Mother' in family_list: genetic_score += 0.35
if 'Father' in family_list: genetic_score += 0.35
if 'Sister/Brother' in family_list: genetic_score += 0.35
if 'Child' in family_list: genetic_score += 0.25
if 'Grandparent' in family_list: genetic_score += 0.15
if 'Uncle/Aunt/Cousin' in family_list: genetic_score += 0.10

diabetes_pedigree = dpf_base + genetic_score
diabetes_pedigree = min(diabetes_pedigree, 2.42)

# --- D. DISPLAY EXPLANATION TO USER ---
with st.expander("ℹ️ How did I calculate Skin Thickness & Insulin?"):
    st.markdown("""
    **Since most users do not have medical lab data at home, I estimated these values based on clinical correlations:**
    
    1. **Skin Thickness (Triceps Skinfold):** * What it is:* A measure of body fat.
       * *Our Estimation:* I calculated this using your **BMI** and **Age**, as subcutaneous fat thickness generally correlates with these factors.
    
    2. **Insulin Level (2-Hour Serum):**
       * *What it is:* How much insulin is in your blood.
       * *Our Estimation:* I derived this from your **Fasting Glucose** and **BMI**. 
       * *Logic:* If you have high BMI and high Glucose, your body likely produces *more* insulin (hyperinsulinemia) to try to loIr your blood sugar.
       
    3. **Diabetes Pedigree Function:**
       * I calculated a genetic risk score betIen **0.08** and **2.42** based on the relatives you selected.
    """)
# -------------------------
# 4️⃣ Create input dataframe
# -------------------------
input_df = pd.DataFrame({
    "Pregnancies": [pregnancies],
    "Glucose": [glucose],
    "BloodPressure": [blood_pressure],
    "SkinThickness": [skin_thickness],
    "Insulin": [insulin],
    "BMI": [bmi],
    "DiabetesPedigreeFunction": [diabetes_pedigree],
    "Age": [age]
})

# -------------------------
# 5️⃣ Preprocessing
# -------------------------
cols_with_zeros = ['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI']
input_df[cols_with_zeros] = input_df[cols_with_zeros].replace(0, np.nan)

X_scaled = scaler.transform(input_df)
X_imputed = imputer.transform(X_scaled)
X_final = pd.DataFrame(scaler.inverse_transform(X_imputed), columns=input_df.columns)

# -------------------------
# 6️⃣ Prediction
# -------------------------
prediction_proba = model.predict_proba(X_final)[:, 1][0]
prediction_class = model.predict(X_final)[0]

st.subheader("Prediction Result:")
st.write(f"Probability of Diabetes: {prediction_proba:.2f}")
st.write("Predicted Class:", "Diabetic" if prediction_class == 1 else "Non-Diabetic")

# -------------------------
# 7️⃣ SHAP Feature Contribution
# -------------------------
explainer = shap.TreeExplainer(model)
shap_values = explainer.shap_values(X_final)

st.subheader("Feature Contribution (SHAP)")

# Save SHAP bar plot as PNG and display
shap_fig_path = "evaluation/shap_feature_importance.png"
if os.path.exists(shap_fig_path):
    st.image(shap_fig_path, caption="SHAP Feature Importance (Saved from Training)")
else:
    fig = plt.figure(figsize=(8,5))
    shap.summary_plot(shap_values, X_final, plot_type="bar", show=False)
    plt.savefig(shap_fig_path)
    st.pyplot(fig)

# -------------------------
# 8️⃣ Show evaluation plots
# -------------------------
st.subheader("Evaluation Plots from Training")
eval_dir = "evaluation"
for file_name in ["roc_curve.png", "mean_metrics.png"]:
    file_path = os.path.join(eval_dir, file_name)
    if os.path.exists(file_path):
        st.image(file_path, caption=file_name.replace(".png",""))

# -------------------------
# 9️⃣ Display input values
# -------------------------
st.subheader("Your Entered Data")
st.dataframe(X_final)

# -------------------------
# 🔟 Save user data to SQLite (for academic purposes)
# -------------------------
# FIX: Convert the list output from st.multiselect into a string for SQLite
family_history_str = ", ".join(family_history)


conn = sqlite3.connect("user_inputs.db")
c = conn.cursor()
c.execute('''
    CREATE TABLE IF NOT EXISTS inputs (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        Pregnancies INTEGER,
        Glucose REAL,
        BloodPressure REAL,
        SkinThickness REAL,
        Insulin REAL,
        BMI REAL,
        DiabetesPedigreeFunction REAL,
        Age INTEGER,
        FamilyHistory TEXT,   -- This column expects a single text string
        PredictedClass TEXT,
        Probability REAL
    )
''')
conn.commit()

c.execute('''
    INSERT INTO inputs (
        Pregnancies, Glucose, BloodPressure, SkinThickness, Insulin, BMI,
        DiabetesPedigreeFunction, Age, FamilyHistory, PredictedClass, Probability
    ) VALUES (?,?,?,?,?,?,?,?,?,?,?)
''', (
    pregnancies, glucose, blood_pressure, skin_thickness, insulin, bmi,
    diabetes_pedigree, age, family_history_str,  # <-- CORRECTED: Use family_history_str
    "Diabetic" if prediction_class==1 else "Non-Diabetic",
    float(prediction_proba)
))
conn.commit()
conn.close()

st.info("For academic purposes only: your entered data has been recorded in our database.")