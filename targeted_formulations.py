import pandas as pd
import numpy as np
import joblib
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Load with weights
model = joblib.load("release_stacked_model.pkl")
scaler = joblib.load("release_scaler.pkl")

# Test File
excel_path = "D:/JJ/Drug Dissolution Research/Paper 2/data/preprocessing/IBMECA data prediction 3.xlsx"
df = pd.read_excel(excel_path, engine="openpyxl")
df.columns = df.columns.str.strip()
y_test= df["Actual"] 
feature_cols = [
    'Drug MW', 'Drug TPSA', 'Drug LogP', 'Polymer MW', 'LA/GA',
    'Initial Drug-to-Polymer Ratio', 'Particle Size',
    'Drug Loading Capacity', 'Drug Encapsulation Efficiency',
    'Solubility Enhancer Concentration', 'Time'
]

missing = [c for c in feature_cols if c not in df.columns]
if missing:
    raise ValueError(f"Missing required feature columns: {missing}")

df_clean = df.dropna(subset=feature_cols).copy()

# Scale & Predict
X = df_clean[feature_cols]
X_scaled = scaler.transform(X)

predictions = model.predict(X_scaled)
predictions = np.clip(predictions, 0, 1)

# Storing predictions in same dataframe 
df_clean['Predicted Release'] = predictions
R2  = r2_score(y_test, predictions)
print(R2)
