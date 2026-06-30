import joblib
import json
import pandas as pd
import numpy as np
from sklearn.metrics import roc_auc_score, f1_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

# Load the deployed pipeline and its metadata
pipeline = joblib.load("models/heart_disease_model.pkl")
meta = json.load(open("models/meta.json"))

print("Pipeline steps:", list(pipeline.named_steps.keys()))
print("Meta metrics:", meta.get("metrics"))

# Load and preprocess data the SAME way as the notebook
df = pd.read_csv("heart_disease_uci.csv")
df['target'] = (df['num'] > 0).astype(int)

df_clean = df.drop(columns=['id', 'dataset', 'num'])

numerical_cols = df_clean.select_dtypes(include=[np.number]).columns.drop('target')
for col in numerical_cols:
    df_clean[col] = df_clean[col].fillna(df_clean[col].median())

categorical_cols = df_clean.select_dtypes(include=['object']).columns
for col in categorical_cols:
    df_clean[col] = df_clean[col].fillna(df_clean[col].mode()[0])

for col in categorical_cols:
    le = LabelEncoder()
    df_clean[col] = le.fit_transform(df_clean[col].astype(str))

X = df_clean.drop('target', axis=1)
y = df_clean['target']

if hasattr(pipeline, "feature_names_in_"):
    X = X[pipeline.feature_names_in_]

# Reproduce the notebook's train/test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

proba_test = pipeline.predict_proba(X_test)[:, 1]
pred_test = pipeline.predict(X_test)

print("\n--- Scored on TEST split (184 rows) ---")
print("ROC-AUC:", roc_auc_score(y_test, proba_test))
print("F1:", f1_score(y_test, pred_test))