"""Quick test script to verify SHAP integration works with the trained model."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

import json
import numpy as np
import pandas as pd
from app import app, get_pipeline
from shap_utils import compute_shap_explanation

with app.app_context():
    model, meta = get_pipeline()

    sample_input = {
        "age": "58",
        "sex": "Male",
        "cp": "atypical angina",
        "trestbps": "130",
        "chol": "245",
        "fbs": "True",
        "restecg": "normal",
        "thalch": "150",
        "exang": "False",
        "oldpeak": "2.0",
        "slope": "flat",
        "ca": "1",
        "thal": "reversable defect",
    }

    numeric = meta["numeric_features"]
    categorical = meta["categorical_features"]
    categories = meta.get("categories", {})

    row = {}
    for feat in numeric:
        row[feat] = float(sample_input.get(feat, 0))
    for feat in categorical:
        raw = sample_input.get(feat, "").strip()
        cats = categories.get(feat, [])
        row[feat] = float(cats.index(raw)) if raw in cats else -1.0

    X = pd.DataFrame([row])
    if hasattr(model, "feature_names_in_"):
        X = X[model.feature_names_in_]

    pred = int(model.predict(X)[0])
    proba = float(model.predict_proba(X)[0][1])
    print(f"Prediction: {pred}, Probability: {proba:.3f}")

    shap_result = compute_shap_explanation(model, X, meta)

    if shap_result:
        print(f"\nSHAP explanation computed successfully with {len(shap_result)} features:")
        for item in shap_result[:5]:
            direction = "+ increases risk" if item["direction"] == "increases" else "- decreases risk"
            print(f"  {item['display_name']}: {item['shap_value']:+.4f} ({direction})")
        print("\nSHAP integration is working correctly!")
    else:
        print("\nERROR: SHAP explanation returned None")
        sys.exit(1)
