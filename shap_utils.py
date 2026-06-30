"""
SHAP explanation utilities for heart disease predictions.

Provides cached explainer initialization and per-prediction SHAP value
computation with human-readable feature labels.
"""

import numpy as np

_explainer = None


def _get_feature_display_names(feature_names, meta):
    """Build human-readable display names for features."""
    display = {}
    for feat in feature_names:
        if feat in meta.get("categories", {}):
            cats = meta["categories"][feat]
            display[feat] = {
                "base": feat,
                "categories": cats,
            }
        else:
            display[feat] = {"base": feat, "categories": None}
    return display


def get_explainer(pipeline):
    """
    Initialise and cache a SHAP TreeExplainer for the pipeline's
    GradientBoostingClassifier step.
    """
    global _explainer
    if _explainer is None:
        import shap

        model_step = None
        for step_name, step_obj in pipeline.named_steps.items():
            if hasattr(step_obj, "predict_proba"):
                model_step = step_obj
                break

        if model_step is None:
            raise RuntimeError(
                "Could not find a tree-based model step in the pipeline."
            )

        _explainer = shap.TreeExplainer(model_step)
    return _explainer


def build_background_data(pipeline, meta):
    """
    Create a small background dataset from feature medians / mode values.
    SHAP needs a background reference for certain explanations.
    """
    numeric = meta["numeric_features"]
    categorical = meta["categorical_features"]
    all_feats = meta["all_features"]
    categories = meta.get("categories", {})

    rows = []
    # Use category indices as numeric placeholders for background
    for i in range(20):
        row = {}
        for feat in numeric:
            # Spread background values around typical ranges
            if feat == "age":
                row[feat] = 40 + (i % 5) * 8
            elif feat == "trestbps":
                row[feat] = 120 + (i % 5) * 10
            elif feat == "chol":
                row[feat] = 200 + (i % 5) * 20
            elif feat == "thalch":
                row[feat] = 140 - (i % 5) * 8
            elif feat == "oldpeak":
                row[feat] = 1.0 + (i % 4) * 0.5
            elif feat == "ca":
                row[feat] = float(i % 3)
            else:
                row[feat] = 0.0
        for feat in categorical:
            cats = categories.get(feat, ["0"])
            row[feat] = float(i % len(cats))
        rows.append(row)

    import pandas as pd
    bg_df = pd.DataFrame(rows)
    if hasattr(pipeline, "feature_names_in_"):
        bg_df = bg_df[pipeline.feature_names_in_]

    return bg_df


def compute_shap_explanation(pipeline, X, meta):
    """
    Compute SHAP values for a single prediction.

    Returns a list of dicts sorted by absolute SHAP value (descending):
        [
            {"feature": "age", "display_name": "Age",
             "shap_value": 0.15, "direction": "increases"},
            ...
        ]

    Falls back gracefully if SHAP is unavailable or computation fails.
    """
    try:
        import shap
    except ImportError:
        return None

    try:
        explainer = get_explainer(pipeline)

        X_scaled = pipeline.named_steps.get("scaler", None)
        if X_scaled is not None:
            X_for_shap = X_scaled.transform(X)
        else:
            X_for_shap = X.values if hasattr(X, "values") else np.array(X)

        feature_names = list(X.columns) if hasattr(X, "columns") else meta["all_features"]

        shap_values = explainer.shap_values(X_for_shap)

        if isinstance(shap_values, list):
            shap_values = shap_values[1]

        if shap_values.ndim == 2:
            shap_values = shap_values[0]

        display_names = _get_feature_display_names(feature_names, meta)
        categories = meta.get("categories", {})

        explanations = []
        for i, feat in enumerate(feature_names):
            sv = float(shap_values[i])
            info = display_names.get(feat, {"base": feat, "categories": None})

            direction = "increases" if sv > 0 else "decreases"

            display_name = _humanize_feature_name(feat, info, X, i, categories)

            explanations.append({
                "feature": feat,
                "display_name": display_name,
                "shap_value": round(sv, 4),
                "abs_shap_value": abs(round(sv, 4)),
                "direction": direction,
            })

        explanations.sort(key=lambda x: x["abs_shap_value"], reverse=True)
        return explanations

    except Exception:
        return None


def _humanize_feature_name(feat, info, X, idx, categories):
    """Create a user-friendly display name for a feature."""
    friendly_names = {
        "age": "Age",
        "sex": "Sex",
        "cp": "Chest Pain Type",
        "trestbps": "Resting Blood Pressure",
        "chol": "Serum Cholesterol",
        "fbs": "Fasting Blood Sugar",
        "restecg": "Resting ECG",
        "thalch": "Max Heart Rate",
        "exang": "Exercise-Induced Angina",
        "oldpeak": "ST Depression (Oldpeak)",
        "slope": "ST Slope",
        "ca": "Major Vessels (CA)",
        "thal": "Myocardial Perfusion Scan",
    }

    base_name = friendly_names.get(feat, feat.title())

    if info["categories"] is not None:
        try:
            val = int(X.iloc[0, idx]) if hasattr(X, "iloc") else int(X[0][idx])
            if 0 <= val < len(info["categories"]):
                cat_label = info["categories"][val]
                return f"{base_name}: {cat_label}"
        except (ValueError, IndexError, TypeError):
            pass

    return base_name
