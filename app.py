"""
app.py  –  Flask backend for the Heart Disease Prediction web app
=================================================================

Architecture
------------
  /                → Home / landing page
  /predict         → GET: prediction form  |  POST: JSON API endpoint
  /history         → Table of past predictions (from SQLite via SQLAlchemy)
  /about           → Explains the model and features

The /predict endpoint accepts BOTH:
  • Regular HTML form submissions (returns rendered result page)
  • JSON requests  (returns JSON – useful for fetch/HTMX calls)

Run locally
-----------
  pip install flask flask-sqlalchemy joblib scikit-learn pandas numpy
  python train_pipeline.py --data heart_disease_uci.csv   # first time only
  python app.py
"""

import json
import os
from datetime import datetime
from pathlib import Path

import requests


import joblib
import numpy as np
import pandas as pd
import shap  

from flask import (Flask, jsonify, render_template, request,
                   redirect, url_for, flash)
from flask_sqlalchemy import SQLAlchemy


def get_heart_news():
    api_key = os.getenv("NEWS_API_KEY")  # set this in your environment
    if not api_key:
        return []

    url = "https://newsapi.org/v2/everything"
    params = {
        "q": '"heart disease" OR cardiology OR cardiovascular OR cholesterol OR hypertension',
        "language": "en",
        "sortBy": "publishedAt",
        "pageSize": 6,
        "apiKey": api_key,
    }

    try:
        response = requests.get(url, params=params, timeout=10)
        response.raise_for_status()
        data = response.json()

        articles = data.get("articles", [])

        cleaned_articles = []
        for article in articles:
            title = article.get("title")
            link = article.get("url")
            image = article.get("urlToImage")
            description = article.get("description")
            source = article.get("source", {}).get("name")
            published_at = article.get("publishedAt")

            if not title or not link:
                continue

            cleaned_articles.append({
                "title": title,
                "url": link,
                "image": image,
                "description": description or "Read the latest update on heart health and cardiovascular research.",
                "source": source or "Health News",
                "published_at": published_at[:10] if published_at else ""
            })

        return cleaned_articles

    except requests.RequestException:
        return []
    
# ── app setup ──────────────────────────────────────────────────────────────
app = Flask(__name__)
app.secret_key = os.environ.get("SECRET_KEY", "dev-secret-change-in-prod")

# SQLite database stored in the instance/ folder (gitignored)
BASE_DIR = Path(__file__).parent
app.config["SQLALCHEMY_DATABASE_URI"] = (
    f"sqlite:///{BASE_DIR / 'instance' / 'predictions.db'}"
)
app.config["SQLALCHEMY_TRACK_MODIFICATIONS"] = False

db = SQLAlchemy(app)

# ── load model & metadata ─────────────────────────────────────────────────
PIPELINE_PATH = BASE_DIR / "models" / "heart_disease_model.pkl"
META_PATH     = BASE_DIR / "models" / "meta.json"

pipeline = None   # loaded lazily on first request (faster startup)
meta     = {}

def get_pipeline():
    """Load pipeline from disk once, then cache it in the module global."""
    global pipeline, meta
    if pipeline is None:
        if not PIPELINE_PATH.exists():
            raise FileNotFoundError(
                f"Model not found at {PIPELINE_PATH}. "
                "Run: python train_pipeline.py --data heart_disease_uci.csv"
            )

        pipeline = joblib.load(PIPELINE_PATH)

        # 👇 ADD THESE LINES
        print("✅ MODEL LOADED FROM:", PIPELINE_PATH)
        print("🔎 MODEL TYPE:", type(pipeline))

        meta = json.loads(META_PATH.read_text())

    return pipeline, meta


# ── database model ────────────────────────────────────────────────────────
class Prediction(db.Model):
    """Stores every prediction made through the web app."""
    __tablename__ = "predictions"

    id          = db.Column(db.Integer, primary_key=True)
    created_at  = db.Column(db.DateTime, default=datetime.utcnow)

    # Input features (stored as JSON string for flexibility)
    input_data  = db.Column(db.Text, nullable=False)

    # Outputs
    prediction  = db.Column(db.Integer, nullable=False)   # 0 or 1
    probability = db.Column(db.Float,   nullable=False)   # P(disease)
    risk_level  = db.Column(db.String(20), nullable=False) # Low / Medium / High

    def to_dict(self):
        return {
            "id":          self.id,
            "created_at":  self.created_at.strftime("%Y-%m-%d %H:%M"),
            "input_data":  json.loads(self.input_data),
            "prediction":  self.prediction,
            "probability": round(self.probability * 100, 1),
            "risk_level":  self.risk_level,
        }


# ── helpers ────────────────────────────────────────────────────────────────
def risk_label(prob: float) -> str:
    """Convert probability to a human-readable risk label."""
    if prob < 0.35:
        return "Low"
    elif prob < 0.65:
        return "Medium"
    return "High"


def parse_form(form_data: dict) -> pd.DataFrame:
    """
    Convert raw form/JSON values into a single-row DataFrame
    that matches the feature names the pipeline was trained on.
    """
    _, m = get_pipeline()
    row = {}

    # numeric fields
    for feat in m["numeric_features"]:
        raw = form_data.get(feat, "")
        try:
            row[feat] = float(raw)
        except (ValueError, TypeError):
            row[feat] = np.nan     # imputer will fill with training median

    # categorical fields – keep as strings
    for feat in m["categorical_features"]:
        row[feat] = str(form_data.get(feat, "")).strip()

    return pd.DataFrame([row], columns=m["all_features"])


# ── routes ────────────────────────────────────────────────────────────────
@app.route("/")
def index():
    news = get_heart_news()
    return render_template("index.html", news=news)

@app.route("/predict", methods=["GET", "POST"])
def predict():
    """
    GET  → Render the prediction form.
    POST → Accept form data or JSON, run inference, persist result, respond.
    """
    model, m = get_pipeline()

    if request.method == "GET":
        return render_template("predict.html", meta=m)

    # ── determine input source ────────────────────────────────────────────
    if request.is_json:
        data = request.get_json(force=True)
    else:
        data = request.form.to_dict()

    # ── inference ─────────────────────────────────────────────────────────
    try:
        X = parse_form(data)
        pred  = int(model.predict(X)[0])
        proba = float(model.predict_proba(X)[0][1])

        
    except Exception as exc:
        err = {"error": str(exc)}
        if request.is_json:
            return jsonify(err), 400
        flash(f"Prediction error: {exc}", "danger")
        return redirect(url_for("predict"))

    risk = risk_label(proba)

    # ── persist ───────────────────────────────────────────────────────────
    record = Prediction(
        input_data  = json.dumps(data),
        prediction  = pred,
        probability = proba,
        risk_level  = risk,
    )
    db.session.add(record)
    db.session.commit()

    result = {
        "prediction":  pred,
        "probability": round(proba * 100, 1),
        "risk_level":  risk,
        "record_id":   record.id,
    }

    # ── respond ───────────────────────────────────────────────────────────
    if request.is_json:
        return jsonify(result)

    return render_template("result.html", result=result, inputs=data, meta=m)


@app.route("/history")
def history():
    """Show a table of all past predictions."""
    records = (Prediction.query
               .order_by(Prediction.created_at.desc())
               .limit(100)
               .all())
    return render_template("history.html",
                           records=[r.to_dict() for r in records])


@app.route("/history/clear", methods=["POST"])
def clear_history():
    """Delete all prediction records (for demo convenience)."""
    Prediction.query.delete()
    db.session.commit()
    flash("History cleared.", "info")
    return redirect(url_for("history"))


@app.route("/about")
def about():
    """Model information page."""
    _, m = get_pipeline()
    return render_template("about.html", meta=m)


@app.route("/api/predict", methods=["POST"])
def api_predict():
    """
    Pure JSON API endpoint (same logic as /predict POST with JSON).
    Useful if you embed this in a larger app or call from JS.

    Example curl:
        curl -X POST http://localhost:5000/api/predict \\
             -H 'Content-Type: application/json' \\
             -d '{"age":63,"sex":"Male","cp":"typical angina",...}'
    """
    return predict()   # reuses the same logic


# ── app entry point ───────────────────────────────────────────────────────
with app.app_context():
    db.create_all()    # creates predictions.db if it doesn't exist

if __name__ == "__main__":
    loaded_model, loaded_meta = get_pipeline()
    app.run(debug=True, host="0.0.0.0", port=5000)
