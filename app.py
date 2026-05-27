"""
app.py  –  Flask backend for the Heart Disease Prediction web app
================================================================

Architecture
------------
  /                  → Home / landing page
  /predict           → GET: prediction form  |  POST: JSON API endpoint
  /history           → Table of past predictions (from SQLite via SQLAlchemy)
  /about              → Explains the model and features

  Authentication Routes:
  /auth/login         → User login
  /auth/register      → User registration
  /auth/logout        → User logout

  Dashboard Routes (protected):
  /dashboard/         → Redirect to role-specific dashboard
  /dashboard/user     → User dashboard
  /dashboard/doctor   → Doctor dashboard (requires approval)
  /dashboard/admin    → Admin dashboard

The /predict endpoint accepts BOTH:
  • Regular HTML form submissions (returns rendered result page)
  • JSON requests  (returns JSON – useful for fetch/HTMX calls)

Run locally
-----------
  pip install flask flask-sqlalchemy flask-login joblib scikit-learn pandas numpy werkzeug
  python train_pipeline.py --data heart_disease_uci.csv   # first time only
  python seed_admin.py   # create admin user (first time only)
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

from flask import Flask, jsonify, render_template, request, redirect, url_for, flash
from flask_login import login_required, current_user

from models import db, User, Prediction
from auth import init_login
from auth_routes import auth
from dashboard_routes import dashboard
from assistant_routes import assistant_bp
from shap_utils import compute_shap_explanation

def get_heart_news():
    try:
        import feedparser
        rss_feeds = [
            "https://www.medicalnewstoday/rss/category/heart-disease/rss.xml",
            "https://feeds.feedburner.com/healthline/heart-disease"
        ]

        articles = []
        for feed_url in rss_feeds:
            try:
                feed = feedparser.parse(feed_url)
                for entry in feed.entries[:3]:
                    articles.append({
                        "title": entry.get("title", "Heart Health News"),
                        "url": entry.get("link", "#"),
                        "image": None,
                        "description": entry.get("summary", "Read the latest update on heart health and cardiovascular research.")[:200],
                        "source": feed.feed.get("title", "Health News"),
                        "published_at": entry.get("published", "")[:10] if entry.get("published") else ""
                    })
            except Exception:
                continue

        if articles:
            return articles[:6]
    except ImportError:
        pass

    api_key = os.getenv("NEWS_API_KEY")
    if not api_key:
        return get_fallback_news()

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
        return get_fallback_news()


def get_fallback_news():
    """Return curated heart health articles when no API is available."""
    return [
        {
            "title": "Understanding Heart Disease: Risk Factors and Prevention",
            "url": "https://www.cdc.gov/heartdisease/risk_factors.htm",
            "image": "https://images.unsplash.com/photo-1559757175-5700dde675bc?w=600&h=400&fit=crop",
            "description": "Learn about the major risk factors for heart disease and steps you can take to protect your heart health.",
            "source": "CDC",
            "published_at": "2024-01-15"
        },
        {
            "title": "New Guidelines for Managing High Blood Pressure",
            "url": "https://www.heart.org/en/health-topics/high-blood-pressure",
            "image": "https://images.unsplash.com/photo-1576091160399-112ba8d25d1d?w=600&h=400&fit=crop",
            "description": "American Heart Association releases updated guidelines for preventing and managing hypertension.",
            "source": "American Heart Association",
            "published_at": "2024-02-01"
        },
        {
            "title": "The Role of Diet and Exercise in Heart Health",
            "url": "https://www.nhlbi.nih.gov/health/heart-healthy-living",
            "image": "https://images.unsplash.com/photo-1571019614242-c5c5dee9f50b?w=600&h=400&fit=crop",
            "description": "Discover how lifestyle changes can significantly reduce your risk of developing heart disease.",
            "source": "NHLBI",
            "published_at": "2024-01-20"
        },
        {
            "title": "Cholesterol and Heart Disease: What You Need to Know",
            "url": "https://www.mayoclinic.org/diseases-conditions/high-blood-cholesterol",
            "image": "https://images.unsplash.com/photo-1505576399279-565b52d4ac71?w=600&h=400&fit=crop",
            "description": "Understanding cholesterol levels and how to manage them for better heart health.",
            "source": "Mayo Clinic",
            "published_at": "2024-02-10"
        },
        {
            "title": "Warning Signs of Heart Attack You Shouldn't Ignore",
            "url": "https://www.heart.org/en/health-topics/heart-attack",
            "image": "https://images.unsplash.com/photo-1631815588090-d4bfec5b1ccb?w=600&h=400&fit=crop",
            "description": "Know the warning signs of a heart attack and when to seek emergency medical care.",
            "source": "American Heart Association",
            "published_at": "2024-01-25"
        },
        {
            "title": "The Connection Between Stress and Heart Health",
            "url": "https://www.apa.org/topics/stress/body",
            "image": "https://images.unsplash.com/photo-1499728603263-13726abce5fd?w=600&h=400&fit=crop",
            "description": "Research shows how chronic stress can impact your cardiovascular system and what you can do about it.",
            "source": "APA",
            "published_at": "2024-02-05"
        }
    ]

# ── app setup ──────────────────────────────────────────────────────────────
app = Flask(__name__)
app.secret_key = os.environ.get("SECRET_KEY", "dev-secret-change-in-prod")

# SQLite database stored in the instance/ folder (gitignored)
BASE_DIR = Path(__file__).parent
app.config["SQLALCHEMY_DATABASE_URI"] = (
    f"sqlite:///{BASE_DIR / 'instance' / 'predictions.db'}"
)
app.config["SQLALCHEMY_TRACK_MODIFICATIONS"] = False

db.init_app(app)
init_login(app)

app.register_blueprint(auth)
app.register_blueprint(dashboard)
app.register_blueprint(assistant_bp)

# ── load model & metadata ─────────────────────────────────────────────────
PIPELINE_PATH = BASE_DIR / "models" / "heart_disease_model.pkl"
META_PATH     = BASE_DIR / "models" / "meta.json"
DATA_PATH     = BASE_DIR / "heart_disease_uci.csv"

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
        print("MODEL LOADED FROM:", PIPELINE_PATH)
        print("MODEL TYPE:", type(pipeline))

        meta = json.loads(META_PATH.read_text())

    return pipeline, meta


# ── helpers ────────────────────────────────────────────────────────────────
def risk_label(prob: float) -> str:
    """Convert probability to a human-readable risk label."""
    if prob < 0.35:
        return "Low"
    elif prob < 0.65:
        return "Medium"
    return "High"


def parse_form(form_data: dict) -> pd.DataFrame:
    pipe, m = get_pipeline()
    row = {}

    for feat in m["numeric_features"]:
        raw = form_data.get(feat, "")
        try:
            row[feat] = float(raw)
        except (ValueError, TypeError):
            row[feat] = np.nan

    for feat in m["categorical_features"]:
        raw = str(form_data.get(feat, "")).strip()
        cats = m["categories"].get(feat, [])
        if raw in cats:
            row[feat] = cats.index(raw)
        else:
            row[feat] = -1  # Unknown category

    df = pd.DataFrame([row])

    if hasattr(pipe, "feature_names_in_"):
        df = df[pipe.feature_names_in_]
    else:
        df = df[m["all_features"]]

    return df


# ── routes ────────────────────────────────────────────────────────────────
@app.route("/")
def index():
    news = get_heart_news()
    return render_template("index.html", news=news)

@app.route("/predict", methods=["GET", "POST"])
@login_required
def predict():
    model, m = get_pipeline()

    if request.method == "GET":
        return render_template("predict.html", meta=m)

    if request.is_json:
        data = request.get_json(force=True)
    else:
        data = request.form.to_dict()

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

    shap_explanation = None
    try:
        shap_explanation = compute_shap_explanation(model, X, m)
    except Exception:
        pass

    record = Prediction(
        user_id     = current_user.id if current_user.is_authenticated else None,
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
        "shap_explanation": shap_explanation,
    }

    if request.is_json:
        return jsonify(result)

    is_htmx = request.headers.get("HX-Request") == "true"
    if is_htmx:
        return render_template("_result_content.html", result=result, inputs=data, meta=m, shap_explanation=shap_explanation)

    return render_template("result.html", result=result, inputs=data, meta=m, shap_explanation=shap_explanation)


@app.route("/history")
@login_required
def history():
    if current_user.role == 'admin':
        records = Prediction.query.order_by(Prediction.created_at.desc()).limit(100).all()
    elif current_user.role == 'doctor' and current_user.is_approved_doctor:
        records = Prediction.query.order_by(Prediction.created_at.desc()).limit(100).all()
    else:
        records = Prediction.query.filter_by(user_id=current_user.id).order_by(Prediction.created_at.desc()).limit(100).all()
    return render_template("history.html", records=[r.to_dict() for r in records])


@app.route("/history/clear", methods=["POST"])
@login_required
def clear_history():
    if current_user.role != 'admin':
        flash("Only admins can clear history.", "danger")
        return redirect(url_for("history"))

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
    return predict()


# ── error handlers ─────────────────────────────────────────────────────────
@app.errorhandler(401)
def unauthorized(e):
    if request.is_json:
        return jsonify({"error": "Authentication required"}), 401
    flash("Please log in to access this page.", "warning")
    return redirect(url_for("auth.login"))

@app.errorhandler(403)
def forbidden(e):
    if request.is_json:
        return jsonify({"error": "Access denied"}), 403
    flash("You don't have permission to access this page.", "danger")
    return redirect(url_for("dashboard.redirect_dashboard"))


# ── app entry point ───────────────────────────────────────────────────────
with app.app_context():
    db.create_all()

if __name__ == "__main__":
    loaded_model, loaded_meta = get_pipeline()
    app.run(debug=True, host="0.0.0.0", port=5000)