# Heart-Disease-Intelligent-Predictor

An AI-powered web application that predicts heart disease risk using clinical health indicators. The system features role-based access for patients, doctors, and administrators, with explainable AI predictions powered by SHAP (SHapley Additive exPlanations).

## Features

### Patient Features
- **Heart Health Assessment** – Enter 13 clinical measurements and receive an instant AI-powered risk prediction (Low, Medium, or High risk)
- **Explainable Results** – View which health factors contributed most to the prediction, displayed as a clear color-coded bar chart ("Pushes Risk Up" / "Pushes Risk Down")
- **Prediction History** – Track all past assessments with timestamps and risk levels
- **Dashboard** – Personal overview with stats on total assessments and risk distribution

### Doctor Features
- **Patient Monitoring** – View all assessments across the system with patient identification (unique Patient ID, e.g. PT-002)
- **Patient Profiles** – Click any patient to see their full assessment history, including all tests and trends over time
- **Doctor's Notes** – Add private clinical notes to Medium and High risk assessments for follow-up and recommendations
- **Analytics Dashboard** – Interactive charts showing:
  - Risk distribution (doughnut chart)
  - Monthly average risk trends (bar chart, color-coded)
  - Assessment timeline (line chart)
  - Age vs. risk scatter plot
  - Patient risk breakdown (stacked bar chart)
- **Assessment Details** – View full measurement data and SHAP feature impact for any prediction

### Admin Features
- **User Management** – View all registered users, roles, and statuses
- **Doctor Approvals** – Review and approve/reject pending doctor registration requests
- **System Oversight** – Access all predictions and manage user accounts (activate/deactivate)

## Tech Stack

| Layer | Technology |
|-------|-----------|
| **Backend** | Python 3, Flask |
| **Database** | SQLite with SQLAlchemy ORM |
| **ML Model** | Scikit-learn (Gradient Boosting Classifier in a Pipeline with StandardScaler) |
| **Explainability** | SHAP (TreeExplainer) |
| **Authentication** | Flask-Login with Werkzeug password hashing (PBKDF2-SHA256) |
| **Chatbot** | Mistral AI API (mistral-medium) |
| **Frontend** | HTML5, CSS3, Bootstrap 5, Bootstrap Icons |
| **Charts** | Chart.js 4 |
| **Dynamic UI** | HTMX (for inline prediction results without page reload) |

## Project Structure

```
Heart-Disease-Intelligent-Predictor/
├── .env                        # Environment variables (MISTRAL_API_KEY, SECRET_KEY)
├── .gitignore                  # Git ignore rules
├── app.py                      # Main Flask application, routes, model loading
├── assistant_engine.py         # Mistral AI-powered chatbot response engine
├── assistant_routes.py         # Chatbot API blueprint with session history
├── auth.py                     # Flask-Login setup, password hashing
├── auth_routes.py              # Login, register, logout routes
├── check_model.py              # Model validation/evaluation script
├── dashboard_routes.py         # User/Doctor/Admin dashboards, analytics, notes
├── decorators.py               # Role-based access decorators
├── models.py                   # SQLAlchemy models (User, Prediction, PredictionNote)
├── migrate_add_user_id.py      # Database migration script
├── seed_admin.py               # Script to create initial admin user
├── shap_utils.py               # SHAP computation and explanation utilities
├── test_shap.py                # SHAP explanation test script
├── heart_disease_uci.csv       # UCI Heart Disease dataset
├── models/
│   ├── heart_disease_model.pkl # Trained ML model (pipeline + classifier)
│   └── meta.json               # Feature metadata (names, categories, metrics)
├── static/
│   └── images/
│       └── heart-bg.png
└── templates/
    ├── base.html               # Base layout with navigation
    ├── index.html              # Landing page with heart health info and news
    ├── predict.html            # Assessment form (HTMX-powered)
    ├── result.html             # Full-page result view
    ├── history.html            # Prediction history table
    ├── about.html              # Model information page
    ├── assistant.html          # Chatbot assistant page
    ├── _error.html             # HTMX error partial
    ├── _prediction_detail.html # Modal content for assessment details + notes
    ├── _recommendations.html   # Health recommendations partial
    ├── _result_card.html       # Risk banner and probability bar
    ├── _result_content.html    # HTMX-injected result (no base template)
    ├── _shap_explanation.html  # Feature impact bar chart
    ├── auth/
    │   ├── login.html
    │   └── register.html
    └── dashboard/
        ├── user_dashboard.html
        ├── doctor_dashboard.html
        ├── admin_dashboard.html
        ├── patient_profile.html
        └── analytics.html
```

## Models & Data

### User Model
| Field | Type | Description |
|-------|------|-------------|
| `id` | Integer | Primary key |
| `username` | String(80) | Unique username |
| `email` | String(120) | Unique email |
| `password_hash` | String(256) | PBKDF2-SHA256 hash |
| `role` | String(20) | `user`, `doctor`, or `admin` |
| `status` | String(20) | `active`, `pending`, `approved`, `rejected`, `inactive` |
| `created_at` | DateTime | Registration timestamp |
| `approved_at` | DateTime | Doctor approval timestamp |

### Prediction Model
| Field | Type | Description |
|-------|------|-------------|
| `id` | Integer | Primary key |
| `user_id` | Integer | FK to users (nullable) |
| `input_data` | Text | JSON of all 13 clinical measurements |
| `prediction` | Integer | 0 (no disease) or 1 (disease) |
| `probability` | Float | Risk probability (0–1) |
| `risk_level` | String(20) | Low, Medium, or High |
| `created_at` | DateTime | Assessment timestamp |

### PredictionNote Model
| Field | Type | Description |
|-------|------|-------------|
| `id` | Integer | Primary key |
| `prediction_id` | Integer | FK to predictions |
| `doctor_id` | Integer | FK to users (doctor) |
| `note_text` | Text | Clinical note content |
| `created_at` | DateTime | Note timestamp |

### Clinical Features (13)
| Feature | Type | Range/Values |
|---------|------|-------------|
| Age | Numeric | 18–100 |
| Sex | Categorical | Female, Male |
| Chest Pain Type | Categorical | asymptomatic, atypical angina, non-anginal, typical angina |
| Resting Blood Pressure | Numeric | 80–250 mm Hg |
| Cholesterol | Numeric | 0–700 mg/dl |
| Fasting Blood Sugar | Categorical | False, True |
| Resting ECG | Categorical | lv hypertrophy, normal, st-t abnormality |
| Max Heart Rate | Numeric | 60–220 bpm |
| Exercise Angina | Categorical | False, True |
| ST Depression | Numeric | -5 to 7 |
| ST Slope | Categorical | downsloping, flat, upsloping |
| Major Vessels | Numeric | 0–3 |
| Thalassemia | Categorical | fixed defect, normal, reversable defect |

## Getting Started

### Prerequisites
- Python 3.10+
- pip

### Installation

1. Clone the repository:
```bash
git clone https://github.com/your-username/Heart-Disease-Intelligent-Predictor.git
cd Heart-Disease-Intelligent-Predictor
```

2. Create a virtual environment and install dependencies:
```bash
python -m venv venv
# Windows:
venv\Scripts\activate
# macOS/Linux:
source venv/bin/activate

pip install -r requirements.txt
```

3. Configure environment variables:
```bash
# Create a .env file in the project root with the following:
MISTRAL_API_KEY=your_mistral_api_key
SECRET_KEY=your_secret_key_here
```
Get a Mistral API key from [console.mistral.ai](https://console.mistral.ai).

4. Run the admin seed script (first time only):
```bash
python seed_admin.py
```

5. Start the application:
```bash
python app.py
```

6. Open your browser to `http://localhost:5000`

**Note:** After running `seed_admin.py`, the default admin credentials will be printed to the console. Change the password immediately after first login.

### Risk Levels
| Level | Probability | Description |
|-------|------------|-------------|
| Low | < 35% | Lower likelihood of heart disease |
| Medium | 35%–65% | Moderate risk, medical evaluation recommended |
| High | > 65% | Significant risk factors, consult a healthcare professional |

## How It Works

1. **Model Pipeline** – The trained `GradientBoostingClassifier` (with `StandardScaler`) processes 13 clinical features to predict heart disease presence and probability.
2. **SHAP Explanations** – For each prediction, `TreeExplainer` computes SHAP values showing how much each feature contributed to pushing the result higher or lower.
3. **Role-Based Access** – Patients see only their own data. Doctors see all assessments and can manage patients. Admins manage users and approve doctors.

## Disclaimer

This tool is for **educational purposes only** and does not constitute medical advice. Always consult a licensed healthcare professional for any health concerns.
