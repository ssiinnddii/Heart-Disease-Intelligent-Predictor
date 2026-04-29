from datetime import datetime
from flask_sqlalchemy import SQLAlchemy

db = SQLAlchemy()

class User(db.Model):
    __tablename__ = "users"

    id           = db.Column(db.Integer, primary_key=True)
    username     = db.Column(db.String(80), unique=True, nullable=False)
    email        = db.Column(db.String(120), unique=True, nullable=False)
    password_hash = db.Column(db.String(256), nullable=False)
    role         = db.Column(db.String(20), nullable=False, default='user')
    status       = db.Column(db.String(20), nullable=False, default='active')
    created_at   = db.Column(db.DateTime, default=datetime.utcnow)
    approved_at  = db.Column(db.DateTime, nullable=True)

    def is_active(self) -> bool:
        return self.status == 'active' or self.status == 'approved'

    def is_authenticated(self) -> bool:
        return True

    def is_anonymous(self) -> bool:
        return False

    def get_id(self) -> str:
        return str(self.id)

    def is_doctor(self) -> bool:
        return self.role == 'doctor'

    def is_admin(self) -> bool:
        return self.role == 'admin'

    def is_approved_doctor(self) -> bool:
        return self.role == 'doctor' and self.status == 'approved'

    def is_pending_doctor(self) -> bool:
        return self.role == 'doctor' and self.status == 'pending'

    def approve(self):
        self.status = 'approved'
        self.approved_at = datetime.utcnow()

    def to_dict(self):
        return {
            'id': self.id,
            'username': self.username,
            'email': self.email,
            'role': self.role,
            'status': self.status,
            'created_at': self.created_at.strftime('%Y-%m-%d %H:%M'),
            'approved_at': self.approved_at.strftime('%Y-%m-%d %H:%M') if self.approved_at else None,
        }


class Prediction(db.Model):
    __tablename__ = "predictions"

    id           = db.Column(db.Integer, primary_key=True)
    user_id      = db.Column(db.Integer, db.ForeignKey('users.id'), nullable=True)
    created_at   = db.Column(db.DateTime, default=datetime.utcnow)
    input_data   = db.Column(db.Text, nullable=False)
    prediction   = db.Column(db.Integer, nullable=False)
    probability  = db.Column(db.Float, nullable=False)
    risk_level   = db.Column(db.String(20), nullable=False)

    def to_dict(self):
        import json
        return {
            "id":          self.id,
            "created_at":  self.created_at.strftime("%Y-%m-%d %H:%M"),
            "input_data":  json.loads(self.input_data),
            "prediction":  self.prediction,
            "probability": round(self.probability * 100, 1),
            "risk_level":  self.risk_level,
        }