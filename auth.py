from flask import Flask
from flask_login import LoginManager
from werkzeug.security import generate_password_hash, check_password_hash

login_manager = LoginManager()

@login_manager.user_loader
def load_user(user_id):
    from models import User
    return User.query.get(int(user_id))

def init_login(app: Flask):
    login_manager.init_app(app)
    login_manager.login_view = 'auth.login'
    login_manager.login_message = 'Please log in to access this page.'

def hash_password(password: str) -> str:
    return generate_password_hash(password, method='pbkdf2:sha256')

def verify_password(password_hash: str, password: str) -> bool:
    return check_password_hash(password_hash, password)