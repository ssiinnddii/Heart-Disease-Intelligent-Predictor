from flask import Blueprint, render_template, redirect, url_for, flash, request
from flask_login import login_user, logout_user, login_required, current_user
from models import db, User
from auth import hash_password, verify_password

auth = Blueprint('auth', __name__, url_prefix='/auth')

@auth.route('/login', methods=['GET', 'POST'])
def login():
    if current_user.is_authenticated:
        return redirect(url_for('dashboard.redirect_dashboard'))

    if request.method == 'POST':
        email = request.form.get('email', '').strip()
        password = request.form.get('password', '')

        user = User.query.filter_by(email=email).first()

        if user and verify_password(user.password_hash, password):
            login_user(user)

            if user.role == 'doctor' and user.status == 'pending':
                logout_user()
                flash('Your doctor account is pending approval. Please wait for admin approval.', 'warning')
                return redirect(url_for('auth.login'))

            next_page = request.args.get('next')
            if next_page:
                return redirect(next_page)
            return redirect(url_for('dashboard.redirect_dashboard'))

        flash('Invalid email or password.', 'danger')

    return render_template('auth/login.html')

@auth.route('/register', methods=['GET', 'POST'])
def register():
    if current_user.is_authenticated:
        return redirect(url_for('dashboard.redirect_dashboard'))

    if request.method == 'POST':
        username = request.form.get('username', '').strip()
        email = request.form.get('email', '').strip()
        password = request.form.get('password', '')
        confirm_password = request.form.get('confirm_password', '')
        role = request.form.get('role', 'user')

        errors = []
        if len(username) < 3:
            errors.append('Username must be at least 3 characters.')
        if '@' not in email:
            errors.append('Please enter a valid email address.')
        if len(password) < 8:
            errors.append('Password must be at least 8 characters.')
        if password != confirm_password:
            errors.append('Passwords do not match.')
        if role not in ['user', 'doctor']:
            errors.append('Invalid role selected.')

        if User.query.filter_by(username=username).first():
            errors.append('Username already exists.')
        if User.query.filter_by(email=email).first():
            errors.append('Email already registered.')

        if errors:
            for error in errors:
                flash(error, 'danger')
            return render_template('auth/register.html')

        status = 'pending' if role == 'doctor' else 'active'
        user = User(
            username=username,
            email=email,
            password_hash=hash_password(password),
            role=role,
            status=status
        )
        db.session.add(user)
        db.session.commit()

        if role == 'doctor':
            flash('Registration successful! Your doctor account is pending admin approval.', 'info')
        else:
            flash('Registration successful! You can now log in.', 'success')

        return redirect(url_for('auth.login'))

    return render_template('auth/register.html')

@auth.route('/logout')
@login_required
def logout():
    logout_user()
    flash('You have been logged out.', 'info')
    return redirect(url_for('auth.login'))