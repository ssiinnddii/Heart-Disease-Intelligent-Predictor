from flask import Blueprint, render_template, redirect, url_for, flash, request, abort, jsonify
from flask_login import login_required, current_user
from models import db, User
from decorators import role_required

dashboard = Blueprint('dashboard', __name__, url_prefix='/dashboard')

@dashboard.route('/')
@login_required
def redirect_dashboard():
    if current_user.role == 'admin':
        return redirect(url_for('dashboard.admin_dashboard'))
    elif current_user.role == 'doctor':
        if current_user.status != 'approved':
            flash('Your doctor account is pending approval.', 'warning')
            return redirect(url_for('auth.login'))
        return redirect(url_for('dashboard.doctor_dashboard'))
    else:
        return redirect(url_for('dashboard.user_dashboard'))

@dashboard.route('/user')
@login_required
@role_required('user')
def user_dashboard():
    from models import Prediction
    records = Prediction.query.filter_by(user_id=current_user.id).order_by(Prediction.created_at.desc()).all()
    stats = {
        'total': len(records),
        'high_risk': sum(1 for r in records if r.risk_level == 'High'),
        'medium_risk': sum(1 for r in records if r.risk_level == 'Medium'),
        'low_risk': sum(1 for r in records if r.risk_level == 'Low'),
    }
    return render_template('dashboard/user_dashboard.html', records=[r.to_dict() for r in records], stats=stats)

@dashboard.route('/doctor')
@login_required
@role_required('doctor')
def doctor_dashboard():
    from models import Prediction
    records = Prediction.query.order_by(Prediction.created_at.desc()).limit(50).all()
    stats = {
        'total': Prediction.query.count(),
        'high_risk': Prediction.query.filter_by(risk_level='High').count(),
        'pending': Prediction.query.filter_by(risk_level='Medium').count(),
        'cleared': Prediction.query.filter_by(risk_level='Low').count(),
    }
    return render_template('dashboard/doctor_dashboard.html', records=[r.to_dict() for r in records], stats=stats)

@dashboard.route('/admin')
@login_required
@role_required('admin')
def admin_dashboard():
    pending_doctors = User.query.filter_by(role='doctor', status='pending').all()
    approved_doctors = User.query.filter_by(role='doctor', status='approved').all()
    all_users = User.query.filter(User.role != 'admin').order_by(User.created_at.desc()).all()
    stats = {
        'total_users': User.query.count(),
        'total_doctors': User.query.filter_by(role='doctor').count(),
        'pending_approvals': len(pending_doctors),
        'active_users': User.query.filter_by(role='user', status='active').count(),
    }
    return render_template('dashboard/admin_dashboard.html',
                          pending_doctors=pending_doctors,
                          approved_doctors=approved_doctors,
                          all_users=all_users,
                          stats=stats)

@dashboard.route('/admin/approve/<int:user_id>', methods=['POST'])
@login_required
@role_required('admin')
def approve_doctor(user_id):
    user = User.query.get_or_404(user_id)
    if user.role != 'doctor' or user.status == 'approved':
        flash('Invalid operation.', 'danger')
        return redirect(url_for('dashboard.admin_dashboard'))

    user.approve()
    db.session.commit()
    flash(f'Doctor account for {user.username} has been approved.', 'success')
    return redirect(url_for('dashboard.admin_dashboard'))

@dashboard.route('/admin/reject/<int:user_id>', methods=['POST'])
@login_required
@role_required('admin')
def reject_doctor(user_id):
    user = User.query.get_or_404(user_id)
    if user.role != 'doctor' or user.status == 'approved':
        flash('Invalid operation.', 'danger')
        return redirect(url_for('dashboard.admin_dashboard'))

    user.status = 'rejected'
    db.session.commit()
    flash(f'Doctor account for {user.username} has been rejected.', 'warning')
    return redirect(url_for('dashboard.admin_dashboard'))

@dashboard.route('/admin/toggle-status/<int:user_id>', methods=['POST'])
@login_required
@role_required('admin')
def toggle_user_status(user_id):
    user = User.query.get_or_404(user_id)
    if user.role == 'admin':
        flash('Cannot modify admin status.', 'danger')
        return redirect(url_for('dashboard.admin_dashboard'))

    user.status = 'inactive' if user.status == 'active' else 'active'
    db.session.commit()
    flash(f'User {user.username} status updated to {user.status}.', 'success')
    return redirect(url_for('dashboard.admin_dashboard'))


@dashboard.route('/prediction/<int:prediction_id>')
@login_required
def view_prediction(prediction_id):
    from models import Prediction
    record = Prediction.query.get_or_404(prediction_id)

    if current_user.role == 'user' and record.user_id != current_user.id:
        abort(403)

    import json
    record_data = record.to_dict()
    record_data['input_data'] = json.loads(record.input_data) if isinstance(record.input_data, str) else record.input_data

    is_htmx = request.headers.get("HX-Request") == "true"
    if is_htmx:
        return render_template('_prediction_detail.html', record=record_data)

    return render_template('prediction_detail.html', record=record_data)