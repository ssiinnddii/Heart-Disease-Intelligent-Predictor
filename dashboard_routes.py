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
    from models import Prediction, PredictionNote
    from collections import defaultdict

    records = Prediction.query.order_by(Prediction.created_at.desc()).limit(50).all()
    stats = {
        'total': Prediction.query.count(),
        'high_risk': Prediction.query.filter_by(risk_level='High').count(),
        'pending': Prediction.query.filter_by(risk_level='Medium').count(),
        'cleared': Prediction.query.filter_by(risk_level='Low').count(),
    }

    patient_map = {}
    for r in records:
        uid = r.user_id
        if uid and uid not in patient_map:
            patient = User.query.get(uid)
            if patient:
                patient_map[uid] = {
                    'id': patient.id,
                    'username': patient.username,
                    'email': patient.email,
                    'assessment_count': Prediction.query.filter_by(user_id=uid).count(),
                    'high_risk_count': Prediction.query.filter_by(user_id=uid, risk_level='High').count(),
                }

    records_data = []
    for r in records:
        d = r.to_dict()
        note_count = PredictionNote.query.filter_by(prediction_id=r.id).count()
        d['notes_count'] = note_count
        d['patient'] = patient_map.get(r.user_id)
        records_data.append(d)

    patients_summary = sorted(patient_map.values(), key=lambda p: p['high_risk_count'], reverse=True)

    return render_template('dashboard/doctor_dashboard.html', records=records_data, stats=stats, patients_summary=patients_summary)

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
    from models import Prediction, PredictionNote
    record = Prediction.query.get_or_404(prediction_id)

    if current_user.role == 'user' and record.user_id != current_user.id:
        abort(403)

    import json
    record_data = record.to_dict()
    record_data['input_data'] = json.loads(record.input_data) if isinstance(record.input_data, str) else record.input_data

    notes = PredictionNote.query.filter_by(prediction_id=prediction_id).order_by(PredictionNote.created_at.desc()).all()
    record_data['notes'] = [n.to_dict() for n in notes]
    record_data['is_doctor'] = current_user.role == 'doctor'

    if record.user_id:
        patient = User.query.get(record.user_id)
        if patient:
            record_data['patient'] = {
                'id': patient.id,
                'username': patient.username,
                'email': patient.email,
                'total_assessments': Prediction.query.filter_by(user_id=patient.id).count(),
            }

    is_htmx = request.headers.get("HX-Request") == "true"
    if is_htmx:
        return render_template('_prediction_detail.html', record=record_data)

    return render_template('prediction_detail.html', record=record_data)


@dashboard.route('/patient/<int:user_id>')
@login_required
@role_required('doctor')
def patient_profile(user_id):
    from models import Prediction, PredictionNote
    patient = User.query.get_or_404(user_id)

    if patient.role != 'user':
        abort(404)

    records = Prediction.query.filter_by(user_id=user_id).order_by(Prediction.created_at.desc()).all()
    stats = {
        'total': len(records),
        'high_risk': sum(1 for r in records if r.risk_level == 'High'),
        'medium_risk': sum(1 for r in records if r.risk_level == 'Medium'),
        'low_risk': sum(1 for r in records if r.risk_level == 'Low'),
    }
    avg_prob = round(sum(r.probability for r in records) / len(records) * 100, 1) if records else 0

    records_data = []
    for r in records:
        d = r.to_dict()
        notes = PredictionNote.query.filter_by(prediction_id=r.id).order_by(PredictionNote.created_at.desc()).all()
        d['notes'] = [n.to_dict() for n in notes]
        records_data.append(d)

    return render_template('dashboard/patient_profile.html',
                           patient=patient,
                           records=records_data,
                           stats=stats,
                           avg_prob=avg_prob)


@dashboard.route('/prediction/<int:prediction_id>/note', methods=['POST'])
@login_required
@role_required('doctor')
def add_prediction_note(prediction_id):
    from models import Prediction, PredictionNote
    Prediction.query.get_or_404(prediction_id)

    data = request.get_json() or request.form
    note_text = data.get('note_text', '').strip()

    if not note_text:
        if request.is_json:
            return jsonify({"error": "Note text is required"}), 400
        flash("Note cannot be empty.", "danger")
        return redirect(request.referrer or url_for('dashboard.doctor_dashboard'))

    note = PredictionNote(
        prediction_id=prediction_id,
        doctor_id=current_user.id,
        note_text=note_text,
    )
    db.session.add(note)
    db.session.commit()

    if request.is_json:
        return jsonify({"success": True, "note": note.to_dict()})

    flash("Note added successfully.", "success")
    return redirect(request.referrer or url_for('dashboard.doctor_dashboard'))


@dashboard.route('/prediction/<int:prediction_id>/note/<int:note_id>', methods=['DELETE'])
@login_required
@role_required('doctor')
def delete_prediction_note(prediction_id, note_id):
    from models import PredictionNote
    note = PredictionNote.query.get_or_404(note_id)

    if note.prediction_id != prediction_id:
        abort(404)
    if note.doctor_id != current_user.id:
        abort(403)

    db.session.delete(note)
    db.session.commit()

    return jsonify({"success": True})


@dashboard.route('/analytics')
@login_required
@role_required('doctor')
def analytics():
    from models import Prediction
    import json
    from collections import defaultdict

    all_records = Prediction.query.order_by(Prediction.created_at).all()

    risk_dist = {
        'Low': Prediction.query.filter_by(risk_level='Low').count(),
        'Medium': Prediction.query.filter_by(risk_level='Medium').count(),
        'High': Prediction.query.filter_by(risk_level='High').count(),
    }

    dates_probs = []
    dates_risk = []
    for r in all_records:
        dates_probs.append({
            'date': r.created_at.strftime('%Y-%m-%d'),
            'probability': round(r.probability * 100, 1)
        })
        risk_val = {'Low': 1, 'Medium': 2, 'High': 3}.get(r.risk_level, 0)
        dates_risk.append({
            'date': r.created_at.strftime('%Y-%m-%d'),
            'risk_level': r.risk_level
        })

    monthly_avg = {}
    monthly_count = {}
    for r in all_records:
        key = r.created_at.strftime('%Y-%m')
        monthly_avg[key] = monthly_avg.get(key, 0) + r.probability
        monthly_count[key] = monthly_count.get(key, 0) + 1
    monthly_labels = sorted(monthly_avg.keys())
    monthly_data = [round(monthly_avg[k] / monthly_count[k] * 100, 1) for k in monthly_labels]

    patient_risks = defaultdict(lambda: {'username': '', 'total': 0, 'high': 0, 'medium': 0, 'avg_prob': 0, 'probs': []})
    for r in all_records:
        uid = r.user_id
        if uid:
            patient_risks[uid]['probs'].append(r.probability)
            patient_risks[uid]['total'] += 1
            if r.risk_level == 'High':
                patient_risks[uid]['high'] += 1
            elif r.risk_level == 'Medium':
                patient_risks[uid]['medium'] += 1
    for uid in patient_risks:
        pr = patient_risks[uid]
        pr['avg_prob'] = round(sum(pr['probs']) / len(pr['probs']) * 100, 1) if pr['probs'] else 0
        del pr['probs']
        patient = User.query.get(uid)
        if patient:
            pr['username'] = patient.username

    patients_list = sorted(patient_risks.items(), key=lambda x: x[1]['high'], reverse=True)[:8]
    patient_names = []
    patient_high = []
    patient_medium = []
    patient_low = []
    for uid, pr in patients_list:
        name = pr['username'] if pr['username'] else f'Patient #{uid}'
        patient_names.append(f"PT-{uid:03d} ({name})")
        patient_high.append(pr['high'])
        patient_medium.append(pr['medium'])
        patient_low.append(pr['total'] - pr['high'] - pr['medium'])

    ages_vs_probs = []
    for r in all_records:
        try:
            inputs = json.loads(r.input_data)
            age = inputs.get('age')
            if age:
                ages_vs_probs.append({'age': int(float(age)), 'probability': round(r.probability * 100, 1)})
        except (json.JSONDecodeError, ValueError, TypeError):
            pass
    ages_vs_probs.sort(key=lambda x: x['age'])

    overall_avg = round(sum(r.probability for r in all_records) / len(all_records) * 100, 1) if all_records else 0

    return render_template('dashboard/analytics.html',
                           risk_dist=risk_dist,
                           dates_probs=dates_probs,
                           dates_risk=dates_risk,
                           monthly_labels=monthly_labels,
                           monthly_data=monthly_data,
                           patient_names=patient_names,
                           patient_high=patient_high,
                           patient_medium=patient_medium,
                           patient_low=patient_low,
                           ages_vs_probs=ages_vs_probs,
                           overall_avg=overall_avg,
                           total=len(all_records),
                           stats={
                                'total': Prediction.query.count(),
                                'high_risk': risk_dist['High'],
                                'medium_risk': risk_dist['Medium'],
                                'low_risk': risk_dist['Low'],
                           })