"""
Eye Disease Detection – Eye Disease Detection Web Application
=====================================================
Academic project – East West University, Dept. of CSE
Supervisor: Md Khalid Mahabub Khan
Team: Ruhana Haque Erin (2020-2-55-004), Zuboraj Seedratul Fardeen (2020-2-55-003)
"""

import os
import io
import hashlib
import json
import threading
import uuid
import logging
from functools import wraps
from datetime import datetime

from flask import (Flask, render_template, request, redirect,
                   url_for, session, jsonify, flash, Response, g)
from typing import Optional
from werkzeug.utils import secure_filename

from config import Config
import database as db
import model_utils as mu

# ── App init ───────────────────────────────────────────────────────────────────
app = Flask(__name__)
app.config.from_object(Config)
logging.basicConfig(level=logging.INFO)

os.makedirs(Config.UPLOAD_FOLDER, exist_ok=True)
db.init_db()

# ── TF pre-warm ────────────────────────────────────────────────────────────────
# Import TF now (during gunicorn --preload) so workers inherit the already-loaded
# module via copy-on-write fork. Without this the first prediction request pays
# the full 20-30 s cold-start cost and Render's proxy times out → 502.
def _prewarm():
    tf = mu._get_tf()
    if tf:
        logging.getLogger(__name__).info(
            "TensorFlow %s pre-warmed at startup.", tf.__version__)

threading.Thread(target=_prewarm, daemon=True).start()


# ── Context helpers ────────────────────────────────────────────────────────────

def _build_models_config():
    """Merge Config + DB overrides to get active model list."""
    cfg = {}
    for key, info in Config.AVAILABLE_MODELS.items():
        cfg[key] = {
            **info,
            'model_dir': Config.MODELS_DIR,
            'enabled':   db.get_model_enabled(key),
        }
    return cfg


def _get_settings():
    s = db.get_all_settings()
    return {
        'confidence_threshold': float(s.get('confidence_threshold', Config.CONFIDENCE_THRESHOLD)),
        'fallback_enabled':     s.get('fallback_enabled', '1') == '1',
        'fallback_priority':    s.get('fallback_priority', 'huggingface'),
        'hf_api_key':           s.get('hf_api_key', ''),
        'gemini_api_key':       s.get('gemini_api_key', ''),
        'admin_username':       s.get('admin_username', Config.ADMIN_USERNAME),
        'admin_password_hash':  s.get('admin_password_hash', Config.ADMIN_PASSWORD_HASH),
    }


@app.context_processor
def inject_globals():
    return {
        'project_title':    Config.PROJECT_TITLE,
        'project_subtitle': Config.PROJECT_SUBTITLE,
        'university':       Config.UNIVERSITY,
        'department':       Config.DEPARTMENT,
        'supervisor':       Config.SUPERVISOR,
        'supervisor_title': Config.SUPERVISOR_TITLE,
        'team_members':     Config.TEAM_MEMBERS,
        'academic_year':    Config.ACADEMIC_YEAR,
        'class_labels':     Config.CLASS_LABELS,
        'class_colors':     Config.CLASS_COLORS,
        'now':              datetime.utcnow(),
    }


# ── Auth helpers ───────────────────────────────────────────────────────────────

def admin_required(f):
    @wraps(f)
    def decorated(*args, **kwargs):
        if not session.get('admin_logged_in'):
            return redirect(url_for('admin_login', next=request.path))
        return f(*args, **kwargs)
    return decorated


def _check_password(plain: str, hashed: str) -> bool:
    return hashlib.sha256(plain.encode()).hexdigest() == hashed


def allowed_file(filename: str) -> bool:
    return ('.' in filename and
            filename.rsplit('.', 1)[1].lower() in Config.ALLOWED_EXTENSIONS)


# ══════════════════════════════════════════════════════════════════════════════
# PUBLIC ROUTES
# ══════════════════════════════════════════════════════════════════════════════

@app.route('/')
def index():
    stats = db.get_prediction_stats()
    return render_template('index.html', stats=stats)


@app.route('/detect')
def detect():
    return render_template('detect.html')


@app.route('/about')
def about():
    return render_template('about.html')


# ── Prediction API ─────────────────────────────────────────────────────────────

@app.route('/api/predict', methods=['POST'])
def api_predict():
    """
    Accepts multipart/form-data with either:
      • file    – uploaded image file
      • image   – base64 data URL (from webcam capture)
    """
    try:
        return _api_predict_inner()
    except MemoryError:
        logging.getLogger(__name__).error(
            "OOM during prediction — server is low on memory.")
        return jsonify({
            'error': (
                'Server ran out of memory loading AI models. '
                'The analysis will use API fallback on the next request. '
                'Please try again.'
            )
        }), 503
    except Exception as exc:
        logging.getLogger(__name__).error(
            "Unhandled error in /api/predict: %s", exc, exc_info=True)
        return jsonify({'error': 'Internal server error. Please try again.'}), 500


def _api_predict_inner():
    file_bytes = None  # type: Optional[bytes]
    filename = 'capture.jpg'

    if 'file' in request.files and request.files['file'].filename:
        f = request.files['file']
        if not allowed_file(f.filename):
            return jsonify({'error': 'File type not allowed.'}), 400
        filename = secure_filename(f.filename)
        file_bytes = f.read()
    elif 'image' in request.form:
        data_url = request.form['image']
        try:
            header, encoded = data_url.split(',', 1)
            import base64
            file_bytes = base64.b64decode(encoded)
            filename   = f'webcam_{uuid.uuid4().hex[:8]}.jpg'
        except Exception:
            return jsonify({'error': 'Invalid image data.'}), 400
    else:
        return jsonify({'error': 'No image provided.'}), 400

    # Save uploaded file
    save_path = os.path.join(Config.UPLOAD_FOLDER, filename)
    with open(save_path, 'wb') as fp:
        fp.write(file_bytes)

    # Load live settings
    s      = _get_settings()
    models = _build_models_config()

    # Run prediction
    result = mu.predict(
        file_bytes           = file_bytes,
        models_config        = models,
        class_labels         = Config.CLASS_LABELS,
        confidence_threshold = s['confidence_threshold'],
        fallback_enabled     = s['fallback_enabled'],
        fallback_priority    = s['fallback_priority'],
        hf_api_url           = Config.HF_EYE_MODEL_URL,
        hf_api_key           = s['hf_api_key'],
        gemini_api_key       = s['gemini_api_key'],
        gemini_api_url       = Config.GEMINI_URL,
        image_size           = Config.IMAGE_SIZE,
    )

    if result.get('error'):
        return jsonify({'error': result['error']}), 500

    # Enrich with clinical info
    pred  = result['prediction']
    result['description'] = Config.CLASS_DESCRIPTIONS.get(pred, '')
    result['symptoms']    = Config.CLASS_SYMPTOMS.get(pred, [])
    result['treatment']   = Config.CLASS_TREATMENT.get(pred, '')
    result['urgency']     = Config.CLASS_URGENCY.get(pred, 'moderate')
    result['color']       = Config.CLASS_COLORS.get(pred, '#333')
    result['filename']    = filename
    result['image_url']   = url_for('static', filename=f'uploads/{filename}')

    # Persist to DB
    db.log_prediction(
        filename       = filename,
        method         = result.get('method', 'ensemble'),
        prediction     = pred,
        confidence     = result['confidence'],
        all_probs      = result['probs'],
        fallback_used  = result.get('fallback_used', False),
        fallback_note  = result.get('fallback_note', ''),
        ip_address     = request.remote_addr,
    )

    return jsonify(result)


# ══════════════════════════════════════════════════════════════════════════════
# ADMIN ROUTES
# ══════════════════════════════════════════════════════════════════════════════

@app.route('/admin/login', methods=['GET', 'POST'])
def admin_login():
    if session.get('admin_logged_in'):
        return redirect(url_for('admin_dashboard'))
    error = None
    if request.method == 'POST':
        s = _get_settings()
        username = request.form.get('username', '')
        password = request.form.get('password', '')
        if (username == s['admin_username'] and
                _check_password(password, s['admin_password_hash'])):
            session['admin_logged_in'] = True
            session.permanent = True
            next_page = request.args.get('next', url_for('admin_dashboard'))
            return redirect(next_page)
        error = 'Invalid username or password.'
    return render_template('admin/login.html', error=error)


@app.route('/admin/logout')
def admin_logout():
    session.pop('admin_logged_in', None)
    return redirect(url_for('index'))


@app.route('/admin')
@app.route('/admin/dashboard')
@admin_required
def admin_dashboard():
    stats  = db.get_prediction_stats()
    models = _build_models_config()
    jobs   = db.get_retrain_jobs(limit=5)
    return render_template('admin/dashboard.html',
                           stats=stats, models=models, jobs=jobs)


@app.route('/admin/predictions')
@admin_required
def admin_predictions():
    page  = int(request.args.get('page', 1))
    limit = 20
    preds = db.get_predictions(limit=limit, offset=(page - 1) * limit)
    return render_template('admin/predictions.html',
                           predictions=preds, page=page, limit=limit)


# ── Model management ───────────────────────────────────────────────────────────

@app.route('/admin/models')
@admin_required
def admin_models():
    models = _build_models_config()
    model_statuses = {}
    for key, info in models.items():
        path   = os.path.join(Config.MODELS_DIR, info['filename'])
        exists = os.path.exists(path)
        size_mb= round(os.path.getsize(path) / 1e6, 1) if exists else 0
        loaded = key in mu._loaded_models
        model_statuses[key] = {
            **info,
            'exists':  exists,
            'size_mb': size_mb,
            'loaded':  loaded,
        }
    return render_template('admin/models.html', models=model_statuses)


@app.route('/admin/models/toggle', methods=['POST'])
@admin_required
def admin_model_toggle():
    key     = request.form.get('model_key')
    enabled = request.form.get('enabled') == '1'
    if key in Config.AVAILABLE_MODELS:
        db.set_model_enabled(key, enabled)
        if not enabled:
            mu.unload_model(key)
        flash(f"Model '{Config.AVAILABLE_MODELS[key]['display_name']}' "
              f"{'enabled' if enabled else 'disabled'}.", 'success')
    return redirect(url_for('admin_models'))


@app.route('/admin/models/unload', methods=['POST'])
@admin_required
def admin_model_unload():
    key = request.form.get('model_key')
    mu.unload_model(key)
    flash(f"Model unloaded from memory.", 'info')
    return redirect(url_for('admin_models'))


# ── Retraining ─────────────────────────────────────────────────────────────────

@app.route('/admin/retrain', methods=['GET', 'POST'])
@admin_required
def admin_retrain():
    jobs = db.get_retrain_jobs(limit=10)
    if request.method == 'POST':
        cfg = {
            'epochs':      int(request.form.get('epochs', 10)),
            'lr':          float(request.form.get('lr', 0.0001)),
            'batch_size':  int(request.form.get('batch_size', 32)),
            'dropout':     float(request.form.get('dropout', 0.3)),
            'dense_units': int(request.form.get('dense_units', 256)),
            'base_model':  request.form.get('base_model', 'MobileNetV2'),
            'image_size':  int(request.form.get('image_size', 224)),
        }
        dataset_path  = request.form.get('dataset_path', '').strip()
        output_name   = request.form.get('output_name', 'retrained_model').strip()
        if not output_name.endswith('.h5'):
            output_name += '.h5'
        output_path   = os.path.join(Config.MODELS_DIR, output_name)

        job_id = db.create_retrain_job({**cfg,
                                         'dataset_path': dataset_path,
                                         'output_path':  output_path})

        def _run():
            mu.run_retrain(
                job_id        = job_id,
                cfg           = cfg,
                dataset_path  = dataset_path,
                output_path   = output_path,
                db_update_fn  = db.update_retrain_job,
            )

        t = threading.Thread(target=_run, daemon=True)
        t.start()
        flash(f'Retraining job #{job_id} queued. Monitor progress below.', 'success')
        return redirect(url_for('admin_retrain'))

    defaults = Config.RETRAIN_DEFAULTS
    return render_template('admin/retrain.html', jobs=jobs, defaults=defaults)


@app.route('/admin/retrain/<int:job_id>/log')
@admin_required
def admin_retrain_log(job_id):
    """Returns current log text for a job (JSON) – polled by frontend."""
    job = db.get_retrain_job(job_id)
    if not job:
        return jsonify({'error': 'Job not found'}), 404
    return jsonify({'status': job['status'], 'log': job['log_text']})


@app.route('/admin/retrain/<int:job_id>/add-model', methods=['POST'])
@admin_required
def admin_retrain_add_model(job_id):
    """Register a completed retrained model into the running config."""
    job = db.get_retrain_job(job_id)
    if not job or job['status'] != 'done':
        flash('Job not found or not completed.', 'danger')
        return redirect(url_for('admin_retrain'))
    output_path = job['config'].get('output_path', '')
    model_name  = os.path.basename(output_path).replace('.h5', '')
    # Add to available models in memory (runtime-only; persists until restart)
    Config.AVAILABLE_MODELS[model_name] = {
        'filename':     os.path.basename(output_path),
        'display_name': f'{model_name} (retrained)',
        'description':  f"Retrained model – job #{job_id}",
        'enabled':      True,
    }
    flash(f"Model '{model_name}' added to ensemble.", 'success')
    return redirect(url_for('admin_models'))


# ── Settings ────────────────────────────────────────────────────────────────────

@app.route('/admin/settings', methods=['GET', 'POST'])
@admin_required
def admin_settings():
    if request.method == 'POST':
        action = request.form.get('action')

        if action == 'api_keys':
            db.set_setting('hf_api_key',       request.form.get('hf_api_key', ''))
            db.set_setting('gemini_api_key',    request.form.get('gemini_api_key', ''))
            db.set_setting('fallback_enabled',  '1' if request.form.get('fallback_enabled') else '0')
            db.set_setting('fallback_priority', request.form.get('fallback_priority', 'huggingface'))
            db.set_setting('confidence_threshold',
                           request.form.get('confidence_threshold', str(Config.CONFIDENCE_THRESHOLD)))
            flash('API & prediction settings saved.', 'success')

        elif action == 'change_password':
            s        = _get_settings()
            current  = request.form.get('current_password', '')
            new_pw   = request.form.get('new_password', '')
            confirm  = request.form.get('confirm_password', '')
            if not _check_password(current, s['admin_password_hash']):
                flash('Current password is incorrect.', 'danger')
            elif new_pw != confirm:
                flash('New passwords do not match.', 'danger')
            elif len(new_pw) < 6:
                flash('Password must be at least 6 characters.', 'danger')
            else:
                new_hash = hashlib.sha256(new_pw.encode()).hexdigest()
                db.set_setting('admin_password_hash', new_hash)
                flash('Password changed successfully.', 'success')

        elif action == 'change_username':
            new_user = request.form.get('new_username', '').strip()
            if new_user:
                db.set_setting('admin_username', new_user)
                flash(f'Username changed to "{new_user}".', 'success')

        return redirect(url_for('admin_settings'))

    s = _get_settings()
    return render_template('admin/settings.html', settings=s)


# ── Health check ───────────────────────────────────────────────────────────────

@app.route('/api/health')
def health():
    models = _build_models_config()
    loaded = list(mu._loaded_models.keys())
    tf_info = mu.tf_status()
    return jsonify({
        'status':        'ok',
        'models_loaded': loaded,
        'db':            os.path.exists(Config.DATABASE_PATH),
        'tensorflow':    tf_info,
    })


if __name__ == '__main__':
    port  = int(os.environ.get('PORT', 8080))
    debug = os.environ.get('FLASK_ENV', 'production') != 'production'
    app.run(debug=debug, host='0.0.0.0', port=port)
