import os
import hashlib
import secrets

BASE_DIR  = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.dirname(BASE_DIR)   # parent folder with the .h5 files


class Config:
    # ── Flask ──────────────────────────────────────────────────────────────────
    SECRET_KEY = os.environ.get('SECRET_KEY', secrets.token_hex(32))
    MAX_CONTENT_LENGTH = 16 * 1024 * 1024        # 16 MB
    UPLOAD_FOLDER      = os.path.join(BASE_DIR, 'static', 'uploads')
    ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'bmp', 'tiff', 'webp'}

    # ── Database ───────────────────────────────────────────────────────────────
    DATABASE_PATH = os.path.join(BASE_DIR, 'app.db')

    # ── Models ─────────────────────────────────────────────────────────────────
    MODELS_DIR = PROJECT_DIR

    AVAILABLE_MODELS = {
        'densenet': {
            'filename':     'densenet.h5',
            'display_name': 'DenseNet121',
            'description':  'Dense Convolutional Network – 121 layers, high accuracy',
            'enabled':      True,
        },
        'mobilenet': {
            'filename':     'mobilenet.h5',
            'display_name': 'MobileNetV2',
            'description':  'Lightweight mobile-optimised network, fast inference',
            'enabled':      True,
        },
    }

    # ── Prediction ─────────────────────────────────────────────────────────────
    IMAGE_SIZE  = (224, 224)
    CLASS_LABELS = ['Cataract', 'Diabetic Retinopathy', 'Glaucoma', 'Normal']

    CLASS_DESCRIPTIONS = {
        'Cataract':            'Clouding of the eye\'s natural lens behind the iris and pupil.',
        'Diabetic Retinopathy':'Damage to retinal blood vessels caused by chronic diabetes.',
        'Glaucoma':            'Group of conditions damaging the optic nerve, often from high eye pressure.',
        'Normal':              'No disease signs detected. Eye appears clinically healthy.',
    }
    CLASS_SYMPTOMS = {
        'Cataract':            ['Blurry or cloudy vision', 'Faded colours', 'Poor night vision', 'Glare/halos'],
        'Diabetic Retinopathy':['Blurred vision', 'Floaters', 'Dark patches', 'Sudden vision loss'],
        'Glaucoma':            ['Loss of peripheral vision', 'Tunnel vision', 'Eye pain/redness', 'Headache'],
        'Normal':              ['Clear, sharp vision', 'Full visual field', 'No retinal abnormalities'],
    }
    CLASS_TREATMENT = {
        'Cataract':            'Outpatient lens-replacement surgery; results are highly successful.',
        'Diabetic Retinopathy':'Blood-sugar control + laser photocoagulation, anti-VEGF injections, or vitrectomy.',
        'Glaucoma':            'Prescription eye drops, selective laser trabeculoplasty, or surgical drainage.',
        'Normal':              'Annual eye check-ups recommended. Maintain a healthy diet and lifestyle.',
    }
    CLASS_URGENCY = {      # low | moderate | high
        'Cataract':            'moderate',
        'Diabetic Retinopathy':'high',
        'Glaucoma':            'high',
        'Normal':              'low',
    }
    CLASS_COLORS = {
        'Cataract':            '#f39c12',
        'Diabetic Retinopathy':'#e74c3c',
        'Glaucoma':            '#9b59b6',
        'Normal':              '#27ae60',
    }

    # ── Fallback API ───────────────────────────────────────────────────────────
    CONFIDENCE_THRESHOLD = 0.65   # below this → call backup API
    # Hugging Face free inference endpoint (no key required for public models)
    HF_EYE_MODEL_URL  = ('https://api-inference.huggingface.co/models/'
                          'Kaludi/Eye-Disease-Classification')
    HF_API_KEY        = os.environ.get('HUGGINGFACE_API_KEY', '')
    # Google Gemini free tier
    GEMINI_API_KEY    = os.environ.get('GEMINI_API_KEY', '')
    GEMINI_URL        = ('https://generativelanguage.googleapis.com/v1beta/'
                          'models/gemini-1.5-flash:generateContent')

    # ── Admin ──────────────────────────────────────────────────────────────────
    ADMIN_USERNAME     = os.environ.get('ADMIN_USERNAME', 'admin')
    _default_pw_hash   = hashlib.sha256('admin123'.encode()).hexdigest()
    ADMIN_PASSWORD_HASH= os.environ.get('ADMIN_PASSWORD_HASH', _default_pw_hash)

    # ── Retraining defaults ────────────────────────────────────────────────────
    RETRAIN_DEFAULTS = {
        'epochs':       10,
        'lr':           0.0001,
        'batch_size':   32,
        'dropout':      0.3,
        'dense_units':  256,
        'base_model':   'MobileNetV2',
        'image_size':   224,
    }

    # ── Academic / project metadata ────────────────────────────────────────────
    PROJECT_TITLE    = 'Eye Disease Detection'
    PROJECT_SUBTITLE = 'Using Machine Learning Models'
    UNIVERSITY       = 'East West University'
    DEPARTMENT       = 'Department of Computer Science & Engineering'
    SUPERVISOR       = 'Md Khalid Mahabub Khan'
    SUPERVISOR_TITLE = 'Senior Lecturer, Dept. of CSE'
    TEAM_MEMBERS     = [
        {'name': 'Ruhana Haque Erin',       'id': '2020-2-55-004'},
        {'name': 'Zuboraj Seedratul Fardeen','id': '2020-2-55-003'},
    ]
    ACADEMIC_YEAR    = '2024–2025'
