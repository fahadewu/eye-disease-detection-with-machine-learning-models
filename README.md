# Eye Disease Detection Using Machine Learning Models

> Academic Final Year Project — East West University, Department of CSE

A professional web application for detecting eye diseases from retinal (fundus) photographs using an ensemble of deep learning CNN models.

---

## Project Information

| | |
|---|---|
| **University** | East West University |
| **Department** | Department of Computer Science & Engineering |
| **Supervisor** | Md Khalid Mahabub Khan *(Senior Lecturer, Dept. of CSE)* |
| **Academic Year** | 2024–2025 |

### Team Members
| Name | Student ID |
|---|---|
| Ruhana Haque Erin | 2020-2-55-004 |
| Zuboraj Seedratul Fardeen | 2020-2-55-003 |

---

## Features

- **Upload or Live Camera Scan** — Upload a fundus image or use your webcam directly
- **Ensemble AI Inference** — Probability-averaging across DenseNet121 and MobileNetV2
- **CLAHE Preprocessing** — Matches training pipeline exactly (LAB colour space, L-channel enhancement)
- **4-Class Classification** — Cataract, Diabetic Retinopathy, Glaucoma, Normal
- **Confidence Visualisation** — Bar chart probability breakdown per class
- **Fallback APIs** — Low-confidence results re-validated via Hugging Face & Google Gemini (free tier)
- **Clinical Info** — Description, symptoms, treatment, and urgency per prediction
- **Downloadable Report** — Text report with full breakdown
- **Admin Panel** — Dashboard, prediction history, model toggle, settings, and retraining UI

---

## Detectable Conditions

| Condition | Urgency |
|---|---|
| Cataract | Moderate |
| Diabetic Retinopathy | High |
| Glaucoma | High |
| Normal | Routine |

---

## Technology Stack

| Layer | Technology |
|---|---|
| Backend | Python 3.9 · Flask 2.3 |
| ML Framework | TensorFlow 2.20 · Keras 3 |
| Image Processing | OpenCV 4.13 (CLAHE) · Pillow |
| Database | SQLite (predictions log, settings) |
| Frontend | Bootstrap 5 · Chart.js · WebRTC Camera |
| Fallback APIs | Hugging Face Inference API · Google Gemini Vision |

---

## Model Architecture

```
Input Image (224×224 RGB)
        ↓
  CLAHE Enhancement
        ↓
 ┌──────────────┐  ┌──────────────┐
 │  DenseNet121 │  │  MobileNetV2 │   ← ImageNet pre-trained backbones
 └──────┬───────┘  └──────┬───────┘
        └──────────────────┘
        ↓  Probability Averaging
  Final Classification
  (Cataract / DR / Glaucoma / Normal)
```

Each backbone uses a custom head:
`GlobalAveragePooling2D → BatchNorm → Dense(256, relu) → Dropout(0.3) → Dense(4, softmax)`

Training: Adam (lr=0.0001), 10 epochs, EarlyStopping, ReduceLROnPlateau

---

## Quick Start

### 1. Clone the repository
```bash
git clone https://github.com/fahadewu/eye-disease-detection-webapp.git
cd eye-disease-detection-webapp
```

### 2. Create virtual environment
```bash
python3 -m venv venv
source venv/bin/activate       # macOS/Linux
# venv\Scripts\activate        # Windows
```

### 3. Install dependencies
```bash
python install.py
```

`install.py` auto-detects your Python version and platform and installs the right TensorFlow build:

| Environment | Installed automatically |
|---|---|
| macOS Apple Silicon (M1/M2/M3/M4) | `tensorflow-macos` + `tensorflow-metal` |
| macOS Intel | `tensorflow` |
| Linux / Windows x86-64 | `tensorflow-cpu` |
| Python 3.13+ or unsupported platform | Core deps only — app uses API fallback |

> **No TF? No problem.** The app runs in fallback mode using free Hugging Face / Gemini Vision APIs. Set your keys in Admin → Settings.

### 4. Add model weights
Place the `.h5` model files in the **parent directory** (one level above this folder):
```
parent-folder/
├── densenet.h5       ← required
├── mobilenet.h5      ← required
└── eye-disease-detection-webapp/   ← this repo
    ├── app.py
    └── ...
```

### 5. (Optional) Configure API keys
```bash
cp .env.example .env
# Edit .env and add your Hugging Face / Gemini API keys
```

### 6. Run
```bash
python app.py
# or
bash run.sh
```

Open **http://localhost:5000** in your browser.

---

## Admin Panel

URL: `http://localhost:5000/admin`  
Default credentials: `admin` / `admin123`  
*(Change immediately in Settings after first login)*

### Admin features
- **Dashboard** — total predictions, class distribution chart, daily trend, model status
- **Prediction History** — paginated log with image thumbnails, confidence, method used
- **Model Manager** — enable/disable individual models, unload from RAM
- **Retrain** — configure and launch a new training job (background thread, live log)
- **Settings** — confidence threshold, fallback API keys, change username/password

---

## Fallback System

When ensemble confidence < threshold (default 65%):
1. Calls **Hugging Face Inference API** (free, no key needed for public models)
2. Falls back to **Google Gemini Vision API** (free tier: 1500 req/day)
3. Blends results: 60% ensemble + 40% API (weighted average)

Configure priority and keys in Admin → Settings.

---

## Dataset

- **Source:** [Eye Diseases Classification — Kaggle](https://www.kaggle.com/datasets/gunavenkatdoddi/eye-diseases-classification)
- **Training images:** 3,376 · **Validation:** 841
- **Classes:** Cataract, Diabetic Retinopathy, Glaucoma, Normal
- **Augmentation:** rotation, width/height shifts, horizontal flip

---

## Disclaimer

> This system is developed for **academic and research purposes only**. It does not constitute medical advice and should not replace professional ophthalmological diagnosis. Always consult a certified eye care professional for clinical assessment.

---

*East West University — Department of CSE — 2024–2025*
