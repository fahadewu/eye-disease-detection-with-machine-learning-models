#!/usr/bin/env bash
# ─────────────────────────────────────────────────────────────────────────────
#  build.sh — Render build script
#  Eye Disease Detection Using Machine Learning Models
#  East West University | Dept. of CSE
# ─────────────────────────────────────────────────────────────────────────────
set -e

echo "════════════════════════════════════════════════════"
echo "  Build: Eye Disease Detection Web App"
echo "  Python: $(python --version 2>&1)"
echo "  Platform: $(uname -s) $(uname -m)"
echo "════════════════════════════════════════════════════"

# ── 1. Upgrade pip ──────────────────────────────────────────────────────────
echo ""
echo "[1/4] Upgrading pip..."
pip install --upgrade pip --quiet

# ── 2. Install base dependencies ────────────────────────────────────────────
echo "[2/4] Installing core dependencies..."
pip install \
    "flask>=2.3.0" \
    "Werkzeug>=2.3.0" \
    "gunicorn>=21.2.0" \
    "opencv-python-headless>=4.8.0" \
    "Pillow>=10.0.0" \
    "numpy>=1.24.0" \
    "requests>=2.31.0" \
    "python-dotenv>=1.0.0" \
    --quiet
echo "  Core deps installed."

# ── 3. Install TensorFlow ────────────────────────────────────────────────────
echo "[3/4] Installing TensorFlow..."
PYTHON_MINOR=$(python -c "import sys; print(sys.version_info.minor)")
PYTHON_MAJOR=$(python -c "import sys; print(sys.version_info.major)")

# Detect available memory (Render exports RENDER_INSTANCE_TYPE or check /proc)
AVAILABLE_RAM_MB=0
if [ -f /proc/meminfo ]; then
    AVAILABLE_RAM_MB=$(awk '/MemTotal/ {print int($2/1024)}' /proc/meminfo)
fi

echo "  Detected RAM: ${AVAILABLE_RAM_MB} MB"
echo "  Python: ${PYTHON_MAJOR}.${PYTHON_MINOR}"

if [ "$PYTHON_MAJOR" -eq 3 ] && [ "$PYTHON_MINOR" -ge 9 ] && [ "$PYTHON_MINOR" -le 12 ]; then
    # Try tensorflow-cpu (smaller than full TF, no GPU driver needed on Render)
    echo "  Attempting tensorflow-cpu install..."
    if pip install "tensorflow-cpu>=2.13.0" --quiet 2>/dev/null; then
        TF_VERSION=$(python -c "import tensorflow as tf; print(tf.__version__)" 2>/dev/null || echo "unknown")
        echo "  tensorflow-cpu ${TF_VERSION} installed."
    else
        # Fallback to full tensorflow
        echo "  tensorflow-cpu unavailable, trying tensorflow..."
        if pip install "tensorflow>=2.13.0" --quiet 2>/dev/null; then
            TF_VERSION=$(python -c "import tensorflow as tf; print(tf.__version__)" 2>/dev/null || echo "unknown")
            echo "  tensorflow ${TF_VERSION} installed."
        else
            echo "  WARNING: TensorFlow could not be installed."
            echo "  The app will run using Hugging Face / Gemini API fallback."
            echo "  Configure API keys in Admin -> Settings after deployment."
        fi
    fi
else
    echo "  Python ${PYTHON_MAJOR}.${PYTHON_MINOR} not supported by TensorFlow."
    echo "  The app will run in API-fallback-only mode."
fi

# ── 4. Pull Git LFS model files ─────────────────────────────────────────────
echo "[4/4] Checking model files (Git LFS)..."
if command -v git-lfs &>/dev/null || git lfs version &>/dev/null 2>&1; then
    echo "  Git LFS found — pulling model weights..."
    git lfs pull --include="models/*.h5" 2>/dev/null || true
    # Verify
    for f in models/densenet.h5 models/mobilenet.h5; do
        if [ -f "$f" ] && [ "$(wc -c < "$f")" -gt 1000 ]; then
            echo "  ✓ $f ($(du -sh "$f" | cut -f1))"
        else
            echo "  ⚠ $f missing or is an LFS pointer — models will use fallback API"
        fi
    done
else
    echo "  Git LFS not available — models may be LFS pointer files."
    echo "  App will use Hugging Face / Gemini API fallback."
fi

echo ""
echo "════════ Build complete ════════"
