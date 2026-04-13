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

# ── 3. TensorFlow (now declared in requirements.txt — skip redundant install) ──
echo "[3/4] Verifying TensorFlow..."
TF_VERSION=$(python -c "import tensorflow as tf; print(tf.__version__)" 2>/dev/null || echo "")
if [ -n "$TF_VERSION" ]; then
    echo "  tensorflow ${TF_VERSION} ready."
else
    echo "  WARNING: TensorFlow import failed after pip install."
    echo "  App will run using Hugging Face / Gemini API fallback."
    echo "  Check that tensorflow-cpu is listed in requirements.txt."
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
