#!/usr/bin/env bash
# ─────────────────────────────────────────────────────────────────────────────
#  Eye Disease Detection — Quick Start Script
#  East West University | Dept. of CSE
# ─────────────────────────────────────────────────────────────────────────────
set -e
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

echo "════════════════════════════════════════════════════"
echo "  Eye Disease Detection Using Machine Learning Models"
echo "  East West University | Dept. of CSE"
echo "════════════════════════════════════════════════════"
echo ""
echo "  Python : $(python3 --version 2>&1)"
echo "  Platform: $(uname -s) $(uname -m)"
echo ""

# ── Virtual environment ────────────────────────────────────────────────────────
if [ ! -d "venv" ]; then
  echo "→ Creating virtual environment…"
  python3 -m venv venv
fi

# Activate venv
if [ -f "venv/bin/activate" ]; then
  source venv/bin/activate
elif [ -f "venv/Scripts/activate" ]; then          # Windows Git Bash
  source venv/Scripts/activate
fi

# ── Smart install ──────────────────────────────────────────────────────────────
echo "→ Running smart installer…"
python install.py

# ── Start server ───────────────────────────────────────────────────────────────
echo ""
echo "→ Starting Flask server on http://0.0.0.0:5000"
echo "→ App URL   : http://localhost:5000"
echo "→ Admin URL : http://localhost:5000/admin  (admin / admin123)"
echo ""
python app.py
