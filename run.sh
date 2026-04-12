#!/usr/bin/env bash
# Eye Disease Detection – Quick start script
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

echo "═══════════════════════════════════════════════"
echo "  Eye Disease Detection – Eye Disease Detection System"
echo "  East West University | Dept. of CSE"
echo "═══════════════════════════════════════════════"

# Virtual environment
if [ ! -d "venv" ]; then
  echo "→ Creating virtual environment..."
  python3 -m venv venv
fi

source venv/bin/activate

echo "→ Installing / verifying dependencies..."
pip install -q -r requirements.txt

echo "→ Starting Flask server on http://0.0.0.0:8080"
echo "→ Admin panel: http://localhost:8080/admin"
echo "→ Default credentials: admin / admin123"
echo ""
python app.py
