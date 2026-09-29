#!/usr/bin/env bash
# One-time setup (Linux/macOS). Usage: ./setup.sh [--rebuild]
#   --rebuild  re-download CUAD, rebuild datasets, retrain the classifier and rebuild the vector DB
set -e
cd "$(dirname "$0")"
PY=${PYTHON:-python3}
[ -d .venv ] || $PY -m venv .venv
. .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
[ -f .env ] || cp .env.example .env
command -v tesseract >/dev/null || echo "NOTE: Tesseract not found — scanned PDFs/images need it (sudo apt install tesseract-ocr / brew install tesseract)."
if [ "$1" = "--rebuild" ]; then
  python scripts/download_datasets.py
  python scripts/build_datasets.py
  python scripts/train_classifier.py
  python scripts/build_vector_db.py --force
else
  python scripts/refresh_models.py      # re-fits models only if your scikit-learn version differs
  python scripts/build_vector_db.py
fi
if command -v npm >/dev/null; then
  (cd frontend && npm install && npm run build)
else
  echo "NOTE: Node.js not found — using the prebuilt frontend in frontend/dist."
fi
python -m pytest backend/tests -q
echo "Setup complete. Start with ./run.sh and open http://localhost:8000"
