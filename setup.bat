@echo off
REM One-time setup (Windows). Usage: setup.bat [--rebuild]
REM   --rebuild  re-download CUAD, rebuild datasets, retrain all models (Transformers: ~35 min on a GTX 1650 GPU,
REM              much longer on CPU), re-run the evaluation and rebuild the vector DB
cd /d "%~dp0"
if not exist .venv python -m venv .venv
call .venv\Scripts\activate.bat
python -m pip install --upgrade pip
pip install -r requirements.txt || exit /b 1
if not exist .env copy .env.example .env
where tesseract >nul 2>nul || echo NOTE: Tesseract not found. Install from https://github.com/UB-Mannheim/tesseract/wiki and set TESSERACT_CMD in .env
if "%1"=="--rebuild" (
  python scripts\download_datasets.py || exit /b 1
  python scripts\build_datasets.py || exit /b 1
  python scripts\prepare_datasets.py || exit /b 1
  python scripts\train_classifier.py || exit /b 1
  python scripts\train_transformer.py --task clause || exit /b 1
  python scripts\train_transformer.py --task risk || exit /b 1
  python scripts\fit_ensembles.py || exit /b 1
  python scripts\evaluate_models.py || exit /b 1
  python scripts\build_vector_db.py --force || exit /b 1
) else (
  python scripts\refresh_models.py || exit /b 1
  python scripts\build_vector_db.py || exit /b 1
)
where npm >nul 2>nul && (cd frontend && call npm install && call npm run build && cd ..) || echo NOTE: Node.js not found - using prebuilt frontend\dist
python -m pytest backend\tests -q
echo Setup complete. Start with run.bat and open http://localhost:8000
