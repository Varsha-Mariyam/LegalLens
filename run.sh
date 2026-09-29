#!/usr/bin/env bash
# Starts the API and serves the built React app at http://localhost:8000
# For frontend development instead: (cd frontend && npm run dev) -> http://localhost:5173
cd "$(dirname "$0")"
[ -d .venv ] && . .venv/bin/activate
exec python -m uvicorn backend.main:app --host 0.0.0.0 --port "${PORT:-8000}" "$@"
