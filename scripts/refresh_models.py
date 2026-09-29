"""Re-fit the saved scikit-learn models if the installed scikit-learn differs from the one they were built with.

Pickled scikit-learn models are only guaranteed to load in the same scikit-learn version. setup.sh/setup.bat run this
automatically. It needs only the files shipped in the project (no downloads):
  1. re-fits the LSA embedder (models/lsa_embedder.joblib) from the local corpus,
  2. re-trains and re-evaluates the clause classifier from data/datasets/clause_train.csv / clause_test.csv,
  3. rebuilds the vector DB (embeddings changed).
Usage: python scripts/refresh_models.py [--force]
"""
import argparse
import json
import subprocess
import sys

import _bootstrap  # noqa: F401
import sklearn

from backend.config import settings

VERSIONS = settings.MODELS_DIR / "versions.json"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()
    built = json.loads(VERSIONS.read_text()).get("scikit-learn") if VERSIONS.exists() else None
    if built == sklearn.__version__ and not args.force:
        print(f"Models match the installed scikit-learn {sklearn.__version__}; nothing to do.")
        return
    print(f"Models were built with scikit-learn {built}, installed is {sklearn.__version__}: re-fitting (about a minute)...")
    from backend.services.embedding_service import LSAEmbedder, build_corpus
    LSAEmbedder.fit_and_save(build_corpus())
    for script in ("train_classifier.py", "build_vector_db.py"):
        cmd = [sys.executable, str(settings.ROOT / "scripts" / script)] + (["--force"] if script == "build_vector_db.py" else [])
        subprocess.run(cmd, check=True)
    record_versions()
    print("Models refreshed.")


def record_versions():
    import numpy
    VERSIONS.write_text(json.dumps({"scikit-learn": sklearn.__version__, "numpy": numpy.__version__}, indent=2))


if __name__ == "__main__":
    main()
