"""Download the Sentence Transformers embedding model into models/sentence_transformer/.

Usage:  pip install sentence-transformers
        python scripts/download_models.py [--model sentence-transformers/all-MiniLM-L6-v2]
Then set EMBEDDING_BACKEND=sbert (or leave 'auto') and rebuild the vector DB:
        python scripts/build_vector_db.py --force
If huggingface.co is unreachable, LegalLens keeps working with its LSA (TF-IDF + SVD) fallback embedder.
"""
import argparse
import sys

import _bootstrap  # noqa: F401

from backend.config import settings


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=settings.EMBEDDING_MODEL)
    args = ap.parse_args()
    try:
        from sentence_transformers import SentenceTransformer
    except ImportError:
        sys.exit("sentence-transformers is not installed. Run: pip install sentence-transformers")
    target = settings.MODELS_DIR / "sentence_transformer"
    print(f"Downloading {args.model} ...")
    try:
        model = SentenceTransformer(args.model)
    except Exception as exc:  # noqa: BLE001
        sys.exit(f"Download failed ({exc}). Check your internet connection / access to huggingface.co. "
                 "LegalLens will continue to use the LSA fallback embedder.")
    model.save(str(target))
    dim = model.get_sentence_embedding_dimension()
    print(f"Saved to {target} (dimension {dim}). Now run: python scripts/build_vector_db.py --force")


if __name__ == "__main__":
    main()
