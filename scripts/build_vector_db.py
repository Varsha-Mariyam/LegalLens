"""Build (or rebuild) the RAG vector database from data/legal_knowledge and data/standard_documents.

Usage:  python scripts/build_vector_db.py [--force]
The index is also built automatically on first use; this script lets you do it ahead of time and
prints a retrieval check using the guide's example clause (7-day termination notice).
"""
import argparse
import json

import _bootstrap  # noqa: F401

from backend.services import embedding_service, rag_service


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--force", action="store_true", help="rebuild even if the fingerprint is unchanged")
    args = ap.parse_args()
    print("embedding backend:", json.dumps(embedding_service.status()))
    meta = rag_service.build_index(force=args.force)
    print("index:", json.dumps(meta, indent=2))
    query = "Either party can terminate with 7 days notice."
    print(f"\nRetrieval check for: {query!r}")
    for r in rag_service.retrieve_knowledge(query, "employment", "Termination", k=3):
        print(f"  knowledge {r['score']:.3f}  {r['metadata'].get('source')} — {r['metadata'].get('title')}")
    for r in rag_service.retrieve_standard(query, "employment", "Termination", k=1):
        print(f"  standard  {r['score']:.3f}  {r['text'][:120]}")


if __name__ == "__main__":
    main()
