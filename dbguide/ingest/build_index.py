
"""
Script to build the vector and BM25 indexes for DBGuide from markdown corpus.
Ensures all metadata is properly formatted for ChromaDB.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, List

from dbguide.app.retrieval import (
    read_markdown_docs,
    build_vector_index,
    build_bm25,
    save_bm25,
)

BASE_DIR: Path = Path(__file__).resolve().parent.parent
CORPUS_DIR: Path = BASE_DIR / "corpus"

def main() -> None:
    """
    Reads all markdown files, normalizes metadata, builds vector and BM25 indexes, and saves them.
    """
    docs: List[Any] = read_markdown_docs(str(CORPUS_DIR))
    if not docs:
        raise SystemExit("No .md files found in ./corpus. Create some cards first.")

    # Ensure all metadata values are primitives (str, int, float, bool, None)
    for d in docs:
        for k, v in list(d.metadata.items()):
            if isinstance(v, list):
                # Convert lists to comma-separated string
                d.metadata[k] = ','.join(map(str, v))
        if 'dialect' not in d.metadata:
            d.metadata['dialect'] = None

    build_vector_index(docs, chroma_dir="data/chroma", collection_name="sql_cards")
    bm25 = build_bm25(docs)
    save_bm25(bm25, docs, path="data/bm25.pkl")

    print(f"OK: indexed {len(docs)} docs.")


if __name__ == "__main__":
    main()
