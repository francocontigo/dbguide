from __future__ import annotations

from pathlib import Path

from dbguide.app.retrieval import (
    read_markdown_docs,
    build_vector_index,
    build_bm25,
    save_bm25,
)
BASE_DIR = Path(__file__).resolve().parent.parent
CORPUS_DIR = BASE_DIR / "corpus"


def main() -> None:
    # Ensure we read .md files from dbguide/corpus even when
    # the script is executed from the project root.
    docs = read_markdown_docs(str(CORPUS_DIR))
    if not docs:
        raise SystemExit("No .md files found in ./corpus. Create some cards first.")

    build_vector_index(docs, chroma_dir="data/chroma", collection_name="sql_cards")
    bm25 = build_bm25(docs)
    save_bm25(bm25, docs, path="data/bm25.pkl")

    print(f"OK: indexed {len(docs)} docs.")


if __name__ == "__main__":
    main()
