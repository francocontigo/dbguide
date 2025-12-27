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
    # Garante que lemos os .md da pasta dbguide/corpus, mesmo quando
    # o script e executado a partir da raiz do projeto.
    docs = read_markdown_docs(str(CORPUS_DIR))
    if not docs:
        raise SystemExit("Nenhum .md encontrado em ./corpus. Crie alguns cards primeiro.")

    build_vector_index(docs, chroma_dir="data/chroma", collection_name="sql_cards")
    bm25 = build_bm25(docs)
    save_bm25(bm25, docs, path="data/bm25.pkl")

    print(f"OK: indexado {len(docs)} docs.")


if __name__ == "__main__":
    main()
