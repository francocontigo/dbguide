"""
Script to build vector and BM25 indexes for DBGuide from markdown corpus.
Refactored with dependency injection and service-oriented architecture.
"""
from __future__ import annotations

from pathlib import Path

from dbguide.models.document import Document
from dbguide.services.document_loader import DocumentLoader
from dbguide.services.indexing import VectorIndexBuilder, BM25IndexBuilder


BASE_DIR: Path = Path(__file__).resolve().parent.parent
CORPUS_DIR: Path = BASE_DIR / "corpus"


def normalize_metadata(documents: list[Document]) -> None:
    """
    Normalize metadata to ensure ChromaDB compatibility.

    Args:
        documents: List of documents to normalize.
    """
    for doc in documents:
        # Convert list values to comma-separated strings
        for key, value in list(doc.metadata.items()):
            if isinstance(value, list):
                doc.metadata[key] = ','.join(map(str, value))

        # Ensure dialect field exists
        if 'dialect' not in doc.metadata:
            doc.metadata['dialect'] = None


def main() -> None:
    """
    Main function to build indexes.
    Loads documents, normalizes metadata, and builds both vector and BM25 indexes.
    """
    # Load documents
    loader = DocumentLoader(str(CORPUS_DIR))
    documents = loader.load_documents()

    if not documents:
        raise SystemExit("No .md files found in ./corpus. Create some cards first.")

    print(f"Loaded {len(documents)} documents from corpus.")

    # Normalize metadata for ChromaDB compatibility
    normalize_metadata(documents)

    # Build vector index
    print("Building vector index...")
    vector_builder = VectorIndexBuilder(
        chroma_dir="data/chroma"
    )
    vector_builder.build_index(documents, collection_name="sql_cards")
    print("Vector index built successfully.")

    # Build BM25 index
    print("Building BM25 index...")
    bm25_builder = BM25IndexBuilder()
    bm25_index = bm25_builder.build_index(documents)
    bm25_builder.save_index(bm25_index, documents, path="data/bm25.pkl")
    print("BM25 index built successfully.")

    print(f"✓ Indexed {len(documents)} documents successfully.")


if __name__ == "__main__":
    main()
