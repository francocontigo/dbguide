"""
Backward compatibility module for retrieval functions.
Provides the original function signatures wrapping the new service-based architecture.
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

import chromadb
from rank_bm25 import BM25Okapi

from dbguide.models.document import Document
from dbguide.services.document_loader import DocumentLoader, get_all_metadata_keys_and_values
from dbguide.services.indexing import VectorIndexBuilder, BM25IndexBuilder
from dbguide.services.retrieval_service import HybridRetrievalService


# Type alias for backward compatibility
Doc = Document


def read_markdown_docs(corpus_dir: str = "corpus") -> List[Doc]:
    """
    Read all markdown files in a directory (recursively).

    Args:
        corpus_dir: Directory containing markdown files.

    Returns:
        List of Doc objects.
    """
    loader = DocumentLoader(corpus_dir)
    return loader.load_documents()


def tokenize(text: str) -> List[str]:
    """
    Tokenize text for BM25.

    Args:
        text: Text to tokenize.

    Returns:
        List of tokens.
    """
    return BM25IndexBuilder.tokenize(text)


def build_vector_index(
    docs: List[Doc],
    chroma_dir: str = "data/chroma",
    collection_name: str = "sql_cards",
) -> chromadb.api.models.Collection.Collection:
    """
    Build a vector index using SentenceTransformer and ChromaDB.

    Args:
        docs: List of documents to index.
        chroma_dir: Directory for ChromaDB persistence.
        collection_name: Name of the collection.

    Returns:
        ChromaDB collection.
    """
    builder = VectorIndexBuilder(chroma_dir=chroma_dir)
    return builder.build_index(docs, collection_name=collection_name)


def build_bm25(docs: List[Doc]) -> BM25Okapi:
    """
    Build a BM25 index from documents.

    Args:
        docs: List of documents.

    Returns:
        BM25Okapi index.
    """
    builder = BM25IndexBuilder()
    return builder.build_index(docs)


def save_bm25(bm25: BM25Okapi, docs: List[Doc], path: str = "data/bm25.pkl") -> None:
    """
    Save BM25 index and documents to disk.

    Args:
        bm25: BM25 index.
        docs: Documents list.
        path: File path.
    """
    BM25IndexBuilder.save_index(bm25, docs, path)


def load_bm25(path: str = "data/bm25.pkl") -> Tuple[BM25Okapi, List[Doc]]:
    """
    Load BM25 index and documents from disk.

    Args:
        path: File path.

    Returns:
        Tuple of (BM25 index, documents list).
    """
    return BM25IndexBuilder.load_index(path)


def hybrid_search(
    query: str,
    col: chromadb.api.models.Collection.Collection,
    bm25: BM25Okapi,
    docs: List[Doc],
    top_k: int = 6,
    alpha: float = 0.55,
    metadata_filter: Optional[Dict[str, Any]] = None,
) -> List[Dict[str, Any]]:
    """
    Perform hybrid vector + BM25 search.

    Args:
        query: User query string.
        col: ChromaDB collection.
        bm25: BM25 index.
        docs: List of documents.
        top_k: Number of results.
        alpha: Weight for vector score (0..1).
        metadata_filter: Optional metadata filter.

    Returns:
        List of dicts: {"id", "score", "text"}
    """
    service = HybridRetrievalService(col, bm25, docs)
    results = service.search(query, top_k, alpha, metadata_filter)

    # Convert SearchResult objects to dict format for backward compatibility
    return [
        {"id": r.id, "score": r.score, "text": r.text}
        for r in results
    ]
