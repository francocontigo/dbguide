"""
Indexing services for building and managing search indexes.
"""
from __future__ import annotations

import os
import pickle
from typing import List

import chromadb
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer

from dbguide.models.document import Document


class VectorIndexBuilder:
    """Builds and manages vector-based search indexes using ChromaDB."""

    def __init__(
        self,
        model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
        chroma_dir: str = "data/chroma",
    ):
        """
        Initialize vector index builder.

        Args:
            model_name: SentenceTransformer model name.
            chroma_dir: Directory for ChromaDB persistence.
        """
        self.model_name = model_name
        self.chroma_dir = chroma_dir
        self._model = None

    @property
    def model(self) -> SentenceTransformer:
        """Lazy-load the sentence transformer model."""
        if self._model is None:
            self._model = SentenceTransformer(self.model_name)
        return self._model

    def build_index(
        self,
        documents: List[Document],
        collection_name: str = "sql_cards",
    ) -> chromadb.api.models.Collection.Collection:
        """
        Build a vector index from documents.

        Args:
            documents: List of documents to index.
            collection_name: Name of the ChromaDB collection.

        Returns:
            ChromaDB collection with indexed documents.
        """
        os.makedirs(self.chroma_dir, exist_ok=True)

        # Generate embeddings
        embeddings = self.model.encode(
            [doc.text for doc in documents],
            normalize_embeddings=True
        ).tolist()

        # Create/get ChromaDB collection
        client = chromadb.PersistentClient(path=self.chroma_dir)
        collection = client.get_or_create_collection(name=collection_name)

        # Upsert documents with embeddings
        collection.upsert(
            ids=[doc.id for doc in documents],
            documents=[doc.text for doc in documents],
            embeddings=embeddings,
            metadatas=[
                {**{"path": doc.id}, **doc.metadata}
                for doc in documents
            ],
        )

        return collection


class BM25IndexBuilder:
    """Builds and manages BM25 keyword-based search indexes."""

    @staticmethod
    def tokenize(text: str) -> List[str]:
        """
        Tokenize text for BM25 indexing.

        Args:
            text: Text to tokenize.

        Returns:
            List of tokens (lowercased, length >= 3).
        """
        # Remove code fences and normalize whitespace
        text = text.replace("```", " ").replace("\n", " ")
        parts = [p.strip().lower() for p in text.split()]
        return [p for p in parts if len(p) >= 3]

    def build_index(self, documents: List[Document]) -> BM25Okapi:
        """
        Build a BM25 index from documents.

        Args:
            documents: List of documents to index.

        Returns:
            BM25Okapi index.
        """
        tokenized = [self.tokenize(doc.text) for doc in documents]
        return BM25Okapi(tokenized)

    @staticmethod
    def save_index(
        bm25: BM25Okapi,
        documents: List[Document],
        path: str = "data/bm25.pkl"
    ) -> None:
        """
        Save BM25 index and documents to disk.

        Args:
            bm25: BM25 index to save.
            documents: Documents associated with the index.
            path: File path for saving.
        """
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump({"bm25": bm25, "docs": documents}, f)

    @staticmethod
    def load_index(path: str = "data/bm25.pkl") -> tuple[BM25Okapi, List[Document]]:
        """
        Load BM25 index and documents from disk.

        Args:
            path: File path to load from.

        Returns:
            Tuple of (BM25 index, documents list).
        """
        with open(path, "rb") as f:
            obj = pickle.load(f)
        return obj["bm25"], obj["docs"]
