"""
Hybrid retrieval service combining vector and keyword search.
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional

import chromadb
from rank_bm25 import BM25Okapi

from dbguide.domain.interfaces import RetrievalService, SearchResult
from dbguide.models.document import Document
from dbguide.services.indexing import BM25IndexBuilder


class HybridRetrievalService(RetrievalService):
    """
    Implements hybrid search combining vector similarity and BM25 keyword matching.
    Follows the Single Responsibility Principle by focusing only on retrieval.
    """

    def __init__(
        self,
        collection: chromadb.api.models.Collection.Collection,
        bm25_index: BM25Okapi,
        documents: List[Document],
    ):
        """
        Initialize hybrid retrieval service.

        Args:
            collection: ChromaDB collection for vector search.
            bm25_index: BM25 index for keyword search.
            documents: List of indexed documents.
        """
        self.collection = collection
        self.bm25_index = bm25_index
        self.documents = documents
        self._doc_map = {doc.id: doc for doc in documents}

    def search(
        self,
        query: str,
        top_k: int = 6,
        alpha: float = 0.55,
        metadata_filter: Optional[Dict[str, Any]] = None,
    ) -> List[SearchResult]:
        """
        Perform hybrid search combining vector and keyword approaches.

        Args:
            query: Search query string.
            top_k: Number of results to return.
            alpha: Weight for vector score (0..1). (1-alpha) is weight for BM25.
            metadata_filter: Optional metadata filter dictionary.

        Returns:
            List of search results ordered by relevance score.
        """
        # 1) Vector-based search with optional metadata filter
        vector_scores = self._vector_search(query, top_k, metadata_filter)

        # 2) BM25 keyword-based search
        bm25_scores = self._bm25_search(query, metadata_filter)

        # 3) Combine scores using alpha weighting
        combined_results = self._combine_scores(
            vector_scores,
            bm25_scores,
            alpha,
            top_k
        )

        self._log_results(combined_results, top_k)

        return combined_results

    def _vector_search(
        self,
        query: str,
        top_k: int,
        metadata_filter: Optional[Dict[str, Any]],
    ) -> Dict[str, float]:
        """
        Perform vector similarity search.

        Args:
            query: Search query.
            top_k: Number of results.
            metadata_filter: Optional filter.

        Returns:
            Dictionary mapping document IDs to similarity scores.
        """
        query_kwargs = {
            "query_texts": [query],
            "n_results": top_k
        }

        if metadata_filter:
            print(f"[VectorSearch] Applying metadata filter: {metadata_filter}")
            query_kwargs["where"] = metadata_filter

        results = self.collection.query(**query_kwargs)

        doc_ids = results["ids"][0]
        distances = results["distances"][0]

        # Convert distance to score (lower distance = higher score)
        scores = {
            doc_id: 1.0 / (1e-6 + float(dist))
            for doc_id, dist in zip(doc_ids, distances)
        }

        return scores

    def _bm25_search(
        self,
        query: str,
        metadata_filter: Optional[Dict[str, Any]],
    ) -> Dict[str, float]:
        """
        Perform BM25 keyword search.

        Args:
            query: Search query.
            metadata_filter: Optional filter.

        Returns:
            Dictionary mapping document IDs to BM25 scores.
        """
        query_tokens = BM25IndexBuilder.tokenize(query)
        all_scores = self.bm25_index.get_scores(query_tokens)

        scores = {}
        for i, doc in enumerate(self.documents):
            # Apply metadata filter if provided
            if metadata_filter and not doc.matches_filter(metadata_filter):
                continue

            scores[doc.id] = float(all_scores[i])

        return scores

    def _combine_scores(
        self,
        vector_scores: Dict[str, float],
        bm25_scores: Dict[str, float],
        alpha: float,
        top_k: int,
    ) -> List[SearchResult]:
        """
        Combine vector and BM25 scores using weighted average.

        Args:
            vector_scores: Vector similarity scores.
            bm25_scores: BM25 keyword scores.
            alpha: Weight for vector scores (0..1).
            top_k: Number of results to return.

        Returns:
            List of search results sorted by combined score.
        """
        all_doc_ids = set(vector_scores.keys()) | set(bm25_scores.keys())

        combined = []
        for doc_id in all_doc_ids:
            vec_score = vector_scores.get(doc_id, 0.0)
            bm25_score = bm25_scores.get(doc_id, 0.0)

            # Weighted combination
            final_score = alpha * vec_score + (1 - alpha) * bm25_score

            combined.append((doc_id, final_score))

        # Sort by score descending
        combined.sort(key=lambda x: x[1], reverse=True)

        # Convert to SearchResult objects
        results = []
        for doc_id, score in combined[:top_k]:
            doc = self._doc_map.get(doc_id)
            if doc:
                results.append(SearchResult(
                    id=doc_id,
                    score=score,
                    text=doc.text,
                    metadata=doc.metadata
                ))

        return results

    def _log_results(self, results: List[SearchResult], top_k: int) -> None:
        """Log search results for debugging."""
        print(f"[HybridSearch] Top {top_k} results:")
        for result in results:
            print(f"  - {result.id} | score={result.score:.4f}")
