"""
Domain interfaces for DBGuide.
Defines abstractions following the Dependency Inversion Principle (SOLID).
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional
from dataclasses import dataclass


@dataclass
class SearchResult:
    """Represents a single search result from retrieval."""
    id: str
    score: float
    text: str
    metadata: Dict[str, Any]


class LLMProvider(ABC):
    """Abstract base class for LLM providers."""

    @abstractmethod
    def chat(
        self,
        model: str,
        system: str,
        user: str,
        temperature: float = 0.2,
        timeout_s: Optional[int] = None,
    ) -> str:
        """
        Send a chat request to the LLM provider.

        Args:
            model: Model identifier.
            system: System prompt.
            user: User prompt.
            temperature: Sampling temperature.
            timeout_s: Request timeout in seconds.

        Returns:
            The LLM response as a string.
        """
        pass


class RetrievalService(ABC):
    """Abstract base class for document retrieval."""

    @abstractmethod
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
            alpha: Weight for vector score (0..1). (1-alpha) is weight for keyword.
            metadata_filter: Optional metadata filter dictionary.

        Returns:
            List of search results ordered by relevance.
        """
        pass


class MetadataFilterService(ABC):
    """Abstract base class for metadata filtering strategies."""

    @abstractmethod
    def suggest_filter(
        self,
        query: str,
        available_metadata: Dict[str, List[Any]],
    ) -> Optional[Dict[str, Any]]:
        """
        Suggest metadata filter based on query and available metadata.

        Args:
            query: User query string.
            available_metadata: Available metadata keys and their possible values.

        Returns:
            Suggested filter dictionary or None if no filter is appropriate.
        """
        pass


class SQLValidator(ABC):
    """Abstract base class for SQL validation."""

    @abstractmethod
    def validate(self, sql: str) -> List[str]:
        """
        Validate SQL query and return list of issues.

        Args:
            sql: SQL query string to validate.

        Returns:
            List of validation issues. Empty list if no issues found.
        """
        pass


class PromptBuilder(ABC):
    """Abstract base class for building prompts."""

    @abstractmethod
    def build_system_prompt(self, dialect: str) -> str:
        """
        Build system prompt for the given SQL dialect.

        Args:
            dialect: SQL dialect (e.g., 'mysql', 'redshift').

        Returns:
            System prompt string.
        """
        pass

    @abstractmethod
    def build_user_prompt(
        self,
        question: str,
        retrieved_cards: List[SearchResult],
        dialect: str,
    ) -> str:
        """
        Build user prompt with question and retrieved context.

        Args:
            question: User's question.
            retrieved_cards: Retrieved search results.
            dialect: SQL dialect.

        Returns:
            User prompt string.
        """
        pass
