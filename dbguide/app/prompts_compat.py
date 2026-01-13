"""
Backward compatibility module for prompts.
Provides the original function signatures wrapping the new service-based architecture.
"""
from __future__ import annotations

from typing import Dict, List

from dbguide.services.prompt_builder import SQLPromptBuilder
from dbguide.domain.interfaces import SearchResult


def system_prompt(dialect: str) -> str:
    """
    Build system prompt for SQL generation.

    Args:
        dialect: SQL dialect (e.g., 'mysql', 'redshift').

    Returns:
        System prompt string.
    """
    builder = SQLPromptBuilder()
    return builder.build_system_prompt(dialect)


def user_prompt(question: str, retrieved_cards: List[Dict], dialect: str) -> str:
    """
    Build user prompt with question and retrieved cards.

    Args:
        question: User's question.
        retrieved_cards: List of card dictionaries with 'id', 'score', 'text'.
        dialect: SQL dialect.

    Returns:
        User prompt string.
    """
    builder = SQLPromptBuilder()

    # Convert dict format to SearchResult objects
    search_results = [
        SearchResult(
            id=card.get('id', ''),
            score=card.get('score', 0.0),
            text=card.get('text', ''),
            metadata=card.get('metadata', {})
        )
        for card in retrieved_cards
    ]

    return builder.build_user_prompt(question, search_results, dialect)
