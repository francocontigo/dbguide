"""
Prompt building service for SQL generation.
"""
from __future__ import annotations

from typing import Dict, List

from dbguide.domain.interfaces import PromptBuilder, SearchResult


class SQLPromptBuilder(PromptBuilder):
    """
    Builds system and user prompts for SQL generation.
    Follows the Single Responsibility Principle - focused only on prompt construction.
    """

    MAX_TOTAL_CONTEXT_CHARS = 9000
    MAX_CARD_CHARS = 3000

    def build_system_prompt(self, dialect: str) -> str:
        """
        Build system prompt for the given SQL dialect.

        Args:
            dialect: SQL dialect (e.g., 'mysql', 'redshift').

        Returns:
            System prompt string with rules and format requirements.
        """
        return f"""
You must write read-only SQL queries (SELECT only).
Target dialect: {dialect}

Mandatory rules:
- NEVER use: DROP, DELETE, UPDATE, INSERT, TRUNCATE, ALTER, CREATE.
- Prefer queries with a date filter whenever there is a temporal column (e.g., created_at).
- Avoid cardinality explosion: when joining 1:N tables, deduplicate to 1 row when appropriate.
- At the end, return exactly 3 sections:

[SQL]
<only the query>

[EXPLANATION]
- 3 bullet points describing what the query does

[CHECKS]
- 2 short SQL checks to validate duplicates/volume
""".strip()

    def build_user_prompt(
        self,
        question: str,
        retrieved_cards: List[SearchResult],
        dialect: str,
    ) -> str:
        """
        Build user prompt with question and retrieved context cards.

        Args:
            question: User's question.
            retrieved_cards: Retrieved search results from RAG.
            dialect: SQL dialect.

        Returns:
            User prompt string with question and context.
        """
        context_parts: List[str] = []
        current_chars = 0

        for card in retrieved_cards:
            # Compact the card text if needed
            compact_text = self._compact_card_text(card.text)

            card_block = f"### CARD: {card.id}\n{compact_text}"
            new_len = current_chars + len(card_block)

            # Stop if we exceed context limit
            if new_len > self.MAX_TOTAL_CONTEXT_CHARS:
                break

            context_parts.append(card_block)
            current_chars = new_len

        context = "\n\n ---\n\n".join(context_parts)

        return f"""
{question}

Database: {dialect}

Use the cards below as reference. Adapt table/column names when necessary; if a table/column is missing, use generic names and let the user complete it.

Cards:
{context}
""".strip()

    def _compact_card_text(self, text: str) -> str:
        """
        Reduce card text size while preserving important content.

        Strategy:
        - Keep the first block (title/introduction).
        - Prioritize paragraphs containing SQL keywords.
        - Fill with other paragraphs until character limit is reached.

        Args:
            text: Original card text.

        Returns:
            Compacted text within MAX_CARD_CHARS limit.
        """
        if len(text) <= self.MAX_CARD_CHARS:
            return text

        paragraphs = text.split("\n\n")
        if not paragraphs:
            return text[:self.MAX_CARD_CHARS]

        title = paragraphs[0]
        rest = paragraphs[1:]

        # SQL-related keywords to prioritize
        sql_keywords = [
            "select ",
            " join ",
            " where ",
            " group by ",
            " having ",
            " window ",
            " over(",
        ]

        # Find paragraphs containing SQL
        important: List[str] = []
        for para in rest:
            para_lower = para.lower()
            if any(keyword in para_lower for keyword in sql_keywords):
                important.append(para)

        # Build compacted text
        chunks: List[str] = [title]

        def total_len(parts: List[str]) -> int:
            return len("\n\n".join(parts))

        # Add important SQL-containing paragraphs first
        for para in important:
            if total_len(chunks + [para]) > self.MAX_CARD_CHARS:
                break
            chunks.append(para)

        # Fill remaining space with other paragraphs
        if total_len(chunks) < self.MAX_CARD_CHARS:
            for para in rest:
                if para in important:
                    continue
                if total_len(chunks + [para]) > self.MAX_CARD_CHARS:
                    break
                chunks.append(para)

        compact = "\n\n".join(chunks)

        # Hard limit at max chars
        if len(compact) > self.MAX_CARD_CHARS:
            compact = compact[:self.MAX_CARD_CHARS]

        return compact
