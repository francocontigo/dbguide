"""
SQL validation and safety checking services.
"""
from __future__ import annotations

import re
from typing import List

from dbguide.domain.interfaces import SQLValidator


class BasicSQLValidator(SQLValidator):
    """
    Basic SQL validator checking for dangerous commands and missing WHERE clauses.
    Follows the Single Responsibility Principle - focused only on SQL validation.
    """

    # Regex pattern for disallowed SQL commands
    DISALLOWED_PATTERN = re.compile(
        r"\b(drop|delete|truncate|alter|update|insert|create)\b",
        re.IGNORECASE
    )

    def validate(self, sql: str) -> List[str]:
        """
        Validate SQL query and return list of issues.

        Args:
            sql: SQL query string to validate.

        Returns:
            List of validation issues. Empty list if no issues found.
        """
        issues = []

        if not sql:
            return issues

        sql_lower = sql.lower()

        # Check for disallowed DML/DDL commands
        if self.DISALLOWED_PATTERN.search(sql):
            issues.append(
                "Detected forbidden DML/DDL command (DROP/DELETE/UPDATE/INSERT/...)."
            )

        # Check for SELECT without WHERE clause
        if "select" in sql_lower and "where" not in sql_lower:
            issues.append(
                "Query without a WHERE clause (may scan the entire table)."
            )

        return issues


class SQLOutputParser:
    """
    Parser for extracting structured output from LLM responses.
    Handles [SQL], [EXPLANATION], and [CHECKS] sections.
    """

    @staticmethod
    def extract_sql_block(raw: str) -> str:
        """
        Extract the content between the [SQL] tag and the next section.

        Args:
            raw: Raw LLM response text.

        Returns:
            Extracted SQL query or full text if no [SQL] tag found.
        """
        if not raw:
            return ""

        lower = raw.lower()
        start = lower.find("[sql]")

        if start == -1:
            return raw.strip()

        after = raw[start + len("[SQL]"):]

        # Find next section marker
        next_markers = ["[explanation]", "[explicacao]", "[explicação]", "[checks]"]
        end_positions = []

        for marker in next_markers:
            pos = after.lower().find(marker)
            if pos != -1:
                end_positions.append(pos)

        end = min(end_positions) if end_positions else len(after)

        return after[:end].strip()

    @staticmethod
    def split_structured_output(raw: str) -> dict[str, str]:
        """
        Split LLM output into [SQL], [EXPLANATION] and [CHECKS] sections.

        Args:
            raw: Raw LLM response text.

        Returns:
            Dictionary with keys "sql", "explanation" and "checks".
            Empty strings for missing sections.
        """
        sections: dict[str, List[str]] = {
            "sql": [],
            "explanation": [],
            "checks": []
        }

        current: str | None = None

        for line in (raw or "").splitlines():
            tag = line.strip().lower()

            if tag == "[sql]":
                current = "sql"
                continue
            elif tag in ("[explanation]", "[explicacao]", "[explicação]"):
                current = "explanation"
                continue
            elif tag == "[checks]":
                current = "checks"
                continue

            if current is not None:
                sections[current].append(line)

        return {key: "\n".join(lines).strip() for key, lines in sections.items()}

    @staticmethod
    def strip_code_fences(text: str | None) -> str:
        """
        Remove all ```...``` code fences from text.

        Args:
            text: Text potentially containing code fences.

        Returns:
            Text with code fences removed.
        """
        if not text:
            return ""

        lines = text.strip().splitlines()
        clean_lines = [
            line for line in lines
            if not line.strip().startswith("```")
        ]

        return "\n".join(clean_lines).strip()
