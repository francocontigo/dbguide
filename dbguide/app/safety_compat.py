"""
Backward compatibility module for safety functions.
Provides the original function signatures wrapping the new service-based architecture.
"""
from __future__ import annotations

from typing import Dict, List

from dbguide.services.sql_validator import BasicSQLValidator, SQLOutputParser


def basic_sql_safety_check(sql: str) -> List[str]:
    """
    Perform basic SQL safety checks.

    Args:
        sql: SQL query string.

    Returns:
        List of issues found.
    """
    validator = BasicSQLValidator()
    return validator.validate(sql)


def extract_sql_block(raw: str) -> str:
    """
    Extract SQL block from LLM response.

    Args:
        raw: Raw LLM response.

    Returns:
        Extracted SQL query.
    """
    parser = SQLOutputParser()
    return parser.extract_sql_block(raw)


def split_structured_output(raw: str) -> Dict[str, str]:
    """
    Split LLM output into structured sections.

    Args:
        raw: Raw LLM response.

    Returns:
        Dictionary with 'sql', 'explicacao', 'checks' keys.
    """
    parser = SQLOutputParser()
    sections = parser.split_structured_output(raw)

    # Map 'explanation' to 'explicacao' for backward compatibility
    return {
        "sql": sections.get("sql", ""),
        "explicacao": sections.get("explanation", ""),
        "checks": sections.get("checks", "")
    }
