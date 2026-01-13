"""
SQL safety utilities for DBGuide.

DEPRECATED: This module is kept for backward compatibility.
Please use dbguide.services.sql_validator for new code.
"""
from __future__ import annotations

from typing import Dict, List

# Import from compatibility layer
from dbguide.app.safety_compat import (
    basic_sql_safety_check,
    extract_sql_block,
    split_structured_output
)

__all__ = ['basic_sql_safety_check', 'extract_sql_block', 'split_structured_output']
