"""
Document and data models for DBGuide retrieval system.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict


@dataclass
class Document:
    """
    Represents a single markdown document/card with optional metadata.
    """
    id: str
    text: str
    metadata: Dict[str, Any] = field(default_factory=dict)

    def matches_filter(self, filter_dict: Dict[str, Any]) -> bool:
        """
        Check if document matches the given metadata filter.

        Args:
            filter_dict: Metadata filter dictionary.

        Returns:
            True if all filter conditions are met.
        """
        return all(
            k in self.metadata and self.metadata[k] == v
            for k, v in filter_dict.items()
        )
