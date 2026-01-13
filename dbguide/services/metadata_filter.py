"""
Metadata filtering strategies for intelligent query routing.
"""
from __future__ import annotations

import json
import logging
from typing import Any, Dict, List, Optional

from dbguide.domain.interfaces import MetadataFilterService, LLMProvider

logger = logging.getLogger("dbguide")


class HeuristicMetadataFilter(MetadataFilterService):
    """
    Heuristic-based metadata filter using keyword matching.
    Follows the Open/Closed Principle - can be extended without modification.
    """

    def suggest_filter(
        self,
        query: str,
        available_metadata: Dict[str, List[Any]],
    ) -> Optional[Dict[str, Any]]:
        """
        Suggest metadata filter by checking if metadata values appear in query.

        Args:
            query: User query string.
            available_metadata: Available metadata keys and their possible values.

        Returns:
            Suggested filter dictionary or None if no matches found.
        """
        query_lower = query.lower()
        filter_dict = {}

        for key, values in available_metadata.items():
            for value in values:
                if value and isinstance(value, str):
                    if value.lower() in query_lower:
                        filter_dict[key] = value

        return filter_dict if filter_dict else None


class LLMMetadataFilter(MetadataFilterService):
    """
    LLM-based metadata filter using AI to understand query intent.
    Demonstrates Dependency Injection with LLMProvider.
    """

    def __init__(
        self,
        llm_provider: LLMProvider,
        model: str = "gpt-3.5-turbo",
    ):
        """
        Initialize LLM-based metadata filter.

        Args:
            llm_provider: LLM provider instance for generating suggestions.
            model: Model name to use for filtering.
        """
        self.llm_provider = llm_provider
        self.model = model

    def suggest_filter(
        self,
        query: str,
        available_metadata: Dict[str, List[Any]],
    ) -> Optional[Dict[str, Any]]:
        """
        Use LLM to intelligently suggest metadata filter.

        Args:
            query: User query string.
            available_metadata: Available metadata keys and their possible values.

        Returns:
            Suggested filter dictionary or None if LLM suggests no filter.
        """
        system_prompt = (
            "You are an assistant for SQL card retrieval. "
            "Given the user's question and available metadata, suggest a metadata filter (in JSON) "
            "to find the most relevant cards. "
            "Respond only with a JSON dictionary of filters, or an empty dictionary if no filter applies."
        )

        user_prompt = (
            f"Question: {query}\n"
            f"Available metadata: {available_metadata}\n"
            "Respond with a JSON filter dictionary only."
        )

        try:
            response = self.llm_provider.chat(
                model=self.model,
                system=system_prompt,
                user=user_prompt,
                temperature=0.0,
            )

            logger.info(f"[LLMMetadataFilter] Raw response: {response}")

            # Parse JSON response
            filter_dict = json.loads(response)

            if not isinstance(filter_dict, dict):
                logger.warning(f"[LLMMetadataFilter] Response is not a dict: {type(filter_dict)}")
                return None

            return filter_dict if filter_dict else None

        except json.JSONDecodeError as e:
            logger.warning(f"[LLMMetadataFilter] Failed to parse JSON: {e}")
            return None
        except Exception as e:
            logger.warning(f"[LLMMetadataFilter] Error suggesting filter: {e}")
            return None


# Factory function for creating filter services
def create_metadata_filter(
    filter_type: str,
    llm_provider: Optional[LLMProvider] = None,
    model: Optional[str] = None,
) -> MetadataFilterService:
    """
    Factory function to create metadata filter services.

    Args:
        filter_type: Type of filter ('heuristic' or 'llm').
        llm_provider: LLM provider (required for 'llm' type).
        model: Model name (optional, for 'llm' type).

    Returns:
        Configured metadata filter service.

    Raises:
        ValueError: If filter_type is invalid or required args are missing.
    """
    if filter_type.lower() == "heuristic":
        return HeuristicMetadataFilter()

    elif filter_type.lower() == "llm":
        if llm_provider is None:
            raise ValueError("llm_provider is required for LLM filter type")

        kwargs = {"llm_provider": llm_provider}
        if model:
            kwargs["model"] = model

        return LLMMetadataFilter(**kwargs)

    else:
        raise ValueError(f"Unknown filter type: {filter_type}")
