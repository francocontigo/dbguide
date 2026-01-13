"""
Backward compatibility module for LLM functions.
Provides the original function signatures wrapping the new service-based architecture.
"""
from __future__ import annotations

from typing import Optional

from dbguide.services.llm_providers import OllamaProvider, OpenAIProvider


def ollama_chat(
    model: str,
    system: str,
    user: str,
    ollama_url: str = "http://localhost:11434",
    temperature: float = 0.2,
    timout_s: int = 100,
) -> str:
    """
    Get a single response from a model running via Ollama.

    Backward compatibility wrapper for OllamaProvider.

    Args:
        model: Model name (e.g. "mistral:7b-instruct").
        system: System prompt.
        user: User prompt.
        ollama_url: Ollama server URL.
        temperature: Sampling temperature.
        timout_s: Timeout in seconds.

    Returns:
        Response string from the model.
    """
    provider = OllamaProvider(base_url=ollama_url)
    return provider.chat(model, system, user, temperature, timout_s)


def openai_chat(
    model: str,
    system: str,
    user: str,
    temperature: float = 0.2,
    timeout_s: Optional[int] = None,
) -> str:
    """
    Get a single response from a model hosted on OpenAI.

    Backward compatibility wrapper for OpenAIProvider.

    Args:
        model: Model name (e.g. "gpt-4o-mini").
        system: System prompt.
        user: User prompt.
        temperature: Sampling temperature.
        timeout_s: Timeout in seconds (optional).

    Returns:
        Response string from the model.
    """
    provider = OpenAIProvider()
    return provider.chat(model, system, user, temperature, timeout_s)
