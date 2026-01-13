"""
LLM provider implementations for DBGuide.
Concrete implementations of the LLMProvider interface.
"""
from __future__ import annotations

import os
from typing import Optional

import requests
from openai import OpenAI

from dbguide.domain.interfaces import LLMProvider


class OllamaProvider(LLMProvider):
    """Ollama LLM provider implementation."""

    def __init__(self, base_url: Optional[str] = None):
        """
        Initialize Ollama provider.

        Args:
            base_url: Ollama server URL. Defaults to env var OLLAMA_URL or localhost.
        """
        self.base_url = base_url or os.getenv("OLLAMA_URL", "http://localhost:11434")

    def chat(
        self,
        model: str,
        system: str,
        user: str,
        temperature: float = 0.2,
        timeout_s: Optional[int] = None,
    ) -> str:
        """
        Get a single response from a model running via Ollama.

        Args:
            model: Model name (e.g. "mistral:7b-instruct").
            system: System prompt.
            user: User prompt.
            temperature: Sampling temperature.
            timeout_s: Timeout in seconds (default 100).

        Returns:
            Response string from the model.
        """
        effective_timeout = timeout_s or 100

        payload = {
            "model": model,
            "stream": False,
            "options": {"temperature": temperature},
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
        }

        resp = requests.post(
            f"{self.base_url}/api/chat",
            json=payload,
            timeout=effective_timeout
        )
        resp.raise_for_status()

        data = resp.json()
        return data.get("message", {}).get("content", "")


class OpenAIProvider(LLMProvider):
    """OpenAI LLM provider implementation."""

    def __init__(self, api_key: Optional[str] = None):
        """
        Initialize OpenAI provider.

        Args:
            api_key: OpenAI API key. If None, will use environment variable.
        """
        if api_key:
            self.client = OpenAI(api_key=api_key)
        else:
            self.client = OpenAI()  # Uses OPENAI_API_KEY from environment

        self.default_timeout = int(os.getenv("OPENAI_TIMEOUT_S", "100"))

    def chat(
        self,
        model: str,
        system: str,
        user: str,
        temperature: float = 0.2,
        timeout_s: Optional[int] = None,
    ) -> str:
        """
        Get a single response from a model hosted on OpenAI.

        Args:
            model: Model name (e.g. "gpt-4o-mini").
            system: System prompt.
            user: User prompt.
            temperature: Sampling temperature.
            timeout_s: Timeout in seconds.

        Returns:
            Response string from the model.
        """
        effective_timeout = timeout_s or self.default_timeout

        resp = self.client.chat.completions.create(
            model=model,
            temperature=temperature,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            timeout=effective_timeout,
        )

        message = resp.choices[0].message
        return message.content or ""


# Factory function for backward compatibility
def create_llm_provider(provider_type: str, **kwargs) -> LLMProvider:
    """
    Factory function to create LLM providers.

    Args:
        provider_type: Type of provider ('ollama' or 'openai').
        **kwargs: Additional arguments passed to provider constructor.

    Returns:
        Configured LLM provider instance.

    Raises:
        ValueError: If provider_type is not supported.
    """
    providers = {
        "ollama": OllamaProvider,
        "openai": OpenAIProvider,
    }

    provider_class = providers.get(provider_type.lower())
    if not provider_class:
        raise ValueError(f"Unknown provider type: {provider_type}")

    return provider_class(**kwargs)
