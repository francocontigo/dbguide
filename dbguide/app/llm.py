"""
LLM utilities for DBGuide: wrappers for Ollama and OpenAI chat APIs.
All functions are type-annotated and documented.

DEPRECATED: This module is kept for backward compatibility.
Please use dbguide.services.llm_providers for new code.
"""
from __future__ import annotations

# Import from compatibility layer
from dbguide.app.llm_compat import ollama_chat, openai_chat

__all__ = ['ollama_chat', 'openai_chat']
