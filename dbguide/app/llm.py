from __future__ import annotations

import os
from typing import Optional

import requests
from openai import OpenAI


DEFAULT_OLLAMA_URL = os.getenv("OLLAMA_URL", "http://localhost:11434")
DEFAULT_OPENAI_TIMEOUT_S = int(os.getenv("OPENAI_TIMEOUT_S", "100"))


def ollama_chat(
    model: str,
    system: str,
    user: str,
    ollama_url: str = DEFAULT_OLLAMA_URL,
    temperature: float = 0.2,
    timout_s: int = 100,
) -> str:
    """Get a single response from a model running via Ollama."""

    payload = {
        "model": model,
        "stream": False,
        "options": {"temperature": temperature},
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
    }
    resp = requests.post(f"{ollama_url}/api/chat", json=payload, timeout=timout_s)
    resp.raise_for_status()
    data = resp.json()
    # {"message": {"role":"assistant","content":"..."}}
    return data.get("message", {}).get("content", "")


def openai_chat(
    model: str,
    system: str,
    user: str,
    temperature: float = 0.2,
    timeout_s: Optional[int] = None,
) -> str:
    """Get a single response from a model hosted on OpenAI."""

    client = OpenAI()
    effective_timeout = timeout_s or DEFAULT_OPENAI_TIMEOUT_S

    resp = client.chat.completions.create(
        model=model,
        temperature=temperature,
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
        timeout=effective_timeout,
    )

    message = resp.choices[0].message
    # message.content can be None if the SDK ever returns it so
    return message.content or ""
