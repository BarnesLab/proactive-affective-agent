"""Unified LLM interface supporting OpenAI, Anthropic, and local endpoints.

Provides a consistent API regardless of the underlying LLM provider.
"""

from __future__ import annotations

from typing import Any


class LLMClient:
    """Unified interface for LLM calls across providers."""

    def __init__(self, provider: str = "openai", config: dict | None = None) -> None:
        self.provider = provider
        self.config = config or {}
        self._client = self._init_client()

    def _init_client(self) -> Any:
        """Initialize the appropriate provider client."""
        raise NotImplementedError

    def generate(self, messages: list[dict], **kwargs) -> str:
        """Generate a completion from the LLM.

        Args:
            messages: List of message dicts (role, content).
            **kwargs: Provider-specific overrides (temperature, max_tokens, etc.)

        Returns:
            Generated text response.
        """
        raise NotImplementedError

    def generate_structured(self, messages: list[dict], schema: dict, **kwargs) -> dict:
        """Generate a structured (JSON) response from the LLM.

        Args:
            messages: List of message dicts.
            schema: Expected output JSON schema.
            **kwargs: Provider-specific overrides.

        Returns:
            Parsed dict matching the schema.
        """
        raise NotImplementedError
