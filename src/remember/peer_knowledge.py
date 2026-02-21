"""Cross-user pattern queries: peer case library and population norms.

Allows an agent to query:
- Peer cases: anonymized cases from users with similar profiles
- Population norms: statistical summaries across all users
- Cross-user patterns: patterns that generalize across the population
"""

from __future__ import annotations

from typing import Any


class PeerKnowledge:
    """Shared knowledge base across all users."""

    def __init__(self) -> None:
        self.population_norms: dict | None = None
        self.peer_cases: list[dict] = []

    def build_norms(self, all_user_data: dict) -> None:
        """Compute population-level norms from all users' data."""
        raise NotImplementedError

    def find_similar_cases(self, pattern: dict, top_k: int = 5) -> list[dict]:
        """Find peer cases with similar sensing/behavioral patterns.

        Args:
            pattern: Current user's sensing pattern.
            top_k: Number of similar cases to return.

        Returns:
            List of anonymized peer cases with outcomes.
        """
        raise NotImplementedError

    def get_population_norm(self, feature: str) -> dict:
        """Get population-level statistics for a specific feature."""
        raise NotImplementedError

    def get_cross_user_patterns(self) -> list[str]:
        """Return patterns that generalize across most users."""
        raise NotImplementedError
