"""Per-user and population normalization for sensing features.

Normalizes features so they can be meaningfully compared:
- Within-user: relative to the user's own baseline
- Across-users: relative to population statistics
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import pandas as pd


class FeatureNormalizer:
    """Normalizes sensing features at per-user and population levels."""

    def __init__(self) -> None:
        self.user_stats: dict[str, dict] = {}
        self.population_stats: dict | None = None

    def fit_population(self, all_features: pd.DataFrame) -> None:
        """Compute population-level statistics from all users' features."""
        raise NotImplementedError

    def fit_user(self, user_id: str, user_features: pd.DataFrame) -> None:
        """Compute per-user baseline statistics."""
        raise NotImplementedError

    def normalize(self, user_id: str, features: dict) -> dict:
        """Normalize features relative to the user's baseline and population norms.

        Returns:
            Dict with both raw and normalized values, plus deviation indicators.
        """
        raise NotImplementedError
