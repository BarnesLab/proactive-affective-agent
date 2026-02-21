"""Evaluation metrics: BA, MAE, F1, receptivity accuracy.

Computes standard metrics for emotional state prediction and
receptivity classification.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import numpy as np


class MetricsCalculator:
    """Computes evaluation metrics for agent predictions."""

    @staticmethod
    def balanced_accuracy(y_true: list, y_pred: list) -> float:
        """Balanced accuracy for categorical predictions."""
        raise NotImplementedError

    @staticmethod
    def mae(y_true: list[float], y_pred: list[float]) -> float:
        """Mean absolute error for continuous predictions."""
        raise NotImplementedError

    @staticmethod
    def f1_score(y_true: list, y_pred: list, average: str = "binary") -> float:
        """F1 score for classification (especially receptivity)."""
        raise NotImplementedError

    @staticmethod
    def receptivity_accuracy(y_true: list[bool], y_pred: list[bool]) -> dict:
        """Receptivity-specific metrics: accuracy, precision, recall, F1."""
        raise NotImplementedError

    @staticmethod
    def calibration_error(confidences: list[float], accuracies: list[bool]) -> float:
        """Expected calibration error (ECE)."""
        raise NotImplementedError
