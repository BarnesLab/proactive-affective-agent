"""Self-evaluation: compare agent predictions against EMA ground truth.

After each EMA arrives, the agent evaluates its own performance and
identifies areas for improvement.
"""

from __future__ import annotations


class SelfEvaluator:
    """Compares predictions to ground truth and generates learning signals."""

    def evaluate(self, prediction: dict, ground_truth: dict) -> dict:
        """Compare a single prediction to its ground truth.

        Args:
            prediction: Agent's predicted emotional states and receptivity.
            ground_truth: Actual EMA responses.

        Returns:
            Evaluation dict with accuracy, error magnitude, and observations.
        """
        raise NotImplementedError

    def evaluate_batch(self, predictions: list[dict], ground_truths: list[dict]) -> dict:
        """Evaluate a batch of predictions. Returns aggregate metrics."""
        raise NotImplementedError

    def generate_reflection(self, evaluation: dict) -> str:
        """Generate a natural language reflection on the evaluation results.

        Used to update agent memory with self-awareness insights.
        """
        raise NotImplementedError
