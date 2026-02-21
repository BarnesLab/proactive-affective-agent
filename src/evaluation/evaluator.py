"""Run retrospective evaluation across all users.

Orchestrates the full evaluation pipeline:
1. For each fold, run simulation on test users
2. Collect predictions and ground truth
3. Compute metrics
4. Generate reports
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from src.evaluation.metrics import MetricsCalculator
    from src.evaluation.reporter import Reporter


class Evaluator:
    """Runs evaluation across all users and folds."""

    def __init__(self, data_dir: Path, output_dir: Path, agent_version: str = "v1") -> None:
        self.data_dir = data_dir
        self.output_dir = output_dir
        self.agent_version = agent_version

    def run_fold(self, fold: int) -> dict:
        """Run evaluation on a single fold. Returns fold-level metrics."""
        raise NotImplementedError

    def run_all_folds(self) -> dict:
        """Run evaluation across all 5 folds. Returns aggregate metrics."""
        raise NotImplementedError

    def compare_versions(self) -> dict:
        """Compare V1 vs V2 agent performance across all folds."""
        raise NotImplementedError
