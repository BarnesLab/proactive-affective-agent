"""Generate evaluation reports in CSV, JSON, and summary formats."""

from __future__ import annotations

from pathlib import Path


class Reporter:
    """Generates structured reports from evaluation results."""

    def __init__(self, output_dir: Path) -> None:
        self.output_dir = output_dir

    def save_csv(self, results: list[dict], filename: str) -> Path:
        """Save results as CSV."""
        raise NotImplementedError

    def save_json(self, results: dict, filename: str) -> Path:
        """Save results as JSON."""
        raise NotImplementedError

    def generate_summary(self, results: dict) -> str:
        """Generate a human-readable summary of evaluation results."""
        raise NotImplementedError

    def save_all(self, results: dict, prefix: str = "eval") -> dict[str, Path]:
        """Save all report formats. Returns {format: path}."""
        raise NotImplementedError
