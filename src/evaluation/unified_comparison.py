"""Unified comparison across all methods: 5 LLM versions + ML baselines.

Loads results from LLM pilot runs and ML baseline runs, computes unified
metrics, and generates comparison tables (markdown + LaTeX for paper).
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)


class UnifiedComparison:
    """Cross-method comparison: LLM versions vs ML baselines."""

    def __init__(
        self,
        llm_results_dir: Path,
        ml_results_dir: Path,
        output_dir: Path | None = None,
    ) -> None:
        """
        Args:
            llm_results_dir: Directory with LLM pilot outputs (comparison_metrics.json).
            ml_results_dir: Directory with ML baseline outputs (ml_baseline_metrics.json).
            output_dir: Where to save comparison outputs.
        """
        self.llm_dir = Path(llm_results_dir)
        self.ml_dir = Path(ml_results_dir)
        self.output_dir = Path(output_dir) if output_dir else self.llm_dir / "unified"
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def load_llm_metrics(self) -> dict[str, Any]:
        """Load LLM version metrics from comparison_metrics.json."""
        path = self.llm_dir / "comparison_metrics.json"
        if not path.exists():
            logger.warning(f"LLM metrics not found at {path}")
            return {}
        with open(path) as f:
            return json.load(f)

    def load_ml_metrics(self) -> dict[str, Any]:
        """Load ML baseline metrics from ml_baseline_metrics.json."""
        path = self.ml_dir / "ml_baseline_metrics.json"
        if not path.exists():
            logger.warning(f"ML metrics not found at {path}")
            return {}
        with open(path) as f:
            return json.load(f)

    def compute_all(self) -> dict[str, Any]:
        """Load all results and compute unified comparison.

        Returns:
            Dict with per-method metrics and rankings.
        """
        llm = self.load_llm_metrics()
        ml = self.load_ml_metrics()

        unified = {}

        # Add LLM versions
        for version, metrics in llm.items():
            agg = metrics.get("aggregate", {})
            unified[version] = {
                "type": "llm",
                "mean_mae": agg.get("mean_mae"),
                "mean_ba": agg.get("mean_balanced_accuracy"),
                "mean_f1": agg.get("mean_f1"),
                "personal_threshold_ba": agg.get("personal_threshold_mean_ba"),
                "personal_threshold_f1": agg.get("personal_threshold_mean_f1"),
                "continuous": metrics.get("continuous", {}),
                "binary": metrics.get("binary", {}),
            }

        # Add ML baselines
        for model_name, targets in ml.items():
            agg = targets.get("_aggregate", {})
            unified[f"ml_{model_name}"] = {
                "type": "ml",
                "mean_mae": agg.get("mean_mae"),
                "mean_ba": agg.get("mean_ba"),
                "mean_f1": agg.get("mean_f1"),
                "targets": {
                    k: v for k, v in targets.items() if k != "_aggregate"
                },
            }

        # Compute rankings
        unified["_rankings"] = self._compute_rankings(unified)

        # Save
        self._save_results(unified)

        return unified

    def _compute_rankings(self, unified: dict) -> dict[str, list]:
        """Rank methods by each aggregate metric."""
        rankings = {}

        for metric in ["mean_mae", "mean_ba", "mean_f1"]:
            methods = []
            for name, data in unified.items():
                if name.startswith("_"):
                    continue
                val = data.get(metric)
                if val is not None:
                    methods.append((name, val))

            if not methods:
                continue

            # MAE: lower is better; BA/F1: higher is better
            reverse = metric != "mean_mae"
            methods.sort(key=lambda x: x[1], reverse=reverse)
            rankings[metric] = [{"rank": i + 1, "method": m, "value": v} for i, (m, v) in enumerate(methods)]

        return rankings

    def _save_results(self, unified: dict) -> None:
        """Save unified comparison as JSON, markdown, and LaTeX."""
        # JSON
        json_path = self.output_dir / "unified_comparison.json"
        with open(json_path, "w") as f:
            json.dump(unified, f, indent=2, default=str)

        # Markdown
        md_path = self.output_dir / "unified_comparison.md"
        md_path.write_text(self._generate_markdown(unified))

        # LaTeX
        latex_path = self.output_dir / "unified_comparison.tex"
        latex_path.write_text(self._generate_latex(unified))

        logger.info(f"Unified comparison saved to {self.output_dir}")

    def _generate_markdown(self, unified: dict) -> str:
        """Generate markdown comparison table."""
        lines = ["# Unified Comparison: LLM Versions vs ML Baselines\n"]

        # Aggregate metrics table
        lines.append("## Aggregate Metrics\n")
        methods = [k for k in unified if not k.startswith("_")]
        lines.append("| Method | Type | Mean MAE | Mean BA | Mean F1 |")
        lines.append("|--------|------|----------|---------|---------|")

        for method in methods:
            data = unified[method]
            mtype = data.get("type", "?")
            mae = data.get("mean_mae")
            ba = data.get("mean_ba")
            f1 = data.get("mean_f1")
            lines.append(
                f"| {method} | {mtype} | "
                f"{mae:.3f if mae is not None else 'N/A'} | "
                f"{ba:.3f if ba is not None else 'N/A'} | "
                f"{f1:.3f if f1 is not None else 'N/A'} |"
            )

        lines.append("")

        # Rankings
        rankings = unified.get("_rankings", {})
        if rankings:
            lines.append("## Rankings\n")
            for metric, ranking in rankings.items():
                display = metric.replace("mean_", "").upper()
                lines.append(f"### {display}")
                for entry in ranking:
                    marker = " **BEST**" if entry["rank"] == 1 else ""
                    lines.append(
                        f"  {entry['rank']}. {entry['method']}: {entry['value']:.3f}{marker}"
                    )
                lines.append("")

        return "\n".join(lines)

    def _generate_latex(self, unified: dict) -> str:
        """Generate LaTeX table for paper."""
        methods = [k for k in unified if not k.startswith("_")]

        lines = [
            r"\begin{table}[ht]",
            r"\centering",
            r"\caption{Unified comparison: LLM versions vs ML baselines}",
            r"\label{tab:unified}",
            r"\begin{tabular}{llccc}",
            r"\toprule",
            r"Method & Type & Mean MAE $\downarrow$ & Mean BA $\uparrow$ & Mean F1 $\uparrow$ \\",
            r"\midrule",
        ]

        # Find best values for bolding
        maes = [unified[m].get("mean_mae") for m in methods if unified[m].get("mean_mae") is not None]
        bas = [unified[m].get("mean_ba") for m in methods if unified[m].get("mean_ba") is not None]
        f1s = [unified[m].get("mean_f1") for m in methods if unified[m].get("mean_f1") is not None]
        best_mae = min(maes) if maes else None
        best_ba = max(bas) if bas else None
        best_f1 = max(f1s) if f1s else None

        for method in methods:
            data = unified[method]
            mtype = data.get("type", "?")
            mae = data.get("mean_mae")
            ba = data.get("mean_ba")
            f1 = data.get("mean_f1")

            def fmt(val, best, lower_better=False):
                if val is None:
                    return "N/A"
                s = f"{val:.3f}"
                if best is not None and abs(val - best) < 1e-6:
                    s = r"\textbf{" + s + "}"
                return s

            mae_s = fmt(mae, best_mae, lower_better=True)
            ba_s = fmt(ba, best_ba)
            f1_s = fmt(f1, best_f1)

            lines.append(f"{method} & {mtype} & {mae_s} & {ba_s} & {f1_s} \\\\")

        lines.extend([
            r"\bottomrule",
            r"\end{tabular}",
            r"\end{table}",
        ])

        return "\n".join(lines)
