"""Generate evaluation reports: CSV, JSON, and comparison table.

Outputs per-version raw predictions and a cross-version comparison table.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd


class Reporter:
    """Generates structured reports from pilot evaluation results."""

    def __init__(self, output_dir: Path) -> None:
        self.output_dir = output_dir
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def save_predictions_csv(
        self,
        predictions: list[dict],
        ground_truths: list[dict],
        metadata: list[dict],
        filename: str,
    ) -> Path:
        """Save raw predictions + ground truth as CSV.

        Args:
            predictions: List of prediction dicts.
            ground_truths: List of ground truth dicts.
            metadata: List of metadata dicts (study_id, date, etc.).
            filename: Output filename (e.g., "callm_predictions.csv").
        """
        rows = []
        for pred, gt, meta in zip(predictions, ground_truths, metadata):
            row = {}
            row.update({f"meta_{k}": v for k, v in meta.items()})
            row.update({f"pred_{k}": v for k, v in pred.items() if not k.startswith("_")})
            row.update({f"true_{k}": v for k, v in gt.items()})
            rows.append(row)

        df = pd.DataFrame(rows)
        path = self.output_dir / filename
        df.to_csv(path, index=False)
        return path

    def save_metrics_json(self, metrics: dict, filename: str) -> Path:
        """Save metrics dict as JSON."""
        path = self.output_dir / filename
        with open(path, "w") as f:
            json.dump(metrics, f, indent=2, default=str)
        return path

    def save_comparison_table(
        self,
        version_metrics: dict[str, dict],
        filename: str = "comparison_table.md",
    ) -> Path:
        """Generate a human-readable comparison table (CALLM vs V1 vs V2).

        Args:
            version_metrics: {version_name: metrics_dict} from compute_all().
        """
        lines = ["# Pilot Study Results: CALLM vs V1 vs V2\n"]

        # Aggregate summary
        lines.append("## Aggregate Metrics\n")
        lines.append("| Metric | " + " | ".join(version_metrics.keys()) + " |")
        lines.append("|--------|" + "|".join(["--------"] * len(version_metrics)) + "|")

        for metric_name in ["mean_mae", "mean_balanced_accuracy", "mean_f1"]:
            vals = []
            for v_name, m in version_metrics.items():
                val = m.get("aggregate", {}).get(metric_name)
                vals.append(f"{val:.3f}" if val is not None else "N/A")
            display_name = metric_name.replace("mean_", "Mean ").replace("_", " ").title()
            lines.append(f"| {display_name} | " + " | ".join(vals) + " |")

        lines.append("")

        # Continuous targets detail
        lines.append("## Continuous Targets (MAE, lower is better)\n")
        lines.append("| Target | " + " | ".join(version_metrics.keys()) + " |")
        lines.append("|--------|" + "|".join(["--------"] * len(version_metrics)) + "|")

        continuous_targets = set()
        for m in version_metrics.values():
            continuous_targets.update(m.get("continuous", {}).keys())

        for target in sorted(continuous_targets):
            vals = []
            for v_name, m in version_metrics.items():
                t_metrics = m.get("continuous", {}).get(target, {})
                val = t_metrics.get("mae")
                vals.append(f"{val:.2f}" if val is not None else "N/A")
            lines.append(f"| {target} | " + " | ".join(vals) + " |")

        lines.append("")

        # Binary targets detail
        lines.append("## Binary State Targets (Balanced Accuracy / F1)\n")
        lines.append("| Target | " + " | ".join(f"{v} BA / F1" for v in version_metrics.keys()) + " |")
        lines.append("|--------|" + "|".join(["--------"] * len(version_metrics)) + "|")

        binary_targets = set()
        for m in version_metrics.values():
            binary_targets.update(m.get("binary", {}).keys())

        for target in sorted(binary_targets):
            short = target.replace("Individual_level_", "").replace("_State", "")
            vals = []
            for v_name, m in version_metrics.items():
                t_metrics = m.get("binary", {}).get(target, {})
                ba = t_metrics.get("balanced_accuracy")
                f1 = t_metrics.get("f1")
                if ba is not None:
                    vals.append(f"{ba:.2f} / {f1:.2f}")
                else:
                    vals.append("N/A")
            lines.append(f"| {short} | " + " | ".join(vals) + " |")

        lines.append("")

        # Availability
        lines.append("## Availability Prediction\n")
        lines.append("| Metric | " + " | ".join(version_metrics.keys()) + " |")
        lines.append("|--------|" + "|".join(["--------"] * len(version_metrics)) + "|")
        for metric_name in ["balanced_accuracy", "f1"]:
            vals = []
            for v_name, m in version_metrics.items():
                val = m.get("availability", {}).get(metric_name)
                vals.append(f"{val:.3f}" if val is not None else "N/A")
            lines.append(f"| {metric_name} | " + " | ".join(vals) + " |")

        content = "\n".join(lines)
        path = self.output_dir / filename
        path.write_text(content)
        return path

    def save_trace(self, trace_data: dict, study_id: int, entry_idx: int, version: str) -> Path:
        """Save raw LLM trace for debugging."""
        trace_dir = self.output_dir / "traces"
        trace_dir.mkdir(exist_ok=True)
        path = trace_dir / f"{version}_user{study_id}_entry{entry_idx}.json"
        with open(path, "w") as f:
            json.dump(trace_data, f, indent=2, default=str)
        return path

    def save_unified_record(self, record: dict, version: str, study_id: int) -> Path:
        """Append one unified record to the JSONL log.

        Each line is a self-contained JSON object with everything needed for
        post-hoc analysis of a single (user, EMA) prediction.
        """
        path = self.output_dir / f"{version}_user{study_id}_records.jsonl"
        with open(path, "a") as f:
            f.write(json.dumps(record, default=str) + "\n")
        return path
