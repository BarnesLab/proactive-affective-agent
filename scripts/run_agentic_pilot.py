#!/usr/bin/env python3
"""Agentic pilot runner compatibility wrapper.

This script keeps the project docs/examples stable by providing:
  - python scripts/run_agentic_pilot.py --version v2
  - python scripts/run_agentic_pilot.py --version v4
  - python scripts/run_agentic_pilot.py --version all
It delegates execution to ``scripts/run_pilot.py``.

Defaults are backend-aware:
  - CLAUDE modes (v2/v4): default model ``sonnet``
  - GPT modes (gpt-v2/gpt-v4): default model ``gpt-5.1-codex-mini``
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
RUN_PILOT = PROJECT_ROOT / "scripts" / "run_pilot.py"

DEFAULT_RUN_PILOT_MODEL_CLAUDE = "sonnet"
DEFAULT_RUN_PILOT_MODEL_GPT = "gpt-5.1-codex-mini"


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compatibility wrapper for agentic pilot versions (v2/v4).",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""Examples:
  python scripts/run_agentic_pilot.py --version v2 --users 71,164,119 --dry-run
  python scripts/run_agentic_pilot.py --version v4 --users 71,164,119
  python scripts/run_agentic_pilot.py --version gpt-v2 --users 71,164,119 --model gpt-5.1-codex-mini
  python scripts/run_agentic_pilot.py --version all --users 71,164,119 --output-dir outputs/pilot_agentic
""",
    )

    parser.add_argument(
        "--version",
        type=str,
        default="v2,v4",
        help="Which version(s) to run: v2,v4,gpt-v2,gpt-v4, or all (default: v2,v4)",
    )
    parser.add_argument(
        "--users",
        type=str,
        default="71,164",
        help="Comma-separated Study_IDs (default: 71,164)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Run through the pipeline without LLM calls",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="",
        help=(
            f"Model for this run. Defaults to '{DEFAULT_RUN_PILOT_MODEL_CLAUDE}' for CLAUDE "
            f"agentic versions and '{DEFAULT_RUN_PILOT_MODEL_GPT}' for GPT versions."
        ),
    )
    parser.add_argument("--delay", type=float, default=2.0, help="Delay between LLM calls")
    parser.add_argument("--output-dir", type=str, default=None, help="Output directory")
    parser.add_argument("--data-dir", type=str, default=None, help="Data directory")
    parser.add_argument("--verbose", action="store_true", help="Enable verbose logging")

    return parser.parse_args()


def _normalize_versions(raw: str) -> list[str]:
    if not raw:
        return ["v2", "v4"]

    requested = [v.strip().lower() for v in raw.split(",") if v.strip()]
    if not requested:
        return ["v2", "v4"]

    normalized: list[str] = []
    for v in requested:
        if v == "all":
            normalized.extend(["v2", "v4", "gpt-v2", "gpt-v4"])
        elif v in {"v2", "v4", "gpt-v2", "gpt-v4"}:
            normalized.append(v)
        else:
            raise ValueError(f"Unsupported agentic version: {v}")

    # Stable, duplicate-free order
    seen = set()
    return [v for v in normalized if not (v in seen or seen.add(v))]


def _choose_model(versions: list[str], user_model: str) -> str:
    if user_model:
        return user_model

    has_gpt = any(v.startswith("gpt-") for v in versions)
    has_claude = any(not v.startswith("gpt-") for v in versions)

    if has_gpt and not has_claude:
        return DEFAULT_RUN_PILOT_MODEL_GPT
    if has_claude and not has_gpt:
        return DEFAULT_RUN_PILOT_MODEL_CLAUDE
    if has_gpt and has_claude:
        # Ambiguous mixed backends: ask user to run in separate passes.
        raise RuntimeError(
            "Mixed Claude and GPT agentic versions are not supported in one invocation. "
            "Use separate runs, for example '--version v2' and '--version gpt-v2'."
        )
    return DEFAULT_RUN_PILOT_MODEL_CLAUDE


def _build_run_cmd(args: argparse.Namespace, versions: list[str], model: str) -> list[str]:
    cmd = [
        sys.executable,
        str(RUN_PILOT),
        "--version",
        ",".join(versions),
        "--users",
        args.users,
        "--model",
        model,
        "--delay",
        str(args.delay),
    ]

    if args.dry_run:
        cmd.append("--dry-run")
    if args.output_dir:
        cmd.extend(["--output-dir", args.output_dir])
    if args.data_dir:
        cmd.extend(["--data-dir", args.data_dir])
    if args.verbose:
        cmd.append("--verbose")

    return cmd


def main() -> None:
    args = _parse_args()

    try:
        versions = _normalize_versions(args.version)
        model = _choose_model(versions, args.model)
    except Exception as exc:  # validation failure should be explicit and visible
        print(f"Error: {exc}", file=sys.stderr)
        raise SystemExit(1)

    env = os.environ.copy()
    env["PYTHONPATH"] = str(PROJECT_ROOT)
    cmd = _build_run_cmd(args, versions, model)

    print(f"[run_agentic_pilot] Running: {' '.join(cmd)}")
    result = subprocess.run(cmd, cwd=str(PROJECT_ROOT), env=env)
    raise SystemExit(result.returncode)


if __name__ == "__main__":
    main()
