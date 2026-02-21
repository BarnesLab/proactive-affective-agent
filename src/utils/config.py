"""YAML configuration loading with environment variable resolution."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml


def load_config(config_path: Path) -> dict:
    """Load a YAML config file.

    Args:
        config_path: Path to the YAML file.

    Returns:
        Parsed config dict.
    """
    with open(config_path) as f:
        return yaml.safe_load(f)


def load_all_configs(config_dir: Path) -> dict[str, dict]:
    """Load all YAML configs from the config directory.

    Returns:
        Dict with config name (without extension) as key.
    """
    configs = {}
    for path in config_dir.glob("*.yaml"):
        configs[path.stem] = load_config(path)
    return configs


def resolve_env_vars(config: dict) -> dict:
    """Resolve environment variable references in config values.

    Values ending with '_env' are treated as env var names and resolved.
    """
    raise NotImplementedError
