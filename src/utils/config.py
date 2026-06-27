"""Configuration loading and path resolution utilities."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

CONFIG_DIR_NAME = "configs"
DEFAULT_CONFIG_FILES = ("paths.yaml", "features.yaml", "models.yaml", "training.yaml")


def get_project_root(start: Path | None = None) -> Path:
    """Return the repository root (directory that contains ``configs/``)."""
    current = (start or Path.cwd()).resolve()
    for candidate in (current, *current.parents):
        if (candidate / CONFIG_DIR_NAME).is_dir():
            return candidate
    raise FileNotFoundError(
        "Could not locate project root. Expected a 'configs/' directory "
        f"in {current} or one of its parent folders."
    )


def load_yaml(path: Path | str) -> dict[str, Any]:
    """Load a YAML configuration file."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")
    with path.open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle)
    if data is None:
        return {}
    if not isinstance(data, dict):
        raise ValueError(f"Expected mapping at top level of {path}")
    return data


def load_configs(
    project_root: Path | str | None = None,
    config_names: tuple[str, ...] = DEFAULT_CONFIG_FILES,
) -> dict[str, dict[str, Any]]:
    """Load all standard project config files into a nested dictionary."""
    root = Path(project_root) if project_root else get_project_root()
    configs: dict[str, dict[str, Any]] = {}
    for name in config_names:
        stem = Path(name).stem
        configs[stem] = load_yaml(root / CONFIG_DIR_NAME / name)
    return configs


def resolve_path(project_root: Path | str, relative_path: str | Path) -> Path:
    """Resolve a config path relative to the project root."""
    root = Path(project_root).resolve()
    path = Path(relative_path)
    if path.is_absolute():
        return path
    return (root / path).resolve()


def ensure_dir(path: Path | str) -> Path:
    """Create a directory (and parents) if it does not exist."""
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path
