from __future__ import annotations

from pathlib import Path

import yaml

from .models import AppConfig


def load_config(path: str | Path) -> AppConfig:
    config_path = Path(path)
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")
    with config_path.open("r", encoding="utf-8") as handle:
        raw = yaml.safe_load(handle) or {}
    return AppConfig.model_validate(raw)


def sorted_sources(config: AppConfig):
    return sorted(config.sources, key=lambda source: (source.priority, source.id))
