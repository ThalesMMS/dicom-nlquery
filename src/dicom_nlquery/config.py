from __future__ import annotations

import logging
from pathlib import Path

import yaml

from .models import NLQueryConfig


def load_config(path: str | Path) -> NLQueryConfig:
    log = logging.getLogger(__name__)
    config_path = Path(path)
    raw = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    data = raw or {}
    config = NLQueryConfig.model_validate(data)
    log.debug("Config loaded", extra={"extra_data": {"path": str(config_path)}})
    return config
