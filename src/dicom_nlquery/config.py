from __future__ import annotations

import logging
from pathlib import Path

import yaml

from .models import McpServerConfig, NLQueryConfig


def load_config(path: str | Path) -> NLQueryConfig:
    log = logging.getLogger(__name__)
    config_path = Path(path)
    raw = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    data = raw or {}
    config = NLQueryConfig.model_validate(data)
    config = _resolve_mcp_config(config, config_path)
    log.debug("Config loaded", extra={"extra_data": {"path": str(config_path)}})
    return config


def _resolve_mcp_config(config: NLQueryConfig, config_path: Path) -> NLQueryConfig:
    mcp = config.mcp
    if mcp is None:
        default_path = _find_default_mcp_config_path(config_path)
        if default_path is None:
            return config
        mcp = McpServerConfig(config_path=str(default_path))
    else:
        mcp = _resolve_mcp_paths(mcp, config_path.parent)

    return config.model_copy(update={"mcp": mcp})


def _resolve_mcp_paths(mcp: McpServerConfig, base_dir: Path) -> McpServerConfig:
    updates: dict[str, str] = {}
    if mcp.config_path:
        updates["config_path"] = str(_resolve_path(base_dir, mcp.config_path))
    if mcp.cwd:
        updates["cwd"] = str(_resolve_path(base_dir, mcp.cwd))
    if not updates:
        return mcp
    return mcp.model_copy(update=updates)


def _resolve_path(base_dir: Path, value: str) -> Path:
    candidate = Path(value)
    if candidate.is_absolute():
        return candidate
    return (base_dir / candidate).resolve()


def _find_default_mcp_config_path(config_path: Path) -> Path | None:
    candidates = [
        config_path.parent / "dicom-mcp" / "configuration.yaml",
        config_path.parent.parent / "dicom-mcp" / "configuration.yaml",
        Path.cwd() / "dicom-mcp" / "configuration.yaml",
        Path.cwd().parent / "dicom-mcp" / "configuration.yaml",
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate.resolve()
    return None
