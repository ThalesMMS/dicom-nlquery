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
    if not isinstance(data, dict):
        raise ValueError("Configuration must be a mapping")
    data = _merge_llm_config(data, config_path)
    config = NLQueryConfig.model_validate(data)
    config = _resolve_mcp_config(config, config_path)
    config = _resolve_lexicon_config(config, config_path)
    config = _resolve_rag_config(config, config_path)
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


def _merge_llm_config(data: dict[str, object], config_path: Path) -> dict[str, object]:
    llm_path = data.get("llm_path")
    if not llm_path:
        return data
    if not isinstance(llm_path, str):
        raise ValueError("llm_path must be a string")
    resolved = _resolve_path(config_path.parent, llm_path)
    if not resolved.exists():
        raise FileNotFoundError(f"LLM configuration file {resolved} not found")
    llm_raw = yaml.safe_load(resolved.read_text(encoding="utf-8")) or {}
    if not isinstance(llm_raw, dict):
        raise ValueError("LLM configuration file must contain a mapping")
    llm_data = llm_raw.get("llm") if isinstance(llm_raw.get("llm"), dict) else llm_raw
    overrides = data.get("llm") or {}
    if not isinstance(overrides, dict):
        raise ValueError("llm overrides must be a mapping")
    merged = dict(llm_data)
    merged.update(overrides)
    updated = dict(data)
    updated["llm"] = merged
    updated.pop("llm_path", None)
    return updated


def _resolve_lexicon_config(config: NLQueryConfig, config_path: Path) -> NLQueryConfig:
    lexicon = config.lexicon
    if lexicon is None or not lexicon.path:
        return config
    resolved = str(_resolve_path(config_path.parent, lexicon.path))
    if resolved == lexicon.path:
        return config
    return config.model_copy(update={"lexicon": lexicon.model_copy(update={"path": resolved})})


def _resolve_rag_config(config: NLQueryConfig, config_path: Path) -> NLQueryConfig:
    rag = config.rag
    if rag is None or not rag.index_path:
        return config
    resolved = str(_resolve_path(config_path.parent, rag.index_path))
    if resolved == rag.index_path:
        return config
    return config.model_copy(update={"rag": rag.model_copy(update={"index_path": resolved})})


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
        config_path.parent / "configs" / "dicom.yaml",
        config_path.parent.parent / "configs" / "dicom.yaml",
        Path.cwd() / "configs" / "dicom.yaml",
        Path.cwd().parent / "configs" / "dicom.yaml",
        config_path.parent / "dicom-mcp" / "configuration.yaml",
        config_path.parent.parent / "dicom-mcp" / "configuration.yaml",
        Path.cwd() / "dicom-mcp" / "configuration.yaml",
        Path.cwd().parent / "dicom-mcp" / "configuration.yaml",
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate.resolve()
    return None
