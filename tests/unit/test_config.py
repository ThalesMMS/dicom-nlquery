from __future__ import annotations

from pathlib import Path
import textwrap

import pytest
from pydantic import ValidationError

from dicom_nlquery.config import load_config


def _write_config(tmp_path: Path, content: str) -> Path:
    path = tmp_path / "config.yaml"
    path.write_text(textwrap.dedent(content).strip() + "\n", encoding="utf-8")
    return path


def test_load_valid_config(tmp_path: Path) -> None:
    config_path = _write_config(
        tmp_path,
        """
        llm:
          provider: "ollama"
          base_url: "http://127.0.0.1:11434"
          model: "llama3.2:latest"
          temperature: 0
          timeout: 60

        mcp:
          command: "dicom-mcp"
          config_path: "dicom-mcp.yaml"
        """,
    )

    (tmp_path / "dicom-mcp.yaml").write_text("nodes: {}", encoding="utf-8")
    config = load_config(config_path)

    assert config.llm.model == "llama3.2:latest"
    assert config.mcp is not None
    assert config.mcp.command == "dicom-mcp"
    assert config.mcp.config_path == str(tmp_path / "dicom-mcp.yaml")


def test_load_mcp_retry_settings(tmp_path: Path) -> None:
    config_path = _write_config(
        tmp_path,
        """
        llm:
          provider: "ollama"
          base_url: "http://127.0.0.1:11434"
          model: "llama3.2:latest"

        mcp:
          command: "dicom-mcp"
          config_path: "dicom-mcp.yaml"
          tool_timeout_seconds: 15
          retry:
            max_attempts: 2
            backoff_seconds: [0.2, 0.4]
          non_idempotent_tools: ["move_study", "move_series"]
        """,
    )

    (tmp_path / "dicom-mcp.yaml").write_text("nodes: {}", encoding="utf-8")
    config = load_config(config_path)

    assert config.mcp is not None
    assert config.mcp.tool_timeout_seconds == 15
    assert config.mcp.retry.max_attempts == 2
    assert config.mcp.retry.backoff_seconds == [0.2, 0.4]
    assert config.mcp.non_idempotent_tools == ["move_study", "move_series"]


def test_load_search_pipeline_limits(tmp_path: Path) -> None:
    config_path = _write_config(
        tmp_path,
        """
        llm:
          provider: "ollama"
          base_url: "http://127.0.0.1:11434"
          model: "llama3.2:latest"

        search_pipeline:
          server_limit_studies: 123
          server_limit_series: 456
        """,
    )

    config = load_config(config_path)

    assert config.search_pipeline.server_limit_studies == 123
    assert config.search_pipeline.server_limit_series == 456


def test_load_invalid_config_missing_llm(tmp_path: Path) -> None:
    config_path = _write_config(
        tmp_path,
        """
        guardrails:
          study_date_range_default_days: 180
        """,
    )

    with pytest.raises(ValidationError):
        load_config(config_path)


def test_config_applies_guardrails_defaults(tmp_path: Path) -> None:
    config_path = _write_config(
        tmp_path,
        """
        llm:
          provider: "ollama"
          base_url: "http://127.0.0.1:11434"
          model: "llama3.2:latest"

        mcp:
          command: "dicom-mcp"
          config_path: "dicom-mcp.yaml"
        """,
    )

    (tmp_path / "dicom-mcp.yaml").write_text("nodes: {}", encoding="utf-8")
    config = load_config(config_path)

    assert config.guardrails.study_date_range_default_days == 180
    assert config.guardrails.max_studies_scanned_default == 700
    assert config.guardrails.search_timeout_seconds == 120


def test_config_validates_llm_provider(tmp_path: Path) -> None:
    config_path = _write_config(
        tmp_path,
        """
        llm:
          provider: "invalid"
          base_url: "http://127.0.0.1:11434"
          model: "llama3.2:latest"
        """,
    )

    with pytest.raises(ValidationError):
        load_config(config_path)


def test_config_resolves_mcp_relative_paths(tmp_path: Path) -> None:
    config_path = _write_config(
        tmp_path,
        """
        llm:
          provider: "ollama"
          base_url: "http://127.0.0.1:11434"
          model: "llama3.2:latest"

        mcp:
          command: "dicom-mcp"
          config_path: "configs/mcp.yaml"
          cwd: "configs"
        """,
    )

    (tmp_path / "configs").mkdir()
    (tmp_path / "configs" / "mcp.yaml").write_text("nodes: {}", encoding="utf-8")

    config = load_config(config_path)

    assert config.mcp is not None
    assert config.mcp.config_path == str(tmp_path / "configs" / "mcp.yaml")
    assert config.mcp.cwd == str(tmp_path / "configs")
