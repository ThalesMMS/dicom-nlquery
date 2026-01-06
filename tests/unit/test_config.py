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
        nodes:
          orthanc:
            host: "localhost"
            port: 4242
            ae_title: "ORTHANC"
            description: "Orthanc local"

        current_node: "orthanc"
        calling_aet: "TESTSCU"

        llm:
          provider: "ollama"
          base_url: "http://127.0.0.1:11434"
          model: "llama3.2:latest"
          temperature: 0
          timeout: 60
        """,
    )

    config = load_config(config_path)

    assert config.current_node == "orthanc"
    assert config.nodes["orthanc"].host == "localhost"
    assert config.llm.model == "llama3.2:latest"


def test_load_invalid_config_missing_nodes(tmp_path: Path) -> None:
    config_path = _write_config(
        tmp_path,
        """
        current_node: "orthanc"
        calling_aet: "TESTSCU"

        llm:
          provider: "ollama"
          base_url: "http://127.0.0.1:11434"
          model: "llama3.2:latest"
        """,
    )

    with pytest.raises(ValidationError):
        load_config(config_path)


def test_config_applies_guardrails_defaults(tmp_path: Path) -> None:
    config_path = _write_config(
        tmp_path,
        """
        nodes:
          orthanc:
            host: "localhost"
            port: 4242
            ae_title: "ORTHANC"

        current_node: "orthanc"
        calling_aet: "TESTSCU"

        llm:
          provider: "ollama"
          base_url: "http://127.0.0.1:11434"
          model: "llama3.2:latest"
        """,
    )

    config = load_config(config_path)

    assert config.guardrails.study_date_range_default_days == 180
    assert config.guardrails.max_studies_scanned_default == 2000


def test_config_validates_llm_provider(tmp_path: Path) -> None:
    config_path = _write_config(
        tmp_path,
        """
        nodes:
          orthanc:
            host: "localhost"
            port: 4242
            ae_title: "ORTHANC"

        current_node: "orthanc"
        calling_aet: "TESTSCU"

        llm:
          provider: "invalid"
          base_url: "http://127.0.0.1:11434"
          model: "llama3.2:latest"
        """,
    )

    with pytest.raises(ValidationError):
        load_config(config_path)


def test_config_accepts_custom_synonyms(tmp_path: Path) -> None:
    config_path = _write_config(
        tmp_path,
        """
        nodes:
          orthanc:
            host: "localhost"
            port: 4242
            ae_title: "ORTHANC"

        current_node: "orthanc"
        calling_aet: "TESTSCU"

        llm:
          provider: "ollama"
          base_url: "http://127.0.0.1:11434"
          model: "llama3.2:latest"

        matching:
          head_keywords: ["cranio", "cabeca", "head", "brain"]
          synonyms:
            axial: ["axial", "ax"]
            pos: ["pos", "post"]
        """,
    )

    config = load_config(config_path)

    assert "axial" in config.matching.synonyms
    assert config.matching.synonyms["axial"] == ["axial", "ax"]
