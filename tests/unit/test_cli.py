from __future__ import annotations

import json
import textwrap
from pathlib import Path

from click.testing import CliRunner

from dicom_nlquery import cli as cli_module
from dicom_nlquery.models import PatientFilter, SearchCriteria, SearchResult, SearchStats


def _write_config(tmp_path: Path) -> Path:
    content = """
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
    """
    path = tmp_path / "config.yaml"
    path.write_text(textwrap.dedent(content).strip() + "\n", encoding="utf-8")
    return path


def test_cli_dry_run_outputs_json(monkeypatch, tmp_path: Path) -> None:
    config_path = _write_config(tmp_path)
    runner = CliRunner()
    criteria = SearchCriteria(
        patient=PatientFilter(sex="F", age_min=20, age_max=40),
        head_keywords=["cranio"],
    )

    def fake_parse(_query, _llm):
        return criteria

    def fail_execute(*_args, **_kwargs):
        raise AssertionError("execute_search should not be called in dry-run")

    monkeypatch.setattr(cli_module, "parse_nl_to_criteria", fake_parse)
    monkeypatch.setattr(cli_module, "execute_search", fail_execute, raising=False)

    result = runner.invoke(
        cli_module.main, ["--config", str(config_path), "dry-run", "query"]
    )

    assert result.exit_code == 0
    payload = json.loads(result.output)
    assert "criteria" in payload
    assert "search_plan" in payload
    assert "PatientName" not in result.output
    assert "PatientID" not in result.output
    assert "PatientBirthDate" not in result.output


def test_cli_json_flag_output_format(monkeypatch, tmp_path: Path) -> None:
    config_path = _write_config(tmp_path)
    runner = CliRunner()
    criteria = SearchCriteria(patient=PatientFilter(sex="F"))

    def fake_parse(_query, _llm):
        return criteria

    monkeypatch.setattr(cli_module, "parse_nl_to_criteria", fake_parse)

    result = runner.invoke(
        cli_module.main,
        ["--config", str(config_path), "--json", "dry-run", "query"],
    )

    assert result.exit_code == 0
    payload = json.loads(result.output)
    assert "criteria" in payload
    assert "search_plan" in payload
    assert "dicom_filters" in payload["search_plan"]
    assert "local_filters" in payload["search_plan"]


def _fake_result(accessions: list[str]) -> SearchResult:
    stats = SearchStats(
        studies_scanned=5,
        studies_matched=len(accessions),
        studies_excluded_no_age=0,
        studies_excluded_no_sex=0,
        limit_reached=False,
        execution_time_seconds=1.0,
        date_range_applied="20200101-20201231",
    )
    return SearchResult(accession_numbers=accessions, stats=stats)


def test_cli_execute_returns_accessions(monkeypatch, tmp_path: Path) -> None:
    config_path = _write_config(tmp_path)
    runner = CliRunner()
    criteria = SearchCriteria(patient=PatientFilter(sex="F"))

    def fake_parse(_query, _llm):
        return criteria

    def fake_execute(*_args, **_kwargs):
        return _fake_result(["ACC001", "ACC002"])

    monkeypatch.setattr(cli_module, "parse_nl_to_criteria", fake_parse)
    monkeypatch.setattr(cli_module, "_create_dicom_client", lambda _cfg: object())
    monkeypatch.setattr(cli_module, "execute_search", fake_execute)

    result = runner.invoke(
        cli_module.main, ["--config", str(config_path), "execute", "query"]
    )

    assert result.exit_code == 0
    assert "ACC001" in result.output
    assert "ACC002" in result.output


def test_cli_execute_no_results_exits_1(monkeypatch, tmp_path: Path) -> None:
    config_path = _write_config(tmp_path)
    runner = CliRunner()
    criteria = SearchCriteria(patient=PatientFilter(sex="F"))

    monkeypatch.setattr(cli_module, "parse_nl_to_criteria", lambda _q, _l: criteria)
    monkeypatch.setattr(cli_module, "_create_dicom_client", lambda _cfg: object())
    monkeypatch.setattr(cli_module, "execute_search", lambda *_a, **_k: _fake_result([]))

    result = runner.invoke(
        cli_module.main, ["--config", str(config_path), "execute", "query"]
    )

    assert result.exit_code == 1


def test_cli_execute_config_error_exits_3(tmp_path: Path) -> None:
    runner = CliRunner()
    missing = tmp_path / "missing.yaml"

    result = runner.invoke(
        cli_module.main, ["--config", str(missing), "execute", "query"]
    )

    assert result.exit_code == 3


def test_cli_execute_dicom_error_exits_2(monkeypatch, tmp_path: Path) -> None:
    config_path = _write_config(tmp_path)
    runner = CliRunner()
    criteria = SearchCriteria(patient=PatientFilter(sex="F"))

    monkeypatch.setattr(cli_module, "parse_nl_to_criteria", lambda _q, _l: criteria)
    monkeypatch.setattr(cli_module, "_create_dicom_client", lambda _cfg: object())

    def fake_execute(*_args, **_kwargs):
        raise RuntimeError("dicom failure")

    monkeypatch.setattr(cli_module, "execute_search", fake_execute)

    result = runner.invoke(
        cli_module.main, ["--config", str(config_path), "execute", "query"]
    )

    assert result.exit_code == 2


def test_cli_execute_json_output(monkeypatch, tmp_path: Path) -> None:
    config_path = _write_config(tmp_path)
    runner = CliRunner()
    criteria = SearchCriteria(patient=PatientFilter(sex="F"))

    monkeypatch.setattr(cli_module, "parse_nl_to_criteria", lambda _q, _l: criteria)
    monkeypatch.setattr(cli_module, "_create_dicom_client", lambda _cfg: object())
    monkeypatch.setattr(cli_module, "execute_search", lambda *_a, **_k: _fake_result(["ACC100"]))

    result = runner.invoke(
        cli_module.main,
        ["--config", str(config_path), "--json", "execute", "query"],
    )

    assert result.exit_code == 0
    payload = json.loads(result.output)
    assert payload["accession_numbers"] == ["ACC100"]
