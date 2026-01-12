from __future__ import annotations

import json
import textwrap
from pathlib import Path

import click
import pytest
from click.testing import CliRunner

from dicom_nlquery import cli as cli_module
from dicom_nlquery.models import (
    LLMConfig,
    McpServerConfig,
    NLQueryConfig,
    ResolvedRequest,
    ResolverResult,
    ResolverConfig,
    SearchCriteria,
    SearchResult,
    SearchStats,
    StudyQuery,
)


def _write_config(tmp_path: Path) -> Path:
    content = """
    llm_path: "llm.yaml"

    mcp:
      command: "dicom-mcp"
      config_path: "dicom-mcp.yaml"
    """
    path = tmp_path / "config.yaml"
    path.write_text(textwrap.dedent(content).strip() + "\n", encoding="utf-8")
    (tmp_path / "llm.yaml").write_text(
        textwrap.dedent(
            """
            provider: "ollama"
            base_url: "http://127.0.0.1:11434"
            model: "llama3.2:latest"
            """
        ).strip()
        + "\n",
        encoding="utf-8",
    )
    (tmp_path / "dicom-mcp.yaml").write_text("nodes: {}", encoding="utf-8")
    return path


def _write_config_with_resolver(tmp_path: Path) -> Path:
    content = """
    llm_path: "llm.yaml"

    mcp:
      command: "dicom-mcp"
      config_path: "dicom-mcp.yaml"

    resolver:
      enabled: true
      require_confirmation: false
    """
    path = tmp_path / "config.yaml"
    path.write_text(textwrap.dedent(content).strip() + "\n", encoding="utf-8")
    (tmp_path / "llm.yaml").write_text(
        textwrap.dedent(
            """
            provider: "ollama"
            base_url: "http://127.0.0.1:11434"
            model: "llama3.2:latest"
            """
        ).strip()
        + "\n",
        encoding="utf-8",
    )
    (tmp_path / "dicom-mcp.yaml").write_text("nodes: {}", encoding="utf-8")
    return path


def test_cli_dry_run_outputs_json(monkeypatch, tmp_path: Path) -> None:
    config_path = _write_config(tmp_path)
    runner = CliRunner()
    criteria = SearchCriteria(
        study=StudyQuery(patient_sex="F", study_description="cranial"),
    )

    def fake_parse(_query, _llm, **_kwargs):
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
    criteria = SearchCriteria(study=StudyQuery(patient_sex="F"))

    def fake_parse(_query, _llm, **_kwargs):
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
    assert "dicom_mcp" in payload["search_plan"]


def test_cli_execute_ignores_node_tokens_warning(monkeypatch, tmp_path: Path) -> None:
    config_path = _write_config_with_resolver(tmp_path)
    runner = CliRunner()
    criteria = SearchCriteria(study=StudyQuery(study_description="cranial"))

    def fake_execute(*_args, **_kwargs):
        return _fake_result(["ACC001"])

    def fake_resolve_request(_query, _registry, _llm):
        request = ResolvedRequest(
            source_node="ORTHANC",
            filters={"study_description": "cranial"},
        )
        return ResolverResult(
            request=request,
            needs_confirmation=True,
            unresolved=["node_tokens_removed_from_filters"],
        )

    monkeypatch.setattr(cli_module, "resolve_request", fake_resolve_request)
    monkeypatch.setattr(cli_module, "_load_node_registry", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(cli_module, "execute_search", fake_execute)
    monkeypatch.setattr(cli_module, "parse_nl_to_criteria", lambda *_a, **_k: criteria)

    result = runner.invoke(
        cli_module.main,
        ["--config", str(config_path), "execute", "query"],
    )

    assert result.exit_code == 0
    assert "ACC001" in result.output


def test_resolver_rejects_without_retry(monkeypatch: pytest.MonkeyPatch) -> None:
    def fake_read() -> str:
        return "no"

    def fake_resolve_request(_query, _registry, _llm):
        request = ResolvedRequest(
            source_node="ORTHANC",
            destination_node="RADIANT",
            filters={"study_description": "angiogram"},
        )
        return ResolverResult(request=request, needs_confirmation=True, unresolved=[])

    config = NLQueryConfig(
        llm=LLMConfig(
            provider="ollama",
            base_url="http://127.0.0.1:11434",
            model="llama3.2:latest",
        ),
        mcp=McpServerConfig(config_path="dicom-mcp.yaml"),
        resolver=ResolverConfig(enabled=True, require_confirmation=True),
    )
    criteria = SearchCriteria(study=StudyQuery(study_description="angiogram"))

    monkeypatch.setattr(cli_module, "_load_node_registry", lambda *_a, **_k: None)
    monkeypatch.setattr(cli_module, "create_llm_client", lambda *_a, **_k: object())
    monkeypatch.setattr(cli_module, "resolve_request", fake_resolve_request)
    monkeypatch.setattr(cli_module, "parse_nl_to_criteria", lambda *_a, **_k: criteria)
    monkeypatch.setattr(cli_module.sys.stdin, "isatty", lambda: True)
    monkeypatch.setattr(cli_module, "_read_stdin_line", fake_read)

    with pytest.raises(click.ClickException) as excinfo:
        cli_module._resolve_with_confirmation(
            "query",
            config,
            log=cli_module.logging.getLogger(__name__),
            llm_debug=False,
        )

    assert excinfo.value.message == config.resolver.confirmation.cancel_message


def _fake_result(accessions: list[str]) -> SearchResult:
    stats = SearchStats(
        studies_scanned=5,
        studies_matched=len(accessions),
        studies_filtered_series=0,
        limit_reached=False,
        execution_time_seconds=1.0,
        date_range_applied="20200101-20201231",
        attempts_run=1,
        successful_stage="direct",
        rewrites_tried=[],
    )
    return SearchResult(accession_numbers=accessions, stats=stats)


def _fake_move_result(study_uids: list[str]) -> SearchResult:
    stats = SearchStats(
        studies_scanned=len(study_uids),
        studies_matched=len(study_uids),
        studies_filtered_series=0,
        limit_reached=False,
        execution_time_seconds=1.0,
        date_range_applied="20000101-20221231",
        attempts_run=1,
        successful_stage="direct",
        rewrites_tried=[],
    )
    return SearchResult(accession_numbers=[], study_instance_uids=study_uids, stats=stats)


def test_cli_execute_returns_accessions(monkeypatch, tmp_path: Path) -> None:
    config_path = _write_config(tmp_path)
    runner = CliRunner()
    criteria = SearchCriteria(study=StudyQuery(patient_sex="F"))

    def fake_parse(_query, _llm, **_kwargs):
        return criteria

    def fake_execute(*_args, **_kwargs):
        return _fake_result(["ACC001", "ACC002"])

    monkeypatch.setattr(cli_module, "parse_nl_to_criteria", fake_parse)
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
    criteria = SearchCriteria(study=StudyQuery(patient_sex="F"))

    monkeypatch.setattr(cli_module, "parse_nl_to_criteria", lambda _q, _l, **_k: criteria)
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
    criteria = SearchCriteria(study=StudyQuery(patient_sex="F"))

    monkeypatch.setattr(cli_module, "parse_nl_to_criteria", lambda _q, _l, **_k: criteria)

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
    criteria = SearchCriteria(study=StudyQuery(patient_sex="F"))

    monkeypatch.setattr(cli_module, "parse_nl_to_criteria", lambda _q, _l, **_k: criteria)
    monkeypatch.setattr(cli_module, "execute_search", lambda *_a, **_k: _fake_result(["ACC100"]))

    result = runner.invoke(
        cli_module.main,
        ["--config", str(config_path), "--json", "execute", "query"],
    )

    assert result.exit_code == 0
    payload = json.loads(result.output)
    assert payload["accession_numbers"] == ["ACC100"]


def test_execute_move_allows_multiple_studies(monkeypatch) -> None:
    moves: list[tuple[str, str]] = []

    class FakeSession:
        def __init__(self, *_args, **_kwargs) -> None:
            pass

        async def __aenter__(self):
            return self

        async def __aexit__(self, exc_type, exc, tb) -> None:
            return None

        async def query_series(self, **_kwargs):
            return []

        async def move_study(self, destination_node: str, study_instance_uid: str):
            moves.append((destination_node, study_instance_uid))
            return {
                "success": True,
                "message": "ok",
                "completed": 1,
                "failed": 0,
                "warning": 0,
            }

    config = NLQueryConfig(
        llm=LLMConfig(
            provider="ollama",
            base_url="http://127.0.0.1:11434",
            model="llama3.2:latest",
        ),
        mcp=McpServerConfig(config_path="dicom-mcp.yaml"),
        resolver=ResolverConfig(enabled=True, require_confirmation=False),
    )
    request = ResolvedRequest(
        source_node="ORTHANC",
        destination_node="RADIANT",
        filters={"study_description": "angiogram"},
    )
    criteria = SearchCriteria(study=StudyQuery(study_description="angiogram"))

    monkeypatch.setattr(cli_module, "execute_search", lambda *_a, **_k: _fake_move_result(["1.2.3", "1.2.4"]))
    monkeypatch.setattr(cli_module, "McpSession", FakeSession)

    result = cli_module._execute_move(
        request,
        criteria,
        config,
        date_range=None,
        max_studies=None,
        unlimited=False,
        log=cli_module.logging.getLogger(__name__),
    )

    assert result["total"] == 2
    assert result["success_count"] == 2
    assert moves == [("RADIANT", "1.2.3"), ("RADIANT", "1.2.4")]


def test_execute_move_requires_confirmation_for_single_study(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    moves: list[tuple[str, str]] = []
    confirmation_reads = {"count": 0}

    class FakeSession:
        def __init__(self, *_args, **_kwargs) -> None:
            pass

        async def __aenter__(self):
            return self

        async def __aexit__(self, exc_type, exc, tb) -> None:
            return None

        async def query_series(self, **_kwargs):
            return []

        async def move_study(self, destination_node: str, study_instance_uid: str):
            moves.append((destination_node, study_instance_uid))
            return {
                "success": True,
                "message": "ok",
                "completed": 1,
                "failed": 0,
                "warning": 0,
            }

    def fake_read() -> str:
        confirmation_reads["count"] += 1
        return "yes"

    config = NLQueryConfig(
        llm=LLMConfig(
            provider="ollama",
            base_url="http://127.0.0.1:11434",
            model="llama3.2:latest",
        ),
        mcp=McpServerConfig(config_path="dicom-mcp.yaml"),
        resolver=ResolverConfig(enabled=True, require_confirmation=True),
    )
    request = ResolvedRequest(
        source_node="ORTHANC",
        destination_node="RADIANT",
        filters={"study_description": "angiogram"},
    )
    criteria = SearchCriteria(study=StudyQuery(study_description="angiogram"))

    monkeypatch.setattr(cli_module, "execute_search", lambda *_a, **_k: _fake_move_result(["1.2.3"]))
    monkeypatch.setattr(cli_module, "McpSession", FakeSession)
    monkeypatch.setattr(cli_module.sys.stdin, "isatty", lambda: True)
    monkeypatch.setattr(cli_module, "_read_stdin_line", fake_read)

    result = cli_module._execute_move(
        request,
        criteria,
        config,
        date_range=None,
        max_studies=None,
        unlimited=False,
        log=cli_module.logging.getLogger(__name__),
    )

    assert confirmation_reads["count"] == 1
    assert result["total"] == 1
    assert result["success_count"] == 1
    assert moves == [("RADIANT", "1.2.3")]
