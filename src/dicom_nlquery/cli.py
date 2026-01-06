from __future__ import annotations

import json
import logging
import re
from pathlib import Path

import click
from pydantic import ValidationError

from .config import load_config
from .dicom_search import apply_guardrails, execute_search
from .logging_config import configure_logging
from .nl_parser import parse_nl_to_criteria


def _load_config(config_path: str, node_override: str | None):
    try:
        config = load_config(config_path)
    except FileNotFoundError as exc:
        raise click.ClickException(
            f"Arquivo de configuracao nao encontrado: {config_path}"
        ) from exc
    except (ValueError, ValidationError) as exc:
        raise click.ClickException(f"Configuracao invalida: {exc}") from exc

    if node_override:
        if node_override not in config.nodes:
            raise click.ClickException(
                f"Node '{node_override}' nao encontrado na configuracao"
            )
        config = config.model_copy(update={"current_node": node_override})

    return config


def _create_dicom_client(config):
    try:
        from dicom_mcp.dicom_client import DicomClient
    except Exception:
        from .dicom_client import DicomClient

    node = config.nodes[config.current_node]
    return DicomClient(
        host=node.host,
        port=node.port,
        calling_aet=config.calling_aet,
        called_aet=node.ae_title,
    )


def _build_search_plan(criteria, guardrails, date_range: str | None):
    dicom_filters: dict[str, object] = {}
    if date_range:
        dicom_filters["StudyDate"] = date_range
    if criteria.patient and criteria.patient.sex:
        dicom_filters["PatientSex"] = criteria.patient.sex
    if criteria.study_narrowing and criteria.study_narrowing.modality_in_study:
        dicom_filters["ModalitiesInStudy"] = criteria.study_narrowing.modality_in_study
    if criteria.study_narrowing and criteria.study_narrowing.study_description_keywords:
        dicom_filters["StudyDescription"] = " ".join(
            criteria.study_narrowing.study_description_keywords
        )

    local_filters: list[str] = []
    if criteria.patient and (
        criteria.patient.age_min is not None or criteria.patient.age_max is not None
    ):
        local_filters.append("age_range")
    if criteria.head_keywords:
        local_filters.append("head_keywords")
    if criteria.required_series:
        local_filters.append("required_series")

    return {
        "dicom_filters": dicom_filters,
        "local_filters": local_filters,
    }


def _configure_logging(verbose: bool) -> logging.Logger:
    level = "DEBUG" if verbose else "WARNING"
    return configure_logging(level)


def _validate_date_range(date_range: str | None) -> str | None:
    if date_range is None:
        return None
    if not re.match(r"^\d{8}-\d{8}$", date_range):
        raise click.ClickException("date-range deve usar YYYYMMDD-YYYYMMDD")
    return date_range


@click.group()
@click.option(
    "--config",
    "-c",
    "config_path",
    default="config.yaml",
    type=click.Path(path_type=Path),
    show_default=True,
    help="Arquivo de configuracao",
)
@click.option("--node", "-n", "node", default=None, help="Override do node DICOM")
@click.option("--verbose", "-v", is_flag=True, help="Ativa logs verbosos")
@click.option("--json", "-j", "json_output", is_flag=True, help="Saida JSON")
@click.pass_context
def main(ctx: click.Context, config_path: Path, node: str | None, verbose: bool, json_output: bool) -> None:
    """CLI for dicom-nlquery."""
    ctx.ensure_object(dict)
    ctx.obj["config_path"] = str(config_path)
    ctx.obj["node"] = node
    ctx.obj["verbose"] = verbose
    ctx.obj["json_output"] = json_output


@main.command("dry-run")
@click.option(
    "--config",
    "-c",
    "config_path_override",
    default=None,
    type=click.Path(path_type=Path),
    help="Arquivo de configuracao",
)
@click.argument("query")
@click.pass_context
def dry_run(ctx: click.Context, config_path_override: Path | None, query: str) -> None:
    """Parse query and show criteria without executing DICOM search."""
    config_path = str(config_path_override) if config_path_override else ctx.obj["config_path"]
    config = _load_config(config_path, ctx.obj["node"])

    criteria = parse_nl_to_criteria(query, config.llm)
    date_range, _ = apply_guardrails(config.guardrails)
    search_plan = _build_search_plan(criteria, config.guardrails, date_range)

    payload = {
        "criteria": criteria.model_dump(),
        "search_plan": search_plan,
    }

    click.echo(json.dumps(payload, indent=2, ensure_ascii=True))


@main.command("execute")
@click.option(
    "--date-range",
    "date_range",
    default=None,
    help="Intervalo de datas no formato YYYYMMDD-YYYYMMDD",
)
@click.option(
    "--max-studies",
    "max_studies",
    default=None,
    type=int,
    help="Numero maximo de estudos para varrer",
)
@click.option(
    "--unlimited",
    is_flag=True,
    help="Remove limites de guardrails (nao recomendado)",
)
@click.argument("query")
@click.pass_context
def execute(
    ctx: click.Context,
    date_range: str | None,
    max_studies: int | None,
    unlimited: bool,
    query: str,
) -> None:
    log = _configure_logging(ctx.obj["verbose"])

    try:
        config = _load_config(ctx.obj["config_path"], ctx.obj["node"])
    except click.ClickException as exc:
        click.echo(f"Erro: {exc.message}", err=True)
        ctx.exit(3)

    try:
        date_range = _validate_date_range(date_range)
    except click.ClickException as exc:
        click.echo(f"Erro: {exc.message}", err=True)
        ctx.exit(3)

    if max_studies is not None and max_studies <= 0:
        click.echo("Erro: max-studies deve ser maior que zero", err=True)
        ctx.exit(3)

    try:
        criteria = parse_nl_to_criteria(query, config.llm)
    except Exception as exc:
        click.echo(f"Erro: {exc}", err=True)
        ctx.exit(3)

    try:
        dicom_client = _create_dicom_client(config)
        result = execute_search(
            criteria,
            dicom_client,
            matching_config=config.matching,
            date_range=date_range,
            max_studies=max_studies,
            unlimited=unlimited,
            guardrails_config=config.guardrails,
            logger=log,
        )
    except Exception as exc:
        click.echo(f"Erro: {exc}", err=True)
        ctx.exit(2)

    if ctx.obj["json_output"]:
        click.echo(json.dumps(result.model_dump(), indent=2, ensure_ascii=True))
    else:
        if result.accession_numbers:
            for accession in result.accession_numbers:
                click.echo(accession)
        else:
            click.echo("Nenhum estudo encontrado com os criterios especificados.")
            click.echo(f"Estudos avaliados: {result.stats.studies_scanned}")
            if result.stats.date_range_applied:
                click.echo(f"Data range: {result.stats.date_range_applied}")

    ctx.exit(0 if result.accession_numbers else 1)
