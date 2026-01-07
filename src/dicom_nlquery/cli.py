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
from .rag_bridge import get_rag_suggestions


def _load_config(config_path: str):
    try:
        config = load_config(config_path)
    except FileNotFoundError as exc:
        raise click.ClickException(
            f"Arquivo de configuracao nao encontrado: {config_path}"
        ) from exc
    except (ValueError, ValidationError) as exc:
        raise click.ClickException(f"Configuracao invalida: {exc}") from exc

    return config


def _build_search_plan(criteria, date_range: str | None):
    study_filters = criteria.study.model_dump(exclude_none=True)
    if date_range and "study_date" not in study_filters:
        study_filters["study_date"] = date_range

    plan = {"query_studies": study_filters}
    if criteria.series is not None:
        series_filters = criteria.series.model_dump(exclude_none=True)
        if series_filters:
            plan["query_series"] = series_filters

    return {"dicom_mcp": plan}


def _configure_logging(verbose: bool, llm_debug: bool = False) -> logging.Logger:
    level = "DEBUG" if verbose else "WARNING"
    return configure_logging(level, show_extra=llm_debug)


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
@click.option("--node", "-n", "node", default=None, help="Node do dicom-mcp para consulta")
@click.option("--verbose", "-v", is_flag=True, help="Ativa logs verbosos")
@click.option(
    "--llm-debug",
    is_flag=True,
    help="Mostra o JSON extraido da LLM e os criterios finais (pode conter PHI).",
)
@click.option("--json", "-j", "json_output", is_flag=True, help="Saida JSON")
@click.pass_context
def main(
    ctx: click.Context,
    config_path: Path,
    node: str | None,
    verbose: bool,
    llm_debug: bool,
    json_output: bool,
) -> None:
    """CLI for dicom-nlquery."""
    ctx.ensure_object(dict)
    ctx.obj["config_path"] = str(config_path)
    ctx.obj["node"] = node
    ctx.obj["verbose"] = verbose
    ctx.obj["llm_debug"] = llm_debug
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
    if ctx.obj["llm_debug"] or ctx.obj["verbose"]:
        _configure_logging(ctx.obj["verbose"], ctx.obj["llm_debug"])
    config_path = str(config_path_override) if config_path_override else ctx.obj["config_path"]
    config = _load_config(config_path)

    criteria = parse_nl_to_criteria(
        query,
        config.llm,
        strict_evidence=True,
        debug=ctx.obj["llm_debug"],
    )
    date_range, _ = apply_guardrails(config.guardrails)
    search_plan = _build_search_plan(criteria, date_range)

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
    log = _configure_logging(ctx.obj["verbose"], ctx.obj["llm_debug"])

    try:
        config = _load_config(ctx.obj["config_path"])
    except click.ClickException as exc:
        click.echo(f"Erro: {exc.message}", err=True)
        ctx.exit(3)

    if config.mcp is None:
        click.echo(
            "Erro: mcp.config_path nao configurado. Ajuste o config.yaml para o dicom-mcp.",
            err=True,
        )
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
        criteria = parse_nl_to_criteria(
            query,
            config.llm,
            strict_evidence=True,
            debug=ctx.obj["llm_debug"],
        )
    except Exception as exc:
        click.echo(f"Erro: {exc}", err=True)
        ctx.exit(3)

    try:
        result = execute_search(
            criteria,
            mcp_config=config.mcp,
            date_range=date_range,
            max_studies=max_studies,
            unlimited=unlimited,
            guardrails_config=config.guardrails,
            search_pipeline_config=config.search_pipeline,
            lexicon_config=config.lexicon,
            rag_config=config.rag,
            ranking_config=config.ranking,
            rag_query=query,
            logger=log,
            node_name=ctx.obj["node"],
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
            if result.stats.stages_tried:
                click.echo(
                    "Estagios tentados: " + ", ".join(result.stats.stages_tried)
                )
            if result.stats.rewrites_tried:
                shown = ", ".join(result.stats.rewrites_tried[:3])
                click.echo(f"Reescritas tentadas: {shown}")
            if result.stats.limit_reached:
                click.echo(
                    "Limite de busca atingido. Ajuste --unlimited ou max-studies."
                )
            if config.rag.enable:
                suggestions = get_rag_suggestions(query, config.rag, log)
                if suggestions:
                    shown = ", ".join(suggestions[:3])
                    click.echo(f"Sugestoes RAG: {shown}")

    ctx.exit(0 if result.accession_numbers else 1)
