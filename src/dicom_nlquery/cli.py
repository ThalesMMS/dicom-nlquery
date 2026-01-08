from __future__ import annotations

import json
import logging
import re
import sys
from pathlib import Path

import anyio
import click
from pydantic import ValidationError

from .confirmation import (
    build_confirmation_message,
    build_invalid_response_message,
    classify_confirmation_response,
)
from .config import load_config
from .dicom_search import apply_guardrails, execute_search
from .logging_config import configure_logging
from .llm_client import OllamaClient
from .mcp_client import McpSession, build_stdio_server_params
from .models import ResolvedRequest, SearchCriteria, StudyQuery
from .node_registry import NodeRegistry
from .nl_parser import parse_nl_to_criteria
from .rag_bridge import get_rag_suggestions
from .resolver import resolve_request


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


def _read_stdin_line() -> str:
    stream = click.get_text_stream("stdin")
    line = stream.readline()
    if not line:
        raise click.ClickException("Confirmacao requerida, mas nenhuma entrada foi fornecida.")
    return line.strip()


def _load_node_registry(mcp_config, log: logging.Logger) -> NodeRegistry:
    server_params = build_stdio_server_params(mcp_config)

    async def _fetch_nodes() -> dict[str, object]:
        async with McpSession(server_params, mcp_config) as session:
            payload = await session.list_dicom_nodes()
            if not isinstance(payload, dict):
                raise ValueError("list_dicom_nodes returned invalid payload")
            return payload

    try:
        payload = anyio.run(_fetch_nodes)
    except Exception as exc:  # pragma: no cover - surfaced to CLI
        log.error("Failed to load node registry", extra={"extra_data": {"error": str(exc)}})
        raise click.ClickException("Falha ao carregar list_dicom_nodes do MCP.") from exc
    nodes = payload.get("nodes")
    if not isinstance(nodes, list):
        raise click.ClickException("list_dicom_nodes retornou um formato invalido.")
    return NodeRegistry.from_tool_payload(nodes)


def _resolve_with_confirmation(
    query: str, config, log: logging.Logger
) -> tuple[ResolvedRequest | None, str]:
    resolver_cfg = config.resolver
    if resolver_cfg is None or not resolver_cfg.enabled:
        return None, query
    if config.mcp is None:
        raise click.ClickException("Erro: mcp.config_path nao configurado.")
    if resolver_cfg.require_confirmation and not sys.stdin.isatty():
        raise click.ClickException(
            "Confirmacao requerida, mas stdin nao e um TTY."
        )
    registry = _load_node_registry(config.mcp, log)
    llm = OllamaClient.from_config(config.llm)
    invalid_attempts = 0
    rejection_attempts = 0
    current_query = query

    while True:
        result = resolve_request(current_query, registry, llm)
        if result.unresolved:
            rejection_attempts += 1
            if rejection_attempts >= resolver_cfg.confirmation.max_rejections:
                raise click.ClickException(resolver_cfg.confirmation.cancel_message)
            click.echo(
                "Resolver: necessario esclarecer antes de executar. "
                + ", ".join(result.unresolved)
            )
            if not sys.stdin.isatty():
                raise click.ClickException(
                    "Clarificacao requerida, mas stdin nao e um TTY."
                )
            click.echo(resolver_cfg.confirmation.correction_prompt)
            current_query = _read_stdin_line()
            continue

        if not resolver_cfg.require_confirmation:
            return result.request, current_query

        click.echo(build_confirmation_message(result.request, resolver_cfg.confirmation))
        response = _read_stdin_line()
        decision = classify_confirmation_response(response, resolver_cfg.confirmation)
        log.debug(
            "Confirmation decision",
            extra={"extra_data": {"decision": decision}},
        )
        if decision == "accept":
            return result.request, current_query
        if decision == "reject":
            rejection_attempts += 1
            if rejection_attempts >= resolver_cfg.confirmation.max_rejections:
                raise click.ClickException(resolver_cfg.confirmation.cancel_message)
            click.echo(resolver_cfg.confirmation.correction_prompt)
            current_query = _read_stdin_line()
            continue
        invalid_attempts += 1
        if invalid_attempts >= resolver_cfg.confirmation.max_invalid_responses:
            raise click.ClickException(resolver_cfg.confirmation.cancel_message)
        click.echo(build_invalid_response_message(resolver_cfg.confirmation))


def _execute_move(
    request: ResolvedRequest,
    config,
    date_range: str | None,
    max_studies: int | None,
    unlimited: bool,
    log: logging.Logger,
) -> dict[str, object]:
    if config.mcp is None:
        raise click.ClickException("Erro: mcp.config_path nao configurado.")
    if not request.destination_node:
        raise click.ClickException("Destino nao informado para move.")

    effective_date_range, effective_max_studies = apply_guardrails(
        config.guardrails,
        date_range=date_range,
        max_studies=max_studies,
        unlimited=unlimited,
        logger=log,
    )

    filters = dict(request.filters)
    if effective_date_range and "study_date" not in filters:
        filters["study_date"] = effective_date_range

    server_params = build_stdio_server_params(config.mcp)

    async def _run_move() -> dict[str, object]:
        async with McpSession(server_params, config.mcp) as session:
            studies = await session.query_studies(
                **filters,
                limit=effective_max_studies,
            )
            if not isinstance(studies, list):
                raise click.ClickException("query_studies retornou formato invalido.")
            if not studies:
                return {"success": False, "message": "Nenhum estudo encontrado."}
            if len(studies) > 1:
                raise click.ClickException(
                    "Multiplos estudos encontrados. Refine a busca para mover apenas um estudo."
                )
            study = studies[0]
            uid = study.get("StudyInstanceUID") or study.get("study_instance_uid")
            if not uid:
                raise click.ClickException("StudyInstanceUID nao encontrado no resultado.")
            await session.query_series(study_instance_uid=uid)
            result = await session.move_study(
                destination_node=request.destination_node,
                study_instance_uid=uid,
            )
            if not isinstance(result, dict):
                return {"success": False, "message": "Resposta invalida do move_study."}
            return result

    return anyio.run(_run_move)


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
        resolved_request, query = _resolve_with_confirmation(query, config, log)
        if resolved_request is not None:
            if resolved_request.destination_node:
                move_result = _execute_move(
                    resolved_request,
                    config,
                    date_range,
                    max_studies,
                    unlimited,
                    log,
                )
                if ctx.obj["json_output"]:
                    click.echo(json.dumps(move_result, indent=2, ensure_ascii=True))
                else:
                    if move_result.get("success"):
                        completed = move_result.get("completed")
                        failed = move_result.get("failed")
                        warning = move_result.get("warning")
                        click.echo(
                            "C-MOVE concluido. "
                            f"Completed={completed} Failed={failed} Warning={warning}"
                        )
                    else:
                        click.echo(
                            f"Falha no C-MOVE: {move_result.get('message', 'Erro desconhecido')}",
                            err=True,
                        )
                ctx.exit(0 if move_result.get("success") else 2)
            criteria = SearchCriteria(study=StudyQuery(**resolved_request.filters))
        else:
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
        elif result.study_instance_uids:
            for study_uid in result.study_instance_uids:
                click.echo(study_uid)
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
