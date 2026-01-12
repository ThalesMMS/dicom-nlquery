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
from .llm_client import create_llm_client
from .mcp_client import McpSession, build_stdio_server_params
from .models import ResolvedRequest, SearchCriteria, StudyQuery
from .node_registry import NodeRegistry
from .nl_parser import parse_nl_to_criteria
from .rag_bridge import get_rag_suggestions
from .resolver import resolve_request, strip_node_tokens_from_filters


def _load_config(config_path: str):
    try:
        config = load_config(config_path)
    except FileNotFoundError as exc:
        raise click.ClickException(
            f"Configuration file not found: {config_path}"
        ) from exc
    except (ValueError, ValidationError) as exc:
        raise click.ClickException(f"Invalid configuration: {exc}") from exc

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
        raise click.ClickException("date-range must use YYYYMMDD-YYYYMMDD")
    return date_range


def _read_stdin_line() -> str:
    stream = click.get_text_stream("stdin")
    line = stream.readline()
    if not line:
        raise click.ClickException("Confirmation required, but no input was provided.")
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
        raise click.ClickException("Failed to load list_dicom_nodes from MCP.") from exc
    nodes = payload.get("nodes")
    if not isinstance(nodes, list):
        raise click.ClickException("list_dicom_nodes returned an invalid format.")
    return NodeRegistry.from_tool_payload(nodes)


def _resolve_with_confirmation(
    query: str, config, log: logging.Logger, llm_debug: bool = False
) -> tuple[ResolvedRequest | None, str, SearchCriteria | None]:
    resolver_cfg = config.resolver
    if resolver_cfg is None or not resolver_cfg.enabled:
        return None, query, None
    if config.mcp is None:
        raise click.ClickException("Error: mcp.config_path not configured.")
    if resolver_cfg.require_confirmation and not sys.stdin.isatty():
        raise click.ClickException(
            "Confirmation required, but stdin is not a TTY."
        )
    registry = _load_node_registry(config.mcp, log)
    llm = create_llm_client(config.llm)
    invalid_attempts = 0
    rejection_attempts = 0
    current_query = query

    while True:
        result = resolve_request(current_query, registry, llm)
        unresolved = list(result.unresolved)
        if unresolved == ["node_tokens_removed_from_filters"] and (
            result.request.source_node or result.request.destination_node
        ):
            unresolved = []
        if unresolved:
            rejection_attempts += 1
            if rejection_attempts >= resolver_cfg.confirmation.max_rejections:
                raise click.ClickException(resolver_cfg.confirmation.cancel_message)
            click.echo(
                "Resolver: clarification required before executing. "
                + ", ".join(unresolved)
            )
            if not sys.stdin.isatty():
                raise click.ClickException(
                    "Clarification required, but stdin is not a TTY."
                )
            click.echo(resolver_cfg.confirmation.correction_prompt)
            current_query = _read_stdin_line()
            continue

        # Strip node tokens from the NL query before parsing clinical filters.
        # This prevents models from treating node names (ORTHANC/RADIANT/etc.) as study_description.
        cleaned_query = current_query
        if registry is not None:
            matches = registry.match(current_query)
            if matches:
                cleaned_payload, _removed = strip_node_tokens_from_filters(
                    {"q": current_query},
                    registry,
                    matches,
                )
                cleaned_query = cleaned_payload.get("q") or current_query

        criteria = parse_nl_to_criteria(
            cleaned_query,
            config.llm,
            strict_evidence=True,
            debug=llm_debug,
        )
        study_filters = criteria.study.model_dump(exclude_none=True)
        if study_filters.get("study_description") and registry is not None:
            matches = registry.match(current_query)
            if matches:
                cleaned, _removed = strip_node_tokens_from_filters(
                    {"study_description": study_filters["study_description"]},
                    registry,
                    matches,
                )
                if "study_description" in cleaned:
                    study_filters["study_description"] = cleaned["study_description"]
                else:
                    study_filters.pop("study_description", None)
                criteria = criteria.model_copy(
                    update={"study": StudyQuery(**study_filters)}
                )
        request = result.request.model_copy(update={"filters": study_filters})

        if not resolver_cfg.require_confirmation:
            return request, current_query, criteria

        click.echo(build_confirmation_message(request, resolver_cfg.confirmation))
        response = _read_stdin_line()
        decision = classify_confirmation_response(response, resolver_cfg.confirmation)
        log.debug(
            "Confirmation decision",
            extra={"extra_data": {"decision": decision}},
        )
        if decision == "accept":
            return request, current_query, criteria
        if decision == "reject":
            raise click.ClickException(resolver_cfg.confirmation.cancel_message)
        invalid_attempts += 1
        if invalid_attempts >= resolver_cfg.confirmation.max_invalid_responses:
            raise click.ClickException(resolver_cfg.confirmation.cancel_message)
        click.echo(build_invalid_response_message(resolver_cfg.confirmation))


def _execute_move(
    request: ResolvedRequest,
    criteria: SearchCriteria | None,
    config,
    date_range: str | None,
    max_studies: int | None,
    unlimited: bool,
    log: logging.Logger,
) -> dict[str, object]:
    if config.mcp is None:
        raise click.ClickException("Error: mcp.config_path not configured.")
    if not request.destination_node:
        raise click.ClickException("Destination not provided for move.")

    if criteria is None:
        criteria = SearchCriteria(study=StudyQuery(**request.filters))

    search_result = execute_search(
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
        rag_query=None,
        logger=log,
        node_name=request.source_node,
    )

    study_uids = search_result.study_instance_uids
    if not study_uids:
        return {"success": False, "message": "No studies found."}

    resolver_cfg = config.resolver
    if resolver_cfg is not None and resolver_cfg.require_confirmation and study_uids:
        if not sys.stdin.isatty():
            raise click.ClickException(
                "Confirmation required, but stdin is not a TTY."
            )
        preview_limit = 5
        preview = study_uids[:preview_limit]
        lines = [
            f"Matched {len(study_uids)} studies.",
            f"Source: {request.source_node or 'default'}",
            f"Destination: {request.destination_node}",
            "StudyInstanceUIDs:",
        ]
        lines.extend(f"- {uid}" for uid in preview)
        if len(study_uids) > preview_limit:
            lines.append(f"... and {len(study_uids) - preview_limit} more")
        confirmation = resolver_cfg.confirmation
        invalid_attempts = 0
        while True:
            click.echo("\n".join(lines))
            click.echo(
                f"Confirm? ({'/'.join(confirmation.accept_tokens)}"
                f"/{'/'.join(confirmation.reject_tokens)})"
            )
            response = _read_stdin_line()
            decision = classify_confirmation_response(response, confirmation)
            if decision == "accept":
                break
            if decision == "reject":
                raise click.ClickException(confirmation.cancel_message)
            invalid_attempts += 1
            if invalid_attempts >= confirmation.max_invalid_responses:
                raise click.ClickException(confirmation.cancel_message)
            click.echo(build_invalid_response_message(confirmation))

    async def _run_move() -> dict[str, object]:
        server_params = build_stdio_server_params(config.mcp)
        async with McpSession(server_params, config.mcp) as session:
            results: list[dict[str, object]] = []
            for study_uid in study_uids:
                await session.query_series(study_instance_uid=study_uid)
                try:
                    result = await session.move_study(
                        destination_node=request.destination_node,
                        study_instance_uid=study_uid,
                    )
                except Exception as exc:
                    result = {
                        "success": False,
                        "message": str(exc),
                        "completed": 0,
                        "failed": 1,
                        "warning": 0,
                    }
                if not isinstance(result, dict):
                    result = {
                        "success": False,
                        "message": "Invalid response from move_study.",
                        "completed": 0,
                        "failed": 1,
                        "warning": 0,
                    }
                results.append(
                    {
                        "study_instance_uid": study_uid,
                        "result": result,
                    }
                )
            completed = sum(
                int(item["result"].get("completed") or 0) for item in results
            )
            failed = sum(int(item["result"].get("failed") or 0) for item in results)
            warning = sum(int(item["result"].get("warning") or 0) for item in results)
            success_count = sum(
                1 for item in results if item["result"].get("success")
            )
            total = len(results)
            return {
                "success": success_count == total,
                "message": f"Moved {success_count}/{total} studies.",
                "completed": completed,
                "failed": failed,
                "warning": warning,
                "success_count": success_count,
                "total": total,
                "studies": results,
            }

    return anyio.run(_run_move)


@click.group()
@click.option(
    "--config",
    "-c",
    "config_path",
    default="config.yaml",
    type=click.Path(path_type=Path),
    show_default=True,
    help="Configuration file",
)
@click.option("--node", "-n", "node", default=None, help="dicom-mcp node for query")
@click.option("--verbose", "-v", is_flag=True, help="Enable verbose logs")
@click.option(
    "--llm-debug",
    is_flag=True,
    help="Show LLM-extracted JSON and final criteria (may contain PHI).",
)
@click.option("--json", "-j", "json_output", is_flag=True, help="JSON output")
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
    help="Configuration file",
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
    if config.mcp is not None and config.resolver is not None and config.resolver.enabled:
        log = logging.getLogger(__name__)
        try:
            registry = _load_node_registry(config.mcp, log)
            matches = registry.match(query)
            payload["node_matches"] = {
                "nodes": sorted({match.node_id for match in matches}),
                "matches": [
                    {
                        "node_id": match.node_id,
                        "start": match.start,
                        "end": match.end,
                        "source": match.source,
                    }
                    for match in matches
                ],
            }
        except Exception as exc:  # pragma: no cover - depends on MCP availability
            payload["node_matches"] = {"error": str(exc)}

    click.echo(json.dumps(payload, indent=2, ensure_ascii=True))


@main.command("execute")
@click.option(
    "--date-range",
    "date_range",
    default=None,
    help="Date range in YYYYMMDD-YYYYMMDD format",
)
@click.option(
    "--max-studies",
    "max_studies",
    default=None,
    type=int,
    help="Maximum number of studies to scan",
)
@click.option(
    "--unlimited",
    is_flag=True,
    help="Remove guardrail limits (not recommended)",
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
        click.echo(f"Error: {exc.message}", err=True)
        ctx.exit(3)

    if config.mcp is None:
        click.echo(
            "Error: mcp.config_path not configured. Update config.yaml for dicom-mcp.",
            err=True,
        )
        ctx.exit(3)

    try:
        date_range = _validate_date_range(date_range)
    except click.ClickException as exc:
        click.echo(f"Error: {exc.message}", err=True)
        ctx.exit(3)

    if max_studies is not None and max_studies <= 0:
        click.echo("Error: max-studies must be greater than zero", err=True)
        ctx.exit(3)

    try:
        resolved_request, query, criteria = _resolve_with_confirmation(
            query,
            config,
            log,
            llm_debug=ctx.obj["llm_debug"],
        )
        if resolved_request is not None:
            if resolved_request.destination_node:
                move_result = _execute_move(
                    resolved_request,
                    criteria,
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
                            "C-MOVE completed. "
                            f"Completed={completed} Failed={failed} Warning={warning}"
                        )
                    else:
                        click.echo(
                            f"C-MOVE failed: {move_result.get('message', 'Unknown error')}",
                            err=True,
                        )
                ctx.exit(0 if move_result.get("success") else 2)
            if criteria is None:
                criteria = SearchCriteria(study=StudyQuery(**resolved_request.filters))
        else:
            criteria = parse_nl_to_criteria(
                query,
                config.llm,
                strict_evidence=True,
                debug=ctx.obj["llm_debug"],
            )
    except Exception as exc:
        click.echo(f"Error: {exc}", err=True)
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
        click.echo(f"Error: {exc}", err=True)
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
            click.echo("No studies found with the specified criteria.")
            click.echo(f"Studies evaluated: {result.stats.studies_scanned}")
            if result.stats.date_range_applied:
                click.echo(f"Date range: {result.stats.date_range_applied}")
            if result.stats.stages_tried:
                click.echo(
                    "Stages tried: " + ", ".join(result.stats.stages_tried)
                )
            if result.stats.rewrites_tried:
                shown = ", ".join(result.stats.rewrites_tried[:3])
                click.echo(f"Rewrites tried: {shown}")
            if result.stats.limit_reached:
                click.echo(
                    "Search limit reached. Adjust --unlimited or max-studies."
                )
            if config.rag.enable:
                suggestions = get_rag_suggestions(query, config.rag, log)
                if suggestions:
                    shown = ", ".join(suggestions[:3])
                    click.echo(f"RAG suggestions: {shown}")

    ctx.exit(0 if result.accession_numbers else 1)


@main.command("smoke-test")
@click.option(
    "--llm-config",
    "llm_config_path",
    default=None,
    type=click.Path(path_type=Path),
    help="LLM config file (default: uses main config's llm section)",
)
@click.option(
    "--max-time",
    "-t",
    type=float,
    default=10.0,
    help="Maximum acceptable response time in seconds",
)
@click.option(
    "--query",
    "-q",
    help="Run a single custom query instead of test cases",
)
@click.pass_context
def smoke_test(
    ctx: click.Context,
    llm_config_path: Path | None,
    max_time: float,
    query: str | None,
) -> None:
    """Run smoke tests to validate LLM backend connectivity and response quality."""
    from datetime import date
    import time

    from .nl_parser import extract_json, SYSTEM_PROMPT

    # Load LLM config
    if llm_config_path:
        import yaml
        with open(llm_config_path) as f:
            data = yaml.safe_load(f)
        from .models import LLMConfig
        llm_config = LLMConfig.model_validate(data)
    else:
        config = _load_config(ctx.obj["config_path"])
        llm_config = config.llm

    client = create_llm_client(llm_config)

    click.echo(click.style("\nLLM Smoke Test", bold=True))
    click.echo(f"  Provider: {llm_config.provider}")
    click.echo(f"  Model: {llm_config.model}")
    click.echo(f"  Base URL: {llm_config.base_url}")
    click.echo(f"  Max time: {max_time}s")
    click.echo()

    # Test cases
    test_cases = [
        {
            "name": "CT angiogram with routing and age",
            "query": "studies from year 2000 until 2022 of CT chest angiograms from ORTHANC to RADIANT, patients age 20 to 80",
            "must_have": ["study_description", "study_date", "patient_birth_date", "modality_in_study"],
        },
        {
            "name": "MRI with sex filter",
            "query": "cranial MRI for women ages 30 to 50",
            "must_have": ["modality_in_study", "patient_birth_date"],
        },
        {
            "name": "Ultrasound with body part",
            "query": "obstetric ultrasound exams",
            "must_have": ["modality_in_study"],
        },
    ]

    if query:
        test_cases = [{"name": "Custom query", "query": query, "must_have": []}]

    system_prompt = f"{SYSTEM_PROMPT}\nToday: {date.today():%Y-%m-%d}"
    passed = 0
    failed = 0

    for test in test_cases:
        start = time.perf_counter()
        try:
            raw = client.chat(system_prompt, test["query"], json_mode=True)
            duration = time.perf_counter() - start

            data = extract_json(raw)
            study = data.get("study", {})

            # Check required fields
            missing = [f for f in test["must_have"] if not study.get(f)]

            if missing:
                click.echo(click.style(f"  [FAIL] {test['name']} ({duration:.2f}s)", fg="red"))
                click.echo(f"        Missing: {', '.join(missing)}")
                failed += 1
            elif duration > max_time:
                click.echo(click.style(f"  [WARN] {test['name']} ({duration:.2f}s)", fg="yellow"))
                click.echo(f"        Slow response (>{max_time}s)")
                passed += 1
            else:
                click.echo(click.style(f"  [PASS] {test['name']} ({duration:.2f}s)", fg="green"))
                passed += 1

            if ctx.obj["llm_debug"]:
                click.echo(f"        Response: {json.dumps(data, indent=2)[:300]}")

        except Exception as e:
            duration = time.perf_counter() - start
            click.echo(click.style(f"  [FAIL] {test['name']} ({duration:.2f}s)", fg="red"))
            click.echo(f"        Error: {e}")
            failed += 1

    click.echo()
    click.echo(click.style("Summary:", bold=True))
    click.echo(f"  Passed: {passed}/{len(test_cases)}")
    click.echo(f"  Failed: {failed}/{len(test_cases)}")

    if failed > 0:
        click.echo(click.style(f"\n{failed} test(s) failed!", fg="red"))
        ctx.exit(1)
    else:
        click.echo(click.style("\nAll tests passed!", fg="green"))
        ctx.exit(0)
