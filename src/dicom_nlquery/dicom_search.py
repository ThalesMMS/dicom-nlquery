from __future__ import annotations

from datetime import date, timedelta
import logging
import time

import anyio

from .lexicon import load_lexicon
from .mcp_client import McpSession, build_stdio_server_params
from .models import (
    GuardrailsConfig,
    LexiconConfig,
    McpServerConfig,
    RagConfig,
    RankingConfig,
    SearchCriteria,
    SearchPipelineConfig,
    SearchResult,
)
from .rag_bridge import get_rag_suggestions
from .search_pipeline import run_pipeline_async, run_pipeline_sync


def apply_guardrails(
    guardrails: GuardrailsConfig,
    date_range: str | None = None,
    max_studies: int | None = None,
    unlimited: bool = False,
    logger: logging.Logger | None = None,
    today: date | None = None,
) -> tuple[str | None, int | None]:
    log = logger or logging.getLogger(__name__)

    if unlimited:
        log.warning(
            "AVISO: Varredura ilimitada solicitada. Isso pode sobrecarregar o PACS."
        )
        return date_range, None

    effective_date_range = date_range
    if effective_date_range is None:
        end_date = today or date.today()
        start_date = end_date - timedelta(days=guardrails.study_date_range_default_days)
        effective_date_range = f"{start_date:%Y%m%d}-{end_date:%Y%m%d}"

    effective_max_studies = max_studies
    if effective_max_studies is None:
        effective_max_studies = guardrails.max_studies_scanned_default

    return effective_date_range, effective_max_studies


def execute_search(
    criteria: SearchCriteria,
    query_client: object | None = None,
    mcp_config: McpServerConfig | None = None,
    date_range: str | None = None,
    max_studies: int | None = None,
    unlimited: bool = False,
    guardrails_config: GuardrailsConfig | None = None,
    search_pipeline_config: SearchPipelineConfig | None = None,
    lexicon_config: LexiconConfig | None = None,
    rag_config: RagConfig | None = None,
    ranking_config: RankingConfig | None = None,
    rag_query: str | None = None,
    logger: logging.Logger | None = None,
    node_name: str | None = None,
) -> SearchResult:
    start_time = time.time()
    guardrails = guardrails_config or GuardrailsConfig()
    log = logger or logging.getLogger(__name__)
    effective_date_range, effective_max_studies = apply_guardrails(
        guardrails,
        date_range=date_range,
        max_studies=max_studies,
        unlimited=unlimited,
        logger=log,
    )
    resolved_study_date = date_range or criteria.study.study_date or effective_date_range
    explicit_study_date = date_range or criteria.study.study_date
    log.info(
        "Starting DICOM search",
        extra={
            "extra_data": {
                "date_range": resolved_study_date,
                "max_studies": effective_max_studies,
                "unlimited": unlimited,
            }
        },
    )
    pipeline_config = search_pipeline_config or SearchPipelineConfig()
    ranking = ranking_config or RankingConfig()
    lexicon = None
    if lexicon_config is not None:
        lexicon = load_lexicon(lexicon_config.path, lexicon_config.synonyms)
    rag_suggestions: list[str] = []
    if rag_config is not None:
        rag_text = rag_query or criteria.study.study_description
        rag_suggestions = get_rag_suggestions(rag_text, rag_config, log)
    timeout_seconds = guardrails.search_timeout_seconds

    if query_client is None and mcp_config is None:
        raise ValueError("mcp_config is required when query_client is not provided")

    if query_client is None:
        return anyio.run(
            _execute_with_mcp,
            criteria,
            mcp_config,
            resolved_study_date,
            explicit_study_date,
            effective_max_studies,
            pipeline_config,
            lexicon,
            rag_suggestions,
            ranking,
            timeout_seconds,
            log,
            start_time,
            node_name,
        )

    return _execute_with_client(
        criteria,
        query_client,
        resolved_study_date,
        explicit_study_date,
        effective_max_studies,
        pipeline_config,
        lexicon,
        rag_suggestions,
        ranking,
        timeout_seconds,
        log,
        start_time,
    )


def _execute_with_client(
    criteria: SearchCriteria,
    query_client: object,
    study_date: str | None,
    explicit_study_date: str | None,
    max_studies: int | None,
    pipeline_config: SearchPipelineConfig,
    lexicon: object | None,
    rag_suggestions: list[str],
    ranking: RankingConfig,
    timeout_seconds: int | None,
    log: logging.Logger,
    start_time: float,
) -> SearchResult:
    return run_pipeline_sync(
        criteria,
        query_client,
        study_date,
        explicit_study_date,
        max_studies,
        pipeline_config,
        lexicon,
        rag_suggestions,
        ranking,
        timeout_seconds,
        log,
        start_time,
    )


async def _execute_with_mcp(
    criteria: SearchCriteria,
    mcp_config: McpServerConfig,
    study_date: str | None,
    explicit_study_date: str | None,
    max_studies: int | None,
    pipeline_config: SearchPipelineConfig,
    lexicon: object | None,
    rag_suggestions: list[str],
    ranking: RankingConfig,
    timeout_seconds: int | None,
    log: logging.Logger,
    start_time: float,
    node_name: str | None,
) -> SearchResult:
    server_params = build_stdio_server_params(mcp_config)
    async with McpSession(server_params) as client:
        if node_name:
            await client.switch_dicom_node(node_name)
        return await run_pipeline_async(
            criteria,
            client,
            study_date,
            explicit_study_date,
            max_studies,
            pipeline_config,
            lexicon,
            rag_suggestions,
            ranking,
            timeout_seconds,
            log,
            start_time,
        )
