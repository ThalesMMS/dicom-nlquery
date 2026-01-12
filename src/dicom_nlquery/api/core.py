"""Public API for dicom-nlquery.

This module provides high-level functions that can be used by both CLI and web interfaces.
All functions are designed to be reusable without stdin/tty dependencies.
"""

from __future__ import annotations

import logging
from typing import Any

import anyio

from ..config import load_config
from ..dicom_search import execute_search as _execute_search
from ..llm_client import create_llm_client
from ..mcp_client import McpSession, build_stdio_server_params
from ..models import (
    LLMConfig,
    McpServerConfig,
    MoveResult,
    NLQueryConfig,
    ResolvedRequest,
    ResolverResult,
    SearchCriteria,
    SearchResult,
    StudyQuery,
)
from ..nl_parser import parse_nl_to_criteria
from ..node_registry import NodeRegistry
from ..resolver import resolve_request as _resolve_request
from ..resolver import strip_node_tokens_from_filters


async def load_node_registry(mcp_config: McpServerConfig) -> NodeRegistry:
    """Load the node registry from MCP server.

    Args:
        mcp_config: MCP server configuration.

    Returns:
        NodeRegistry populated with available DICOM nodes.

    Raises:
        ValueError: If the MCP response is invalid.
        Exception: If connection to MCP fails.
    """
    server_params = build_stdio_server_params(mcp_config)
    async with McpSession(server_params, mcp_config) as session:
        payload = await session.list_dicom_nodes()
        if not isinstance(payload, dict):
            raise ValueError("list_dicom_nodes returned invalid payload")
        nodes = payload.get("nodes")
        if not isinstance(nodes, list):
            raise ValueError("list_dicom_nodes returned an invalid format")
        return NodeRegistry.from_tool_payload(nodes)


def load_node_registry_sync(mcp_config: McpServerConfig) -> NodeRegistry:
    """Synchronous wrapper for load_node_registry.

    Args:
        mcp_config: MCP server configuration.

    Returns:
        NodeRegistry populated with available DICOM nodes.
    """
    return anyio.run(load_node_registry, mcp_config)


def resolve_query(
    query: str,
    registry: NodeRegistry,
    llm_config: LLMConfig,
) -> ResolverResult:
    """Resolve source/destination nodes from a natural language query.

    This function uses the LLM to extract node routing information from the query.
    It does NOT handle confirmation flow - that's the caller's responsibility.

    Args:
        query: Natural language query (e.g., "CT chest from ORTHANC to RADIANT").
        registry: Node registry with available DICOM nodes.
        llm_config: LLM configuration.

    Returns:
        ResolverResult with extracted nodes and any unresolved issues.
    """
    llm = create_llm_client(llm_config)
    return _resolve_request(query, registry, llm)


def parse_query(
    query: str,
    llm_config: LLMConfig,
    registry: NodeRegistry | None = None,
    strict_evidence: bool = True,
    debug: bool = False,
) -> SearchCriteria:
    """Parse a natural language query into SearchCriteria.

    If a registry is provided, node tokens (ORTHANC, RADIANT, etc.) will be
    stripped from the query before parsing to prevent them from polluting
    clinical filters.

    Args:
        query: Natural language query.
        llm_config: LLM configuration.
        registry: Optional node registry for token stripping.
        strict_evidence: Whether to validate evidence in responses.
        debug: Whether to enable debug logging.

    Returns:
        SearchCriteria with parsed study/series filters.
    """
    cleaned_query = query

    if registry is not None:
        matches = registry.match(query)
        if matches:
            cleaned_payload, _removed = strip_node_tokens_from_filters(
                {"q": query},
                registry,
                matches,
            )
            cleaned_query = cleaned_payload.get("q") or query

    criteria = parse_nl_to_criteria(
        cleaned_query,
        llm_config,
        strict_evidence=strict_evidence,
        debug=debug,
    )

    # Double-check: strip node tokens from study_description if present
    if registry is not None and criteria.study.study_description:
        matches = registry.match(query)
        if matches:
            study_filters = criteria.study.model_dump(exclude_none=True)
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

    return criteria


def resolve_and_parse(
    query: str,
    config: NLQueryConfig,
    debug: bool = False,
) -> tuple[ResolverResult | None, SearchCriteria]:
    """Resolve nodes and parse query in one step.

    This is the recommended entry point for processing natural language queries.
    It handles both node resolution and clinical filter parsing.

    Args:
        query: Natural language query.
        config: Full NLQuery configuration.
        debug: Whether to enable debug logging.

    Returns:
        Tuple of (ResolverResult or None, SearchCriteria).
        ResolverResult is None if resolver is disabled.
    """
    resolver_config = config.resolver
    registry: NodeRegistry | None = None
    resolved: ResolverResult | None = None

    # Load registry and resolve if enabled
    if (
        resolver_config is not None
        and resolver_config.enabled
        and config.mcp is not None
    ):
        registry = load_node_registry_sync(config.mcp)
        resolved = resolve_query(query, registry, config.llm)

    # Parse with node stripping
    criteria = parse_query(
        query,
        config.llm,
        registry=registry,
        strict_evidence=True,
        debug=debug,
    )

    return resolved, criteria


def search_studies(
    criteria: SearchCriteria,
    config: NLQueryConfig,
    node_name: str | None = None,
    date_range: str | None = None,
    max_studies: int | None = None,
    unlimited: bool = False,
    rag_query: str | None = None,
    logger: logging.Logger | None = None,
) -> SearchResult:
    """Execute a DICOM study search.

    Args:
        criteria: Search criteria with study/series filters.
        config: Full NLQuery configuration.
        node_name: Optional source node name.
        date_range: Optional date range override (YYYYMMDD-YYYYMMDD).
        max_studies: Optional max studies override.
        unlimited: Whether to remove guardrail limits.
        rag_query: Original query for RAG suggestions.
        logger: Optional logger instance.

    Returns:
        SearchResult with matched studies.
    """
    if config.mcp is None:
        raise ValueError("MCP configuration is required for search")

    return _execute_search(
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
        rag_query=rag_query,
        logger=logger,
        node_name=node_name,
    )


async def move_studies(
    study_uids: list[str],
    destination_node: str,
    mcp_config: McpServerConfig,
    source_node: str | None = None,
) -> MoveResult:
    """Move studies to a destination node via C-MOVE.

    This function does NOT handle confirmation - that's the caller's responsibility.

    Args:
        study_uids: List of StudyInstanceUIDs to move.
        destination_node: Target DICOM node name.
        mcp_config: MCP server configuration.
        source_node: Optional source node (uses default if None).

    Returns:
        MoveResult with success/failure counts.
    """
    if not study_uids:
        return MoveResult(success=False, message="No studies to move")

    server_params = build_stdio_server_params(mcp_config)
    async with McpSession(server_params, mcp_config) as session:
        # Switch to source node if specified
        if source_node:
            await session.switch_dicom_node(source_node)

        results: list[dict[str, Any]] = []
        for study_uid in study_uids:
            # Query series first (for metadata)
            await session.query_series(study_instance_uid=study_uid)
            try:
                result = await session.move_study(
                    destination_node=destination_node,
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
                    "message": "Invalid response from move_study",
                    "completed": 0,
                    "failed": 1,
                    "warning": 0,
                }
            results.append({
                "study_instance_uid": study_uid,
                "result": result,
            })

        # Aggregate results
        completed = sum(int(item["result"].get("completed") or 0) for item in results)
        failed = sum(int(item["result"].get("failed") or 0) for item in results)
        warning = sum(int(item["result"].get("warning") or 0) for item in results)
        success_count = sum(1 for item in results if item["result"].get("success"))
        total = len(results)

        return MoveResult(
            success=success_count == total,
            message=f"Moved {success_count}/{total} studies",
            completed=completed,
            failed=failed,
            warning=warning,
            studies=results,
        )


def move_studies_sync(
    study_uids: list[str],
    destination_node: str,
    mcp_config: McpServerConfig,
    source_node: str | None = None,
) -> MoveResult:
    """Synchronous wrapper for move_studies.

    Args:
        study_uids: List of StudyInstanceUIDs to move.
        destination_node: Target DICOM node name.
        mcp_config: MCP server configuration.
        source_node: Optional source node.

    Returns:
        MoveResult with success/failure counts.
    """
    return anyio.run(
        move_studies,
        study_uids,
        destination_node,
        mcp_config,
        source_node,
    )


async def execute_nl_query(
    query: str,
    config: NLQueryConfig,
    mode: str = "search",
    date_range: str | None = None,
    max_studies: int | None = None,
    unlimited: bool = False,
    debug: bool = False,
    logger: logging.Logger | None = None,
) -> dict[str, Any]:
    """Execute a complete natural language DICOM query.

    This is the highest-level API function that combines resolution, parsing,
    search, and optionally move into a single call. Ideal for web interfaces.

    Args:
        query: Natural language query.
        config: Full NLQuery configuration.
        mode: "search" or "move" (move requires destination node in query).
        date_range: Optional date range override.
        max_studies: Optional max studies override.
        unlimited: Whether to remove guardrail limits.
        debug: Whether to enable debug logging.
        logger: Optional logger instance.

    Returns:
        Dictionary with:
        - resolved: ResolverResult dict or None
        - criteria: SearchCriteria dict
        - search_result: SearchResult dict
        - move_result: MoveResult dict (only if mode="move" and destination found)
        - needs_confirmation: bool (True if move pending confirmation)
    """
    # Step 1: Resolve and parse
    resolved, criteria = resolve_and_parse(query, config, debug=debug)

    # Extract node info
    source_node: str | None = None
    destination_node: str | None = None
    if resolved is not None:
        source_node = resolved.request.source_node
        destination_node = resolved.request.destination_node

    # Step 2: Search
    search_result = search_studies(
        criteria,
        config,
        node_name=source_node,
        date_range=date_range,
        max_studies=max_studies,
        unlimited=unlimited,
        rag_query=query,
        logger=logger,
    )

    result: dict[str, Any] = {
        "resolved": resolved.model_dump() if resolved else None,
        "criteria": criteria.model_dump(),
        "search_result": search_result.model_dump(),
        "move_result": None,
        "needs_confirmation": False,
    }

    # Step 3: Handle move if requested and destination present
    if mode == "move" and destination_node and search_result.study_instance_uids:
        # For web context, we return needs_confirmation=True
        # The caller should confirm and then call move_studies directly
        result["needs_confirmation"] = True
        result["pending_move"] = {
            "study_uids": search_result.study_instance_uids,
            "destination_node": destination_node,
            "source_node": source_node,
        }

    return result


def execute_nl_query_sync(
    query: str,
    config: NLQueryConfig,
    mode: str = "search",
    date_range: str | None = None,
    max_studies: int | None = None,
    unlimited: bool = False,
    debug: bool = False,
    logger: logging.Logger | None = None,
) -> dict[str, Any]:
    """Synchronous wrapper for execute_nl_query.

    Args:
        query: Natural language query.
        config: Full NLQuery configuration.
        mode: "search" or "move".
        date_range: Optional date range override.
        max_studies: Optional max studies override.
        unlimited: Whether to remove guardrail limits.
        debug: Whether to enable debug logging.
        logger: Optional logger instance.

    Returns:
        Same as execute_nl_query.
    """
    return anyio.run(
        execute_nl_query,
        query,
        config,
        mode,
        date_range,
        max_studies,
        unlimited,
        debug,
        logger,
    )
