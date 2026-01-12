"""Public API for dicom-nlquery.

This module provides high-level functions that can be used by both CLI and web interfaces.

Example usage:

    from dicom_nlquery.api import execute_nl_query_sync, load_config
    from dicom_nlquery.config import load_config

    config = load_config("config.yaml")
    result = execute_nl_query_sync("CT chest from ORTHANC", config)
    print(result["search_result"]["accession_numbers"])
"""

from .core import (
    execute_nl_query,
    execute_nl_query_sync,
    load_node_registry,
    load_node_registry_sync,
    move_studies,
    move_studies_sync,
    parse_query,
    resolve_and_parse,
    resolve_query,
    search_studies,
)

__all__ = [
    # Async functions
    "load_node_registry",
    "move_studies",
    "execute_nl_query",
    # Sync wrappers
    "load_node_registry_sync",
    "move_studies_sync",
    "execute_nl_query_sync",
    # Sync functions
    "resolve_query",
    "parse_query",
    "resolve_and_parse",
    "search_studies",
]
