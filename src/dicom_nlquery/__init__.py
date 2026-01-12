"""DICOM NL Query package."""

from .config import load_config
from .dicom_search import execute_search
from .models import (
    MoveResult,
    NLQueryConfig,
    ResolvedRequest,
    ResolverResult,
    SearchCriteria,
    SearchResult,
)
from .nl_parser import parse_nl_to_criteria

# Public API - high-level functions for CLI and web
from .api import (
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

__version__ = "0.1.0"

__all__ = [
    # Core functions (low-level)
    "execute_search",
    "parse_nl_to_criteria",
    "load_config",
    # Models
    "SearchCriteria",
    "SearchResult",
    "MoveResult",
    "NLQueryConfig",
    "ResolvedRequest",
    "ResolverResult",
    # Public API (high-level)
    "load_node_registry",
    "load_node_registry_sync",
    "resolve_query",
    "parse_query",
    "resolve_and_parse",
    "search_studies",
    "move_studies",
    "move_studies_sync",
    "execute_nl_query",
    "execute_nl_query_sync",
]
