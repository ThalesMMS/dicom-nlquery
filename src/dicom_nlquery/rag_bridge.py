from __future__ import annotations

from functools import lru_cache
import logging

from .models import RagConfig


def get_rag_suggestions(
    query: str | None,
    config: RagConfig,
    logger: logging.Logger | None = None,
) -> list[str]:
    log = logger or logging.getLogger(__name__)
    if not config.enable:
        return []
    if not query:
        return []
    if not config.index_path:
        log.warning("RAG enabled but index_path is not configured")
        return []

    try:
        return list(
            _cached_retrieve(
                config.index_path,
                query,
                config.top_k,
                config.min_score,
                config.provider,
                config.model,
                config.base_url,
                config.embed_dim,
            )
        )
    except Exception as exc:  # pragma: no cover - optional dependency
        log.warning("RAG unavailable: %s", exc)
        return []


@lru_cache(maxsize=128)
def _cached_retrieve(
    index_path: str,
    query: str,
    top_k: int,
    min_score: float,
    provider: str,
    model: str | None,
    base_url: str | None,
    embed_dim: int,
) -> tuple[str, ...]:
    from pacs_rag.embedder import build_embedder
    from pacs_rag.retrieve import retrieve

    embedder = build_embedder(
        provider=provider,
        model=model,
        base_url=base_url,
        dim=embed_dim,
    )
    suggestions = retrieve(
        index_path=index_path,
        query=query,
        top_k=top_k,
        min_score=min_score,
        embedder=embedder,
    )
    return tuple(suggestion.text for suggestion in suggestions)
