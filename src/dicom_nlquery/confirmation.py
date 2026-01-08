from __future__ import annotations

from typing import Any

from .models import ConfirmationConfig, ResolvedRequest


def build_confirmation_message(request: ResolvedRequest, config: ConfirmationConfig) -> str:
    mode = "move" if request.destination_node else "search-only"
    source_node = request.source_node or "unknown"
    destination_node = request.destination_node or "none"
    filters_text = _format_filters(request.filters)
    return config.prompt_template.format(
        mode=mode,
        source_node=source_node,
        destination_node=destination_node,
        filters=filters_text,
        accept_tokens=", ".join(config.accept_tokens),
        reject_tokens=", ".join(config.reject_tokens),
    )


def classify_confirmation_response(response: str, config: ConfirmationConfig) -> str:
    normalized = _normalize_token(response)
    accept_tokens = {_normalize_token(token) for token in config.accept_tokens}
    reject_tokens = {_normalize_token(token) for token in config.reject_tokens}
    if normalized in accept_tokens:
        return "accept"
    if normalized in reject_tokens:
        return "reject"
    return "invalid"


def build_invalid_response_message(config: ConfirmationConfig) -> str:
    return config.invalid_response.format(
        accept_tokens=", ".join(config.accept_tokens),
        reject_tokens=", ".join(config.reject_tokens),
    )


def _normalize_token(value: str) -> str:
    return value.strip().lower()


def _format_filters(filters: dict[str, Any]) -> str:
    if not filters:
        return "none"
    lines: list[str] = []
    for key in sorted(filters.keys()):
        value = filters[key]
        formatted = _format_filter_value(value)
        if not formatted:
            continue
        lines.append(f"- {key}: {formatted}")
    if not lines:
        return "none"
    return "\n".join(lines)


def _format_filter_value(value: Any) -> str | None:
    if value is None:
        return None
    if isinstance(value, list):
        cleaned = [str(item).strip() for item in value if str(item).strip()]
        if not cleaned:
            return None
        return ", ".join(cleaned)
    text = str(value).strip()
    return text or None
