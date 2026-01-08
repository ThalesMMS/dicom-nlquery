from __future__ import annotations

from dataclasses import dataclass
import json
import logging
import re
from typing import Any

from .models import NodeMatch, ResolvedRequest, ResolverResult
from .node_registry import NodeRegistry


TOKEN_RE = re.compile(r"[\w]+", re.UNICODE)

RESOLVER_SYSTEM_PROMPT = (
    "You are a DICOM request resolver. "
    "Return JSON only with keys: source_node, destination_node, filters, unmatched_tokens. "
    "Matched nodes are nodes only and must never be placed inside filters. "
    "Do not infer clinical filters that are not explicitly stated."
)


@dataclass(frozen=True)
class _TokenSpan:
    value: str
    start: int
    end: int


def resolve_request(
    raw_query: str,
    registry: NodeRegistry,
    llm_client: Any,
) -> ResolverResult:
    log = logging.getLogger(__name__)
    matches = registry.match(raw_query)
    log.debug(
        "Node matches resolved",
        extra={"extra_data": {"count": len(matches), "nodes": [m.node_id for m in matches]}},
    )
    user_prompt = _build_user_prompt(raw_query, matches, registry)
    response = llm_client.chat(RESOLVER_SYSTEM_PROMPT, user_prompt)
    request = _parse_llm_response(response)
    unresolved: list[str] = []
    request, node_issues = _normalize_nodes(request, registry)
    unresolved.extend(node_issues)
    cleaned_filters, removed_tokens = strip_node_tokens_from_filters(
        request.filters, registry, matches
    )
    if removed_tokens:
        unresolved.append("node_tokens_removed_from_filters")
    request = request.model_copy(update={"filters": cleaned_filters})
    if matches and not request.destination_node:
        unresolved.append("missing_destination_node")
    if (
        request.source_node
        and request.destination_node
        and request.source_node == request.destination_node
    ):
        unresolved.append("same_source_and_destination")
    log.debug(
        "Resolver output prepared",
        extra={
            "extra_data": {
                "filters": sorted(cleaned_filters.keys()),
                "unresolved": unresolved,
            }
        },
    )
    return ResolverResult(request=request, needs_confirmation=True, unresolved=unresolved)


def strip_node_tokens_from_filters(
    filters: dict[str, Any],
    registry: NodeRegistry,
    matches: list[NodeMatch],
) -> tuple[dict[str, Any], list[str]]:
    if not filters or not matches:
        return dict(filters), []
    sequences = _collect_node_sequences(registry, matches)
    cleaned: dict[str, Any] = {}
    removed_tokens: list[str] = []
    for key, value in filters.items():
        if isinstance(value, str):
            stripped, removed = _strip_node_tokens_from_text(value, sequences)
            if stripped:
                cleaned[key] = stripped
            removed_tokens.extend(removed)
        elif isinstance(value, list):
            cleaned_list: list[str] = []
            for item in value:
                if not isinstance(item, str):
                    continue
                stripped, removed = _strip_node_tokens_from_text(item, sequences)
                if stripped:
                    cleaned_list.append(stripped)
                removed_tokens.extend(removed)
            if cleaned_list:
                cleaned[key] = cleaned_list
        else:
            cleaned[key] = value
    return cleaned, _dedupe(removed_tokens)


def _build_user_prompt(
    raw_query: str, matches: list[NodeMatch], registry: NodeRegistry
) -> str:
    payload = {
        "query": raw_query,
        "matched_nodes": [
            {
                "node_id": match.node_id,
                "start": match.start,
                "end": match.end,
                "source": match.source,
            }
            for match in matches
        ],
        "available_nodes": registry.node_ids,
    }
    return json.dumps(payload, ensure_ascii=True)


def _parse_llm_response(response: str) -> ResolvedRequest:
    raw = _extract_json(response)
    data = json.loads(raw)
    if not isinstance(data, dict):
        raise ValueError("resolver response must be a JSON object")
    if data.get("filters") is None:
        data["filters"] = {}
    return ResolvedRequest.model_validate(data)


def _extract_json(response: str) -> str:
    start = response.find("{")
    end = response.rfind("}")
    if start == -1 or end == -1 or end <= start:
        raise ValueError("resolver response does not contain JSON")
    return response[start : end + 1]


def _normalize_nodes(
    request: ResolvedRequest, registry: NodeRegistry
) -> tuple[ResolvedRequest, list[str]]:
    unresolved: list[str] = []
    updates: dict[str, str | None] = {}
    for field_name in ("source_node", "destination_node"):
        value = getattr(request, field_name)
        if value is None:
            continue
        canonical = registry.canonicalize(value)
        if canonical is None:
            unresolved.append(f"unknown_{field_name}")
            updates[field_name] = None
        else:
            updates[field_name] = canonical
    if updates:
        request = request.model_copy(update=updates)
    return request, unresolved


def _collect_node_sequences(
    registry: NodeRegistry, matches: list[NodeMatch]
) -> list[tuple[str, ...]]:
    sequences: list[tuple[str, ...]] = []
    for node_id in {match.node_id for match in matches}:
        sequences.extend(registry.patterns_for_node(node_id))
    return sequences


def _strip_node_tokens_from_text(
    text: str, sequences: list[tuple[str, ...]]
) -> tuple[str | None, list[str]]:
    tokens = _tokenize(text)
    if not tokens:
        return None, []
    normalized = [token.value.lower() for token in tokens]
    candidates: list[tuple[int, int]] = []
    for sequence in sequences:
        size = len(sequence)
        if size == 0 or size > len(tokens):
            continue
        for index in range(len(tokens) - size + 1):
            if tuple(normalized[index : index + size]) == sequence:
                candidates.append((index, index + size - 1))
    selected = _select_longest_spans(candidates)
    if not selected:
        return text.strip() or None, []
    removed: list[str] = []
    removed_indexes: set[int] = set()
    for start, end in selected:
        removed.append(" ".join(token.value for token in tokens[start : end + 1]))
        removed_indexes.update(range(start, end + 1))
    remaining = [
        token.value for index, token in enumerate(tokens) if index not in removed_indexes
    ]
    cleaned = " ".join(remaining).strip()
    return cleaned or None, removed


def _select_longest_spans(candidates: list[tuple[int, int]]) -> list[tuple[int, int]]:
    if not candidates:
        return []
    ordered = sorted(candidates, key=lambda span: (-(span[1] - span[0]), span[0]))
    selected: list[tuple[int, int]] = []
    for span in ordered:
        if any(_overlaps(span, kept) for kept in selected):
            continue
        selected.append(span)
    selected.sort(key=lambda span: (span[0], span[1]))
    return selected


def _overlaps(left: tuple[int, int], right: tuple[int, int]) -> bool:
    return not (left[1] < right[0] or left[0] > right[1])


def _tokenize(text: str) -> list[_TokenSpan]:
    return [
        _TokenSpan(match.group(0), match.start(), match.end())
        for match in TOKEN_RE.finditer(text)
    ]


def _dedupe(values: list[str]) -> list[str]:
    seen: set[str] = set()
    result: list[str] = []
    for value in values:
        if value in seen:
            continue
        seen.add(value)
        result.append(value)
    return result
