from __future__ import annotations

from dataclasses import dataclass
import json
import logging
import re
from typing import Any

from .llm_client import LLMClient
from .models import NodeMatch, ResolvedRequest, ResolverResult
from .node_registry import NodeRegistry


TOKEN_RE = re.compile(r"[\w]+", re.UNICODE)

RESOLVER_SYSTEM_PROMPT = (
    "Return JSON only with keys: source_node, destination_node, filters, unmatched_tokens. "
    "Input JSON keys: q (query), m (matched node ids), n (available node ids). "
    "filters must be a JSON object. "
    "Nodes must not be placed inside filters. "
    "Do not infer clinical filters."
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
    # All LLMClient implementations support json_mode
    response = llm_client.chat(RESOLVER_SYSTEM_PROMPT, user_prompt, json_mode=True)
    request, issues = _parse_llm_response(response)
    unresolved: list[str] = []
    unresolved.extend(issues)
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
        "q": raw_query,
        "m": sorted({match.node_id for match in matches}),
        "n": registry.node_ids,
    }
    return json.dumps(payload, ensure_ascii=True, separators=(",", ":"))


def _parse_llm_response(response: str) -> tuple[ResolvedRequest, list[str]]:
    raw = _extract_json(response)
    data = json.loads(raw)
    if not isinstance(data, dict):
        raise ValueError("resolver response must be a JSON object")
    issues: list[str] = []
    filters = data.get("filters")
    if filters is None:
        data["filters"] = {}
    elif not isinstance(filters, dict):
        issues.append("invalid_filters")
        data["filters"] = {}
        if isinstance(filters, list):
            extracted: list[str] = []
            for item in filters:
                if isinstance(item, dict):
                    node_id = item.get("node_id") or item.get("node")
                    if isinstance(node_id, str) and node_id.strip():
                        extracted.append(node_id.strip())
                elif isinstance(item, str):
                    cleaned = item.strip()
                    if cleaned:
                        extracted.append(cleaned)
            if extracted:
                data.setdefault("unmatched_tokens", [])
                data["unmatched_tokens"].extend(extracted)
    if data.get("filters") is None:
        data["filters"] = {}
    return ResolvedRequest.model_validate(data), issues


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
