from __future__ import annotations

from dataclasses import dataclass
import logging
import re
from typing import Any

from .models import NodeMatch


TOKEN_RE = re.compile(r"[\w]+", re.UNICODE)


@dataclass(frozen=True)
class _TokenSpan:
    value: str
    start: int
    end: int


@dataclass(frozen=True)
class _NodePattern:
    node_id: str
    tokens: tuple[str, ...]
    source: str
    text: str


def _tokenize(text: str) -> list[_TokenSpan]:
    return [
        _TokenSpan(match.group(0), match.start(), match.end())
        for match in TOKEN_RE.finditer(text)
    ]


def _normalize_tokens(tokens: list[_TokenSpan]) -> tuple[str, ...]:
    return tuple(token.value.lower() for token in tokens)


def _dedupe(values: list[str]) -> list[str]:
    seen: set[str] = set()
    result: list[str] = []
    for value in values:
        if value in seen:
            continue
        seen.add(value)
        result.append(value)
    return result


class NodeRegistry:
    def __init__(self, node_ids: list[str], aliases: dict[str, list[str]] | None = None) -> None:
        cleaned_ids = _dedupe([node.strip() for node in node_ids if node.strip()])
        if not cleaned_ids:
            raise ValueError("node_ids must not be empty")
        self.node_ids = cleaned_ids
        self.aliases = self._normalize_aliases(aliases or {}, cleaned_ids)
        self._patterns: list[_NodePattern] = []
        self._patterns_by_node: dict[str, list[_NodePattern]] = {}
        self._canonical_lookup: dict[str, str] = {}
        self.normalized_index = self._build_index()

    @classmethod
    def from_tool_payload(cls, payload: list[dict[str, Any]]) -> "NodeRegistry":
        node_ids: list[str] = []
        aliases: dict[str, list[str]] = {}
        for entry in payload:
            if not isinstance(entry, dict):
                raise ValueError("node registry entries must be objects")
            node_id = entry.get("name") or entry.get("node_id") or entry.get("id")
            if not isinstance(node_id, str) or not node_id.strip():
                raise ValueError("node registry entries must include a name")
            node_id = node_id.strip()
            node_ids.append(node_id)
            alias_items: list[str] = []
            raw_aliases = entry.get("aliases")
            if isinstance(raw_aliases, str):
                alias_items.append(raw_aliases)
            elif isinstance(raw_aliases, list):
                alias_items.extend([str(item) for item in raw_aliases])
            ae_title = entry.get("ae_title") or entry.get("aeTitle")
            if isinstance(ae_title, str) and ae_title.strip():
                alias_items.append(ae_title)
            if alias_items:
                aliases[node_id] = alias_items
        registry = cls(node_ids=node_ids, aliases=aliases)
        logging.getLogger(__name__).debug(
            "Node registry loaded",
            extra={"extra_data": {"count": len(registry.node_ids)}},
        )
        return registry

    def canonicalize(self, value: str) -> str | None:
        normalized = value.strip().lower()
        if not normalized:
            return None
        return self._canonical_lookup.get(normalized)

    def patterns_for_node(self, node_id: str) -> list[tuple[str, ...]]:
        patterns = self._patterns_by_node.get(node_id, [])
        return [pattern.tokens for pattern in patterns]

    def match(self, query: str) -> list[NodeMatch]:
        tokens = _tokenize(query)
        if not tokens:
            return []
        normalized = [token.value.lower() for token in tokens]
        candidates: list[NodeMatch] = []
        for index, token_value in enumerate(normalized):
            for pattern in self.normalized_index.get(token_value, []):
                size = len(pattern.tokens)
                if index + size > len(tokens):
                    continue
                if tuple(normalized[index : index + size]) != pattern.tokens:
                    continue
                start = tokens[index].start
                end = tokens[index + size - 1].end
                matched_text = query[start:end]
                candidates.append(
                    NodeMatch(
                        node_id=pattern.node_id,
                        start=start,
                        end=end,
                        source=matched_text,
                    )
                )
        return self._select_longest(candidates)

    def _normalize_aliases(
        self, aliases: dict[str, list[str]], node_ids: list[str]
    ) -> dict[str, list[str]]:
        normalized: dict[str, list[str]] = {}
        for node_id in node_ids:
            raw = aliases.get(node_id, [])
            cleaned_items: list[str] = []
            for item in raw:
                if item is None:
                    continue
                text = str(item).strip()
                if text:
                    cleaned_items.append(text)
            cleaned = _dedupe(cleaned_items)
            if cleaned:
                normalized[node_id] = cleaned
        return normalized

    def _build_index(self) -> dict[str, list[_NodePattern]]:
        patterns: list[_NodePattern] = []
        for node_id in self.node_ids:
            self._canonical_lookup[node_id.lower()] = node_id
            node_tokens = _normalize_tokens(_tokenize(node_id))
            if node_tokens:
                patterns.append(
                    _NodePattern(
                        node_id=node_id,
                        tokens=node_tokens,
                        source="canonical",
                        text=node_id,
                    )
                )
            for alias in self.aliases.get(node_id, []):
                alias_tokens = _normalize_tokens(_tokenize(alias))
                if not alias_tokens:
                    continue
                patterns.append(
                    _NodePattern(
                        node_id=node_id,
                        tokens=alias_tokens,
                        source="alias",
                        text=alias,
                    )
                )
                self._canonical_lookup[alias.lower()] = node_id
        deduped: list[_NodePattern] = []
        seen: set[tuple[str, tuple[str, ...]]] = set()
        for pattern in patterns:
            key = (pattern.node_id, pattern.tokens)
            if key in seen:
                continue
            seen.add(key)
            deduped.append(pattern)
        self._patterns = deduped
        for pattern in deduped:
            self._patterns_by_node.setdefault(pattern.node_id, []).append(pattern)
        index: dict[str, list[_NodePattern]] = {}
        for pattern in deduped:
            index.setdefault(pattern.tokens[0], []).append(pattern)
        for token_value, items in index.items():
            items.sort(
                key=lambda item: (-len(item.tokens), item.node_id, item.source, item.text)
            )
            index[token_value] = items
        return index

    def _select_longest(self, candidates: list[NodeMatch]) -> list[NodeMatch]:
        if not candidates:
            return []
        ordered = sorted(
            candidates,
            key=lambda item: (
                -(item.end - item.start),
                item.start,
                item.node_id,
                item.end,
            ),
        )
        selected: list[NodeMatch] = []
        for match in ordered:
            if any(self._overlaps(match, kept) for kept in selected):
                continue
            selected.append(match)
        selected.sort(key=lambda item: (item.start, item.end, item.node_id))
        return selected

    @staticmethod
    def _overlaps(left: NodeMatch, right: NodeMatch) -> bool:
        return not (left.end <= right.start or left.start >= right.end)
