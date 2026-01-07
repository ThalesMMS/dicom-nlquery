from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import re
import unicodedata

import yaml


def normalize_text(text: str) -> str:
    normalized = unicodedata.normalize("NFKD", text)
    normalized = "".join(c for c in normalized if not unicodedata.combining(c))
    normalized = normalized.lower()
    normalized = re.sub(r"[^a-z0-9]+", " ", normalized)
    return " ".join(normalized.split())


def normalize_token(token: str) -> str:
    return normalize_text(token).replace(" ", "")


def _tokenize(text: str) -> list[str]:
    return [token for token in normalize_text(text).split() if token]


@dataclass
class Lexicon:
    _expansions: dict[str, list[str]]

    def __init__(self) -> None:
        self._expansions = {}

    @classmethod
    def from_yaml(cls, path: str | Path) -> "Lexicon":
        data = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
        raw = data.get("synonyms", data) if isinstance(data, dict) else {}
        lexicon = cls()
        if isinstance(raw, dict):
            lexicon.update(raw)
        return lexicon

    def update(self, synonyms: dict[str, list[str]]) -> None:
        for key, values in synonyms.items():
            if not key:
                continue
            key_norm = normalize_text(str(key))
            if not key_norm:
                continue
            raw_values = values or []
            group = [key_norm]
            for value in raw_values:
                if value is None:
                    continue
                value_norm = normalize_text(str(value))
                if value_norm and value_norm not in group:
                    group.append(value_norm)
            for token in group:
                expansions = [token] + [item for item in group if item != token]
                self._expansions[token] = expansions

    def has_entries(self) -> bool:
        return bool(self._expansions)

    def expand(self, term: str) -> list[str]:
        if term is None:
            return []
        normalized = normalize_text(term)
        if not normalized:
            return []
        return list(self._expansions.get(normalized, [normalized]))

    def equivalent(self, left: str, right: str) -> bool:
        left_norm = normalize_text(left)
        right_norm = normalize_text(right)
        if not left_norm or not right_norm:
            return False
        if left_norm == right_norm:
            return True
        return right_norm in set(self.expand(left_norm))

    def match_text(self, query: str, candidate: str | None) -> bool:
        if not query:
            return True
        if not candidate:
            return False
        candidate_tokens = [token for token in _tokenize(candidate) if token]
        candidate_norm = " ".join(candidate_tokens)
        candidate_with_bounds = f" {candidate_norm} "
        tokens = [token for token in _tokenize(query) if len(token) > 2]
        if not tokens:
            return True
        for token in tokens:
            expanded = self.expand(token)
            matched = False
            for term in expanded:
                term_norm = normalize_text(term)
                if not term_norm:
                    continue
                if " " in term_norm:
                    if f" {term_norm} " in candidate_with_bounds:
                        matched = True
                        break
                elif term_norm in candidate_tokens:
                    matched = True
                    break
            if not matched:
                return False
        return True

    def expand_text(self, text: str, max_candidates: int | None = None) -> list[str]:
        if not text:
            return []
        normalized = normalize_text(text)
        if normalized in self._expansions:
            expansions = [item for item in self._expansions[normalized] if item != normalized]
            return expansions[: max_candidates or len(expansions)]

        tokens = _tokenize(text)
        if not tokens:
            return []

        expansions_per_token = [self.expand(token) for token in tokens]
        beam: list[tuple[list[str], int]] = [(list(tokens), 0)]
        for idx, options in enumerate(expansions_per_token):
            if not options:
                continue
            next_beam: list[tuple[list[str], int]] = []
            for candidate_tokens, replacements in beam:
                for option in options:
                    updated = list(candidate_tokens)
                    updated[idx] = option
                    updated_replacements = replacements + (1 if option != tokens[idx] else 0)
                    next_beam.append((updated, updated_replacements))
            next_beam.sort(key=lambda item: (item[1], " ".join(item[0])))
            if max_candidates:
                next_beam = next_beam[:max_candidates]
            beam = next_beam

        candidates: list[str] = []
        for candidate_tokens, replacements in beam:
            if replacements == 0:
                continue
            candidate = " ".join(candidate_tokens)
            if candidate not in candidates:
                candidates.append(candidate)
            if max_candidates and len(candidates) >= max_candidates:
                break
        return candidates


def load_lexicon(path: str | None = None, synonyms: dict[str, list[str]] | None = None) -> Lexicon | None:
    lexicon = Lexicon()
    if path:
        lexicon = Lexicon.from_yaml(path)
    if synonyms:
        lexicon.update(synonyms)
    return lexicon if lexicon.has_entries() else None
