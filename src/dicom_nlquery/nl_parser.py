from __future__ import annotations

import json
import logging
import re

from pydantic import ValidationError

from .llm_client import OllamaClient
from .models import LLMConfig, SearchCriteria, StudyNarrowing
from .normalizer import normalize


log = logging.getLogger(__name__)


SYSTEM_PROMPT = """Voce e um assistente que converte consultas em linguagem natural para criterios estruturados de busca DICOM.

Sua tarefa e extrair criterios de busca do texto fornecido e retornar APENAS um objeto JSON valido seguindo este schema:

{
  "patient": {
    "sex": "F" | "M" | "O" | null,
    "age_min": <numero inteiro ou null>,
    "age_max": <numero inteiro ou null>
  },
  "head_keywords": ["lista", "de", "keywords"],
  "required_series": [
    {
      "name": "identificador da serie",
      "modality": "MR" | "CT" | null,
      "within_head": true | false,
      "all_keywords": ["todos", "obrigatorios"],
      "any_keywords": ["qualquer", "um"]
    }
  ],
  "study_narrowing": {
    "modality_in_study": ["MR", "CT"] | null,
    "study_description_keywords": ["keywords"] | null
  }
}

Regras:
1. Retorne APENAS o JSON, sem explicacoes adicionais
2. Use null para campos nao mencionados na consulta
3. Para sexo: "mulher/feminino" = "F", "homem/masculino" = "M", "outro" = "O"
4. Para cranio/cabeca: inclua em head_keywords se mencionado
5. Para series especificas (ex: "axial MT pos"), crie um SeriesRequirement
6. Normalize keywords para lowercase sem acentos
7. Se a consulta menciona faixa etaria (ex: "20 a 40 anos"), use age_min e age_max
"""

_MODALITY_SYNONYMS = {
    "MR": {"rm", "mri", "ressonancia", "ressonancia magnetica"},
    "CT": {"ct", "tc", "tomografia", "tomografia computadorizada"},
    "US": {"us", "ultrassom", "ultrasom"},
    "CR": {"rx", "raio x", "raiox", "radiografia"},
}


def _infer_modalities(query: str, criteria: SearchCriteria) -> SearchCriteria:
    if criteria.study_narrowing and criteria.study_narrowing.modality_in_study:
        return criteria

    modalities: set[str] = set()
    for requirement in criteria.required_series or []:
        if requirement.modality:
            modalities.add(requirement.modality.upper())

    if not modalities:
        normalized = normalize(query)
        tokens = set(re.findall(r"[a-z0-9]+", normalized))
        for modality, synonyms in _MODALITY_SYNONYMS.items():
            for term in synonyms:
                if " " in term:
                    if term in normalized:
                        modalities.add(modality)
                        break
                elif term in tokens:
                    modalities.add(modality)
                    break

    if not modalities:
        return criteria

    narrowing = criteria.study_narrowing or StudyNarrowing()
    if not narrowing.modality_in_study:
        narrowing = narrowing.model_copy(
            update={"modality_in_study": sorted(modalities)}
        )
    updated = criteria.model_copy(update={"study_narrowing": narrowing})
    log.debug(
        "Modalities inferred from query",
        extra={"extra_data": {"modalities": sorted(modalities)}},
    )
    return updated


def extract_json(text: str) -> dict:
    start = text.find("{")
    if start == -1:
        raise ValueError("No JSON object found")

    depth = 0
    in_string = False
    escape = False
    for idx in range(start, len(text)):
        char = text[idx]
        if in_string:
            if escape:
                escape = False
            elif char == "\\":
                escape = True
            elif char == '"':
                in_string = False
            continue

        if char == '"':
            in_string = True
        elif char == "{":
            depth += 1
        elif char == "}":
            depth -= 1
            if depth == 0:
                candidate = text[start : idx + 1]
                try:
                    return json.loads(candidate)
                except json.JSONDecodeError as exc:
                    raise ValueError("Invalid JSON object") from exc

    raise ValueError("No JSON object found")


def parse_nl_to_criteria(query: str, llm: LLMConfig | object) -> SearchCriteria:
    log.debug("Parsing NL query", extra={"extra_data": {"query_length": len(query)}})
    if hasattr(llm, "chat"):
        client = llm
    elif isinstance(llm, LLMConfig):
        client = OllamaClient.from_config(llm)
    else:
        raise TypeError("llm must be an LLMConfig or client with chat()")

    raw = client.chat(SYSTEM_PROMPT, query)
    data = extract_json(raw)
    if isinstance(data, dict):
        required_series = data.get("required_series")
        if isinstance(required_series, list):
            cleaned = []
            for entry in required_series:
                if not isinstance(entry, dict):
                    continue
                name = entry.get("name")
                if not name:
                    continue
                cleaned.append(entry)
            data["required_series"] = cleaned
    try:
        criteria = SearchCriteria.model_validate(data)
    except ValidationError:
        raise
    criteria = _infer_modalities(query, criteria)
    log.debug(
        "NL criteria parsed",
        extra={
            "extra_data": {
                "has_patient": criteria.patient is not None,
                "head_keywords": len(criteria.head_keywords or []),
                "required_series": len(criteria.required_series or []),
            }
        },
    )
    return criteria
