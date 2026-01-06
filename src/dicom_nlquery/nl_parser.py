from __future__ import annotations

import json
import logging

from pydantic import ValidationError

from .llm_client import OllamaClient
from .models import LLMConfig, SearchCriteria


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
      "modality": "codigo de modalidade DICOM" | null,
      "within_head": true | false,
      "all_keywords": ["todos", "obrigatorios"],
      "any_keywords": ["qualquer", "um"]
    }
  ],
  "study_narrowing": {
    "modality_in_study": ["codigos", "de", "modalidade"] | null,
    "study_description_keywords": ["keywords"] | null
  }
}

Regras:
1. Retorne APENAS o JSON, sem explicacoes adicionais
2. Use null para campos nao mencionados na consulta
3. Para sexo: "mulher/feminino" = "F", "homem/masculino" = "M", "outro" = "O"
4. Se a consulta indicar filtro por cabeca, use head_keywords com termos relevantes da consulta
5. Para series especificas (ex: "axial MT pos"), crie um SeriesRequirement
6. Normalize keywords para lowercase sem acentos
7. Se a consulta menciona faixa etaria (ex: "20 a 40 anos"), use age_min e age_max
8. Para modalidades, use codigos DICOM (ex: MR, CT, US, CR) quando mencionados
"""


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
    log.debug(
        "NL criteria parsed",
        extra={
            "extra_data": {
                "has_patient": criteria.patient is not None,
                "required_series": len(criteria.required_series or []),
                "has_study_narrowing": bool(
                    criteria.study_narrowing
                    and (
                        criteria.study_narrowing.modality_in_study
                        or criteria.study_narrowing.study_description_keywords
                    )
                ),
            }
        },
    )
    return criteria
