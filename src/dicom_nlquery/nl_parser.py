from __future__ import annotations

import json
import logging
from datetime import date

from pydantic import ValidationError

from .llm_client import OllamaClient
from .models import LLMConfig, SearchCriteria


log = logging.getLogger(__name__)


SYSTEM_PROMPT = """Voce e um assistente que converte consultas em linguagem natural para criterios estruturados de busca DICOM.

Sua tarefa e extrair criterios de busca do texto fornecido e retornar APENAS um objeto JSON valido seguindo este schema (dicom-mcp):

{
  "study": {
    "patient_id": "string" | null,
    "patient_sex": "F" | "M" | "O" | null,
    "patient_birth_date": "YYYYMMDD" | "YYYYMMDD-YYYYMMDD" | null,
    "study_date": "YYYYMMDD" | "YYYYMMDD-YYYYMMDD" | null,
    "modality_in_study": "MR" | "CT" | "US" | "CR" | "MR\\\\CT" | null,
    "study_description": "texto livre" | null,
    "accession_number": "string" | null,
    "study_instance_uid": "string" | null
  },
  "series": {
    "modality": "MR" | "CT" | "US" | "CR" | null,
    "series_number": "string" | null,
    "series_description": "texto livre" | null,
    "series_instance_uid": "string" | null
  } | null
}

Regras:
1. Retorne APENAS o JSON, sem explicacoes adicionais
2. Use null para campos nao mencionados na consulta
3. Para sexo: "mulher/feminino" = "F", "homem/masculino" = "M", "outro" = "O"
4. Datas devem estar no formato DICOM YYYYMMDD ou intervalo YYYYMMDD-YYYYMMDD
5. Para modalidades, use codigos DICOM (ex: MR, CT, US, CR). Use barra invertida para multiplas modalidades
6. Para texto livre de exame/parte do corpo (ex: cranio, abdome), prefira study_description
7. Use campos de series SOMENTE se a consulta pedir explicitamente algo de serie (ex: numero de serie, sequencia, fase, serie X). Caso contrario, use series=null
8. Voce pode usar wildcard "*" nos campos de descricao se fizer sentido
9. Se a consulta mencionar idade/faixa etaria, converta para patient_birth_date usando a data atual
10. Nao invente modalidades: use apenas as que forem citadas no texto
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

    system_prompt = f"{SYSTEM_PROMPT}\nHoje: {date.today():%Y-%m-%d}"
    raw = client.chat(system_prompt, query)
    data = extract_json(raw)
    try:
        criteria = SearchCriteria.model_validate(data)
    except ValidationError:
        raise
    log.debug(
        "NL criteria parsed",
        extra={
            "extra_data": {
                "has_study_filters": any(
                    [
                        criteria.study.patient_id,
                        criteria.study.patient_sex,
                        criteria.study.patient_birth_date,
                        criteria.study.study_date,
                        criteria.study.modality_in_study,
                        criteria.study.study_description,
                        criteria.study.accession_number,
                        criteria.study.study_instance_uid,
                    ]
                ),
                "has_series_filters": criteria.series is not None,
            }
        },
    )
    return criteria
