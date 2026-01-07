from __future__ import annotations

import json
import logging
import re
import unicodedata
from datetime import date

from pydantic import ValidationError

from .llm_client import OllamaClient
from .models import DATE_RE, LLMConfig, QueryStudiesArgs, SearchCriteria


log = logging.getLogger(__name__)


SYSTEM_PROMPT = """Voce e um assistente que converte consultas em linguagem natural para criterios estruturados de busca DICOM.

Sua tarefa e extrair criterios de busca do texto fornecido e retornar APENAS um objeto JSON valido seguindo este schema (dicom-mcp):

{
  "study": {
    "patient_id": "string" | null,
    "patient_name": "string" | null,
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
11. Nao inferir filtros. Se o usuario nao declarou explicitamente sexo, data, modalidade, descricao, IDs ou nome, retorne null.
12. Nunca use a data de hoje como StudyDate padrão.
"""

SEX_EVIDENCE = {
    "M": ["masculino", "homem", "male"],
    "F": ["feminino", "mulher", "female", "gestante"],
    "O": ["outro", "outros", "other"],
}
MODALITY_EVIDENCE = {
    "MR": ["rm", "ressonancia", "mri"],
    "CT": ["tc", "tomografia", "ct"],
    "US": ["us", "ultrassom", "ultrasom", "usg"],
    "CR": ["rx", "raio x", "raio-x", "raiox"],
    "DX": ["rx", "raio x", "raio-x", "raiox"],
    "SR": ["relatorio", "relatorio estruturado", "structured report", "sr"],
    "PDF": ["pdf", "documento"],
}
STOPWORDS = {
    "de",
    "da",
    "do",
    "dos",
    "das",
    "para",
    "pra",
    "pro",
    "no",
    "na",
    "nos",
    "nas",
}
DATE_HINT_RE = re.compile(
    r"\b(hoje|ontem|semana(s)?|mes(es)?|dia(s)?|passad[ao]s?|ultim[ao]s?|recente(s)?|recentemente)\b"
)
DATE_TOKEN_RE = re.compile(r"\b\d{8}\b|\b\d{4}[-/]\d{2}[-/]\d{2}\b|\b\d{8}-\d{8}\b")
AGE_RANGE_RE = re.compile(r"\b\d{1,3}\s*a\s*\d{1,3}\s*anos?\b")
AGE_RE = re.compile(r"\b\d{1,3}\s*anos?\b")
DESTINATION_RE = re.compile(r"\bpara\s+([a-z0-9_-]+)\b")
NAME_START_RE = re.compile(r"\b(?:exames?|estudos?)\s+de\s+(.+)$")
PATIENT_START_RE = re.compile(r"\b(?:paciente|pacientes)\s+(?:de\s+)?(.+)$")
NAME_CLAUSE_BREAK_RE = re.compile(
    r"\b(para|com|sem|no|na|nos|nas|sexo|feminino|masculino|gestante|idade|anos?|ano|meses?|mes|dia|dias"
    r"|rm|mri|tc|ct|tomografia|ressonancia|ultrassom|usg|raio|rx)\b"
)
NAME_BLOCKLIST = {
    "paciente",
    "pacientes",
    "sexo",
    "feminino",
    "masculino",
    "gestante",
    "exame",
    "exames",
    "estudo",
    "estudos",
}
NAME_PREPOSITIONS = {"de", "da", "do", "dos", "das"}


def _normalize_text(text: str) -> str:
    normalized = unicodedata.normalize("NFKD", text)
    return "".join(c for c in normalized if not unicodedata.combining(c)).lower()


def _normalize_token(token: str) -> str:
    normalized = _normalize_text(token)
    return re.sub(r"[^a-z0-9]+", "", normalized)


def _has_id_evidence(value: str, query_norm: str) -> bool:
    value_norm = _normalize_text(value)
    return bool(value_norm) and value_norm in query_norm


def _has_name_evidence(value: str, query_norm: str) -> bool:
    value_norm = _normalize_text(value).replace("^", " ")
    tokens = [token for token in value_norm.split() if len(token) >= 3]
    return any(token in query_norm for token in tokens)


def _has_sex_evidence(value: str, query_norm: str) -> bool:
    tokens = SEX_EVIDENCE.get(value.upper(), [])
    return any(token in query_norm for token in tokens)


def _has_date_evidence(value: str, query_norm: str) -> bool:
    digits = re.findall(r"\d{4}", value)
    if any(d in query_norm for d in digits):
        return True
    return bool(DATE_HINT_RE.search(query_norm))


def _has_birth_date_evidence(value: str, query_norm: str) -> bool:
    digits = re.findall(r"\d{4}", value)
    if any(d in query_norm for d in digits):
        return True
    if AGE_RANGE_RE.search(query_norm) or AGE_RE.search(query_norm):
        return True
    if "nasc" in query_norm:
        return True
    return False


def _query_mentions_date(query_norm: str) -> bool:
    return bool(DATE_TOKEN_RE.search(query_norm) or DATE_HINT_RE.search(query_norm))


def _normalize_study_date(value: str | None) -> str | None:
    if value is None:
        return None
    normalized = value.strip()
    if not normalized:
        return None
    if DATE_RE.match(normalized):
        return normalized
    if re.match(r"^\d{4}[-/]\d{2}[-/]\d{2}$", normalized):
        return normalized.replace("-", "").replace("/", "")
    parts = re.findall(r"\d{4}[-/]\d{2}[-/]\d{2}", normalized)
    if len(parts) == 2:
        start = parts[0].replace("-", "").replace("/", "")
        end = parts[1].replace("-", "").replace("/", "")
        return f"{start}-{end}"
    return None


def _normalize_modality_value(value: str) -> str | None:
    try:
        parsed = QueryStudiesArgs.model_validate({"modality_in_study": value})
    except ValidationError:
        return None
    return parsed.modality_in_study


def _extract_modalities_from_query(query_norm: str) -> list[str]:
    matched: list[str] = []
    for code, tokens in MODALITY_EVIDENCE.items():
        if any(token in query_norm for token in tokens):
            matched.append(code)
    return matched


def _extract_patient_name(query: str) -> str | None:
    query_norm = _normalize_text(query)
    match = NAME_START_RE.search(query_norm) or PATIENT_START_RE.search(query_norm)
    if not match:
        return None
    tail = match.group(1).strip()
    if not tail:
        return None
    tail = NAME_CLAUSE_BREAK_RE.split(tail, maxsplit=1)[0].strip()
    if not tail:
        return None
    tokens = [token for token in tail.split() if token]
    if not tokens:
        return None
    if all(token in NAME_BLOCKLIST or token in NAME_PREPOSITIONS for token in tokens):
        return None
    tokens = [token for token in tokens if token not in NAME_BLOCKLIST]
    if not tokens:
        return None
    return " ".join(tokens).upper()


def _filter_modalities_by_query(value: str, query_norm: str) -> str | None:
    normalized = _normalize_modality_value(value)
    if not normalized:
        return None
    parts = [part for part in normalized.split("\\") if part]
    kept: list[str] = []
    for part in parts:
        tokens = MODALITY_EVIDENCE.get(part.upper(), [])
        if any(token in query_norm for token in tokens):
            if part not in kept:
                kept.append(part)
    if not kept:
        return None
    return "\\".join(kept)


def _strip_modality_tokens(description: str, modalities: list[str]) -> str | None:
    if not description:
        return description
    remove_tokens: set[str] = set()
    for modality in modalities:
        for token in MODALITY_EVIDENCE.get(modality, []):
            normalized = _normalize_token(token)
            if normalized:
                remove_tokens.add(normalized)
    if not remove_tokens:
        return description

    tokens = description.split()
    filtered: list[str] = []
    for token in tokens:
        normalized = _normalize_token(token)
        if normalized in remove_tokens:
            continue
        if normalized in STOPWORDS:
            continue
        filtered.append(token)

    if not filtered:
        return None
    return " ".join(filtered)


def apply_evidence_guardrails(
    criteria: SearchCriteria | dict[str, object], query: str
) -> SearchCriteria:
    query_norm = _normalize_text(query)
    if isinstance(criteria, SearchCriteria):
        data = criteria.model_dump()
    else:
        data = dict(criteria)
    study = dict(data.get("study") or {})

    if study.get("patient_id") and not _has_id_evidence(study["patient_id"], query_norm):
        study["patient_id"] = None
    if study.get("accession_number") and not _has_id_evidence(
        study["accession_number"], query_norm
    ):
        study["accession_number"] = None
    if study.get("study_instance_uid") and not _has_id_evidence(
        study["study_instance_uid"], query_norm
    ):
        study["study_instance_uid"] = None
    if study.get("patient_name") and not _has_name_evidence(
        study["patient_name"], query_norm
    ):
        study["patient_name"] = None
    if study.get("patient_sex") and not _has_sex_evidence(
        study["patient_sex"], query_norm
    ):
        study["patient_sex"] = None
    if study.get("modality_in_study"):
        study["modality_in_study"] = _filter_modalities_by_query(
            study["modality_in_study"], query_norm
        )
    if study.get("study_date") and not _has_date_evidence(study["study_date"], query_norm):
        study["study_date"] = None
    if study.get("patient_birth_date") and not _has_birth_date_evidence(
        study["patient_birth_date"], query_norm
    ):
        study["patient_birth_date"] = None
    if not study.get("modality_in_study"):
        fallback_modalities = _extract_modalities_from_query(query_norm)
        if fallback_modalities:
            study["modality_in_study"] = "\\".join(fallback_modalities)

    modalities = []
    if study.get("modality_in_study"):
        modalities = [part for part in str(study["modality_in_study"]).split("\\") if part]
    if study.get("study_description") and modalities:
        study["study_description"] = _strip_modality_tokens(
            study["study_description"],
            modalities,
        )
    if study.get("patient_name"):
        match = DESTINATION_RE.search(query_norm)
        if match and _normalize_text(study["patient_name"]) == match.group(1):
            study["patient_name"] = None
    if not study.get("patient_name"):
        extracted_name = _extract_patient_name(query)
        if extracted_name:
            study["patient_name"] = extracted_name
    if study.get("study_description"):
        match = DESTINATION_RE.search(query_norm)
        if match and _normalize_text(study["study_description"]) == match.group(1):
            study["study_description"] = None

    has_study_filters = any(
        study.get(field)
        for field in (
            "patient_id",
            "patient_name",
            "patient_sex",
            "patient_birth_date",
            "study_date",
            "modality_in_study",
            "study_description",
            "accession_number",
            "study_instance_uid",
        )
    )
    if not has_study_filters and study.get("modality_in_study"):
        has_study_filters = True

    data["study"] = study
    series = data.get("series")
    if isinstance(series, dict):
        if not any(
            series.get(field)
            for field in (
                "modality",
                "series_number",
                "series_description",
                "series_instance_uid",
            )
        ):
            data["series"] = None

    if not has_study_filters and data.get("series") is None:
        raise ValueError("Consulta nao especifica filtros validos.")

    try:
        return SearchCriteria.model_validate(data)
    except ValidationError as exc:
        raise ValueError("Consulta nao especifica filtros validos.") from exc


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


def parse_nl_to_criteria(
    query: str,
    llm: LLMConfig | object,
    strict_evidence: bool = False,
    debug: bool = False,
) -> SearchCriteria:
    log.debug("Parsing NL query", extra={"extra_data": {"query_length": len(query)}})
    if hasattr(llm, "chat"):
        client = llm
    elif isinstance(llm, LLMConfig):
        client = OllamaClient.from_config(llm)
    else:
        raise TypeError("llm must be an LLMConfig or client with chat()")

    system_prompt = SYSTEM_PROMPT
    query_norm = _normalize_text(query)
    if AGE_RANGE_RE.search(query_norm) or AGE_RE.search(query_norm):
        system_prompt = f"{SYSTEM_PROMPT}\nHoje: {date.today():%Y-%m-%d}"
    raw = client.chat(system_prompt, query)
    data = extract_json(raw)
    if debug:
        log.info("LLM JSON extracted", extra={"extra_data": {"llm_json": data}})
    if strict_evidence:
        criteria = apply_evidence_guardrails(data, query)
    else:
        try:
            criteria = SearchCriteria.model_validate(data)
        except ValidationError:
            raise
    if criteria.study.study_date:
        if _query_mentions_date(query_norm):
            criteria.study.study_date = _normalize_study_date(criteria.study.study_date)
        else:
            criteria.study.study_date = None
    if criteria.study.study_description:
        criteria.study.study_description = _normalize_text(criteria.study.study_description)
    if debug:
        log.info(
            "NL criteria resolved",
            extra={"extra_data": {"criteria": criteria.model_dump()}},
        )

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
