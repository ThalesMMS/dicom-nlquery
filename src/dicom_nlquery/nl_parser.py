from __future__ import annotations

import json
import logging
import re
import unicodedata
from datetime import date, timedelta
from typing import Any

import yaml
from pydantic import ValidationError

from .llm_client import LLMClient, OllamaClient, create_llm_client
from .models import DATE_RE, LLMConfig, QueryStudiesArgs, SearchCriteria


log = logging.getLogger(__name__)


SYSTEM_PROMPT = """You convert a natural-language imaging request into structured DICOM search criteria.

Return ONLY a valid JSON object that matches this schema:

{
  "study": {
    "patient_id": "string" | null,
    "patient_name": "string" | null,
    "patient_sex": "F" | "M" | "O" | null,
    "patient_birth_date": "YYYYMMDD" | "YYYYMMDD-YYYYMMDD" | null,
    "study_date": "YYYYMMDD" | "YYYYMMDD-YYYYMMDD" | null,
    "modality_in_study": "MR" | "CT" | "US" | "CR" | "MR\\\\CT" | null,
    "study_description": "free text" | null,
    "accession_number": "string" | null,
    "study_instance_uid": "string" | null
  },
  "series": null
}

Keys:
- study: patient_id, patient_name, patient_sex(F/M/O), patient_birth_date(YYYYMMDD or range),
  study_date(YYYYMMDD or range), modality_in_study(MR/CT/US/CR or MR\\\\CT),
  study_description, accession_number, study_instance_uid
- series: modality, series_number, series_description, series_instance_uid (omit unless explicitly requested)

Rules:
1) Always return ONLY the JSON object (no prose, no markdown).
2) Use null for fields not mentioned.
3) Use DICOM dates and modality codes.
4) Put body-part terms and procedure/exam terms mentioned by the user into study.study_description.
5) When using study_description, prefer wildcard matching by wrapping key terms with "*" (e.g. "*term*").
6) If the query includes routing ("from X to Y"), X/Y are node names and MUST NOT appear inside filters.
7) Age/range -> patient_birth_date using Today: YYYY-MM-DD.
8) Do not invent filters; never default StudyDate.

Examples (return only the JSON object, not the example labels):

Query: "CT chest angiogram from NODE_A to NODE_B between 2000 and 2022"
Output:
{"study":{"patient_id":null,"patient_name":null,"patient_sex":null,"patient_birth_date":null,"study_date":"20000101-20221231","modality_in_study":"CT","study_description":"*chest*angiogram*","accession_number":null,"study_instance_uid":null},"series":null}

Query: "MR brain study for patient ID 12345 on 20230101"
Output:
{"study":{"patient_id":"12345","patient_name":null,"patient_sex":null,"patient_birth_date":null,"study_date":"20230101","modality_in_study":"MR","study_description":"*brain*study*","accession_number":null,"study_instance_uid":null},"series":null}

Query: "ultrasound abdomen exams for female patients from NODE_X to NODE_Y"
Output:
{"study":{"patient_id":null,"patient_name":null,"patient_sex":"F","patient_birth_date":null,"study_date":null,"modality_in_study":"US","study_description":"*abdomen*exam*","accession_number":null,"study_instance_uid":null},"series":null}
"""


def _build_response_schema() -> dict[str, Any]:
    study_props = {
        "patient_id": {"type": ["string", "null"]},
        "patient_name": {"type": ["string", "null"]},
        "patient_sex": {"type": ["string", "null"], "enum": ["F", "M", "O", None]},
        "patient_birth_date": {"type": ["string", "null"]},
        "study_date": {"type": ["string", "null"]},
        "modality_in_study": {
            "type": ["string", "null"],
            "enum": ["MR", "CT", "US", "CR", "MR\\CT", None],
        },
        "study_description": {"type": ["string", "null"]},
        "accession_number": {"type": ["string", "null"]},
        "study_instance_uid": {"type": ["string", "null"]},
    }
    series_props = {
        "modality": {"type": ["string", "null"]},
        "series_number": {"type": ["string", "null"]},
        "series_description": {"type": ["string", "null"]},
        "series_instance_uid": {"type": ["string", "null"]},
    }
    return {
        "type": "object",
        "additionalProperties": False,
        "properties": {
            "study": {
                "type": "object",
                "additionalProperties": False,
                "properties": study_props,
                "required": list(study_props.keys()),
            },
            "series": {
                "type": ["object", "null"],
                "additionalProperties": False,
                "properties": series_props,
                "required": list(series_props.keys()),
            },
        },
        "required": ["study", "series"],
    }


RESPONSE_SCHEMA = _build_response_schema()

SEX_EVIDENCE = {
    "M": ["male", "man", "men"],
    "F": ["female", "woman", "women", "pregnant"],
    "O": ["other", "nonbinary", "non-binary"],
}
MODALITY_EVIDENCE = {
    "MR": ["mr", "mri", "magnetic resonance"],
    "CT": ["ct", "computed tomography", "cat"],
    "US": ["us", "ultrasound", "sonogram", "sonography"],
    "CR": ["x-ray", "xray", "radiograph", "radiography"],
    "DX": ["x-ray", "xray", "radiograph", "radiography"],
    "SR": ["report", "structured report", "sr"],
    "PDF": ["pdf", "document"],
}
STOPWORDS = {
    "of",
    "the",
    "for",
    "to",
    "in",
    "on",
    "with",
    "without",
    "and",
    "or",
    "a",
    "an",
}
FALLBACK_DESCRIPTION_STOPWORDS = {
    "age",
    "ages",
    "between",
    "during",
    "exam",
    "exams",
    "from",
    "patient",
    "patients",
    "study",
    "studies",
    "through",
    "until",
    "year",
    "years",
}
BODY_PART_TERMS = {"pelvis", "pelvic"}
DATE_HINT_RE = re.compile(
    r"\b(today|yesterday|week(s)?|month(s)?|day(s)?|last|past|recent|recently)\b"
)
DATE_TOKEN_RE = re.compile(r"\b\d{8}\b|\b\d{4}[-/]\d{2}[-/]\d{2}\b|\b\d{8}-\d{8}\b")
AGE_RANGE_RE = re.compile(r"\b\d{1,3}\s*(?:to|-)\s*\d{1,3}\s*(?:years?|yrs?|yr)?\b")
AGE_RE = re.compile(r"\b\d{1,3}\s*(?:years?|yrs?|yr|y/o|yo)\b")
AGE_RANGE_CAPTURE_RE = re.compile(
    r"\b(\d{1,3})\s*(?:to|-)\s*(\d{1,3})\s*(?:years?|yrs?|yr|y/o|yo)?\b"
)
AGE_CAPTURE_RE = re.compile(r"\b(\d{1,3})\s*(?:years?|yrs?|yr|y/o|yo)\b")
YEAR_RANGE_RE = re.compile(
    r"\b(?:from|between)?\s*(?:year\s*)?(\d{4})\s*"
    r"(?:to|until|through|and|-)\s*(?:year\s*)?(\d{4})\b"
)
YEAR_SINCE_RE = re.compile(r"\b(?:since|from)\s*(?:year\s*)?(\d{4})\b")
YEAR_ONLY_RE = re.compile(r"\b(?:in|during|year)\s*(\d{4})\b")
SOURCE_RE = re.compile(r"\bfrom\s+([a-z0-9_-]+)\b")
DESTINATION_RE = re.compile(r"\bto\s+([a-z0-9_-]+)\b")
NAME_START_RE = re.compile(r"\b(?:exams?|studies)\s+(?:of|for)\s+(.+)$")
PATIENT_START_RE = re.compile(r"\b(?:patient|patients)\s+(?:of\s+)?(.+)$")
NAME_CLAUSE_BREAK_RE = re.compile(
    r"\b(to|with|without|in|sex|female|male|pregnant|age|years?|months?|days?"
    r"|mr|mri|ct|computed|tomography|ultrasound|x[- ]?ray|radiograph|cr|dx)\b"
)
NAME_BLOCKLIST = {
    "patient",
    "patients",
    "sex",
    "female",
    "male",
    "pregnant",
    "exam",
    "exams",
    "study",
    "studies",
}
NAME_PREPOSITIONS = {"of", "for"}
SERIES_EVIDENCE_RE = re.compile(
    r"\b(series(?:\s*(?:number|no|#))?|sequence(?:\s*(?:number|no|#))?|phase)\b"
)


def _normalize_text(text: str) -> str:
    normalized = unicodedata.normalize("NFKD", text)
    return "".join(c for c in normalized if not unicodedata.combining(c)).lower()


def _normalize_token(token: str) -> str:
    normalized = _normalize_text(token)
    return re.sub(r"[^a-z0-9]+", "", normalized)


def _is_wildcard_only(value: str | None) -> bool:
    if value is None:
        return False
    stripped = value.strip()
    if not stripped:
        return False
    collapsed = stripped.replace(" ", "")
    return bool(collapsed) and all(char in "*?" for char in collapsed)


def _extract_body_part_from_query(query_norm: str) -> str | None:
    tokens = [token for token in query_norm.split() if token]
    for token in tokens:
        normalized = _normalize_token(token)
        if not normalized or normalized in STOPWORDS:
            continue
        if normalized in BODY_PART_TERMS:
            return normalized
    return None


def _extract_routing_tokens(query_norm: str) -> set[str]:
    tokens: set[str] = set()
    for regex in (SOURCE_RE, DESTINATION_RE):
        for match in regex.finditer(query_norm):
            token = _normalize_token(match.group(1))
            if token:
                tokens.add(token)
    return tokens


def _extract_fallback_description(query_norm: str, study: dict[str, object]) -> str | None:
    remove_tokens = set(STOPWORDS)
    remove_tokens.update(FALLBACK_DESCRIPTION_STOPWORDS)
    remove_tokens.update(_extract_routing_tokens(query_norm))
    for tokens in SEX_EVIDENCE.values():
        for token in tokens:
            normalized = _normalize_token(token)
            if normalized:
                remove_tokens.add(normalized)
    for tokens in MODALITY_EVIDENCE.values():
        for token in tokens:
            normalized = _normalize_token(token)
            if normalized:
                remove_tokens.add(normalized)
    patient_name = study.get("patient_name")
    if isinstance(patient_name, str) and patient_name.strip():
        for token in _normalize_text(patient_name).split():
            normalized = _normalize_token(token)
            if normalized:
                remove_tokens.add(normalized)

    filtered: list[str] = []
    seen: set[str] = set()
    for raw_token in query_norm.split():
        normalized = _normalize_token(raw_token)
        if not normalized or normalized in remove_tokens:
            continue
        if not any(char.isalpha() for char in normalized):
            continue
        if normalized in seen:
            continue
        seen.add(normalized)
        filtered.append(normalized)
    if not filtered:
        return None
    return "*" + "*".join(filtered) + "*"


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
    return bool(
        DATE_TOKEN_RE.search(query_norm)
        or DATE_HINT_RE.search(query_norm)
        or YEAR_RANGE_RE.search(query_norm)
        or YEAR_SINCE_RE.search(query_norm)
        or YEAR_ONLY_RE.search(query_norm)
    )


def _subtract_years(value: date, years: int) -> date:
    target_year = value.year - years
    try:
        return value.replace(year=target_year)
    except ValueError:
        return value.replace(year=target_year, month=2, day=28)


def _birth_date_range_for_ages(
    min_age: int, max_age: int, today: date
) -> str | None:
    if min_age < 0 or max_age < 0:
        return None
    if min_age > max_age:
        min_age, max_age = max_age, min_age
    if min_age > 130 or max_age > 130:
        return None
    earliest = _subtract_years(today, max_age + 1) + timedelta(days=1)
    latest = _subtract_years(today, min_age)
    if earliest > latest:
        return None
    return f"{earliest:%Y%m%d}-{latest:%Y%m%d}"


def _extract_birth_date_from_query(query_norm: str, today: date | None = None) -> str | None:
    today_value = today or date.today()
    match = AGE_RANGE_CAPTURE_RE.search(query_norm)
    if match:
        return _birth_date_range_for_ages(int(match.group(1)), int(match.group(2)), today_value)
    match = AGE_CAPTURE_RE.search(query_norm)
    if match:
        return _birth_date_range_for_ages(int(match.group(1)), int(match.group(1)), today_value)
    return None


def _format_year_range(start_year: int, end_year: int) -> str:
    if start_year > end_year:
        start_year, end_year = end_year, start_year
    return f"{start_year:04d}0101-{end_year:04d}1231"


def _extract_study_date_from_query(query_norm: str, today: date | None = None) -> str | None:
    today_value = today or date.today()
    match = re.search(r"\b(\d{8})\s*-\s*(\d{8})\b", query_norm)
    if match:
        return f"{match.group(1)}-{match.group(2)}"
    tokens = re.findall(r"\b\d{8}\b|\b\d{4}[-/]\d{2}[-/]\d{2}\b", query_norm)
    if len(tokens) >= 2:
        start = _normalize_study_date(tokens[0])
        end = _normalize_study_date(tokens[1])
        if start and end:
            return f"{start}-{end}"
    if len(tokens) == 1:
        normalized = _normalize_study_date(tokens[0])
        if normalized:
            return normalized
    match = YEAR_RANGE_RE.search(query_norm)
    if match:
        return _format_year_range(int(match.group(1)), int(match.group(2)))
    match = YEAR_SINCE_RE.search(query_norm)
    if match:
        return f"{int(match.group(1)):04d}0101-{today_value:%Y%m%d}"
    match = YEAR_ONLY_RE.search(query_norm)
    if match:
        year = int(match.group(1))
        return f"{year:04d}0101-{year:04d}1231"
    return None


def _has_series_evidence(query_norm: str) -> bool:
    return bool(SERIES_EVIDENCE_RE.search(query_norm))


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
    stripped = description.strip()
    prefix = ""
    for char in stripped:
        if char in "*?":
            prefix += char
        else:
            break
    suffix = ""
    for char in reversed(stripped):
        if char in "*?":
            suffix = char + suffix
        else:
            break
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
    cleaned = " ".join(filtered)
    if prefix and cleaned and cleaned[0] not in "*?":
        cleaned = f"{prefix}{cleaned}"
    if suffix and cleaned and cleaned[-1] not in "*?":
        cleaned = f"{cleaned}{suffix}"
    return cleaned


def apply_evidence_guardrails(
    criteria: SearchCriteria | dict[str, object], query: str
) -> SearchCriteria:
    query_norm = _normalize_text(query)
    if isinstance(criteria, SearchCriteria):
        data = criteria.model_dump()
    else:
        data = dict(criteria)
    study = dict(data.get("study") or {})
    series = data.get("series")
    if isinstance(series, dict):
        if _is_wildcard_only(series.get("series_description")):
            series["series_description"] = None
        if not _has_series_evidence(query_norm):
            data["series"] = None
            series = None

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
    extracted_birth_date = _extract_birth_date_from_query(query_norm)
    if extracted_birth_date:
        study["patient_birth_date"] = extracted_birth_date
    extracted_study_date = _extract_study_date_from_query(query_norm)
    if extracted_study_date:
        study["study_date"] = extracted_study_date
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
    if study.get("study_description") and _is_wildcard_only(study["study_description"]):
        study["study_description"] = None
    if not study.get("study_description"):
        body_part = _extract_body_part_from_query(query_norm)
        if body_part:
            study["study_description"] = body_part
    if not study.get("study_description"):
        fallback = _extract_fallback_description(query_norm, study)
        if fallback:
            study["study_description"] = fallback

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
        else:
            data["series"] = series

    if not has_study_filters and data.get("series") is None:
        raise ValueError("Query does not specify valid filters.")

    try:
        return SearchCriteria.model_validate(data)
    except ValidationError as exc:
        raise ValueError("Query does not specify valid filters.") from exc


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
                    if _has_missing_values(candidate):
                        raise ValueError("Invalid JSON object") from exc
                    try:
                        data = yaml.safe_load(candidate)
                    except yaml.YAMLError as yaml_exc:
                        raise ValueError("Invalid JSON object") from yaml_exc
                    if not isinstance(data, dict):
                        raise ValueError("Invalid JSON object") from exc
                    return data

    raise ValueError("No JSON object found")


def _has_missing_values(candidate: str) -> bool:
    in_double = False
    in_single = False
    escape = False
    for idx, char in enumerate(candidate):
        if in_double:
            if escape:
                escape = False
            elif char == "\\":
                escape = True
            elif char == '"':
                in_double = False
            continue
        if in_single:
            if char == "'":
                in_single = False
            continue
        if char == '"':
            in_double = True
            continue
        if char == "'":
            in_single = True
            continue
        if char == ":":
            j = idx + 1
            while j < len(candidate) and candidate[j].isspace():
                j += 1
            if j < len(candidate) and candidate[j] in ",}]":
                return True
    return False


def parse_nl_to_criteria(
    query: str,
    llm: LLMConfig | LLMClient | object,
    strict_evidence: bool = False,
    debug: bool = False,
) -> SearchCriteria:
    log.debug("Parsing NL query", extra={"extra_data": {"query_length": len(query)}})
    if hasattr(llm, "chat"):
        client = llm
    elif isinstance(llm, LLMConfig):
        client = create_llm_client(llm)
    else:
        raise TypeError("llm must be an LLMConfig or client with chat()")

    system_prompt = SYSTEM_PROMPT
    query_norm = _normalize_text(query)
    if AGE_RANGE_RE.search(query_norm) or AGE_RE.search(query_norm):
        system_prompt = f"{SYSTEM_PROMPT}\nToday: {date.today():%Y-%m-%d}"
    # Use json_mode for clients that support it
    json_schema = None
    response_format = getattr(client, "response_format", None)
    if response_format in {"json_schema", "auto"}:
        json_schema = RESPONSE_SCHEMA
    raw = client.chat(system_prompt, query, json_mode=True, json_schema=json_schema)
    try:
        data = extract_json(raw)
    except ValueError as exc:
        if debug:
            log.info(
                "LLM raw response",
                extra={"extra_data": {"response": raw}},
            )
        raise ValueError(
            "LLM response did not contain a JSON object. "
            "Verify stop tokens/max_tokens or rerun with --llm-debug."
        ) from exc
    if not isinstance(data.get("study"), dict):
        data["study"] = {}
    if "series" not in data:
        data["series"] = None
    elif data["series"] is not None and not isinstance(data["series"], dict):
        data["series"] = None
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
