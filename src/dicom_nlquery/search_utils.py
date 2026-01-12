from __future__ import annotations

import re
from typing import Callable, Iterable

from .models import SearchCriteria

STUDY_TEXT_FIELDS = (
    "StudyDescription",
    "RequestedProcedureDescription",
    "ReferringPhysicianName",
    "RequestingPhysician",
    "InstitutionName",
    "Manufacturer",
    "ManufacturerModelName",
    "BodyPartExamined",
    "ProtocolName",
    "RequestedProcedureCodeSequence",
    "RequestAttributesSequence",
)
SERIES_TEXT_FIELDS = (
    "SeriesDescription",
    "BodyPartExamined",
    "ProtocolName",
    "Modality",
    "Manufacturer",
    "ManufacturerModelName",
    "RequestedProcedureDescription",
    "RequestedProcedureCodeSequence",
    "RequestAttributesSequence",
)
STUDY_DESCRIPTION_EXTRA_ATTRS = (
    "Modality",
    "Manufacturer",
    "InstitutionName",
    "ReferringPhysicianName",
    "RequestingPhysician",
    "StudyDescription",
    "SeriesDescription",
    "ManufacturerModelName",
    "BodyPartExamined",
    "ProtocolName",
    "RequestedProcedureDescription",
    "RequestedProcedureCodeSequence",
    "RequestAttributesSequence",
)
SERIES_DESCRIPTION_EXTRA_ATTRS = (
    "Modality",
    "Manufacturer",
    "SeriesDescription",
    "ManufacturerModelName",
    "BodyPartExamined",
    "ProtocolName",
    "RequestedProcedureDescription",
    "RequestedProcedureCodeSequence",
    "RequestAttributesSequence",
)


def _get_attr(obj: object, key: str) -> object | None:
    if isinstance(obj, dict):
        return obj.get(key)
    return getattr(obj, key, None)


def _normalize_str(value: object | None) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def _tokenize_needle(needle: str) -> list[str]:
    return [token for token in re.split(r"[\s*?\^]+", needle.casefold()) if token]


def _contains_casefold(haystack: object | None, needle: str) -> bool:
    if haystack is None:
        return False
    hay = str(haystack).casefold()
    tokens = _tokenize_needle(needle)
    if not tokens:
        return True
    pos = 0
    for token in tokens:
        idx = hay.find(token, pos)
        if idx == -1:
            return False
        pos = idx + len(token)
    return True


def _contains_casefold_unordered(haystack: object | None, tokens: Iterable[str]) -> bool:
    if haystack is None:
        return False
    hay = str(haystack).casefold()
    return all(token in hay for token in tokens)


def _iter_text_values(value: object | None) -> Iterable[str]:
    if value is None:
        return
    if isinstance(value, dict):
        for item in value.values():
            yield from _iter_text_values(item)
        return
    if isinstance(value, (list, tuple, set)):
        for item in value:
            yield from _iter_text_values(item)
        return
    text = str(value).strip()
    if text:
        yield text


def _contains_casefold_any(haystack: object | None, needle: str) -> bool:
    tokens = _tokenize_needle(needle)
    if not tokens:
        return True
    for text in _iter_text_values(haystack):
        if _contains_casefold(text, needle):
            return True
        if len(tokens) > 1 and _contains_casefold_unordered(text, tokens):
            return True
    return False


def _matches_any_field(item: object, fields: Iterable[str], needle: str) -> bool:
    for field in fields:
        if _contains_casefold_any(_get_attr(item, field), needle):
            return True
    return False


def _date_matches(candidate: str | None, filter_value: str) -> bool:
    if candidate is None:
        return False
    cand = candidate.strip()
    if not cand:
        return False
    filt = filter_value.strip()
    if "-" in filt:
        start, end = filt.split("-", 1)
        start = start.strip()
        end = end.strip()
        if start and cand < start:
            return False
        if end and cand > end:
            return False
        return True
    return cand == filt


def _study_matches_criteria(
    study: object,
    criteria: SearchCriteria,
    explicit_study_date: str | None,
    description_override: str | None = None,
    description_matcher: Callable[[object], bool] | None = None,
    skip_description: bool = False,
) -> bool:
    filters = criteria.study

    if filters.patient_id:
        if _normalize_str(_get_attr(study, "PatientID")) != filters.patient_id:
            return False

    if filters.patient_name:
        candidate_name = _get_attr(study, "PatientName")
        if candidate_name is not None and not _contains_casefold(candidate_name, filters.patient_name):
            return False

    if filters.patient_sex:
        if _normalize_str(_get_attr(study, "PatientSex")) != filters.patient_sex:
            return False

    if filters.patient_birth_date:
        candidate_birth_date = _normalize_str(_get_attr(study, "PatientBirthDate"))
        if candidate_birth_date is None or not _date_matches(
            candidate_birth_date, filters.patient_birth_date
        ):
            return False

    if explicit_study_date:
        candidate_date = _normalize_str(_get_attr(study, "StudyDate"))
        if candidate_date is None or not _date_matches(candidate_date, explicit_study_date):
            return False

    if filters.modality_in_study:
        raw_modalities = _get_attr(study, "ModalitiesInStudy")
        if raw_modalities is None:
            return False
        if isinstance(raw_modalities, str):
            values = [m.strip().upper() for m in raw_modalities.split("\\") if m.strip()]
        else:
            values = [str(m).strip().upper() for m in raw_modalities if str(m).strip()]
        if filters.modality_in_study.upper() not in values:
            return False

    description_value = description_override
    if not skip_description:
        if description_matcher is not None:
            if not description_matcher(study):
                return False
        else:
            if description_value is None:
                description_value = filters.study_description
            if description_value:
                if not _matches_any_field(study, STUDY_TEXT_FIELDS, description_value):
                    return False

    if filters.accession_number:
        if _normalize_str(_get_attr(study, "AccessionNumber")) != filters.accession_number:
            return False

    if filters.study_instance_uid:
        if _normalize_str(_get_attr(study, "StudyInstanceUID")) != filters.study_instance_uid:
            return False

    return True


def _series_matches_criteria(series: object, criteria: SearchCriteria) -> bool:
    if criteria.series is None:
        return True
    filters = criteria.series

    if filters.series_instance_uid:
        if _normalize_str(_get_attr(series, "SeriesInstanceUID")) != filters.series_instance_uid:
            return False

    if filters.modality:
        if _normalize_str(_get_attr(series, "Modality")) != filters.modality:
            return False

    if filters.series_number:
        if _normalize_str(_get_attr(series, "SeriesNumber")) != filters.series_number:
            return False

    if filters.series_description:
        if not _contains_casefold(
            _get_attr(series, "SeriesDescription"),
            filters.series_description,
        ):
            return False

    return True


def _build_study_args(
    criteria: SearchCriteria,
    study_date: str | None,
    study_description_override: str | None = None,
    include_description: bool = True,
) -> dict[str, object]:
    args = criteria.study.model_dump(exclude_none=True)
    if study_date:
        args["study_date"] = study_date
    if not include_description:
        args.pop("study_description", None)
    if study_description_override is not None:
        args["study_description"] = study_description_override
    if criteria.study.study_description or study_description_override is not None:
        extras = list(args.get("additional_attributes") or [])
        for attr in STUDY_DESCRIPTION_EXTRA_ATTRS:
            if attr not in extras:
                extras.append(attr)
        args["additional_attributes"] = extras
    if "patient_name" in args:
        name = str(args["patient_name"]).strip()
        if name and "*" not in name and "?" not in name:
            args["patient_name"] = f"*{name.replace(' ', '*')}*"
    if "study_description" in args:
        desc = str(args["study_description"]).strip()
        if desc and "*" not in desc and "?" not in desc:
            args["study_description"] = f"*{desc}*"
    return args


def _build_series_args(criteria: SearchCriteria) -> dict[str, object]:
    args: dict[str, object] = {}
    if _has_series_filters(criteria):
        assert criteria.series is not None
        args.update(criteria.series.model_dump(exclude_none=True))
    if criteria.study.study_description:
        extras = list(args.get("additional_attributes") or [])
        for attr in SERIES_DESCRIPTION_EXTRA_ATTRS:
            if attr not in extras:
                extras.append(attr)
        args["additional_attributes"] = extras
    return args


def _has_series_filters(criteria: SearchCriteria) -> bool:
    if criteria.series is None:
        return False
    return any(
        [
            criteria.series.modality,
            criteria.series.series_number,
            criteria.series.series_description,
            criteria.series.series_instance_uid,
        ]
    )
