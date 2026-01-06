from __future__ import annotations

from datetime import date

from .normalizer import normalize
from .models import SearchCriteria, SeriesRequirement


def _parse_date(value: str | None) -> date | None:
    if not value:
        return None
    if len(value) < 8:
        return None
    try:
        return date(int(value[:4]), int(value[4:6]), int(value[6:8]))
    except (TypeError, ValueError):
        return None


def calculate_age(birth_date: str | None, study_date: str | None) -> int | None:
    birth = _parse_date(birth_date)
    study = _parse_date(study_date)
    if not birth or not study:
        return None
    age = study.year - birth.year
    if (study.month, study.day) < (birth.month, birth.day):
        age -= 1
    return age


def _get_attr(obj: object, key: str) -> object | None:
    if isinstance(obj, dict):
        return obj.get(key)
    return getattr(obj, key, None)


def contains_keyword(text: str | None, keyword: str | None) -> bool:
    if not text or not keyword:
        return False
    return normalize(keyword) in normalize(text)


def series_matches(
    series: object,
    requirement: SeriesRequirement,
    head_keywords: list[str] | None = None,
) -> bool:
    modality = _get_attr(series, "Modality")
    if requirement.modality and (not modality or str(modality).upper() != requirement.modality.upper()):
        return False

    description = _get_attr(series, "SeriesDescription")
    description_text = str(description) if description is not None else ""

    if requirement.within_head:
        if not head_keywords:
            return False
        if not any(contains_keyword(description_text, kw) for kw in head_keywords):
            return False

    if requirement.all_keywords:
        if not all(contains_keyword(description_text, kw) for kw in requirement.all_keywords):
            return False

    if requirement.any_keywords:
        if not any(contains_keyword(description_text, kw) for kw in requirement.any_keywords):
            return False

    return True


def study_matches(
    study: object,
    series_list: list[object],
    criteria: SearchCriteria,
    head_keywords: list[str] | None = None,
) -> bool:
    active_head_keywords = criteria.head_keywords or head_keywords

    if active_head_keywords:
        study_desc = _get_attr(study, "StudyDescription")
        study_text = str(study_desc) if study_desc is not None else ""
        has_head = any(contains_keyword(study_text, kw) for kw in active_head_keywords)
        if not has_head:
            for series in series_list:
                series_desc = _get_attr(series, "SeriesDescription")
                series_text = str(series_desc) if series_desc is not None else ""
                if any(contains_keyword(series_text, kw) for kw in active_head_keywords):
                    has_head = True
                    break
        if not has_head:
            return False

    if criteria.required_series:
        if not series_list:
            return False
        for requirement in criteria.required_series:
            if not any(series_matches(series, requirement, head_keywords=active_head_keywords) for series in series_list):
                return False

    if criteria.study_narrowing:
        narrowing = criteria.study_narrowing
        if narrowing.modality_in_study:
            modalities_value = _get_attr(study, "ModalitiesInStudy")
            if isinstance(modalities_value, str):
                modalities = {m.strip().upper() for m in modalities_value.split("\\\\") if m.strip()}
            elif modalities_value is None:
                modalities = set()
            else:
                modalities = {str(m).upper() for m in modalities_value}
            required = {m.upper() for m in narrowing.modality_in_study}
            if not required.issubset(modalities):
                return False

        if narrowing.study_description_keywords:
            study_desc = _get_attr(study, "StudyDescription")
            study_text = str(study_desc) if study_desc is not None else ""
            if not all(contains_keyword(study_text, kw) for kw in narrowing.study_description_keywords):
                return False

    return True
