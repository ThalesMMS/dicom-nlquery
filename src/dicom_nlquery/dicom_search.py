from __future__ import annotations

from datetime import date, timedelta
import logging
import time

from .logging_config import mask_phi
from .matcher import calculate_age, study_matches
from .models import GuardrailsConfig, MatchingConfig, SearchCriteria, SearchResult, SearchStats


def _get_attr(obj: object, key: str) -> object | None:
    if isinstance(obj, dict):
        return obj.get(key)
    return getattr(obj, key, None)


def _format_modalities(modalities: list[str]) -> str:
    if len(modalities) == 1:
        return modalities[0]
    return "\\".join(modalities)


def apply_guardrails(
    guardrails: GuardrailsConfig,
    date_range: str | None = None,
    max_studies: int | None = None,
    unlimited: bool = False,
    logger: logging.Logger | None = None,
    today: date | None = None,
) -> tuple[str | None, int | None]:
    log = logger or logging.getLogger(__name__)

    if unlimited:
        log.warning(
            "AVISO: Varredura ilimitada solicitada. Isso pode sobrecarregar o PACS."
        )
        return date_range, None

    effective_date_range = date_range
    if effective_date_range is None:
        end_date = today or date.today()
        start_date = end_date - timedelta(days=guardrails.study_date_range_default_days)
        effective_date_range = f"{start_date:%Y%m%d}-{end_date:%Y%m%d}"

    effective_max_studies = max_studies
    if effective_max_studies is None:
        effective_max_studies = guardrails.max_studies_scanned_default

    return effective_date_range, effective_max_studies


class DicomSearchEngine:
    def __init__(self, dicom_client: object) -> None:
        self._client = dicom_client

    def find_studies(self, criteria: SearchCriteria, date_range: str | None = None) -> list[dict]:
        kwargs: dict[str, object] = {}
        if date_range:
            kwargs["study_date"] = date_range

        if criteria.study_narrowing and criteria.study_narrowing.modality_in_study:
            kwargs["modality"] = _format_modalities(criteria.study_narrowing.modality_in_study)

        if criteria.study_narrowing and criteria.study_narrowing.study_description_keywords:
            kwargs["study_description"] = " ".join(
                criteria.study_narrowing.study_description_keywords
            )

        log = logging.getLogger(__name__)
        if log.isEnabledFor(logging.DEBUG):
            log.debug("C-FIND study filters", extra={"extra_data": {"filters": kwargs}})

        extra_attrs = ["PatientSex", "PatientBirthDate", "ModalitiesInStudy"]
        try:
            return list(self._client.query_study(**kwargs, additional_attrs=extra_attrs))
        except TypeError:
            return list(self._client.query_study(**kwargs))

    def find_series(self, study_instance_uid: str) -> list[dict]:
        return list(self._client.query_series(study_instance_uid=study_instance_uid))


def execute_search(
    criteria: SearchCriteria,
    dicom_client: object,
    matching_config: MatchingConfig | None = None,
    date_range: str | None = None,
    max_studies: int | None = None,
    unlimited: bool = False,
    guardrails_config: GuardrailsConfig | None = None,
    logger: logging.Logger | None = None,
) -> SearchResult:
    start_time = time.time()
    engine = DicomSearchEngine(dicom_client)
    guardrails = guardrails_config or GuardrailsConfig()
    log = logger or logging.getLogger(__name__)
    effective_date_range, effective_max_studies = apply_guardrails(
        guardrails,
        date_range=date_range,
        max_studies=max_studies,
        unlimited=unlimited,
        logger=log,
    )
    log.info(
        "Starting DICOM search",
        extra={
            "extra_data": {
                "date_range": effective_date_range,
                "max_studies": effective_max_studies,
                "unlimited": unlimited,
            }
        },
    )
    head_keywords = criteria.head_keywords

    studies = engine.find_studies(criteria, date_range=effective_date_range)
    if log.isEnabledFor(logging.DEBUG):
        modalities = {}
        for study in studies:
            raw = _get_attr(study, "ModalitiesInStudy")
            if isinstance(raw, str):
                values = [m.strip().upper() for m in raw.split("\\") if m.strip()]
            elif raw is None:
                values = []
            else:
                values = [str(m).upper() for m in raw]
            if not values:
                modalities["UNKNOWN"] = modalities.get("UNKNOWN", 0) + 1
            else:
                for value in values:
                    modalities[value] = modalities.get(value, 0) + 1
        log.debug(
            "C-FIND studies returned",
            extra={"extra_data": {"count": len(studies), "modalities": modalities}},
        )
    stats = SearchStats(
        studies_scanned=0,
        studies_matched=0,
        studies_excluded_no_age=0,
        studies_excluded_no_sex=0,
        limit_reached=False,
        execution_time_seconds=0.0,
        date_range_applied=effective_date_range or "",
    )
    accession_numbers: list[str] = []

    for study in studies:
        if log.isEnabledFor(logging.DEBUG):
            log.debug("Study candidate", extra={"extra_data": mask_phi(study)})
        if effective_max_studies is not None and stats.studies_scanned >= effective_max_studies:
            stats.limit_reached = True
            remaining = max(len(studies) - stats.studies_scanned, 0)
            log.warning(
                "AVISO: Limite de %s estudos atingido. %s estudos adicionais nao foram avaliados.",
                effective_max_studies,
                remaining,
            )
            break
        stats.studies_scanned += 1
        patient_filter = criteria.patient
        if patient_filter and patient_filter.sex:
            sex = _get_attr(study, "PatientSex")
            if not sex:
                stats.studies_excluded_no_sex += 1
                continue
            if str(sex).upper() != patient_filter.sex:
                continue

        if patient_filter and (patient_filter.age_min is not None or patient_filter.age_max is not None):
            age = calculate_age(
                _get_attr(study, "PatientBirthDate"),
                _get_attr(study, "StudyDate"),
            )
            if age is None:
                stats.studies_excluded_no_age += 1
                continue
            if patient_filter.age_min is not None and age < patient_filter.age_min:
                continue
            if patient_filter.age_max is not None and age > patient_filter.age_max:
                continue

        needs_series = bool(criteria.required_series) or bool(head_keywords)
        series_list = engine.find_series(_get_attr(study, "StudyInstanceUID")) if needs_series else []

        if not study_matches(study, series_list, criteria, head_keywords=head_keywords):
            continue

        stats.studies_matched += 1
        accession = _get_attr(study, "AccessionNumber")
        if accession:
            accession_numbers.append(str(accession))

    stats.execution_time_seconds = time.time() - start_time
    log.info("DICOM search completed", extra={"extra_data": stats.model_dump()})

    return SearchResult(accession_numbers=accession_numbers, stats=stats)
