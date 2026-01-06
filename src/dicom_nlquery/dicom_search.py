from __future__ import annotations

from datetime import date, timedelta
import logging
import re
import time

import anyio

from .logging_config import mask_phi
from .mcp_client import McpSession, build_stdio_server_params
from .models import GuardrailsConfig, McpServerConfig, SearchCriteria, SearchResult, SearchStats


def _get_attr(obj: object, key: str) -> object | None:
    if isinstance(obj, dict):
        return obj.get(key)
    return getattr(obj, key, None)


def _normalize_str(value: object | None) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def _contains_casefold(haystack: object | None, needle: str) -> bool:
    if haystack is None:
        return False
    hay = str(haystack).casefold()
    ned = needle.casefold()
    tokens = [token for token in re.split(r"[\s*?\^]+", ned) if token]
    if not tokens:
        return True
    pos = 0
    for token in tokens:
        idx = hay.find(token, pos)
        if idx == -1:
            return False
        pos = idx + len(token)
    return True


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

    if filters.study_description:
        if not _contains_casefold(_get_attr(study, "StudyDescription"), filters.study_description):
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


def execute_search(
    criteria: SearchCriteria,
    query_client: object | None = None,
    mcp_config: McpServerConfig | None = None,
    date_range: str | None = None,
    max_studies: int | None = None,
    unlimited: bool = False,
    guardrails_config: GuardrailsConfig | None = None,
    logger: logging.Logger | None = None,
    node_name: str | None = None,
) -> SearchResult:
    start_time = time.time()
    guardrails = guardrails_config or GuardrailsConfig()
    log = logger or logging.getLogger(__name__)
    effective_date_range, effective_max_studies = apply_guardrails(
        guardrails,
        date_range=date_range,
        max_studies=max_studies,
        unlimited=unlimited,
        logger=log,
    )
    resolved_study_date = date_range or criteria.study.study_date or effective_date_range
    explicit_study_date = date_range or criteria.study.study_date
    log.info(
        "Starting DICOM search",
        extra={
            "extra_data": {
                "date_range": resolved_study_date,
                "max_studies": effective_max_studies,
                "unlimited": unlimited,
            }
        },
    )

    if query_client is None and mcp_config is None:
        raise ValueError("mcp_config is required when query_client is not provided")

    if query_client is None:
        return anyio.run(
            _execute_with_mcp,
            criteria,
            mcp_config,
            resolved_study_date,
            explicit_study_date,
            effective_max_studies,
            log,
            start_time,
            node_name,
        )

    return _execute_with_client(
        criteria,
        query_client,
        resolved_study_date,
        explicit_study_date,
        effective_max_studies,
        log,
        start_time,
    )


def _execute_with_client(
    criteria: SearchCriteria,
    query_client: object,
    study_date: str | None,
    explicit_study_date: str | None,
    max_studies: int | None,
    log: logging.Logger,
    start_time: float,
) -> SearchResult:
    study_args = _build_study_args(criteria, study_date)
    query_studies = getattr(query_client, "query_studies", None) or getattr(
        query_client, "query_study"
    )
    studies = list(query_studies(**study_args))
    if not studies and criteria.study.study_description:
        relaxed_args = dict(study_args)
        relaxed_args.pop("study_description", None)
        studies = list(query_studies(**relaxed_args))
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
        studies_filtered_series=0,
        limit_reached=False,
        execution_time_seconds=0.0,
        date_range_applied=study_date or "",
    )
    accession_numbers: list[str] = []
    has_series_filters = _has_series_filters(criteria)
    series_args = _build_series_args(criteria)

    for study in studies:
        if log.isEnabledFor(logging.DEBUG):
            log.debug("Study candidate", extra={"extra_data": mask_phi(study)})
        if max_studies is not None and stats.studies_scanned >= max_studies:
            stats.limit_reached = True
            remaining = max(len(studies) - stats.studies_scanned, 0)
            log.warning(
                "AVISO: Limite de %s estudos atingido. %s estudos adicionais nao foram avaliados.",
                max_studies,
                remaining,
            )
            break
        stats.studies_scanned += 1

        if not _study_matches_criteria(study, criteria, explicit_study_date):
            continue

        if has_series_filters:
            study_uid = _get_attr(study, "StudyInstanceUID")
            if not study_uid:
                stats.studies_filtered_series += 1
                continue
            query_series = getattr(query_client, "query_series", None)
            if query_series is None:
                raise RuntimeError("query_client lacks query_series()")
            series_list = list(query_series(study_instance_uid=study_uid, **series_args))
            if not series_list or not any(
                _series_matches_criteria(item, criteria) for item in series_list
            ):
                stats.studies_filtered_series += 1
                continue

        stats.studies_matched += 1
        accession = _get_attr(study, "AccessionNumber")
        if accession:
            accession_numbers.append(str(accession))

    stats.execution_time_seconds = time.time() - start_time
    log.info("DICOM search completed", extra={"extra_data": stats.model_dump()})

    return SearchResult(accession_numbers=accession_numbers, stats=stats)


async def _execute_with_mcp(
    criteria: SearchCriteria,
    mcp_config: McpServerConfig,
    study_date: str | None,
    explicit_study_date: str | None,
    max_studies: int | None,
    log: logging.Logger,
    start_time: float,
    node_name: str | None,
) -> SearchResult:
    server_params = build_stdio_server_params(mcp_config)
    async with McpSession(server_params) as client:
        if node_name:
            await client.switch_dicom_node(node_name)

        study_args = _build_study_args(criteria, study_date)
        studies = list(await client.query_studies(**study_args))
        if not studies and criteria.study.study_description:
            relaxed_args = dict(study_args)
            relaxed_args.pop("study_description", None)
            studies = list(await client.query_studies(**relaxed_args))
        if log.isEnabledFor(logging.DEBUG):
            log.debug(
                "C-FIND studies returned",
                extra={"extra_data": {"count": len(studies)}},
            )

        stats = SearchStats(
            studies_scanned=0,
            studies_matched=0,
            studies_filtered_series=0,
            limit_reached=False,
            execution_time_seconds=0.0,
            date_range_applied=study_date or "",
        )
        accession_numbers: list[str] = []
        has_series_filters = _has_series_filters(criteria)
        series_args = _build_series_args(criteria)

        for study in studies:
            if log.isEnabledFor(logging.DEBUG):
                log.debug("Study candidate", extra={"extra_data": mask_phi(study)})
            if max_studies is not None and stats.studies_scanned >= max_studies:
                stats.limit_reached = True
                remaining = max(len(studies) - stats.studies_scanned, 0)
                log.warning(
                    "AVISO: Limite de %s estudos atingido. %s estudos adicionais nao foram avaliados.",
                    max_studies,
                    remaining,
                )
                break
            stats.studies_scanned += 1

            if not _study_matches_criteria(study, criteria, explicit_study_date):
                continue

            if has_series_filters:
                study_uid = _get_attr(study, "StudyInstanceUID")
                if not study_uid:
                    stats.studies_filtered_series += 1
                    continue
                series_list = list(
                    await client.query_series(
                        study_instance_uid=study_uid,
                        **series_args,
                    )
                )
                if not series_list or not any(
                    _series_matches_criteria(item, criteria) for item in series_list
                ):
                    stats.studies_filtered_series += 1
                    continue

            stats.studies_matched += 1
            accession = _get_attr(study, "AccessionNumber")
            if accession:
                accession_numbers.append(str(accession))

        stats.execution_time_seconds = time.time() - start_time
        log.info("DICOM search completed", extra={"extra_data": stats.model_dump()})

        return SearchResult(accession_numbers=accession_numbers, stats=stats)


def _build_study_args(criteria: SearchCriteria, study_date: str | None) -> dict[str, object]:
    args = criteria.study.model_dump(exclude_none=True)
    if study_date:
        args["study_date"] = study_date
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
    if not _has_series_filters(criteria):
        return {}
    assert criteria.series is not None
    return criteria.series.model_dump(exclude_none=True)


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
