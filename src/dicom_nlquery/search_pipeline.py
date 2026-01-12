from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime
from difflib import SequenceMatcher
import logging
import time
from typing import Iterable

from .logging_config import mask_phi
from .lexicon import Lexicon, normalize_text
from .models import (
    RankingConfig,
    SearchCriteria,
    SearchPipelineConfig,
    SearchResult,
    SearchStats,
    StageMetrics,
)
from .search_utils import (
    _build_series_args,
    _build_study_args,
    _get_attr,
    _has_series_filters,
    _series_matches_criteria,
    _study_matches_criteria,
    _iter_text_values,
    _matches_any_field,
    SERIES_DESCRIPTION_EXTRA_ATTRS,
    SERIES_TEXT_FIELDS,
    STUDY_TEXT_FIELDS,
)


@dataclass(frozen=True)
class SearchAttempt:
    stage: str
    study_args: dict[str, object]
    description_override: str | None = None
    use_lexicon_match: bool = False
    allow_series_probe: bool = False
    rewrite: str | None = None
    reason: str | None = None


@dataclass(frozen=True)
class RewriteCandidate:
    text: str
    source: str
    score: float


def _wildcard_patterns(text: str, modes: Iterable[str]) -> list[str]:
    if not text:
        return []
    if "*" in text or "?" in text:
        return [text]
    cleaned = " ".join(text.split())
    tokens = [token for token in cleaned.split() if token]
    patterns: list[str] = []
    if "contains" in modes:
        patterns.append(f"*{cleaned}*")
    if "token_chain" in modes and len(tokens) > 1:
        patterns.append("*" + "*".join(tokens) + "*")
    if "startswith" in modes:
        patterns.append(f"{cleaned}*")
    if "headword" in modes and tokens:
        patterns.append(f"*{tokens[0]}*")
    return patterns


def _freeze(value: object) -> object:
    if isinstance(value, dict):
        return tuple(sorted((key, _freeze(val)) for key, val in value.items()))
    if isinstance(value, list):
        return tuple(_freeze(item) for item in value)
    if isinstance(value, tuple):
        return tuple(_freeze(item) for item in value)
    if isinstance(value, set):
        return tuple(sorted((_freeze(item) for item in value), key=repr))
    return value


def _dedupe_attempts(attempts: list[SearchAttempt]) -> list[SearchAttempt]:
    seen: set[tuple[str, object, str | None, bool]] = set()
    result: list[SearchAttempt] = []
    for attempt in attempts:
        key = (
            attempt.stage,
            _freeze(attempt.study_args),
            attempt.description_override,
            attempt.use_lexicon_match,
        )
        if key in seen:
            continue
        seen.add(key)
        result.append(attempt)
    return result


def _lexicon_matches_any(
    item: object,
    description: str,
    fields: Iterable[str],
    lexicon: Lexicon,
) -> bool:
    for field in fields:
        value = _get_attr(item, field)
        for text in _iter_text_values(value):
            if lexicon.match_text(description, text):
                return True
    return False


def build_attempts(
    criteria: SearchCriteria,
    study_date: str | None,
    config: SearchPipelineConfig,
    lexicon: Lexicon | None,
    rag_suggestions: list[str] | None = None,
) -> list[SearchAttempt]:
    base_args = _build_study_args(criteria, study_date)
    description = criteria.study.study_description
    if not config.enabled:
        return [SearchAttempt(stage="direct", study_args=base_args, reason="disabled")]

    attempts: list[SearchAttempt] = []
    if description:
        # Order attempts from most structured to most expansive to keep queries selective.
        if config.structured_first:
            structured_args = _build_study_args(
                criteria,
                study_date,
                include_description=False,
            )
            attempts.append(
                SearchAttempt(
                    stage="structured-first",
                    study_args=structured_args,
                    description_override=description,
                    use_lexicon_match=bool(lexicon),
                    allow_series_probe=config.series_probe_enabled,
                    reason="structured-first",
                )
            )

        for pattern in _wildcard_patterns(description, config.wildcard_modes):
            wildcard_args = _build_study_args(
                criteria,
                study_date,
                study_description_override=pattern,
            )
            attempts.append(
                SearchAttempt(
                    stage="wildcard",
                    study_args=wildcard_args,
                    description_override=pattern,
                    use_lexicon_match=bool(lexicon),
                    reason="wildcard",
                )
            )

        # Collect rewrite candidates from lexicon/RAG before expanding to wildcard variants.
        rewrite_candidates: dict[str, RewriteCandidate] = {}
        if lexicon is not None and config.max_rewrites != 0:
            max_candidates = config.max_rewrites if config.max_rewrites > 0 else None
            rewrites = lexicon.expand_text(description, max_candidates=max_candidates)
            for rewrite in rewrites:
                _add_candidate(rewrite_candidates, description, rewrite, "lexicon")
        if rag_suggestions:
            for suggestion in rag_suggestions:
                _add_candidate(rewrite_candidates, description, suggestion, "rag")

        ranked_candidates = sorted(
            rewrite_candidates.values(),
            key=lambda item: (-item.score, item.text),
        )
        max_rewrites = config.max_rewrites if config.max_rewrites > 0 else None
        if max_rewrites is not None:
            ranked_candidates = ranked_candidates[:max_rewrites]

        for candidate in ranked_candidates:
            for pattern in _wildcard_patterns(candidate.text, config.wildcard_modes):
                rewrite_args = _build_study_args(
                    criteria,
                    study_date,
                    study_description_override=pattern,
                )
                attempts.append(
                    SearchAttempt(
                        stage="rag" if candidate.source == "rag" else "rewrite",
                        study_args=rewrite_args,
                        description_override=pattern,
                        use_lexicon_match=bool(lexicon),
                        rewrite=candidate.text,
                        reason=candidate.source,
                    )
                )
    else:
        attempts.append(SearchAttempt(stage="direct", study_args=base_args, reason="no-description"))

    # Dedupe to avoid issuing identical C-FIND requests across stages.
    attempts = _dedupe_attempts(attempts)
    if config.max_attempts and len(attempts) > config.max_attempts:
        attempts = attempts[: config.max_attempts]
    return attempts


def _record_rewrite(stats: SearchStats, rewrite: str | None) -> None:
    if not rewrite:
        return
    if rewrite in stats.rewrites_tried:
        return
    stats.rewrites_tried.append(rewrite)


def _record_stage(stats: SearchStats, stage: str) -> None:
    if stage in stats.stages_tried:
        return
    stats.stages_tried.append(stage)


def _record_stage_metrics(
    stats: SearchStats,
    stage: str,
    latency_seconds: float,
    studies_returned: int,
    studies_matched: int,
    success: bool,
    config: SearchPipelineConfig,
) -> None:
    if not config.telemetry.enabled:
        return
    metrics = stats.stage_metrics.get(stage)
    if metrics is None:
        metrics = StageMetrics()
    metrics.attempts += 1
    metrics.latency_seconds += latency_seconds
    metrics.studies_returned += studies_returned
    metrics.studies_matched += studies_matched
    if success:
        metrics.successes += 1
    stats.stage_metrics[stage] = metrics


def run_pipeline_sync(
    criteria: SearchCriteria,
    query_client: object,
    study_date: str | None,
    explicit_study_date: str | None,
    max_studies: int | None,
    config: SearchPipelineConfig,
    lexicon: Lexicon | None,
    rag_suggestions: list[str] | None,
    ranking: RankingConfig,
    timeout_seconds: int | None,
    log: logging.Logger,
    start_time: float,
) -> SearchResult:
    query_studies = getattr(query_client, "query_studies", None) or getattr(
        query_client, "query_study"
    )
    if query_studies is None:
        raise RuntimeError("query_client lacks query_studies/query_study")

    query_series = getattr(query_client, "query_series", None)

    attempts = build_attempts(criteria, study_date, config, lexicon, rag_suggestions)
    stats = SearchStats(
        studies_scanned=0,
        studies_matched=0,
        studies_filtered_series=0,
        limit_reached=False,
        execution_time_seconds=0.0,
        date_range_applied=study_date or "",
        attempts_run=0,
        successful_stage=None,
        rewrites_tried=[],
    )
    accession_numbers: list[str] = []
    study_instance_uids: list[str] = []
    seen_uids: set[str] = set()
    has_series_filters = _has_series_filters(criteria)
    series_args = _build_series_args(criteria)
    original_description = criteria.study.study_description

    deadline = _deadline_from_timeout(start_time, timeout_seconds)
    for attempt in attempts:
        if _deadline_reached(deadline):
            stats.limit_reached = True
            log.warning("Warning: search timeout reached.")
            break
        stats.attempts_run += 1
        _record_stage(stats, attempt.stage)
        _record_rewrite(stats, attempt.rewrite)
        attempt_start = time.time()
        study_args = {**attempt.study_args}
        if config.server_limit_studies is not None:
            study_args["limit"] = config.server_limit_studies
        studies = list(query_studies(**study_args))
        scanned_this_attempt = 0
        if log.isEnabledFor(logging.DEBUG):
            log.debug(
                "C-FIND studies returned",
                extra={
                    "extra_data": {
                        "count": len(studies),
                        "stage": attempt.stage,
                    }
                },
            )

        attempt_accession_matches: list[tuple[object, str]] = []
        attempt_uid_matches: list[tuple[object, str]] = []
        for study in studies:
            if _deadline_reached(deadline):
                stats.limit_reached = True
                log.warning("Warning: search timeout reached.")
                break
            if log.isEnabledFor(logging.DEBUG):
                log.debug("Study candidate", extra={"extra_data": mask_phi(study)})
            if max_studies is not None and stats.studies_scanned >= max_studies:
                stats.limit_reached = True
                remaining = max(len(studies) - scanned_this_attempt, 0)
                log.warning(
                    "Warning: limit of %s studies reached. %s additional studies were not evaluated.",
                    max_studies,
                    remaining,
                )
                break
            stats.studies_scanned += 1
            scanned_this_attempt += 1

            description_matcher = None
            if attempt.use_lexicon_match and lexicon and original_description:
                description_matcher = lambda item: _lexicon_matches_any(
                    item,
                    original_description,
                    STUDY_TEXT_FIELDS,
                    lexicon,
                )
            matches_study = _study_matches_criteria(
                study,
                criteria,
                explicit_study_date,
                description_override=attempt.description_override,
                description_matcher=description_matcher,
            )
            if not matches_study:
                # Optionally probe a few series to match text when study-level fields are sparse.
                if not (attempt.allow_series_probe and config.series_probe_enabled):
                    continue
                if not _study_matches_criteria(
                    study,
                    criteria,
                    explicit_study_date,
                    description_override=attempt.description_override,
                    description_matcher=description_matcher,
                    skip_description=True,
                ):
                    continue
                study_uid = _get_attr(study, "StudyInstanceUID")
                if not study_uid:
                    stats.studies_filtered_series += 1
                    continue
                if query_series is None:
                    raise RuntimeError("query_client lacks query_series()")
                if config.series_probe_limit <= 0:
                    continue
                series_kwargs = {"study_instance_uid": study_uid}
                if original_description:
                    series_kwargs["additional_attributes"] = list(SERIES_DESCRIPTION_EXTRA_ATTRS)
                if config.server_limit_series is not None:
                    series_kwargs["limit"] = config.server_limit_series
                series_list = list(query_series(**series_kwargs))
                if not series_list:
                    stats.studies_filtered_series += 1
                    continue
                series_match = False
                for item in series_list[: config.series_probe_limit]:
                    if lexicon and original_description:
                        if _lexicon_matches_any(
                            item,
                            original_description,
                            SERIES_TEXT_FIELDS,
                            lexicon,
                        ):
                            series_match = True
                            break
                    elif attempt.description_override:
                        if _matches_any_field(
                            item,
                            SERIES_TEXT_FIELDS,
                            attempt.description_override,
                        ):
                            series_match = True
                            break
                if not series_match:
                    stats.studies_filtered_series += 1
                    continue
                matches_study = True
            if not matches_study:
                continue

            if has_series_filters:
                # Apply series-level filters by requiring at least one matching series.
                if query_series is None:
                    raise RuntimeError("query_client lacks query_series()")
                study_uid = _get_attr(study, "StudyInstanceUID")
                if not study_uid:
                    stats.studies_filtered_series += 1
                    continue
                series_kwargs = {"study_instance_uid": study_uid, **series_args}
                if config.server_limit_series is not None:
                    series_kwargs["limit"] = config.server_limit_series
                series_list = list(query_series(**series_kwargs))
                if not series_list or not any(
                    _series_matches_criteria(item, criteria) for item in series_list
                ):
                    stats.studies_filtered_series += 1
                    continue

            study_uid = _get_attr(study, "StudyInstanceUID")
            study_uid_str = str(study_uid) if study_uid else ""
            if study_uid_str and study_uid_str in seen_uids:
                continue
            if study_uid_str:
                seen_uids.add(study_uid_str)
            stats.studies_matched += 1
            accession = _get_attr(study, "AccessionNumber")
            if accession:
                attempt_accession_matches.append((study, str(accession)))
            if study_uid_str:
                attempt_uid_matches.append((study, study_uid_str))

        attempt_success = bool(attempt_uid_matches)
        _record_stage_metrics(
            stats,
            attempt.stage,
            time.time() - attempt_start,
            len(studies),
            len(attempt_uid_matches),
            attempt_success,
            config,
        )

        if attempt_success:
            # Rank matches for relevance and stop at the first successful stage.
            accession_numbers = _rank_accessions(
                attempt_accession_matches,
                criteria.study.study_description or attempt.description_override,
                ranking,
            )
            study_instance_uids = _rank_study_uids(
                attempt_uid_matches,
                criteria.study.study_description or attempt.description_override,
                ranking,
            )
            stats.successful_stage = attempt.stage
            break

        if stats.limit_reached:
            break

    stats.execution_time_seconds = time.time() - start_time
    log.info("DICOM search completed", extra={"extra_data": stats.model_dump()})
    return SearchResult(
        accession_numbers=accession_numbers,
        study_instance_uids=study_instance_uids,
        stats=stats,
    )


async def run_pipeline_async(
    criteria: SearchCriteria,
    query_client: object,
    study_date: str | None,
    explicit_study_date: str | None,
    max_studies: int | None,
    config: SearchPipelineConfig,
    lexicon: Lexicon | None,
    rag_suggestions: list[str] | None,
    ranking: RankingConfig,
    timeout_seconds: int | None,
    log: logging.Logger,
    start_time: float,
) -> SearchResult:
    attempts = build_attempts(criteria, study_date, config, lexicon, rag_suggestions)
    stats = SearchStats(
        studies_scanned=0,
        studies_matched=0,
        studies_filtered_series=0,
        limit_reached=False,
        execution_time_seconds=0.0,
        date_range_applied=study_date or "",
        attempts_run=0,
        successful_stage=None,
        rewrites_tried=[],
    )
    accession_numbers: list[str] = []
    study_instance_uids: list[str] = []
    seen_uids: set[str] = set()
    has_series_filters = _has_series_filters(criteria)
    series_args = _build_series_args(criteria)
    original_description = criteria.study.study_description

    deadline = _deadline_from_timeout(start_time, timeout_seconds)
    for attempt in attempts:
        if _deadline_reached(deadline):
            stats.limit_reached = True
            log.warning("Warning: search timeout reached.")
            break
        stats.attempts_run += 1
        _record_stage(stats, attempt.stage)
        _record_rewrite(stats, attempt.rewrite)
        attempt_start = time.time()
        study_args = {**attempt.study_args}
        if config.server_limit_studies is not None:
            study_args["limit"] = config.server_limit_studies
        studies = list(await query_client.query_studies(**study_args))
        scanned_this_attempt = 0
        if log.isEnabledFor(logging.DEBUG):
            log.debug(
                "C-FIND studies returned",
                extra={
                    "extra_data": {
                        "count": len(studies),
                        "stage": attempt.stage,
                    }
                },
            )

        attempt_accession_matches: list[tuple[object, str]] = []
        attempt_uid_matches: list[tuple[object, str]] = []
        for study in studies:
            if _deadline_reached(deadline):
                stats.limit_reached = True
                log.warning("Warning: search timeout reached.")
                break
            if log.isEnabledFor(logging.DEBUG):
                log.debug("Study candidate", extra={"extra_data": mask_phi(study)})
            if max_studies is not None and stats.studies_scanned >= max_studies:
                stats.limit_reached = True
                remaining = max(len(studies) - scanned_this_attempt, 0)
                log.warning(
                    "Warning: limit of %s studies reached. %s additional studies were not evaluated.",
                    max_studies,
                    remaining,
                )
                break
            stats.studies_scanned += 1
            scanned_this_attempt += 1

            description_matcher = None
            if attempt.use_lexicon_match and lexicon and original_description:
                description_matcher = lambda item: _lexicon_matches_any(
                    item,
                    original_description,
                    STUDY_TEXT_FIELDS,
                    lexicon,
                )
            matches_study = _study_matches_criteria(
                study,
                criteria,
                explicit_study_date,
                description_override=attempt.description_override,
                description_matcher=description_matcher,
            )
            if not matches_study:
                # Optionally probe a few series to match text when study-level fields are sparse.
                if not (attempt.allow_series_probe and config.series_probe_enabled):
                    continue
                if not _study_matches_criteria(
                    study,
                    criteria,
                    explicit_study_date,
                    description_override=attempt.description_override,
                    description_matcher=description_matcher,
                    skip_description=True,
                ):
                    continue
                study_uid = _get_attr(study, "StudyInstanceUID")
                if not study_uid:
                    stats.studies_filtered_series += 1
                    continue
                if config.series_probe_limit <= 0:
                    continue
                series_kwargs = {"study_instance_uid": study_uid}
                if original_description:
                    series_kwargs["additional_attributes"] = list(SERIES_DESCRIPTION_EXTRA_ATTRS)
                if config.server_limit_series is not None:
                    series_kwargs["limit"] = config.server_limit_series
                series_list = list(await query_client.query_series(**series_kwargs))
                if not series_list:
                    stats.studies_filtered_series += 1
                    continue
                series_match = False
                for item in series_list[: config.series_probe_limit]:
                    if lexicon and original_description:
                        if _lexicon_matches_any(
                            item,
                            original_description,
                            SERIES_TEXT_FIELDS,
                            lexicon,
                        ):
                            series_match = True
                            break
                    elif attempt.description_override:
                        if _matches_any_field(
                            item,
                            SERIES_TEXT_FIELDS,
                            attempt.description_override,
                        ):
                            series_match = True
                            break
                if not series_match:
                    stats.studies_filtered_series += 1
                    continue
                matches_study = True
            if not matches_study:
                continue

            if has_series_filters:
                # Apply series-level filters by requiring at least one matching series.
                study_uid = _get_attr(study, "StudyInstanceUID")
                if not study_uid:
                    stats.studies_filtered_series += 1
                    continue
                series_kwargs = {"study_instance_uid": study_uid, **series_args}
                if config.server_limit_series is not None:
                    series_kwargs["limit"] = config.server_limit_series
                series_list = list(await query_client.query_series(**series_kwargs))
                if not series_list or not any(
                    _series_matches_criteria(item, criteria) for item in series_list
                ):
                    stats.studies_filtered_series += 1
                    continue

            study_uid = _get_attr(study, "StudyInstanceUID")
            study_uid_str = str(study_uid) if study_uid else ""
            if study_uid_str and study_uid_str in seen_uids:
                continue
            if study_uid_str:
                seen_uids.add(study_uid_str)
            stats.studies_matched += 1
            accession = _get_attr(study, "AccessionNumber")
            if accession:
                attempt_accession_matches.append((study, str(accession)))
            if study_uid_str:
                attempt_uid_matches.append((study, study_uid_str))

        attempt_success = bool(attempt_uid_matches)
        _record_stage_metrics(
            stats,
            attempt.stage,
            time.time() - attempt_start,
            len(studies),
            len(attempt_uid_matches),
            attempt_success,
            config,
        )

        if attempt_success:
            # Rank matches for relevance and stop at the first successful stage.
            accession_numbers = _rank_accessions(
                attempt_accession_matches,
                criteria.study.study_description or attempt.description_override,
                ranking,
            )
            study_instance_uids = _rank_study_uids(
                attempt_uid_matches,
                criteria.study.study_description or attempt.description_override,
                ranking,
            )
            stats.successful_stage = attempt.stage
            break

        if stats.limit_reached:
            break

    stats.execution_time_seconds = time.time() - start_time
    log.info("DICOM search completed", extra={"extra_data": stats.model_dump()})
    return SearchResult(
        accession_numbers=accession_numbers,
        study_instance_uids=study_instance_uids,
        stats=stats,
    )


def _score_similarity(original: str, candidate: str) -> float:
    if not original or not candidate:
        return 0.0
    original_norm = normalize_text(original)
    candidate_norm = normalize_text(candidate)
    if not original_norm or not candidate_norm:
        return 0.0
    # Blend character similarity with token overlap to favor near-synonyms.
    ratio = SequenceMatcher(None, original_norm, candidate_norm).ratio()
    original_tokens = set(original_norm.split())
    candidate_tokens = set(candidate_norm.split())
    if not original_tokens or not candidate_tokens:
        return ratio
    overlap = len(original_tokens & candidate_tokens)
    union = len(original_tokens | candidate_tokens)
    jaccard = overlap / union if union else 0.0
    return max(ratio, jaccard)


def _add_candidate(
    candidates: dict[str, RewriteCandidate],
    original: str,
    candidate: str,
    source: str,
) -> None:
    if normalize_text(candidate) == normalize_text(original):
        return
    score = _score_similarity(original, candidate)
    existing = candidates.get(candidate)
    if existing is None or score > existing.score:
        candidates[candidate] = RewriteCandidate(text=candidate, source=source, score=score)


def _rank_accessions(
    matches: list[tuple[object, str]],
    query_text: str | None,
    ranking: RankingConfig,
) -> list[str]:
    if not ranking.enabled:
        return _dedupe_accessions(matches)
    query = query_text or ""
    today = date.today()
    scored: list[tuple[float, str]] = []
    for study, accession in matches:
        description = str(_get_attr(study, "StudyDescription") or "")
        study_date = str(_get_attr(study, "StudyDate") or "")
        text_score = _text_match_score(query, description)
        recency_score = _recency_score(study_date, today)
        score = ranking.text_match_weight * text_score + ranking.recency_weight * recency_score
        scored.append((score, accession))
    scored.sort(key=lambda item: item[0], reverse=True)
    return _dedupe_accessions([(None, accession) for _, accession in scored])


def _rank_study_uids(
    matches: list[tuple[object, str]],
    query_text: str | None,
    ranking: RankingConfig,
) -> list[str]:
    if not ranking.enabled:
        return _dedupe_uids(matches)
    query = query_text or ""
    today = date.today()
    scored: list[tuple[float, str]] = []
    for study, study_uid in matches:
        description = str(_get_attr(study, "StudyDescription") or "")
        study_date = str(_get_attr(study, "StudyDate") or "")
        text_score = _text_match_score(query, description)
        recency_score = _recency_score(study_date, today)
        score = ranking.text_match_weight * text_score + ranking.recency_weight * recency_score
        scored.append((score, study_uid))
    scored.sort(key=lambda item: item[0], reverse=True)
    return _dedupe_uids([(None, study_uid) for _, study_uid in scored])


def _text_match_score(query: str, description: str) -> float:
    if not query or not description:
        return 0.0
    query_tokens = set(normalize_text(query).split())
    desc_tokens = set(normalize_text(description).split())
    if not query_tokens or not desc_tokens:
        return 0.0
    return len(query_tokens & desc_tokens) / len(query_tokens)


def _recency_score(study_date: str, today: date) -> float:
    if not study_date or len(study_date) != 8 or not study_date.isdigit():
        return 0.0
    try:
        parsed = datetime.strptime(study_date, "%Y%m%d").date()
    except ValueError:
        return 0.0
    delta_days = (today - parsed).days
    if delta_days < 0:
        return 0.0
    if delta_days >= 365:
        return 0.0
    return 1.0 - (delta_days / 365.0)


def _deadline_from_timeout(start_time: float, timeout_seconds: int | None) -> float | None:
    if timeout_seconds is None:
        return None
    if timeout_seconds <= 0:
        return start_time
    return start_time + timeout_seconds


def _deadline_reached(deadline: float | None) -> bool:
    if deadline is None:
        return False
    return time.time() >= deadline


def _dedupe_accessions(matches: list[tuple[object | None, str]]) -> list[str]:
    seen: set[str] = set()
    ordered: list[str] = []
    for _, accession in matches:
        if accession in seen:
            continue
        seen.add(accession)
        ordered.append(accession)
    return ordered


def _dedupe_uids(matches: list[tuple[object | None, str]]) -> list[str]:
    seen: set[str] = set()
    ordered: list[str] = []
    for _, uid in matches:
        if not uid:
            continue
        if uid in seen:
            continue
        seen.add(uid)
        ordered.append(uid)
    return ordered
