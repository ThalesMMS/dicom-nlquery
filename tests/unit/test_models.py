from __future__ import annotations

import pytest
from pydantic import ValidationError

from dicom_nlquery.models import QueryStudiesArgs, SearchCriteria, SeriesQuery, StudyQuery


def test_search_criteria_requires_filters() -> None:
    with pytest.raises(ValidationError):
        SearchCriteria(study=StudyQuery())


def test_search_criteria_accepts_patient_filter() -> None:
    criteria = SearchCriteria(study=StudyQuery(patient_sex="F"))

    assert criteria.study.patient_sex == "F"


def test_search_criteria_accepts_patient_name() -> None:
    criteria = SearchCriteria(study=StudyQuery(patient_name="PACIENTE TESTE"))

    assert criteria.study.patient_name == "PACIENTE TESTE"


def test_study_query_validates_sex() -> None:
    with pytest.raises(ValidationError):
        StudyQuery(patient_sex="X")


def test_series_query_accepts_filters() -> None:
    criteria = SearchCriteria(
        study=StudyQuery(study_description="cranio"),
        series=SeriesQuery(modality="MR", series_description="AX T1"),
    )

    assert criteria.series is not None
    assert criteria.series.modality == "MR"


def test_query_studies_modality_accepts_extra_text() -> None:
    args = QueryStudiesArgs.model_validate({"modality_in_study": "RM com contraste"})

    assert args.modality_in_study == "MR"
