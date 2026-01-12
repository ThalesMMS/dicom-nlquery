from __future__ import annotations

from dicom_nlquery.models import SearchCriteria, StudyQuery
from dicom_nlquery.search_utils import (
    _build_series_args,
    _build_study_args,
    _study_matches_criteria,
)


def test_study_matches_requested_procedure_description() -> None:
    criteria = SearchCriteria(study=StudyQuery(study_description="angiogram"))
    study = {"RequestedProcedureDescription": "CT ANGIOGRAM CHEST"}

    assert _study_matches_criteria(study, criteria, explicit_study_date=None)


def test_build_study_args_includes_requested_procedure_description() -> None:
    criteria = SearchCriteria(study=StudyQuery(study_description="angiogram"))

    args = _build_study_args(criteria, study_date=None)

    assert "additional_attributes" in args
    assert "RequestedProcedureDescription" in args["additional_attributes"]


def test_build_series_args_includes_extra_attributes() -> None:
    criteria = SearchCriteria(study=StudyQuery(study_description="chest"))

    args = _build_series_args(criteria)

    assert "additional_attributes" in args
    assert "SeriesDescription" in args["additional_attributes"]


def test_study_matches_code_sequence_text() -> None:
    criteria = SearchCriteria(study=StudyQuery(study_description="chest"))
    study = {
        "RequestedProcedureCodeSequence": [
            {"CodeMeaning": "CT ANGIOGRAM CHEST"},
            {"CodeValue": "ABC123"},
        ]
    }

    assert _study_matches_criteria(study, criteria, explicit_study_date=None)


def test_study_matches_out_of_order_tokens() -> None:
    criteria = SearchCriteria(study=StudyQuery(study_description="chest angiogram"))
    study = {"RequestedProcedureDescription": "CT ANGIOGRAM CHEST"}

    assert _study_matches_criteria(study, criteria, explicit_study_date=None)
