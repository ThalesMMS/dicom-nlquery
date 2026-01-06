from __future__ import annotations

import pytest
from pydantic import ValidationError

from dicom_nlquery.models import PatientFilter, SearchCriteria, SeriesRequirement


def test_search_criteria_requires_filters() -> None:
    with pytest.raises(ValidationError):
        SearchCriteria()


def test_search_criteria_accepts_patient_filter() -> None:
    criteria = SearchCriteria(patient=PatientFilter(sex="F"))

    assert criteria.patient is not None
    assert criteria.patient.sex == "F"


def test_patient_filter_validates_sex() -> None:
    with pytest.raises(ValidationError):
        PatientFilter(sex="X")


def test_patient_filter_validates_age_range() -> None:
    with pytest.raises(ValidationError):
        PatientFilter(age_min=40, age_max=20)


def test_series_requirement_requires_match_fields() -> None:
    with pytest.raises(ValidationError):
        SeriesRequirement(name="cranio", within_head=True)


def test_series_requirement_accepts_keywords() -> None:
    requirement = SeriesRequirement(
        name="axial t1",
        within_head=True,
        all_keywords=["axial", "t1"],
    )

    assert requirement.all_keywords == ["axial", "t1"]
