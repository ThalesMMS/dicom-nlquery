from __future__ import annotations

import pytest


@pytest.mark.integration
def test_cfind_study_returns_studies(orthanc_with_data):
    client = orthanc_with_data["client"]
    date_range = orthanc_with_data["date_range"]
    expected = {study.accession_number for study in orthanc_with_data["studies"]}

    studies = client.query_study(
        study_date=date_range,
        additional_attrs=["PatientSex", "PatientBirthDate"],
    )
    accessions = {
        study.get("AccessionNumber") for study in studies if study.get("AccessionNumber")
    }

    assert accessions == expected


@pytest.mark.integration
def test_cfind_series_returns_series(orthanc_with_data):
    client = orthanc_with_data["client"]

    studies = client.query_study(
        accession_number="ACC001",
        additional_attrs=["PatientSex", "PatientBirthDate"],
    )
    assert studies, "Expected to find study for ACC001"
    study_uid = studies[0]["StudyInstanceUID"]

    series_list = client.query_series(study_instance_uid=study_uid)
    descriptions = {
        series.get("SeriesDescription")
        for series in series_list
        if series.get("SeriesDescription")
    }

    assert "AX T1 CRANIAL" in descriptions
