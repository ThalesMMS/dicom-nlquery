from __future__ import annotations

from dicom_nlquery.dicom_search import execute_search
from dicom_nlquery.models import SearchCriteria, SeriesQuery, StudyQuery


class FakeDicomClient:
    def __init__(self, studies, series_by_uid):
        self.studies = studies
        self.series_by_uid = series_by_uid
        self.study_calls = []
        self.series_calls = []

    def query_study(self, **kwargs):
        self.study_calls.append(kwargs)
        return list(self.studies)

    def query_series(self, study_instance_uid, **kwargs):
        self.series_calls.append((study_instance_uid, kwargs))
        return list(self.series_by_uid.get(study_instance_uid, []))

    def query_studies(self, **kwargs):
        return self.query_study(**kwargs)


def test_find_studies_passes_date_range() -> None:
    client = FakeDicomClient([], {})
    criteria = SearchCriteria(study=StudyQuery(patient_sex="F"))

    execute_search(criteria, query_client=client, date_range="20200101-20201231")

    assert client.study_calls
    assert client.study_calls[0]["study_date"] == "20200101-20201231"


def test_execute_search_filters_results() -> None:
    studies = [
        {
            "StudyInstanceUID": "1",
            "AccessionNumber": "ACC001",
            "PatientSex": "F",
            "PatientBirthDate": "19900101",
            "StudyDate": "20200101",
            "StudyDescription": "Cranio",
        },
        {
            "StudyInstanceUID": "2",
            "AccessionNumber": "ACC002",
            "PatientSex": "M",
            "PatientBirthDate": "19800101",
            "StudyDate": "20200101",
            "StudyDescription": "Torax",
        },
    ]
    series_by_uid = {
        "1": [
            {
                "SeriesInstanceUID": "S1",
                "SeriesDescription": "AX T1 CRANIO",
                "Modality": "MR",
            }
        ],
        "2": [
            {
                "SeriesInstanceUID": "S2",
                "SeriesDescription": "AX T1 TORAX",
                "Modality": "MR",
            }
        ],
    }
    client = FakeDicomClient(studies, series_by_uid)
    criteria = SearchCriteria(
        study=StudyQuery(patient_sex="F", study_description="cranio"),
        series=SeriesQuery(modality="MR", series_description="t1"),
    )

    result = execute_search(criteria, query_client=client)

    assert result.accession_numbers == ["ACC001"]
    assert result.stats.studies_scanned == 2
    assert result.stats.studies_matched == 1
    assert client.series_calls


def test_execute_search_relaxes_description_on_empty_results() -> None:
    client = FakeDicomClient([], {})
    criteria = SearchCriteria(
        study=StudyQuery(study_description="RM fetal", modality_in_study="MR")
    )

    execute_search(criteria, query_client=client)

    assert len(client.study_calls) >= 2
    assert "study_description" not in client.study_calls[0]
    assert any("study_description" in call for call in client.study_calls[1:])


def test_execute_search_accepts_wildcard_description() -> None:
    studies = [
        {
            "StudyInstanceUID": "1",
            "AccessionNumber": "ACC001",
            "StudyDescription": "RESSONANCIA MAGNETICA DE CRANIO",
        }
    ]
    client = FakeDicomClient(studies, {})
    criteria = SearchCriteria(study=StudyQuery(study_description="*cranio"))

    result = execute_search(criteria, query_client=client)

    assert result.accession_numbers == ["ACC001"]


def test_execute_search_applies_patient_name_wildcard() -> None:
    client = FakeDicomClient([], {})
    criteria = SearchCriteria(study=StudyQuery(patient_name="Paciente Teste"))

    execute_search(criteria, query_client=client)

    assert client.study_calls
    assert client.study_calls[0]["patient_name"] == "*Paciente*Teste*"


def test_execute_search_filters_by_patient_name() -> None:
    studies = [
        {
            "StudyInstanceUID": "1",
            "AccessionNumber": "ACC001",
            "PatientName": "PACIENTE^TESTE",
        }
    ]
    client = FakeDicomClient(studies, {})
    criteria = SearchCriteria(study=StudyQuery(patient_name="Paciente Teste"))

    result = execute_search(criteria, query_client=client)

    assert result.accession_numbers == ["ACC001"]
