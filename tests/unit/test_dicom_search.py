from __future__ import annotations

from dicom_nlquery.dicom_search import DicomSearchEngine, execute_search
from dicom_nlquery.models import PatientFilter, SearchCriteria, SeriesRequirement


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


def test_find_studies_passes_date_range() -> None:
    client = FakeDicomClient([], {})
    engine = DicomSearchEngine(client)
    criteria = SearchCriteria(patient=PatientFilter(sex="F"))

    engine.find_studies(criteria, date_range="20200101-20201231")

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
        patient=PatientFilter(sex="F", age_min=20, age_max=40),
        head_keywords=["cranio"],
        required_series=[
            SeriesRequirement(
                name="t1 cranio",
                modality="MR",
                within_head=True,
                any_keywords=["t1"],
            )
        ],
    )

    result = execute_search(criteria, client)

    assert result.accession_numbers == ["ACC001"]
    assert result.stats.studies_scanned == 2
    assert result.stats.studies_matched == 1
    assert client.series_calls
