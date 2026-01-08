from __future__ import annotations

from dicom_nlquery.dicom_search import execute_search
from dicom_nlquery.models import (
    LexiconConfig,
    SearchCriteria,
    SearchPipelineConfig,
    SeriesQuery,
    StudyQuery,
)


class FakeDicomClient:
    def __init__(self) -> None:
        self.study_calls: list[dict[str, object]] = []

    def query_study(self, **kwargs):
        self.study_calls.append(kwargs)
        desc = str(kwargs.get("study_description") or "")
        if not desc:
            return []
        if "fetal" not in desc:
            return []
        return [
            {
                "StudyInstanceUID": "1",
                "AccessionNumber": "ACC001",
                "StudyDescription": "MR fetal study",
                "ModalitiesInStudy": "MR",
            }
        ]

    def query_studies(self, **kwargs):
        return self.query_study(**kwargs)

    def query_series(self, study_instance_uid, **kwargs):
        return []


def test_search_pipeline_rewrites_description() -> None:
    client = FakeDicomClient()
    criteria = SearchCriteria(
        study=StudyQuery(study_description="fetus", modality_in_study="MR")
    )
    pipeline_config = SearchPipelineConfig(max_attempts=5, max_rewrites=5)
    lexicon_config = LexiconConfig(synonyms={"fetus": ["fetal"]})

    result = execute_search(
        criteria,
        query_client=client,
        search_pipeline_config=pipeline_config,
        lexicon_config=lexicon_config,
    )

    assert result.accession_numbers == ["ACC001"]
    assert result.study_instance_uids == ["1"]
    assert result.stats.successful_stage == "rewrite"
    assert any(
        "fetal" in str(call.get("study_description")) for call in client.study_calls
    )


def test_search_pipeline_scored_rewrite_prefers_closest() -> None:
    class ScoredClient(FakeDicomClient):
        def query_study(self, **kwargs):
            self.study_calls.append(kwargs)
            desc = str(kwargs.get("study_description") or "")
            if "fetal" in desc:
                return [
                    {
                        "StudyInstanceUID": "2",
                        "AccessionNumber": "ACC002",
                        "StudyDescription": "MR fetal study",
                        "ModalitiesInStudy": "MR",
                    }
                ]
            return []

    client = ScoredClient()
    criteria = SearchCriteria(
        study=StudyQuery(study_description="fetus", modality_in_study="MR")
    )
    pipeline_config = SearchPipelineConfig(max_attempts=6, max_rewrites=1)
    lexicon_config = LexiconConfig(synonyms={"fetus": ["fetal", "pregnancy"]})

    result = execute_search(
        criteria,
        query_client=client,
        search_pipeline_config=pipeline_config,
        lexicon_config=lexicon_config,
    )

    assert result.accession_numbers == ["ACC002"]
    assert result.study_instance_uids == ["2"]


def test_search_pipeline_propagates_server_limits() -> None:
    class LimitClient:
        def __init__(self) -> None:
            self.study_calls: list[dict[str, object]] = []
            self.series_calls: list[dict[str, object]] = []

        def query_studies(self, **kwargs):
            self.study_calls.append(kwargs)
            return [
                {
                    "StudyInstanceUID": "1",
                    "AccessionNumber": "ACC123",
                    "PatientID": "123",
                }
            ]

        def query_series(self, study_instance_uid: str, **kwargs):
            self.series_calls.append({"study_instance_uid": study_instance_uid, **kwargs})
            return [
                {
                    "SeriesInstanceUID": "SERIES1",
                    "SeriesDescription": "AXIAL",
                }
            ]

    client = LimitClient()
    criteria = SearchCriteria(
        study=StudyQuery(patient_id="123"),
        series=SeriesQuery(series_description="AXIAL"),
    )
    pipeline_config = SearchPipelineConfig(
        server_limit_studies=10,
        server_limit_series=20,
    )

    result = execute_search(
        criteria,
        query_client=client,
        search_pipeline_config=pipeline_config,
    )

    assert result.accession_numbers == ["ACC123"]
    assert result.study_instance_uids == ["1"]
    assert client.study_calls
    assert client.study_calls[0]["limit"] == 10
    assert client.series_calls
    assert client.series_calls[0]["limit"] == 20
