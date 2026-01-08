from __future__ import annotations

from dicom_nlquery.dicom_search import execute_search
from dicom_nlquery.models import RagConfig, SearchCriteria, SearchPipelineConfig, StudyQuery


class FakeDicomClient:
    def __init__(self) -> None:
        self.study_calls: list[dict[str, object]] = []

    def query_study(self, **kwargs):
        self.study_calls.append(kwargs)
        desc = str(kwargs.get("study_description") or "")
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


def test_search_pipeline_uses_rag_suggestions(monkeypatch) -> None:
    client = FakeDicomClient()
    criteria = SearchCriteria(study=StudyQuery(study_description="pregnancy"))
    pipeline_config = SearchPipelineConfig(max_attempts=6, max_rewrites=2)
    rag_config = RagConfig(enable=True, index_path="/tmp/fake.sqlite")

    monkeypatch.setattr(
        "dicom_nlquery.dicom_search.get_rag_suggestions",
        lambda _q, _c, _l=None: ["fetal"],
    )

    result = execute_search(
        criteria,
        query_client=client,
        search_pipeline_config=pipeline_config,
        rag_config=rag_config,
    )

    assert result.accession_numbers == ["ACC001"]
    assert any(
        "fetal" in str(call.get("study_description")) for call in client.study_calls
    )
