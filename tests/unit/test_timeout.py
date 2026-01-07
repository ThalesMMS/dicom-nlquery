from __future__ import annotations

from dicom_nlquery.dicom_search import execute_search
from dicom_nlquery.models import GuardrailsConfig, SearchCriteria, StudyQuery


class FakeClient:
    def query_study(self, **kwargs):
        return [
            {
                "StudyInstanceUID": "1",
                "AccessionNumber": "ACC001",
                "StudyDescription": "cranio",
                "StudyDate": "20240101",
            }
        ]

    def query_studies(self, **kwargs):
        return self.query_study(**kwargs)


def test_search_timeout_short_circuits() -> None:
    client = FakeClient()
    criteria = SearchCriteria(study=StudyQuery(study_description="cranio"))
    guardrails = GuardrailsConfig(search_timeout_seconds=0)

    result = execute_search(criteria, query_client=client, guardrails_config=guardrails)

    assert result.accession_numbers == []
    assert result.stats.limit_reached
