from __future__ import annotations

from dicom_nlquery.dicom_search import execute_search
from dicom_nlquery.models import SearchCriteria, SearchPipelineConfig, StudyQuery, TelemetryConfig


class TelemetryClient:
    def query_study(self, **kwargs):
        return [
            {
                "StudyInstanceUID": "1",
                "AccessionNumber": "ACC123",
                "PatientID": "123",
            }
        ]

    def query_studies(self, **kwargs):
        return self.query_study(**kwargs)


def test_stage_metrics_recorded_when_enabled() -> None:
    client = TelemetryClient()
    criteria = SearchCriteria(study=StudyQuery(patient_id="123"))
    pipeline_config = SearchPipelineConfig(telemetry=TelemetryConfig(enabled=True))

    result = execute_search(criteria, query_client=client, search_pipeline_config=pipeline_config)

    metrics = result.stats.stage_metrics
    assert "direct" in metrics
    stage = metrics["direct"]
    assert stage.attempts == 1
    assert stage.successes == 1
    assert stage.studies_returned == 1
    assert stage.studies_matched == 1
    assert stage.latency_seconds >= 0


def test_stage_metrics_disabled() -> None:
    client = TelemetryClient()
    criteria = SearchCriteria(study=StudyQuery(patient_id="123"))
    pipeline_config = SearchPipelineConfig(telemetry=TelemetryConfig(enabled=False))

    result = execute_search(criteria, query_client=client, search_pipeline_config=pipeline_config)

    assert result.stats.stage_metrics == {}
