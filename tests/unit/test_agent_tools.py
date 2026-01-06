from __future__ import annotations

from dicom_nlquery.agent_tools import execute_tool, get_tools_schema
from dicom_nlquery.models import AgentState, ToolName


class FakeClient:
    def __init__(self) -> None:
        self.last_query = None

    def query_studies(self, **kwargs):
        self.last_query = kwargs
        return [
            {
                "StudyInstanceUID": "1.2.3",
                "StudyDate": "20240101",
                "StudyDescription": "TEST",
                "ModalitiesInStudy": "MR",
                "PatientName": "ANON",
            }
        ]

    def query_series(self, **kwargs):
        return [{"SeriesInstanceUID": "1.2.3.4"}]

    def move_study(self, destination_node: str, study_instance_uid: str):
        return {"success": True, "completed": 1, "failed": 0}


def test_query_studies_populates_state() -> None:
    state = AgentState()
    client = FakeClient()
    result = execute_tool("query_studies", {"patient_id": "123"}, client, state)

    assert result.ok is True
    assert state.search_results
    assert state.search_results[0].study_instance_uid == "1.2.3"
    assert "study_date" in state.search_filters
    assert state.guardrail_date_range


def test_invalid_uid_blocked_for_series() -> None:
    state = AgentState()
    client = FakeClient()
    result = execute_tool("query_series", {"study_instance_uid": "bad-uid"}, client, state)

    assert result.ok is False
    assert result.error
    assert result.error.code == "validation_error"


def test_uid_not_in_state_blocks_move() -> None:
    state = AgentState()
    client = FakeClient()
    result = execute_tool(
        "move_study",
        {"study_instance_uid": "1.2.3", "destination_node": "dest"},
        client,
        state,
    )

    assert result.ok is False
    assert result.error
    assert result.error.code == "uid_not_in_state"


def test_extra_fields_rejected() -> None:
    state = AgentState()
    client = FakeClient()
    result = execute_tool(
        "query_studies",
        {"patient_id": "123", "unexpected": "nope"},
        client,
        state,
    )

    assert result.ok is False
    assert result.error
    assert result.error.code == "validation_error"


def test_tools_schema_is_strict() -> None:
    schema = get_tools_schema([ToolName.QUERY_STUDIES])

    assert len(schema) == 1
    parameters = schema[0]["function"]["parameters"]
    assert parameters.get("additionalProperties") is False
