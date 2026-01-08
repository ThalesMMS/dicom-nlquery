from __future__ import annotations

from dicom_nlquery.agent import DicomAgent
from dicom_nlquery.models import ConfirmationConfig, ResolvedRequest, ResolverResult


class FakeClient:
    def __init__(self) -> None:
        self.calls = []

    def query_studies(self, **kwargs):
        self.calls.append(("query_studies", kwargs))
        return [
            {
                "StudyInstanceUID": "1.2.3",
                "StudyDate": "20240101",
                "StudyDescription": "TEST",
                "ModalitiesInStudy": "MR",
            }
        ]

    def query_series(self, **kwargs):
        self.calls.append(("query_series", kwargs))
        return [{"SeriesInstanceUID": "1.2.3.4"}]

    def move_study(self, destination_node: str, study_instance_uid: str):
        self.calls.append(("move_study", destination_node, study_instance_uid))
        return {"success": True, "completed": 1, "failed": 0}


class FakeLLM:
    def __init__(self, responses):
        self._responses = iter(responses)

    def chat_with_tools(self, messages, tools):
        return next(self._responses)


def test_agent_requires_confirmation_before_tools() -> None:
    client = FakeClient()
    llm = FakeLLM(
        [
            {
                "role": "assistant",
                "tool_calls": [
                    {"function": {"name": "query_studies", "arguments": {"patient_id": "123"}}}
                ],
                "content": "",
            },
            {
                "role": "assistant",
                "tool_calls": [
                    {"function": {"name": "query_series", "arguments": {"study_instance_uid": "1.2.3"}}}
                ],
                "content": "",
            },
            {
                "role": "assistant",
                "tool_calls": [
                    {
                        "function": {
                            "name": "move_study",
                            "arguments": {"study_instance_uid": "1.2.3", "destination_node": "dest"},
                        }
                    }
                ],
                "content": "",
            },
        ]
    )
    resolver_result = ResolverResult(
        request=ResolvedRequest(destination_node="dest", filters={"patient_id": "123"}),
        needs_confirmation=True,
        unresolved=[],
    )
    resolver = lambda _: resolver_result
    confirmation = ConfirmationConfig(accept_tokens=["yes"], reject_tokens=["no"])
    agent = DicomAgent(
        llm,
        client,
        max_steps=5,
        resolver=resolver,
        confirmation_config=confirmation,
        require_confirmation=True,
    )

    response = agent.run("Move study for patient 123 to dest")

    assert "Confirm" in response
    assert client.calls == []

    response = agent.run("yes")

    assert "C-MOVE" in response
    assert any(call[0] == "query_studies" for call in client.calls)
