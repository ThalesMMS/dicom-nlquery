from __future__ import annotations

from dicom_nlquery.agent import DicomAgent
from dicom_nlquery.lexicon import Lexicon


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


def test_uid_propagation_allows_move() -> None:
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
    agent = DicomAgent(llm, client, max_steps=5)

    response = agent.run("Move study for patient 123 to dest")

    assert "C-MOVE" in response
    assert agent.state.selected_uid == "1.2.3"
    assert any(call[0] == "query_studies" for call in client.calls)


def test_stringified_arguments_are_parsed() -> None:
    client = FakeClient()
    llm = FakeLLM(
        [
            {
                "role": "assistant",
                "tool_calls": [
                    {
                        "function": {
                            "name": "query_studies",
                            "arguments": '{"patient_id": "123"}',
                        }
                    }
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
    agent = DicomAgent(llm, client, max_steps=5)

    response = agent.run("Move study for patient 123 to dest")

    assert "C-MOVE" in response
    assert any(call[0] == "query_studies" for call in client.calls)


def test_destination_node_is_captured_for_move() -> None:
    client = FakeClient()
    llm = FakeLLM(
        [
            {
                "role": "assistant",
                "tool_calls": [
                    {
                        "function": {
                            "name": "query_studies",
                            "arguments": {"patient_id": "123", "destination_node": "dest"},
                        }
                    }
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
                            "arguments": {"study_instance_uid": "1.2.3"},
                        }
                    }
                ],
                "content": "",
            },
        ]
    )
    agent = DicomAgent(llm, client, max_steps=5)

    response = agent.run("Move study for patient 123 to dest")

    assert "C-MOVE" in response
    assert ("move_study", "dest", "1.2.3") in client.calls


def test_protocol_violation_blocks_json_in_text() -> None:
    client = FakeClient()
    llm = FakeLLM(
        [
            {
                "role": "assistant",
                "content": '{"name": "query_studies", "arguments": {"patient_id": "123"}}',
            }
        ]
    )
    agent = DicomAgent(llm, client, max_steps=1)

    response = agent.run("Search studies")

    assert "PROTOCOL ERROR" in response
    assert client.calls == []


def test_missing_tool_call_when_required_is_blocked() -> None:
    client = FakeClient()
    llm = FakeLLM([{"role": "assistant", "content": "I will search now."}])
    agent = DicomAgent(llm, client, max_steps=1)

    response = agent.run("Search studies")

    assert "PROTOCOL ERROR" in response
    assert client.calls == []


def test_query_series_before_search_is_blocked() -> None:
    client = FakeClient()
    llm = FakeLLM(
        [
            {
                "role": "assistant",
                "tool_calls": [
                    {"function": {"name": "query_series", "arguments": {"study_instance_uid": "1.2.3"}}}
                ],
                "content": "",
            }
        ]
    )
    agent = DicomAgent(llm, client, max_steps=1)

    response = agent.run("Inspect series")

    assert "PROTOCOL ERROR" in response
    assert client.calls == []


def test_evidence_check_blocks_unmentioned_sex_filter() -> None:
    client = FakeClient()
    llm = FakeLLM(
        [
            {
                "role": "assistant",
                "tool_calls": [
                    {"function": {"name": "query_studies", "arguments": {"patient_sex": "F"}}}
                ],
                "content": "",
            }
        ]
    )
    agent = DicomAgent(llm, client, max_steps=1)

    response = agent.run("Search cranial studies")

    assert "PROTOCOL ERROR" in response
    assert "patient_sex" in response
    assert client.calls == []


def test_evidence_check_allows_lexicon_synonym() -> None:
    client = FakeClient()
    llm = FakeLLM(
        [
            {
                "role": "assistant",
                "tool_calls": [
                    {
                        "function": {
                            "name": "query_studies",
                            "arguments": {"study_description": "fetal"},
                        }
                    }
                ],
                "content": "",
            }
        ]
    )
    lexicon = Lexicon()
    lexicon.update({"fetus": ["fetal"]})
    agent = DicomAgent(llm, client, max_steps=1, lexicon=lexicon)

    response = agent.run("Search MR for fetus")

    assert "PROTOCOL ERROR" not in response
    assert client.calls


def test_move_without_search_blocked() -> None:
    client = FakeClient()
    llm = FakeLLM(
        [
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
            }
        ]
    )
    agent = DicomAgent(llm, client, max_steps=1)

    response = agent.run("Move directly")

    assert "PROTOCOL ERROR" in response
    assert client.calls == []


def test_multi_result_requires_explicit_uid() -> None:
    class MultiClient(FakeClient):
        def query_studies(self, **kwargs):
            self.calls.append(("query_studies", kwargs))
            return [
                {"StudyInstanceUID": "1.2.3", "StudyDate": "20240101"},
                {"StudyInstanceUID": "1.2.4", "StudyDate": "20240201"},
            ]

    client = MultiClient()
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
                    {"function": {"name": "query_studies", "arguments": {"patient_id": "123"}}}
                ],
                "content": "",
            },
        ]
    )
    agent = DicomAgent(llm, client, max_steps=2)

    response = agent.run("Search studies for patient 123")

    assert "multiple studies" in response.lower()


def test_inferred_study_date_blocked() -> None:
    client = FakeClient()
    llm = FakeLLM(
        [
            {
                "role": "assistant",
                "tool_calls": [
                    {
                        "function": {
                            "name": "query_studies",
                            "arguments": {"study_date": "20200101-20201231"},
                        }
                    }
                ],
                "content": "",
            }
        ]
    )
    agent = DicomAgent(llm, client, max_steps=1)

    response = agent.run("Search studies")

    assert "study_date" in response
    assert client.calls == []


def test_guardrail_date_range_allowed() -> None:
    client = FakeClient()
    llm = FakeLLM(
        [
            {
                "role": "assistant",
                "tool_calls": [
                    {
                        "function": {
                            "name": "query_studies",
                            "arguments": {"study_date": "20000101-20001231"},
                        }
                    }
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
    agent = DicomAgent(llm, client, max_steps=3)
    agent.state.guardrail_date_range = "20000101-20001231"

    response = agent.run("Search studies")

    assert "C-MOVE" in response
