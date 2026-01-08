from __future__ import annotations

from dicom_nlquery.node_registry import NodeRegistry
from dicom_nlquery.resolver import resolve_request


class StubLLM:
    def __init__(self, response: str) -> None:
        self._response = response

    def chat(self, system_prompt: str, user_prompt: str) -> str:
        return self._response


def _registry() -> NodeRegistry:
    payload = [
        {"name": "ORTHANC", "ae_title": "ORTHANC"},
        {"name": "RADIANT", "ae_title": "RADIANT"},
    ]
    return NodeRegistry.from_tool_payload(payload)


def test_resolver_normalizes_nodes() -> None:
    llm = StubLLM(
        '{"source_node": "orthanc", "destination_node": "radiant", '
        '"filters": {"study_description": "aorta"}}'
    )

    result = resolve_request("move aorta from ORTHANC to RADIANT", _registry(), llm)

    assert result.request.source_node == "ORTHANC"
    assert result.request.destination_node == "RADIANT"
    assert result.request.filters["study_description"] == "aorta"
    assert result.unresolved == []


def test_resolver_strips_node_tokens_from_filters() -> None:
    llm = StubLLM(
        '{"source_node": "ORTHANC", "destination_node": "RADIANT", '
        '"filters": {"study_description": "orthanc aorta"}}'
    )

    result = resolve_request("orthanc aorta to radiant", _registry(), llm)

    assert result.request.filters["study_description"] == "aorta"
    assert "node_tokens_removed_from_filters" in result.unresolved


def test_resolver_flags_missing_destination() -> None:
    llm = StubLLM('{"source_node": "ORTHANC", "filters": {"patient_id": "1"}}')

    result = resolve_request("from ORTHANC patient 1", _registry(), llm)

    assert "missing_destination_node" in result.unresolved
