from __future__ import annotations

from dicom_nlquery.node_registry import NodeRegistry


def test_match_nodes_with_aliases_and_longest_match() -> None:
    payload = [
        {"name": "ORTHANC", "ae_title": "ORTHANC", "aliases": ["local"]},
        {"name": "PACS ONE", "ae_title": "PACS1", "aliases": ["pacs"]},
    ]
    registry = NodeRegistry.from_tool_payload(payload)

    matches = registry.match("move from pacs one to orthanc")

    assert [match.node_id for match in matches] == ["PACS ONE", "ORTHANC"]


def test_match_requires_token_boundaries() -> None:
    registry = NodeRegistry(node_ids=["ORTHANC"], aliases={})

    matches = registry.match("orthancian study")

    assert matches == []


def test_alias_match_is_case_insensitive() -> None:
    payload = [{"name": "ORTHANC", "aliases": ["LOCAL"]}]
    registry = NodeRegistry.from_tool_payload(payload)

    matches = registry.match("send to local")

    assert [match.node_id for match in matches] == ["ORTHANC"]
