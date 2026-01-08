from __future__ import annotations

import asyncio

import pytest

from dicom_nlquery.mcp_client import (
    McpSession,
    McpToolCallError,
    build_stdio_server_params,
)
from dicom_nlquery.models import McpRetryConfig, McpServerConfig


class FakeResult:
    def __init__(self, payload: object, is_error: bool = False) -> None:
        self.isError = is_error
        self.structuredContent = {"result": payload}
        self.content = []


class FakeSession:
    def __init__(self, responses: list[object]) -> None:
        self._responses = list(responses)
        self.calls = 0

    async def call_tool(self, name: str, arguments: dict[str, object]) -> object:
        self.calls += 1
        response = self._responses.pop(0)
        if isinstance(response, Exception):
            raise response
        return response


class TimeoutSession:
    def __init__(self) -> None:
        self.calls = 0

    async def call_tool(self, name: str, arguments: dict[str, object]) -> object:
        self.calls += 1
        event = asyncio.Event()
        await event.wait()


def _make_client(config: McpServerConfig, session: object) -> McpSession:
    server_params = build_stdio_server_params(config)
    client = McpSession(server_params, mcp_config=config)
    client._session = session
    return client


def _run(coro):
    return asyncio.run(coro)


def test_call_tool_retries_on_timeout() -> None:
    config = McpServerConfig(
        tool_timeout_seconds=0.01,
        retry=McpRetryConfig(max_attempts=3, backoff_seconds=[0, 0]),
    )
    session = TimeoutSession()
    client = _make_client(config, session)

    with pytest.raises(McpToolCallError) as excinfo:
        _run(client.call_tool("query_studies", {"patient_id": "123"}))

    assert session.calls == 3
    details = excinfo.value.details
    assert details["tool"] == "query_studies"
    assert details["attempt"] == 3
    assert details["retryable"] is True
    assert details["argument_keys"] == ["patient_id"]


def test_call_tool_no_retry_for_non_idempotent() -> None:
    config = McpServerConfig(
        tool_timeout_seconds=0.01,
        retry=McpRetryConfig(max_attempts=3, backoff_seconds=[0, 0]),
        non_idempotent_tools=["move_study"],
    )
    session = TimeoutSession()
    client = _make_client(config, session)

    with pytest.raises(McpToolCallError) as excinfo:
        _run(client.call_tool("move_study", {"study_instance_uid": "1"}))

    assert session.calls == 1
    assert excinfo.value.details["retryable"] is False


def test_call_tool_retries_on_os_error() -> None:
    config = McpServerConfig(
        tool_timeout_seconds=0.01,
        retry=McpRetryConfig(max_attempts=2, backoff_seconds=[0]),
    )
    session = FakeSession([OSError("boom"), FakeResult({"ok": True})])
    client = _make_client(config, session)

    result = _run(client.call_tool("query_series", {}))

    assert result == {"ok": True}
    assert session.calls == 2


def test_call_tool_does_not_retry_on_tool_error() -> None:
    config = McpServerConfig(
        tool_timeout_seconds=0.01,
        retry=McpRetryConfig(max_attempts=2, backoff_seconds=[0]),
    )
    session = FakeSession([FakeResult({"error": "nope"}, is_error=True)])
    client = _make_client(config, session)

    with pytest.raises(McpToolCallError) as excinfo:
        _run(client.call_tool("query_studies", {}))

    assert session.calls == 1
    assert excinfo.value.details["payload_summary"]["type"] == "dict"
