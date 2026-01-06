from __future__ import annotations

import json

import httpx
import pytest

from dicom_nlquery.llm_client import OllamaClient


def test_llm_client_parses_response() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        assert request.url == httpx.URL("http://ollama/api/chat")
        payload = json.loads(request.content)
        assert payload["model"] == "llama3.2:latest"
        return httpx.Response(200, json={"message": {"content": "{\"ok\": true}"}})

    transport = httpx.MockTransport(handler)
    client = httpx.Client(transport=transport)
    llm = OllamaClient(
        base_url="http://ollama",
        model="llama3.2:latest",
        client=client,
    )

    assert llm.chat("system", "user") == "{\"ok\": true}"


def test_llm_client_connection_error() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("boom", request=request)

    transport = httpx.MockTransport(handler)
    client = httpx.Client(transport=transport)
    llm = OllamaClient(
        base_url="http://ollama",
        model="llama3.2:latest",
        client=client,
    )

    with pytest.raises(httpx.RequestError):
        llm.chat("system", "user")


def test_llm_client_timeout() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ReadTimeout("timeout", request=request)

    transport = httpx.MockTransport(handler)
    client = httpx.Client(transport=transport)
    llm = OllamaClient(
        base_url="http://ollama",
        model="llama3.2:latest",
        client=client,
    )

    with pytest.raises(httpx.TimeoutException):
        llm.chat("system", "user")
