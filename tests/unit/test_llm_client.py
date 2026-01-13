from __future__ import annotations

import json

import httpx
import pytest

from dicom_nlquery.llm_client import (
    OllamaClient,
    OpenAIClient,
    create_llm_client,
)
from dicom_nlquery.models import LLMConfig


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


# OpenAI Client Tests


def test_openai_client_parses_response() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        assert request.url == httpx.URL("http://vllm:8001/v1/chat/completions")
        payload = json.loads(request.content)
        assert payload["model"] == "default"
        assert "Authorization" in request.headers
        return httpx.Response(
            200,
            json={
                "choices": [{"message": {"content": "{\"ok\": true}"}}]
            },
        )

    transport = httpx.MockTransport(handler)
    client = httpx.Client(transport=transport)
    llm = OpenAIClient(
        base_url="http://vllm:8001",
        model="default",
        client=client,
    )

    assert llm.chat("system", "user") == "{\"ok\": true}"


def test_openai_client_includes_stop_and_max_tokens() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        payload = json.loads(request.content)
        assert payload["max_tokens"] == 256
        assert payload["max_completion_tokens"] == 128
        assert payload["stop"] == ["<|eot_id|>"]
        return httpx.Response(
            200,
            json={
                "choices": [{"message": {"content": "ok"}}]
            },
        )

    transport = httpx.MockTransport(handler)
    client = httpx.Client(transport=transport)
    llm = OpenAIClient(
        base_url="http://vllm:8001",
        model="default",
        client=client,
        max_tokens=256,
        max_completion_tokens=128,
        stop=["<|eot_id|>"],
    )

    assert llm.chat("system", "user") == "ok"


def test_openai_client_json_schema_mode() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        payload = json.loads(request.content)
        assert payload.get("response_format", {}).get("type") == "json_schema"
        return httpx.Response(
            200,
            json={
                "choices": [{"message": {"content": "{\"result\": 42}"}}]
            },
        )

    transport = httpx.MockTransport(handler)
    client = httpx.Client(transport=transport)
    llm = OpenAIClient(
        base_url="http://vllm:8001",
        model="default",
        client=client,
        response_format="json_schema",
    )

    schema = {
        "type": "object",
        "properties": {"result": {"type": "integer"}},
        "required": ["result"],
        "additionalProperties": False,
    }

    result = llm.chat("system", "user", json_mode=True, json_schema=schema)
    assert result == "{\"result\": 42}"


def test_openai_client_json_schema_fallback_on_error() -> None:
    call_count = {"count": 0}

    def handler(request: httpx.Request) -> httpx.Response:
        call_count["count"] += 1
        payload = json.loads(request.content)
        if payload.get("response_format", {}).get("type") == "json_schema":
            return httpx.Response(400, json={"error": {"message": "unsupported"}})
        return httpx.Response(
            200,
            json={
                "choices": [{"message": {"content": "{\"ok\": true}"}}]
            },
        )

    transport = httpx.MockTransport(handler)
    client = httpx.Client(transport=transport)
    llm = OpenAIClient(
        base_url="http://vllm:8001",
        model="default",
        client=client,
        response_format="json_schema",
    )

    schema = {
        "type": "object",
        "properties": {"ok": {"type": "boolean"}},
        "required": ["ok"],
        "additionalProperties": False,
    }

    result = llm.chat("system", "user", json_mode=True, json_schema=schema)
    assert result == "{\"ok\": true}"
    assert call_count["count"] == 2


def test_openai_client_json_mode() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        payload = json.loads(request.content)
        assert payload.get("response_format") == {"type": "json_object"}
        return httpx.Response(
            200,
            json={
                "choices": [{"message": {"content": "{\"result\": 42}"}}]
            },
        )

    transport = httpx.MockTransport(handler)
    client = httpx.Client(transport=transport)
    llm = OpenAIClient(
        base_url="http://vllm:8001",
        model="default",
        client=client,
    )

    result = llm.chat("system", "user", json_mode=True)
    assert result == "{\"result\": 42}"


def test_openai_client_json_mode_content_dict() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        payload = json.loads(request.content)
        assert payload.get("response_format") == {"type": "json_object"}
        return httpx.Response(
            200,
            json={
                "choices": [{"message": {"content": {"result": 42}}}]
            },
        )

    transport = httpx.MockTransport(handler)
    client = httpx.Client(transport=transport)
    llm = OpenAIClient(
        base_url="http://vllm:8001",
        model="default",
        client=client,
    )

    result = llm.chat("system", "user", json_mode=True)
    assert result == "{\"result\": 42}"


def test_openai_client_json_mode_tool_call_fallback() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        payload = json.loads(request.content)
        assert payload.get("response_format") == {"type": "json_object"}
        return httpx.Response(
            200,
            json={
                "choices": [
                    {
                        "message": {
                            "content": None,
                            "tool_calls": [
                                {
                                    "id": "call_1",
                                    "function": {
                                        "name": "json_response",
                                        "arguments": "{\"ok\": true}",
                                    },
                                }
                            ],
                        }
                    }
                ]
            },
        )

    transport = httpx.MockTransport(handler)
    client = httpx.Client(transport=transport)
    llm = OpenAIClient(
        base_url="http://vllm:8001",
        model="default",
        client=client,
    )

    result = llm.chat("system", "user", json_mode=True)
    assert result == "{\"ok\": true}"


def test_openai_client_json_mode_tool_call_arguments_at_top_level() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        payload = json.loads(request.content)
        assert payload.get("response_format") == {"type": "json_object"}
        return httpx.Response(
            200,
            json={
                "choices": [
                    {
                        "message": {
                            "content": None,
                            "tool_calls": [
                                {
                                    "id": "call_2",
                                    "name": "json_response",
                                    "arguments": {"ok": True},
                                }
                            ],
                        }
                    }
                ]
            },
        )

    transport = httpx.MockTransport(handler)
    client = httpx.Client(transport=transport)
    llm = OpenAIClient(
        base_url="http://vllm:8001",
        model="default",
        client=client,
    )

    result = llm.chat("system", "user", json_mode=True)
    assert result == "{\"ok\": true}"


def test_openai_client_json_mode_choice_tool_call_fallback() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        payload = json.loads(request.content)
        assert payload.get("response_format") == {"type": "json_object"}
        return httpx.Response(
            200,
            json={
                "choices": [
                    {
                        "message": {
                            "content": None,
                        },
                        "tool_calls": [
                            {
                                "id": "call_2",
                                "function": {
                                    "name": "json_response",
                                    "arguments": {"ok": True},
                                },
                            }
                        ],
                    }
                ]
            },
        )

    transport = httpx.MockTransport(handler)
    client = httpx.Client(transport=transport)
    llm = OpenAIClient(
        base_url="http://vllm:8001",
        model="default",
        client=client,
    )

    result = llm.chat("system", "user", json_mode=True)
    assert result == "{\"ok\": true}"


def test_openai_client_json_mode_function_call_fallback() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        payload = json.loads(request.content)
        assert payload.get("response_format") == {"type": "json_object"}
        return httpx.Response(
            200,
            json={
                "choices": [
                    {
                        "message": {
                            "content": None,
                            "function_call": {"arguments": "{\"ok\": true}"},
                        }
                    }
                ]
            },
        )

    transport = httpx.MockTransport(handler)
    client = httpx.Client(transport=transport)
    llm = OpenAIClient(
        base_url="http://vllm:8001",
        model="default",
        client=client,
    )

    result = llm.chat("system", "user", json_mode=True)
    assert result == "{\"ok\": true}"


def test_openai_client_json_mode_reasoning_fallback() -> None:
    reasoning = "Output JSON: {\"ok\": true}"

    def handler(request: httpx.Request) -> httpx.Response:
        payload = json.loads(request.content)
        assert payload.get("response_format") == {"type": "json_object"}
        return httpx.Response(
            200,
            json={
                "choices": [
                    {
                        "message": {
                            "content": None,
                            "reasoning": reasoning,
                        }
                    }
                ]
            },
        )

    transport = httpx.MockTransport(handler)
    client = httpx.Client(transport=transport)
    llm = OpenAIClient(
        base_url="http://vllm:8001",
        model="default",
        client=client,
    )

    result = llm.chat("system", "user", json_mode=True)
    assert result == reasoning


def test_openai_client_with_tools() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        payload = json.loads(request.content)
        assert "tools" in payload
        return httpx.Response(
            200,
            json={
                "choices": [
                    {
                        "message": {
                            "role": "assistant",
                            "content": "",
                            "tool_calls": [
                                {
                                    "id": "call_123",
                                    "function": {
                                        "name": "query_studies",
                                        "arguments": '{"modality_in_study": "MR"}',
                                    },
                                }
                            ],
                        }
                    }
                ]
            },
        )

    transport = httpx.MockTransport(handler)
    client = httpx.Client(transport=transport)
    llm = OpenAIClient(
        base_url="http://vllm:8001",
        model="default",
        client=client,
    )

    tools = [{"type": "function", "function": {"name": "query_studies"}}]
    result = llm.chat_with_tools([{"role": "user", "content": "test"}], tools)

    assert "tool_calls" in result
    assert len(result["tool_calls"]) == 1
    assert result["tool_calls"][0]["function"]["name"] == "query_studies"
    # Arguments should be parsed from JSON string
    assert result["tool_calls"][0]["function"]["arguments"] == {"modality_in_study": "MR"}


def test_openai_client_connection_error() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("boom", request=request)

    transport = httpx.MockTransport(handler)
    client = httpx.Client(transport=transport)
    llm = OpenAIClient(
        base_url="http://vllm:8001",
        model="default",
        client=client,
    )

    with pytest.raises(httpx.RequestError):
        llm.chat("system", "user")


def test_openai_client_timeout() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ReadTimeout("timeout", request=request)

    transport = httpx.MockTransport(handler)
    client = httpx.Client(transport=transport)
    llm = OpenAIClient(
        base_url="http://vllm:8001",
        model="default",
        client=client,
    )

    with pytest.raises(httpx.TimeoutException):
        llm.chat("system", "user")


# Factory Tests


def test_create_llm_client_openai() -> None:
    config = LLMConfig(
        provider="openai",
        base_url="http://vllm:8001",
        model="default",
    )
    client = create_llm_client(config)
    assert isinstance(client, OpenAIClient)


def test_create_llm_client_ollama() -> None:
    config = LLMConfig(
        provider="ollama",
        base_url="http://localhost:11434",
        model="llama3.2:latest",
    )
    client = create_llm_client(config)
    assert isinstance(client, OllamaClient)


def test_create_llm_client_lmstudio() -> None:
    config = LLMConfig(
        provider="lmstudio",
        base_url="http://localhost:1234",
        model="local-model",
    )
    client = create_llm_client(config)
    # lmstudio uses same API as Ollama
    assert isinstance(client, OllamaClient)
