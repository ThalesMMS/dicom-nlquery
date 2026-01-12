from __future__ import annotations

import json
import logging
import time
import uuid
from abc import ABC, abstractmethod
from typing import Any

import httpx

from .models import LLMConfig


def _build_timeout(timeout: float) -> httpx.Timeout:
    return httpx.Timeout(connect=10.0, read=timeout, write=timeout, pool=timeout)


class LLMClient(ABC):
    """Abstract base class for LLM clients."""

    @abstractmethod
    def chat(
        self,
        system_prompt: str,
        user_prompt: str,
        json_mode: bool = False,
        json_schema: dict[str, Any] | None = None,
    ) -> str:
        """Send a chat request and return the response content."""
        ...

    @abstractmethod
    def chat_with_tools(self, messages: list, tools: list) -> dict:
        """Send a chat request with tool definitions and return the response."""
        ...


class OllamaClient(LLMClient):
    """Client for Ollama API."""

    def __init__(
        self,
        base_url: str,
        model: str,
        temperature: float = 0.1,
        timeout: int = 60,
        client: httpx.Client | None = None,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.temperature = temperature
        self.timeout = timeout
        self._client = client or httpx.Client(timeout=_build_timeout(timeout))

    @classmethod
    def from_config(cls, config: LLMConfig) -> "OllamaClient":
        return cls(
            base_url=config.base_url,
            model=config.model,
            temperature=config.temperature,
            timeout=config.timeout,
        )

    def chat(
        self,
        system_prompt: str,
        user_prompt: str,
        json_mode: bool = False,
        json_schema: dict[str, Any] | None = None,
    ) -> str:
        log = logging.getLogger(__name__)
        log.debug(
            "Sending LLM request",
            extra={
                "extra_data": {
                    "model": self.model,
                    "base_url": self.base_url,
                    "system_prompt_len": len(system_prompt),
                    "user_prompt_len": len(user_prompt),
                }
            },
        )
        payload: dict[str, Any] = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            "stream": False,
            "options": {"temperature": self.temperature},
        }
        if json_mode:
            payload["format"] = "json"
            # Parsing should be deterministic and short to avoid timeouts on small instruct models.
            payload["options"]["temperature"] = 0
            payload["options"]["num_predict"] = 512
        response = self._client.post(f"{self.base_url}/api/chat", json=payload)
        response.raise_for_status()
        data = response.json()
        if "error" in data:
            raise RuntimeError(data["error"])
        message = data.get("message") or {}
        content = message.get("content")
        if content is None:
            raise ValueError("LLM response missing content")
        log.debug(
            "LLM response received",
            extra={"extra_data": {"content_len": len(content)}},
        )
        return content

    def chat_with_tools(self, messages: list, tools: list) -> dict:
        payload = {
            "model": self.model,
            "messages": messages,
            "tools": tools,
            "stream": False,
            "options": {"temperature": 0},
        }
        response = self._client.post(f"{self.base_url}/api/chat", json=payload)
        response.raise_for_status()
        message = response.json().get("message", {})
        if "tool_calls" in message and not isinstance(message.get("tool_calls"), list):
            raise ValueError("tool_calls must be a list")
        for call in message.get("tool_calls", []) or []:
            function = call.get("function") if isinstance(call, dict) else None
            if not function or "name" not in function:
                raise ValueError("tool_call missing function name")
        return message


class OpenAIClient(LLMClient):
    """Client for OpenAI-compatible APIs (OpenAI, vLLM, etc.)."""

    _json_schema_support_cache: dict[tuple[str, str], bool] = {}

    def __init__(
        self,
        base_url: str,
        model: str,
        temperature: float = 0.1,
        timeout: int = 60,
        api_key: str | None = None,
        max_tokens: int | None = None,
        max_completion_tokens: int | None = None,
        stop: list[str] | None = None,
        response_format: str = "json_object",
        client: httpx.Client | None = None,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.temperature = temperature
        self.timeout = timeout
        self.api_key = api_key or "not-needed"
        self.max_tokens = max_tokens
        self.max_completion_tokens = max_completion_tokens
        self.stop = stop
        self.response_format = response_format
        self._client = client or httpx.Client(timeout=_build_timeout(timeout))

    @classmethod
    def from_config(cls, config: LLMConfig) -> "OpenAIClient":
        return cls(
            base_url=config.base_url,
            model=config.model,
            temperature=config.temperature,
            timeout=config.timeout,
            api_key=config.api_key,
            max_tokens=config.max_tokens,
            max_completion_tokens=config.max_completion_tokens,
            stop=config.stop,
            response_format=config.response_format,
        )

    def _headers(self, request_id: str | None = None) -> dict[str, str]:
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}",
        }
        if request_id:
            headers["X-Request-ID"] = request_id
        return headers

    def _should_log_debug_payload(self) -> bool:
        root = logging.getLogger()
        for handler in root.handlers:
            formatter = getattr(handler, "formatter", None)
            if getattr(formatter, "_show_extra", False):
                return True
        return False

    def _get_json_schema_support(self) -> bool:
        key = (self.base_url, self.model)
        return self._json_schema_support_cache.get(key, True)

    def _set_json_schema_support(self, supported: bool) -> None:
        key = (self.base_url, self.model)
        self._json_schema_support_cache[key] = supported

    def _normalize_content(self, value: Any) -> str | None:
        if isinstance(value, str):
            return value
        if isinstance(value, (dict, list)):
            try:
                return json.dumps(value, ensure_ascii=True)
            except TypeError:
                return None
        return None

    def _extract_tool_content(self, message: dict[str, Any]) -> str | None:
        tool_calls = message.get("tool_calls")
        if tool_calls is None:
            tool_calls = message.get("tool_call")
        if isinstance(tool_calls, dict):
            tool_calls = [tool_calls]
        if not isinstance(tool_calls, list) or not tool_calls:
            return None
        first_call = tool_calls[0]
        if not isinstance(first_call, dict):
            return None
        function = first_call.get("function")
        arguments: Any = None
        if isinstance(function, dict):
            arguments = function.get("arguments")
        if arguments is None:
            for key in ("arguments", "args", "input", "parameters"):
                if key in first_call:
                    arguments = first_call.get(key)
                    break
        if arguments is None:
            return None
        if isinstance(arguments, str):
            return arguments
        try:
            return json.dumps(arguments, ensure_ascii=True)
        except TypeError:
            return None

    def _extract_function_call_content(self, message: dict[str, Any]) -> str | None:
        function_call = message.get("function_call")
        if function_call is None:
            return None
        if isinstance(function_call, str):
            return function_call
        if not isinstance(function_call, dict):
            return None
        arguments = function_call.get("arguments")
        if arguments is None:
            return None
        if isinstance(arguments, str):
            return arguments
        try:
            return json.dumps(arguments, ensure_ascii=True)
        except TypeError:
            return None

    def _extract_reasoning_content(self, message: dict[str, Any]) -> str | None:
        for key in ("reasoning_content", "reasoning"):
            candidate = self._normalize_content(message.get(key))
            if candidate:
                return candidate
        return None

    def chat(
        self,
        system_prompt: str,
        user_prompt: str,
        json_mode: bool = False,
        json_schema: dict[str, Any] | None = None,
    ) -> str:
        log = logging.getLogger(__name__)
        request_id = str(uuid.uuid4())
        log.debug(
            "Sending LLM request",
            extra={
                "extra_data": {
                    "request_id": request_id,
                    "model": self.model,
                    "base_url": self.base_url,
                    "system_prompt_len": len(system_prompt),
                    "user_prompt_len": len(user_prompt),
                }
            },
        )
        max_tokens = self.max_tokens
        if max_tokens is None and self.max_completion_tokens is not None:
            max_tokens = self.max_completion_tokens
        if json_mode and max_tokens is None:
            max_tokens = 512
        payload: dict[str, Any] = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            "temperature": self.temperature,
        }
        if max_tokens is not None:
            payload["max_tokens"] = max_tokens
        if self.max_completion_tokens is not None:
            payload["max_completion_tokens"] = self.max_completion_tokens
        if self.stop:
            payload["stop"] = self.stop
        use_json_schema = False
        if json_mode:
            payload["temperature"] = 0
            if (
                json_schema
                and self.response_format in {"json_schema", "auto"}
                and self._get_json_schema_support()
            ):
                payload["response_format"] = {
                    "type": "json_schema",
                    "json_schema": {
                        "name": "search_criteria",
                        "schema": json_schema,
                    },
                }
                use_json_schema = True
            else:
                payload["response_format"] = {"type": "json_object"}

        def _send_request(payload_data: dict[str, Any], current_request_id: str) -> httpx.Response:
            start = time.monotonic()
            try:
                response = self._client.post(
                    f"{self.base_url}/v1/chat/completions",
                    json=payload_data,
                    headers=self._headers(current_request_id),
                )
            except httpx.RequestError as exc:
                duration_ms = int((time.monotonic() - start) * 1000)
                log.debug(
                    "LLM request failed",
                    extra={
                        "extra_data": {
                            "request_id": current_request_id,
                            "duration_ms": duration_ms,
                            "error": str(exc),
                        }
                    },
                )
                raise
            duration_ms = int((time.monotonic() - start) * 1000)
            log.debug(
                "LLM request completed",
                extra={
                    "extra_data": {
                        "request_id": current_request_id,
                        "status_code": response.status_code,
                        "duration_ms": duration_ms,
                    }
                },
            )
            response.raise_for_status()
            return response

        try:
            response = _send_request(payload, request_id)
        except httpx.HTTPStatusError as exc:
            status_code = exc.response.status_code if exc.response else None
            if use_json_schema and status_code in {400, 422}:
                self._set_json_schema_support(False)
                log.debug(
                    "LLM json_schema unsupported; retrying with json_object",
                    extra={
                        "extra_data": {
                            "request_id": request_id,
                            "status_code": status_code,
                        }
                    },
                )
                payload["response_format"] = {"type": "json_object"}
                request_id = str(uuid.uuid4())
                response = _send_request(payload, request_id)
                use_json_schema = False
            else:
                raise

        data = response.json()

        if "error" in data:
            if use_json_schema:
                self._set_json_schema_support(False)
                log.debug(
                    "LLM json_schema error; retrying with json_object",
                    extra={"extra_data": {"request_id": request_id}},
                )
                payload["response_format"] = {"type": "json_object"}
                request_id = str(uuid.uuid4())
                response = _send_request(payload, request_id)
                data = response.json()
            if "error" in data:
                raise RuntimeError(data["error"].get("message", str(data["error"])))

        choices = data.get("choices") or []
        if not choices:
            raise ValueError("LLM response missing choices")

        choice = choices[0]
        message = choice.get("message")
        if not isinstance(message, dict):
            message = {}
        content = self._normalize_content(message.get("content"))
        if content is None or content == "":
            tool_content = (
                self._extract_tool_content(message)
                or self._extract_tool_content(choice)
                or self._extract_function_call_content(message)
                or self._extract_function_call_content(choice)
            )
            if tool_content is not None:
                log.debug(
                    "LLM tool response received",
                    extra={
                        "extra_data": {
                            "request_id": request_id,
                            "content_len": len(tool_content),
                        }
                    },
                )
                return tool_content
            if json_mode:
                reasoning_content = (
                    self._extract_reasoning_content(message)
                    or self._extract_reasoning_content(choice)
                )
                if reasoning_content and "{" in reasoning_content and "}" in reasoning_content:
                    log.debug(
                        "LLM reasoning response received",
                        extra={
                            "extra_data": {
                                "request_id": request_id,
                                "content_len": len(reasoning_content),
                            }
                        },
                    )
                    return reasoning_content
        if content is None:
            text = choice.get("text")
            if isinstance(text, str) and text:
                log.debug(
                    "LLM completion response received",
                    extra={
                        "extra_data": {
                            "request_id": request_id,
                            "content_len": len(text),
                        }
                    },
                )
                return text
            raise ValueError("LLM response missing content")

        if self._should_log_debug_payload():
            log.info(
                "LLM response snippet",
                extra={
                    "extra_data": {
                        "request_id": request_id,
                        "content_snippet": content[:200],
                    }
                },
            )
        log.debug(
            "LLM response received",
            extra={
                "extra_data": {
                    "request_id": request_id,
                    "content_len": len(content),
                }
            },
        )
        return content

    def chat_with_tools(self, messages: list, tools: list) -> dict:
        log = logging.getLogger(__name__)
        request_id = str(uuid.uuid4())
        payload: dict[str, Any] = {
            "model": self.model,
            "messages": messages,
            "temperature": 0,
        }
        if self.max_tokens is not None:
            payload["max_tokens"] = self.max_tokens
        if self.max_completion_tokens is not None:
            payload["max_completion_tokens"] = self.max_completion_tokens
        if self.stop:
            payload["stop"] = self.stop
        if tools:
            payload["tools"] = tools
            payload["tool_choice"] = "auto"

        log.debug(
            "Sending LLM tool request",
            extra={
                "extra_data": {
                    "request_id": request_id,
                    "model": self.model,
                    "message_count": len(messages),
                    "tool_count": len(tools),
                }
            },
        )

        start = time.monotonic()
        response = self._client.post(
            f"{self.base_url}/v1/chat/completions",
            json=payload,
            headers=self._headers(request_id),
        )
        duration_ms = int((time.monotonic() - start) * 1000)
        log.debug(
            "LLM tool request completed",
            extra={
                "extra_data": {
                    "request_id": request_id,
                    "status_code": response.status_code,
                    "duration_ms": duration_ms,
                }
            },
        )
        response.raise_for_status()
        data = response.json()

        if "error" in data:
            raise RuntimeError(data["error"].get("message", str(data["error"])))

        choices = data.get("choices") or []
        if not choices:
            raise ValueError("LLM response missing choices")

        message = choices[0].get("message") or {}

        # Normalize OpenAI tool_calls format to match expected structure
        tool_calls = message.get("tool_calls") or []
        normalized_calls = []
        for call in tool_calls:
            if not isinstance(call, dict):
                continue
            function = call.get("function") or {}
            name = function.get("name")
            if not name:
                raise ValueError("tool_call missing function name")
            # Parse arguments if they're a string
            arguments = function.get("arguments", {})
            if isinstance(arguments, str):
                try:
                    arguments = json.loads(arguments)
                except json.JSONDecodeError:
                    pass
            normalized_calls.append({
                "id": call.get("id"),
                "function": {
                    "name": name,
                    "arguments": arguments,
                },
            })

        result = {
            "role": message.get("role", "assistant"),
            "content": message.get("content") or "",
        }
        if normalized_calls:
            result["tool_calls"] = normalized_calls

        return result


def create_llm_client(config: LLMConfig) -> LLMClient:
    """Factory function to create the appropriate LLM client based on config."""
    if config.provider == "openai":
        return OpenAIClient.from_config(config)
    else:
        # ollama and lmstudio use the same API format
        return OllamaClient.from_config(config)
