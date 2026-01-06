from __future__ import annotations

import logging
import httpx

from .models import LLMConfig


class OllamaClient:
    def __init__(
        self,
        base_url: str,
        model: str,
        temperature: float = 0,
        timeout: int = 60,
        client: httpx.Client | None = None,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.temperature = temperature
        self.timeout = timeout
        self._client = client or httpx.Client(timeout=httpx.Timeout(timeout))

    @classmethod
    def from_config(cls, config: LLMConfig) -> "OllamaClient":
        return cls(
            base_url=config.base_url,
            model=config.model,
            temperature=config.temperature,
            timeout=config.timeout,
        )

    def chat(self, system_prompt: str, user_prompt: str) -> str:
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
        payload = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            "stream": False,
            "options": {"temperature": self.temperature},
        }
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
            "options": {"temperature": 0}
        }
        # Endpoint nativo do Ollama compatível com OpenAI
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
