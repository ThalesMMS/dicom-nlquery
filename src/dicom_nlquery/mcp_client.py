from __future__ import annotations

import json
import logging
from typing import Any

from mcp.client.session import ClientSession
from mcp.client.stdio import StdioServerParameters, stdio_client

from .models import McpServerConfig


log = logging.getLogger(__name__)


def build_stdio_server_params(config: McpServerConfig) -> StdioServerParameters:
    args: list[str] = []
    if config.config_path:
        args.append(config.config_path)
    args.extend(config.args)
    return StdioServerParameters(
        command=config.command,
        args=args,
        cwd=config.cwd,
        env=config.env,
    )


class McpSession:
    def __init__(self, server_params: StdioServerParameters) -> None:
        self._server_params = server_params
        self._stdio_cm = None
        self._session_cm: ClientSession | None = None
        self._session: ClientSession | None = None

    async def __aenter__(self) -> "McpSession":
        self._stdio_cm = stdio_client(self._server_params)
        read_stream, write_stream = await self._stdio_cm.__aenter__()
        self._session_cm = ClientSession(read_stream, write_stream)
        self._session = await self._session_cm.__aenter__()
        await self._session.initialize()
        return self

    async def __aexit__(self, exc_type, exc, tb) -> None:
        if self._session_cm is not None:
            await self._session_cm.__aexit__(exc_type, exc, tb)
        if self._stdio_cm is not None:
            await self._stdio_cm.__aexit__(exc_type, exc, tb)
        self._stdio_cm = None
        self._session_cm = None
        self._session = None

    async def call_tool(self, name: str, arguments: dict[str, Any] | None = None) -> Any:
        if self._session is None:
            raise RuntimeError("MCP session not initialized")
        result = await self._session.call_tool(name=name, arguments=arguments or {})
        payload = _extract_tool_payload(result)
        if result.isError:
            raise RuntimeError(f"MCP tool '{name}' failed: {payload}")
        return payload

    async def query_studies(self, **kwargs) -> Any:
        return await self.call_tool("query_studies", kwargs)

    async def query_series(self, **kwargs) -> Any:
        return await self.call_tool("query_series", kwargs)

    async def switch_dicom_node(self, node_name: str) -> Any:
        return await self.call_tool("switch_dicom_node", {"node_name": node_name})

    async def move_study(self, destination_node: str, study_instance_uid: str) -> Any:
        return await self.call_tool(
            "move_study",
            {"destination_node": destination_node, "study_instance_uid": study_instance_uid},
        )


def _extract_tool_payload(result) -> Any:
    if result.structuredContent is not None:
        payload = result.structuredContent
        if isinstance(payload, dict) and set(payload.keys()) == {"result"}:
            return payload["result"]
        return payload
    for block in result.content:
        if getattr(block, "type", None) == "text":
            text = block.text.strip()
            if not text:
                continue
            try:
                payload = json.loads(text)
                if isinstance(payload, dict) and set(payload.keys()) == {"result"}:
                    return payload["result"]
                return payload
            except json.JSONDecodeError:
                return text
    return None
