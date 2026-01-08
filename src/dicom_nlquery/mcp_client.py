from __future__ import annotations

import asyncio
import json
import logging
from dataclasses import dataclass
from typing import Any

from mcp.client.session import ClientSession
from mcp.client.stdio import StdioServerParameters, stdio_client

from .models import McpServerConfig


log = logging.getLogger(__name__)

MANIFEST_TOOL_NAME = "get_manifest"
REQUIRED_MANIFEST_VERSION = "1.0"
REQUIRED_SCHEMA_VERSION = "1.0"
REQUIRED_TOOL_VERSIONS: dict[str, str] = {
    "list_dicom_nodes": "1.0",
    "query_studies": "1.0",
    "query_series": "1.0",
    "move_study": "1.0",
}


@dataclass(frozen=True)
class McpRetryPolicy:
    timeout_seconds: float
    max_attempts: int
    backoff_seconds: tuple[float, ...]
    non_idempotent_tools: frozenset[str]


DEFAULT_RETRY_POLICY = McpRetryPolicy(
    timeout_seconds=30.0,
    max_attempts=3,
    backoff_seconds=(0.5, 1.0, 2.0),
    non_idempotent_tools=frozenset({"move_study", "move_series"}),
)


class McpToolExecutionError(RuntimeError):
    def __init__(self, tool: str, payload: Any) -> None:
        super().__init__(f"MCP tool '{tool}' returned error")
        self.tool = tool
        self.payload = payload


class McpToolCallError(RuntimeError):
    def __init__(self, message: str, details: dict[str, Any]) -> None:
        super().__init__(message)
        self.details = details


class McpManifestError(RuntimeError):
    """Raised when MCP manifest validation fails."""


def _parse_version(value: str) -> tuple[int, int, int]:
    parts = value.strip().split(".")
    if not parts or any(not part.isdigit() for part in parts):
        raise ValueError(f"invalid version '{value}'")
    numbers = [int(part) for part in parts]
    while len(numbers) < 3:
        numbers.append(0)
    return tuple(numbers[:3])


def _is_compatible(required: str, actual: str) -> bool:
    required_tuple = _parse_version(required)
    actual_tuple = _parse_version(actual)
    if actual_tuple[0] != required_tuple[0]:
        return False
    return actual_tuple >= required_tuple


def validate_manifest_payload(manifest: dict[str, Any]) -> None:
    if not isinstance(manifest, dict):
        raise ValueError("manifest must be a dictionary")
    manifest_version = manifest.get("manifest_version")
    if not isinstance(manifest_version, str):
        raise ValueError("manifest_version must be a string")
    if not _is_compatible(REQUIRED_MANIFEST_VERSION, manifest_version):
        raise ValueError(
            f"manifest_version '{manifest_version}' is incompatible with '{REQUIRED_MANIFEST_VERSION}'"
        )
    schema_version = manifest.get("schema_version")
    if not isinstance(schema_version, str):
        raise ValueError("schema_version must be a string")
    if not _is_compatible(REQUIRED_SCHEMA_VERSION, schema_version):
        raise ValueError(
            f"schema_version '{schema_version}' is incompatible with '{REQUIRED_SCHEMA_VERSION}'"
        )
    tools = manifest.get("tools")
    if not isinstance(tools, dict):
        raise ValueError("tools must be a dictionary")
    required_tools = tools.get("required")
    if not isinstance(required_tools, dict):
        raise ValueError("tools.required must be a dictionary")
    for tool_name, required_version in REQUIRED_TOOL_VERSIONS.items():
        actual_version = required_tools.get(tool_name)
        if actual_version is None:
            raise ValueError(f"manifest missing required tool '{tool_name}'")
        if not isinstance(actual_version, str):
            raise ValueError(f"tool version for '{tool_name}' must be a string")
        if not _is_compatible(required_version, actual_version):
            raise ValueError(
                f"tool '{tool_name}' version '{actual_version}' is incompatible with '{required_version}'"
            )


def _policy_from_config(config: McpServerConfig | None) -> McpRetryPolicy:
    if config is None:
        return DEFAULT_RETRY_POLICY
    backoff = tuple(config.retry.backoff_seconds)
    non_idempotent = frozenset(tool.strip() for tool in config.non_idempotent_tools if tool.strip())
    return McpRetryPolicy(
        timeout_seconds=config.tool_timeout_seconds,
        max_attempts=config.retry.max_attempts,
        backoff_seconds=backoff,
        non_idempotent_tools=non_idempotent,
    )


def _backoff_for_attempt(attempt: int, policy: McpRetryPolicy) -> float:
    if not policy.backoff_seconds:
        return 0.0
    index = min(attempt - 1, len(policy.backoff_seconds) - 1)
    return policy.backoff_seconds[index]


def _summarize_payload(payload: Any) -> dict[str, Any] | None:
    if payload is None:
        return None
    if isinstance(payload, list):
        return {"type": "list", "count": len(payload)}
    if isinstance(payload, dict):
        keys = sorted(str(key) for key in payload.keys())
        return {"type": "dict", "keys": keys[:20]}
    if isinstance(payload, str):
        return {"type": "str", "length": len(payload)}
    return {"type": type(payload).__name__}


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
    def __init__(
        self,
        server_params: StdioServerParameters,
        mcp_config: McpServerConfig | None = None,
    ) -> None:
        self._server_params = server_params
        self._retry_policy = _policy_from_config(mcp_config)
        self._stdio_cm = None
        self._session_cm: ClientSession | None = None
        self._session: ClientSession | None = None

    async def __aenter__(self) -> "McpSession":
        self._stdio_cm = stdio_client(self._server_params)
        read_stream, write_stream = await self._stdio_cm.__aenter__()
        self._session_cm = ClientSession(read_stream, write_stream)
        self._session = await self._session_cm.__aenter__()
        try:
            await self._session.initialize()
            await self._ensure_manifest()
        except Exception as exc:
            await self.__aexit__(type(exc), exc, exc.__traceback__)
            raise
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
        arguments = arguments or {}
        policy = self._retry_policy
        max_attempts = policy.max_attempts
        if name in policy.non_idempotent_tools:
            max_attempts = 1

        for attempt in range(1, max_attempts + 1):
            try:
                return await self._call_tool_once(name, arguments, policy.timeout_seconds)
            except asyncio.CancelledError:
                raise
            except Exception as exc:
                retryable = _is_retryable(name, exc, policy)
                details = _build_error_details(
                    name=name,
                    arguments=arguments,
                    attempt=attempt,
                    max_attempts=max_attempts,
                    policy=policy,
                    retryable=retryable,
                    error=exc,
                )
                if not retryable or attempt >= max_attempts:
                    log.error("MCP tool call failed", extra={"extra_data": details})
                    raise McpToolCallError("MCP tool call failed", details) from exc
                backoff_seconds = _backoff_for_attempt(attempt, policy)
                log.warning(
                    "MCP tool call failed, retrying",
                    extra={"extra_data": {**details, "backoff_seconds": backoff_seconds}},
                )
                if backoff_seconds > 0:
                    await asyncio.sleep(backoff_seconds)
        raise RuntimeError("MCP tool call retry loop exited unexpectedly")

    async def _call_tool_once(
        self,
        name: str,
        arguments: dict[str, Any],
        timeout_seconds: float,
    ) -> Any:
        if self._session is None:
            raise RuntimeError("MCP session not initialized")
        call = self._session.call_tool(name=name, arguments=arguments)
        result = await asyncio.wait_for(call, timeout=timeout_seconds)
        payload = _extract_tool_payload(result)
        if result.isError:
            raise McpToolExecutionError(name, payload)
        return payload

    async def _ensure_manifest(self) -> None:
        try:
            manifest = await self.call_tool(MANIFEST_TOOL_NAME)
            validate_manifest_payload(manifest)
        except Exception as exc:  # pragma: no cover - handled by tests
            raise McpManifestError(f"MCP manifest validation failed: {exc}") from exc

    async def query_studies(self, **kwargs) -> Any:
        return await self.call_tool("query_studies", kwargs)

    async def query_series(self, **kwargs) -> Any:
        return await self.call_tool("query_series", kwargs)

    async def list_dicom_nodes(self) -> Any:
        return await self.call_tool("list_dicom_nodes")

    async def switch_dicom_node(self, node_name: str) -> Any:
        return await self.call_tool("switch_dicom_node", {"node_name": node_name})

    async def move_study(self, destination_node: str, study_instance_uid: str) -> Any:
        return await self.call_tool(
            "move_study",
            {"destination_node": destination_node, "study_instance_uid": study_instance_uid},
        )


def _is_retryable(tool: str, error: BaseException, policy: McpRetryPolicy) -> bool:
    if tool in policy.non_idempotent_tools:
        return False
    if isinstance(error, McpToolExecutionError):
        return False
    if isinstance(error, (asyncio.TimeoutError, TimeoutError, ConnectionError, OSError)):
        return True
    return False


def _build_error_details(
    *,
    name: str,
    arguments: dict[str, Any],
    attempt: int,
    max_attempts: int,
    policy: McpRetryPolicy,
    retryable: bool,
    error: BaseException,
) -> dict[str, Any]:
    payload_summary = None
    if isinstance(error, McpToolExecutionError):
        payload_summary = _summarize_payload(error.payload)
    return {
        "tool": name,
        "attempt": attempt,
        "max_attempts": max_attempts,
        "retryable": retryable,
        "timeout_seconds": policy.timeout_seconds,
        "error_type": type(error).__name__,
        "error_message": str(error),
        "argument_keys": sorted(arguments.keys()),
        "payload_summary": payload_summary,
        "non_idempotent": name in policy.non_idempotent_tools,
    }


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
