from __future__ import annotations

import json
import logging
import re
import unicodedata
from datetime import date
from typing import Any, Callable

from pydantic import ValidationError

from .agent_tools import execute_tool, get_tools_schema
from .confirmation import (
    build_confirmation_message,
    build_invalid_response_message,
    classify_confirmation_response,
)
from .lexicon import Lexicon, normalize_text as lexicon_normalize_text
from .llm_client import OllamaClient
from .models import (
    AgentPhase,
    AgentState,
    ConfirmationConfig,
    QueryStudiesArgs,
    ResolvedRequest,
    ToolName,
    ToolResult,
)

log = logging.getLogger(__name__)

SYSTEM_PROMPT = f"""
You are a specialist agent in Radiology and DICOM. Date: {date.today()}.

CRITICAL OPERATING RULES:
1. **TOOL CALLING**: Use only structured tools (tool_calls). Never send JSON in the text.
2. **NO INFERENCE**: Do not create filters the user did not explicitly mention.
3. **NO HALLUCINATION**: Never invent UIDs; use only UIDs returned by tools.
4. **SEQUENCE**: query_studies -> query_series -> move_study. Never skip steps.
5. **ONE TOOL AT A TIME**: In each turn, make at most 1 tool_call.
6. **REAL UID**: Do not use placeholders (<...>).
7. **SEX**: Only use patient_sex if the user explicitly states it.
8. **GUARDRAILS**: Default date range may be applied as operational protection.
9. **NODES**: Node identifiers come from an external registry and must not appear in filters.
"""

MAX_STEPS = 10
MAX_BROADEN_ATTEMPTS = 2
RELAXATION_ORDER = [
    "study_description",
    "modality_in_study",
    "patient_name",
    "patient_sex",
    "patient_birth_date",
    "accession_number",
    "study_date",
    "patient_id",
]

SEX_EVIDENCE = {
    "M": ["male", "man", "men"],
    "F": ["female", "woman", "women", "pregnant"],
    "O": ["other", "nonbinary", "non-binary"],
}
MODALITY_EVIDENCE = {
    "MR": ["mr", "mri", "magnetic resonance"],
    "CT": ["ct", "computed tomography", "cat"],
    "US": ["us", "ultrasound", "sonogram", "sonography"],
    "CR": ["x-ray", "xray", "radiograph", "radiography"],
    "DX": ["x-ray", "xray", "radiograph", "radiography"],
}


def _protocol_error(message: str) -> str:
    return f"PROTOCOL ERROR: {message}"


def _format_selection_prompt(state: AgentState) -> str:
    lines = ["Multiple studies found. Refine the search or choose a UID:"]
    for item in state.search_results[:5]:
        lines.append(
            f"- UID {item.study_instance_uid} | Date {item.study_date or '-'} | "
            f"Desc {item.study_description or '-'}"
        )
    if len(state.search_results) > 5:
        lines.append("(Showing only the first results.)")
    return "\n".join(lines)


def _format_move_result(result: ToolResult) -> str:
    data = result.data or {}
    if result.ok:
        return (
            "C-MOVE completed. "
            f"Completed={data.get('completed')} Failed={data.get('failed')} Warning={data.get('warning')}"
        )
    return f"C-MOVE failed: {data.get('message', 'Unknown error')}"


def _format_error(result: ToolResult) -> str:
    if result.error:
        return f"Error: {result.error.message}"
    return "Unknown error."


def _format_validation_feedback(details: Any | None) -> str:
    items: list[str] = []
    if isinstance(details, list):
        for entry in details:
            if not isinstance(entry, dict):
                items.append(str(entry))
                continue
            loc = entry.get("loc") or []
            field = ".".join(str(part) for part in loc) if loc else "?"
            msg = entry.get("msg") or entry.get("type") or "Validation error"
            items.append(f"{field}: {msg}")
    elif details:
        items.append(str(details))
    else:
        items.append("Validation error in parameters.")
    summary = "; ".join(items)
    return (
        "SYSTEM: Validation error in tool arguments. "
        "Fix and resend the tool_call with the schema. "
        f"Details: {summary}"
    )


def _normalize_text(text: str) -> str:
    normalized = unicodedata.normalize("NFKD", text)
    return "".join(c for c in normalized if not unicodedata.combining(c)).lower()


def _has_evidence_for_filter(
    field: str,
    value: str,
    query: str,
    lexicon: Lexicon | None = None,
) -> bool:
    query_norm = _normalize_text(query)
    value_norm = _normalize_text(value)

    if field in {"patient_id", "accession_number", "study_instance_uid"}:
        return value_norm and value_norm in query_norm

    if field == "patient_sex":
        tokens = SEX_EVIDENCE.get(value.upper(), [])
        return any(token in query_norm for token in tokens)

    if field == "modality_in_study":
        tokens = MODALITY_EVIDENCE.get(value.upper(), [])
        return any(token in query_norm for token in tokens)

    if field in {"study_date", "patient_birth_date"}:
        digits = re.findall(r"\d{4}", value)
        return any(d in query_norm for d in digits)

    if field == "study_description":
        stripped = value_norm.replace("*", " ")
        tokens = [token for token in stripped.split() if len(token) >= 3]
        if any(token in query_norm for token in tokens):
            return True
        if lexicon:
            query_lex = lexicon_normalize_text(query)
            for token in tokens:
                for expansion in lexicon.expand(token):
                    expansion_norm = lexicon_normalize_text(expansion)
                    if expansion_norm and expansion_norm in query_lex:
                        return True
        return False

    if field == "patient_name":
        stripped = value_norm.replace("^", " ").replace("*", " ")
        tokens = [token for token in stripped.split() if len(token) >= 3]
        return any(token in query_norm for token in tokens)

    return True


def _validate_search_evidence(
    args: QueryStudiesArgs,
    query: str,
    guardrail_date_range: str | None,
    lexicon: Lexicon | None = None,
) -> str | None:
    for field, value in args.model_dump(exclude_none=True).items():
        if not isinstance(value, str):
            continue
        if field == "study_date":
            if guardrail_date_range and value == guardrail_date_range:
                continue
        if not _has_evidence_for_filter(field, value, query, lexicon=lexicon):
            return field
    return None


class DicomAgent:
    def __init__(
        self,
        llm: OllamaClient,
        dicom_client: Any,
        max_steps: int = MAX_STEPS,
        lexicon: Lexicon | None = None,
        resolver: Callable[[str], Any] | None = None,
        confirmation_config: ConfirmationConfig | None = None,
        require_confirmation: bool = True,
    ) -> None:
        self.llm = llm
        self.client = dicom_client
        self.max_steps = max_steps
        self._lexicon = lexicon
        self._resolver = resolver
        self._confirmation_config = confirmation_config or ConfirmationConfig()
        self._require_confirmation = require_confirmation
        self.state = AgentState()
        if self._resolver:
            self.state.phase = AgentPhase.RESOLVE
        self.history = [{"role": "system", "content": SYSTEM_PROMPT}]
        self._user_query: str | None = None
        self._destination_node: str | None = None

    def _allowed_tools(self) -> list[ToolName]:
        if self.state.phase in {AgentPhase.RESOLVE, AgentPhase.CONFIRM}:
            return []
        if self.state.awaiting_confirmation:
            return []
        if self.state.phase == AgentPhase.SEARCH:
            return [ToolName.QUERY_STUDIES]
        if self.state.phase == AgentPhase.INSPECT:
            return [ToolName.QUERY_SERIES]
        if self.state.phase == AgentPhase.MOVE:
            return [ToolName.MOVE_STUDY]
        return []

    def _relax_filters(self) -> tuple[dict[str, Any] | None, str | None]:
        filters = dict(self.state.search_filters)
        for key in RELAXATION_ORDER:
            if key not in filters:
                continue
            if key == "study_date" and self.state.guardrail_date_range:
                if filters.get(key) == self.state.guardrail_date_range:
                    continue
            filters.pop(key, None)
            return filters, key
        return None, None

    def _post_tool_step(self, result: ToolResult) -> str | None:
        if not result.ok:
            self.state.last_error = result.error
            self.state.phase = AgentPhase.ERROR
            return _format_error(result)

        if result.tool == ToolName.QUERY_STUDIES:
            count = int(result.meta.get("count", 0))
            if count == 0:
                if self.state.broaden_attempts < MAX_BROADEN_ATTEMPTS:
                    relaxed, removed = self._relax_filters()
                    if relaxed is not None and removed is not None:
                        self.state.broaden_attempts += 1
                        self.state.search_filters = relaxed
                        self.history.append(
                            {
                                "role": "user",
                                "content": (
                                    "SYSTEM: Empty search. Retry without filter "
                                    f"'{removed}'. Use only these filters: "
                                    f"{json.dumps(relaxed, ensure_ascii=True)}"
                                ),
                            }
                        )
                        self.state.phase = AgentPhase.SEARCH
                        return None
                return "No studies found with the provided filters."

            if count == 1 and self.state.search_results:
                self.state.selected_uid = self.state.search_results[0].study_instance_uid
                self.state.phase = AgentPhase.INSPECT
                return None

            self.state.phase = AgentPhase.SEARCH
            return _format_selection_prompt(self.state)

        if result.tool == ToolName.QUERY_SERIES:
            self.state.phase = AgentPhase.MOVE
            return None

        if result.tool == ToolName.MOVE_STUDY:
            self.state.phase = AgentPhase.DONE
            return _format_move_result(result)

        return None

    def _apply_destination_node(self, tool: ToolName, arguments: dict[str, Any]) -> dict[str, Any]:
        args = dict(arguments)
        destination = args.get("destination_node")
        if isinstance(destination, str):
            destination = destination.strip()
            if destination:
                self._destination_node = destination
                args["destination_node"] = destination
            else:
                args.pop("destination_node", None)
        elif destination is not None:
            args.pop("destination_node", None)

        if tool != ToolName.MOVE_STUDY:
            args.pop("destination_node", None)
        elif "destination_node" not in args and self._destination_node:
            args["destination_node"] = self._destination_node
        return args

    def _apply_resolved_request(self, request: ResolvedRequest) -> None:
        self.state.resolved_request = request
        self.state.search_filters = dict(request.filters)
        if request.destination_node:
            self._destination_node = request.destination_node

    def _prime_history_from_request(self, request: ResolvedRequest) -> None:
        payload = {
            "query": self._user_query,
            "source_node": request.source_node,
            "destination_node": request.destination_node,
            "filters": request.filters,
        }
        self.history.append(
            {
                "role": "user",
                "content": (
                    "SYSTEM: Request resolved. Use only this information and filters. "
                    f"Data: {json.dumps(payload, ensure_ascii=True)}"
                ),
            }
        )

    def _handle_resolution(self, user_query: str) -> str | None:
        if not self._resolver:
            return None
        self._user_query = user_query
        result = self._resolver(user_query)
        self.state.resolved_request = result.request
        if result.unresolved:
            self.state.resolved_request = None
            self.state.awaiting_confirmation = False
            self.state.phase = AgentPhase.RESOLVE
            return "SYSTEM: Ask for clarification. Pending items: " + ", ".join(
                result.unresolved
            )
        if self._require_confirmation:
            self.state.awaiting_confirmation = True
            self.state.phase = AgentPhase.CONFIRM
            return build_confirmation_message(result.request, self._confirmation_config)
        self._apply_resolved_request(result.request)
        self.state.phase = AgentPhase.SEARCH
        self._prime_history_from_request(result.request)
        return None

    def _handle_confirmation(self, user_query: str) -> str | None:
        decision = classify_confirmation_response(user_query, self._confirmation_config)
        log.debug(
            "Confirmation decision",
            extra={"extra_data": {"decision": decision}},
        )
        if decision == "accept":
            self.state.awaiting_confirmation = False
            self.state.phase = AgentPhase.SEARCH
            if self.state.resolved_request is not None:
                self._apply_resolved_request(self.state.resolved_request)
                self._prime_history_from_request(self.state.resolved_request)
            return None
        if decision == "reject":
            self.state.confirmation_rejections += 1
            if (
                self.state.confirmation_rejections
                >= self._confirmation_config.max_rejections
            ):
                self.state.phase = AgentPhase.ERROR
                return self._confirmation_config.cancel_message
            self.state.awaiting_confirmation = False
            self.state.resolved_request = None
            self.state.phase = AgentPhase.RESOLVE
            return self._confirmation_config.correction_prompt
        self.state.confirmation_invalid_attempts += 1
        if (
            self.state.confirmation_invalid_attempts
            >= self._confirmation_config.max_invalid_responses
        ):
            self.state.phase = AgentPhase.ERROR
            return self._confirmation_config.cancel_message
        return build_invalid_response_message(self._confirmation_config)

    def run(self, user_query: str) -> str:
        append_history = True
        if self.state.phase == AgentPhase.CONFIRM:
            message = self._handle_confirmation(user_query)
            if message is not None:
                return message
            append_history = False
        elif self.state.phase == AgentPhase.RESOLVE:
            message = self._handle_resolution(user_query)
            if message is not None:
                return message
            append_history = False

        if append_history:
            self._user_query = user_query
            self.history.append({"role": "user", "content": user_query})

        for _ in range(self.max_steps):
            allowed_tools = self._allowed_tools()
            tools_schema = get_tools_schema(allowed_tools)
            response = self.llm.chat_with_tools(self.history, tools=tools_schema)
            self.history.append(response)

            tool_calls = response.get("tool_calls") or []
            content = response.get("content") or ""

            if not tool_calls:
                if self.state.last_tool is not None and content:
                    return content
                if allowed_tools:
                    if "{" in content:
                        return _protocol_error("Tool calls must be sent via tool_calls.")
                    return _protocol_error("A tool_call is required at this step.")
                return content

            if len(tool_calls) != 1:
                return _protocol_error("Only 1 tool_call per step.")

            tool_call = tool_calls[0]
            function = tool_call.get("function") or {}
            name = function.get("name")
            arguments = function.get("arguments", {})

            if not name:
                return _protocol_error("Tool call missing name.")

            try:
                tool_name = ToolName(name)
            except ValueError:
                return _protocol_error(f"Tool not allowed: {name}")

            if tool_name not in allowed_tools:
                return _protocol_error(
                    f"Tool {name} not allowed in phase {self.state.phase.value}."
                )

            if isinstance(arguments, str):
                try:
                    arguments = json.loads(arguments)
                except json.JSONDecodeError:
                    return _protocol_error("Arguments must be valid JSON.")

            if not isinstance(arguments, dict):
                return _protocol_error("Arguments must be a JSON object.")

            arguments = self._apply_destination_node(tool_name, arguments)

            if tool_name == ToolName.QUERY_STUDIES and self.state.resolved_request is not None:
                resolved_filters = dict(self.state.resolved_request.filters)
                if "study_instance_uid" in arguments:
                    resolved_filters["study_instance_uid"] = arguments["study_instance_uid"]
                arguments = resolved_filters

            if tool_name == ToolName.QUERY_STUDIES and self._user_query:
                try:
                    parsed = QueryStudiesArgs.model_validate(arguments)
                except ValidationError:
                    parsed = None
                if parsed is not None:
                    invalid_field = _validate_search_evidence(
                        parsed,
                        self._user_query,
                        self.state.guardrail_date_range,
                        self._lexicon,
                    )
                    if invalid_field:
                        return _protocol_error(
                            f"Filter '{invalid_field}' was not explicitly requested. Remove it."
                        )
                    if self.state.requires_selection and parsed.study_instance_uid is None:
                        return _protocol_error(
                            "Multiple studies found. Explicitly select a study_instance_uid."
                        )

            result = execute_tool(name, arguments, self.client, self.state)
            payload = result.model_dump()
            self.history.append(
                {
                    "role": "tool",
                    "content": json.dumps(payload, ensure_ascii=True),
                    "name": name,
                }
            )

            if result.error and result.error.code == "validation_error":
                self.history.append(
                    {
                        "role": "user",
                        "content": _format_validation_feedback(result.error.details),
                    }
                )
                continue

            message = self._post_tool_step(result)
            if message is not None:
                return message

        return "Step limit reached."
