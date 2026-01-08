from __future__ import annotations

import json
from typing import Any, Iterable

from pydantic import ValidationError

from .dicom_client import DicomClient
from .dicom_search import apply_guardrails
from .models import (
    AgentState,
    GuardrailsConfig,
    MoveStudyArgs,
    QuerySeriesArgs,
    QueryStudiesArgs,
    StudySummary,
    ToolError,
    ToolName,
    ToolResult,
)

MAX_RESULTS_DISPLAY = 5
MAX_SERIES_DISPLAY = 15

TOOL_DESCRIPTIONS: dict[ToolName, str] = {
    ToolName.QUERY_STUDIES: (
        "Query DICOM studies. Returns a list of UIDs and metadata. "
        "Use filters only when explicitly cited by the user."
    ),
    ToolName.QUERY_SERIES: "List series for a specific study.",
    ToolName.MOVE_STUDY: "Move a study to a destination node.",
}

TOOL_ARGS_MODELS = {
    ToolName.QUERY_STUDIES: QueryStudiesArgs,
    ToolName.QUERY_SERIES: QuerySeriesArgs,
    ToolName.MOVE_STUDY: MoveStudyArgs,
}


def _json_safe(value: Any | None) -> Any | None:
    if value is None:
        return None
    try:
        json.dumps(value)
        return value
    except TypeError:
        return json.loads(json.dumps(value, default=str))


def _tool_error(tool: ToolName, code: str, message: str, details: Any | None = None) -> ToolResult:
    return ToolResult(
        tool=tool,
        ok=False,
        error=ToolError(code=code, message=message, details=_json_safe(details)),
    )


def _ensure_schema_additional_properties(schema: dict[str, Any]) -> dict[str, Any]:
    if "additionalProperties" not in schema:
        schema["additionalProperties"] = False
    return schema


def get_tools_schema(allowed_tools: Iterable[ToolName] | None = None) -> list[dict[str, Any]]:
    tools = []
    selected = list(allowed_tools) if allowed_tools is not None else list(ToolName)
    for tool in selected:
        args_model = TOOL_ARGS_MODELS[tool]
        schema = args_model.model_json_schema()
        schema.pop("title", None)
        schema = _ensure_schema_additional_properties(schema)
        tools.append(
            {
                "type": "function",
                "function": {
                    "name": tool.value,
                    "description": TOOL_DESCRIPTIONS[tool],
                    "parameters": schema,
                },
            }
        )
    return tools


def _summarize_study(study: dict[str, Any]) -> StudySummary | None:
    uid = study.get("StudyInstanceUID")
    if not uid:
        return None
    return StudySummary(
        study_instance_uid=str(uid),
        study_date=study.get("StudyDate"),
        modalities_in_study=study.get("ModalitiesInStudy"),
        study_description=study.get("StudyDescription"),
        patient_name=study.get("PatientName"),
        patient_id=study.get("PatientID"),
        accession_number=study.get("AccessionNumber"),
    )


def execute_tool(name: str, args: dict[str, Any], client: DicomClient, state: AgentState) -> ToolResult:
    try:
        tool = ToolName(name)
    except ValueError:
        return _tool_error(ToolName.QUERY_STUDIES, "unknown_tool", f"Unknown tool: {name}")

    args_model = TOOL_ARGS_MODELS[tool]
    try:
        parsed = args_model.model_validate(args)
    except ValidationError as exc:
        return _tool_error(tool, "validation_error", "Invalid parameters", exc.errors())

    if tool == ToolName.QUERY_STUDIES:
        params = parsed.model_dump(exclude_none=True)
        guardrails = GuardrailsConfig()
        date_range_applied = None
        if "study_date" not in params:
            date_range, _ = apply_guardrails(guardrails)
            if date_range:
                params["study_date"] = date_range
                date_range_applied = date_range
                state.guardrail_date_range = date_range

        try:
            results = client.query_studies(**params)
        except Exception as exc:
            return _tool_error(tool, "connection_error", f"Connection error: {exc}")

        summaries: list[StudySummary] = []
        for study in results:
            summary = _summarize_study(study)
            if summary is not None:
                summaries.append(summary)

        state.search_filters = params
        state.search_results = summaries
        state.selected_uid = None
        state.last_tool = tool
        state.last_error = None
        state.requires_selection = len(summaries) > 1

        data = [item.model_dump() for item in summaries[:MAX_RESULTS_DISPLAY]]
        return ToolResult(
            tool=tool,
            ok=True,
            data=data,
            meta={
                "count": len(summaries),
                "truncated": len(summaries) > MAX_RESULTS_DISPLAY,
                "date_range_applied": date_range_applied,
            },
        )

    if tool == ToolName.QUERY_SERIES:
        uid = parsed.study_instance_uid
        if not state.has_uid(uid):
            return _tool_error(tool, "uid_not_in_state", "UID not found in current state")

        params = parsed.model_dump(exclude_none=True)
        try:
            series = client.query_series(**params)
        except Exception as exc:
            return _tool_error(tool, "connection_error", f"Error querying series: {exc}")

        state.selected_uid = uid
        state.last_tool = tool
        state.last_error = None

        return ToolResult(
            tool=tool,
            ok=True,
            data=series[:MAX_SERIES_DISPLAY],
            meta={"count": len(series), "truncated": len(series) > MAX_SERIES_DISPLAY},
        )

    if tool == ToolName.MOVE_STUDY:
        uid = parsed.study_instance_uid
        if not state.has_uid(uid):
            return _tool_error(tool, "uid_not_in_state", "UID not found in current state")

        try:
            result = client.move_study(
                destination_node=parsed.destination_node,
                study_instance_uid=uid,
            )
        except Exception as exc:
            return _tool_error(tool, "move_failed", f"C-MOVE error: {exc}")

        state.selected_uid = uid
        state.last_tool = tool
        state.last_error = None

        return ToolResult(tool=tool, ok=bool(result.get("success")), data=result)

    return _tool_error(tool, "unsupported_tool", f"Unsupported tool: {tool.value}")
