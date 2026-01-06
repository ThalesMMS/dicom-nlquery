from __future__ import annotations

import json
import logging
import re
import unicodedata
from datetime import date
from typing import Any

from pydantic import ValidationError

from .agent_tools import execute_tool, get_tools_schema
from .llm_client import OllamaClient
from .models import AgentPhase, AgentState, QueryStudiesArgs, ToolName, ToolResult

log = logging.getLogger(__name__)

SYSTEM_PROMPT = f"""
Você é um Agente Especialista em Radiologia e DICOM. Data: {date.today()}.

REGRAS CRÍTICAS DE OPERAÇÃO:
1. **TOOL CALLING**: Use exclusivamente as tools estruturadas (tool_calls). Nunca envie JSON no texto.
2. **SEM INFERÊNCIA**: Não crie filtros que o usuário não mencionou explicitamente.
3. **SEM ALUCINAÇÃO**: Nunca invente UIDs; use apenas UIDs retornados por tools.
4. **SEQUÊNCIA**: query_studies -> query_series -> move_study. Nunca pule etapas.
5. **UMA FERRAMENTA POR VEZ**: Em cada turno faça no máximo 1 tool_call.
6. **UID REAL**: Não use placeholders (<...>).
7. **SEXO**: Só use patient_sex se o usuário declarar explicitamente.
8. **GUARDRAILS**: Date range padrão pode ser aplicado como proteção operacional.
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
    "M": ["masculino", "homem", "male"],
    "F": ["feminino", "mulher", "female", "gestante"],
    "O": ["outro", "outros", "other"],
}
MODALITY_EVIDENCE = {
    "MR": ["rm", "ressonancia", "ressonância", "mri"],
    "CT": ["tc", "tomografia", "ct"],
    "US": ["us", "ultrassom", "ultrasom", "usg"],
    "CR": ["rx", "raio x", "raio-x", "raiox"],
    "DX": ["rx", "raio x", "raio-x", "raiox"],
}


def _protocol_error(message: str) -> str:
    return f"ERRO DE PROTOCOLO: {message}"


def _format_selection_prompt(state: AgentState) -> str:
    lines = ["Foram encontrados multiplos estudos. Refine a busca ou escolha um UID:"]
    for item in state.search_results[:5]:
        lines.append(
            f"- UID {item.study_instance_uid} | Data {item.study_date or '-'} | "
            f"Desc {item.study_description or '-'}"
        )
    if len(state.search_results) > 5:
        lines.append("(Mostrando apenas os primeiros resultados.)")
    return "\n".join(lines)


def _format_move_result(result: ToolResult) -> str:
    data = result.data or {}
    if result.ok:
        return (
            "C-MOVE concluido. "
            f"Completed={data.get('completed')} Failed={data.get('failed')} Warning={data.get('warning')}"
        )
    return f"Falha no C-MOVE: {data.get('message', 'Erro desconhecido')}"


def _format_error(result: ToolResult) -> str:
    if result.error:
        return f"Erro: {result.error.message}"
    return "Erro desconhecido."


def _format_validation_feedback(details: Any | None) -> str:
    items: list[str] = []
    if isinstance(details, list):
        for entry in details:
            if not isinstance(entry, dict):
                items.append(str(entry))
                continue
            loc = entry.get("loc") or []
            field = ".".join(str(part) for part in loc) if loc else "?"
            msg = entry.get("msg") or entry.get("type") or "Erro de validacao"
            items.append(f"{field}: {msg}")
    elif details:
        items.append(str(details))
    else:
        items.append("Erro de validacao nos parametros.")
    summary = "; ".join(items)
    return (
        "SISTEMA: Erro de validacao nos argumentos da tool. "
        "Corrija e reenvie a tool_call com o schema. "
        f"Detalhes: {summary}"
    )


def _normalize_text(text: str) -> str:
    normalized = unicodedata.normalize("NFKD", text)
    return "".join(c for c in normalized if not unicodedata.combining(c)).lower()


def _has_evidence_for_filter(field: str, value: str, query: str) -> bool:
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
        return any(token in query_norm for token in tokens)

    if field == "patient_name":
        stripped = value_norm.replace("^", " ").replace("*", " ")
        tokens = [token for token in stripped.split() if len(token) >= 3]
        return any(token in query_norm for token in tokens)

    return True


def _validate_search_evidence(
    args: QueryStudiesArgs,
    query: str,
    guardrail_date_range: str | None,
) -> str | None:
    for field, value in args.model_dump(exclude_none=True).items():
        if not isinstance(value, str):
            continue
        if field == "study_date":
            if guardrail_date_range and value == guardrail_date_range:
                continue
        if not _has_evidence_for_filter(field, value, query):
            return field
    return None


class DicomAgent:
    def __init__(self, llm: OllamaClient, dicom_client: Any, max_steps: int = MAX_STEPS) -> None:
        self.llm = llm
        self.client = dicom_client
        self.max_steps = max_steps
        self.state = AgentState()
        self.history = [{"role": "system", "content": SYSTEM_PROMPT}]
        self._user_query: str | None = None
        self._destination_node: str | None = None

    def _allowed_tools(self) -> list[ToolName]:
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
                                    "SISTEMA: Busca vazia. Refaça a busca removendo o filtro "
                                    f"'{removed}'. Use apenas estes filtros: {json.dumps(relaxed, ensure_ascii=True)}"
                                ),
                            }
                        )
                        self.state.phase = AgentPhase.SEARCH
                        return None
                return "Nenhum estudo encontrado com os filtros informados."

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

    def run(self, user_query: str) -> str:
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
                        return _protocol_error("Tool calls devem ser enviados pelo canal tool_calls.")
                    return _protocol_error("Uma tool_call e obrigatoria nesta etapa.")
                return content

            if len(tool_calls) != 1:
                return _protocol_error("Apenas 1 tool_call por etapa.")

            tool_call = tool_calls[0]
            function = tool_call.get("function") or {}
            name = function.get("name")
            arguments = function.get("arguments", {})

            if not name:
                return _protocol_error("Tool call sem nome.")

            try:
                tool_name = ToolName(name)
            except ValueError:
                return _protocol_error(f"Ferramenta nao permitida: {name}")

            if tool_name not in allowed_tools:
                return _protocol_error(
                    f"Ferramenta {name} nao permitida na fase {self.state.phase.value}."
                )

            if isinstance(arguments, str):
                try:
                    arguments = json.loads(arguments)
                except json.JSONDecodeError:
                    return _protocol_error("Arguments devem ser JSON valido.")

            if not isinstance(arguments, dict):
                return _protocol_error("Arguments devem ser um objeto JSON.")

            arguments = self._apply_destination_node(tool_name, arguments)

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
                    )
                    if invalid_field:
                        return _protocol_error(
                            f"Filtro '{invalid_field}' nao foi explicitamente solicitado. Remova-o."
                        )
                    if self.state.requires_selection and parsed.study_instance_uid is None:
                        return _protocol_error(
                            "Multiplos estudos encontrados. Selecione um study_instance_uid explicitamente."
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

        return "Limite de passos atingido."
