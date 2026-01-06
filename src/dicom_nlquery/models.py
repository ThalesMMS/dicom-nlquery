from __future__ import annotations

from enum import Enum
import re
import unicodedata
from typing import Any

from pydantic import AliasChoices, BaseModel, ConfigDict, Field, field_validator, model_validator


class DicomNodeConfig(BaseModel):
    host: str
    port: int
    ae_title: str
    description: str | None = None


class LLMConfig(BaseModel):
    provider: str
    base_url: str
    model: str
    temperature: float = 0
    timeout: int = 60

    @field_validator("provider")
    @classmethod
    def validate_provider(cls, value: str) -> str:
        normalized = value.strip().lower()
        if normalized not in {"ollama", "lmstudio"}:
            raise ValueError("provider must be 'ollama' or 'lmstudio'")
        return normalized


class GuardrailsConfig(BaseModel):
    study_date_range_default_days: int = 180
    max_studies_scanned_default: int = 700


class MatchingConfig(BaseModel):
    head_keywords: list[str] = Field(default_factory=list)
    synonyms: dict[str, list[str]] = Field(default_factory=dict)


class McpServerConfig(BaseModel):
    command: str = "dicom-mcp"
    config_path: str | None = None
    args: list[str] = Field(default_factory=list)
    cwd: str | None = None
    env: dict[str, str] | None = None


class NLQueryConfig(BaseModel):
    llm: LLMConfig
    guardrails: GuardrailsConfig = Field(default_factory=GuardrailsConfig)
    mcp: McpServerConfig | None = None
    nodes: dict[str, DicomNodeConfig] | None = None
    current_node: str | None = None
    calling_aet: str | None = None
    matching: MatchingConfig | None = None

    @model_validator(mode="after")
    def validate_current_node(self) -> "NLQueryConfig":
        if self.nodes and self.current_node:
            if self.current_node not in self.nodes:
                raise ValueError("current_node must exist in nodes")
        return self


class StudyQuery(BaseModel):
    model_config = ConfigDict(extra="ignore")

    patient_id: str | None = None
    patient_name: str | None = None
    patient_sex: str | None = None
    patient_birth_date: str | None = None
    study_date: str | None = None
    modality_in_study: str | None = None
    study_description: str | None = None
    accession_number: str | None = None
    study_instance_uid: str | None = None

    @field_validator(
        "patient_id",
        "patient_name",
        "patient_sex",
        "patient_birth_date",
        "study_date",
        "modality_in_study",
        "study_description",
        "accession_number",
        "study_instance_uid",
    )
    @classmethod
    def normalize_empty(cls, value: str | None) -> str | None:
        if value is None:
            return value
        normalized = value.strip()
        return normalized or None

    @field_validator("patient_sex")
    @classmethod
    def validate_patient_sex(cls, value: str | None) -> str | None:
        if value is None:
            return value
        normalized = value.strip().upper()
        if normalized not in {"F", "M", "O"}:
            raise ValueError("patient_sex must be 'F', 'M', or 'O'")
        return normalized

    @field_validator("modality_in_study", mode="before")
    @classmethod
    def normalize_modalities(cls, value: str | list[str] | None) -> str | None:
        if value is None:
            return None
        if isinstance(value, list):
            return "\\".join([str(item).strip().upper() for item in value if str(item).strip()])
        return value


class SeriesQuery(BaseModel):
    model_config = ConfigDict(extra="ignore")

    modality: str | None = None
    series_number: str | None = None
    series_description: str | None = None
    series_instance_uid: str | None = None

    @field_validator("modality")
    @classmethod
    def normalize_series_modality(cls, value: str | None) -> str | None:
        if value is None:
            return value
        normalized = value.strip().upper()
        return normalized or None


class SearchCriteria(BaseModel):
    model_config = ConfigDict(extra="ignore")

    study: StudyQuery
    series: SeriesQuery | None = None

    @model_validator(mode="after")
    def validate_filters(self) -> "SearchCriteria":
        study_filters = any(
            [
                self.study.patient_id,
                self.study.patient_name,
                self.study.patient_sex,
                self.study.patient_birth_date,
                self.study.study_date,
                self.study.modality_in_study,
                self.study.study_description,
                self.study.accession_number,
                self.study.study_instance_uid,
            ]
        )
        series_filters = False
        if self.series is not None:
            series_filters = any(
                [
                    self.series.modality,
                    self.series.series_number,
                    self.series.series_description,
                    self.series.series_instance_uid,
                ]
            )
        if not (study_filters or series_filters):
            raise ValueError("at least one filter must be specified")
        return self


class SearchStats(BaseModel):
    studies_scanned: int
    studies_matched: int
    studies_filtered_series: int
    limit_reached: bool
    execution_time_seconds: float
    date_range_applied: str


class SearchResult(BaseModel):
    accession_numbers: list[str]
    stats: SearchStats


UID_RE = re.compile(r"^\d+(?:\.\d+)+$")
DATE_RE = re.compile(r"^(?:\d{8}|\d{8}-\d{8}|\d{8}-|-\d{8})$")


class ToolName(str, Enum):
    QUERY_STUDIES = "query_studies"
    QUERY_SERIES = "query_series"
    MOVE_STUDY = "move_study"


class AgentPhase(str, Enum):
    SEARCH = "search"
    INSPECT = "inspect"
    MOVE = "move"
    DONE = "done"
    ERROR = "error"


class ToolError(BaseModel):
    code: str
    message: str
    details: Any | None = None


class ToolResult(BaseModel):
    tool: ToolName
    ok: bool
    data: Any | None = None
    error: ToolError | None = None
    meta: dict[str, Any] = Field(default_factory=dict)


class StudySummary(BaseModel):
    study_instance_uid: str
    study_date: str | None = None
    modalities_in_study: str | None = None
    study_description: str | None = None
    patient_name: str | None = None
    patient_id: str | None = None
    accession_number: str | None = None


class AgentState(BaseModel):
    phase: AgentPhase = AgentPhase.SEARCH
    search_filters: dict[str, Any] = Field(default_factory=dict)
    search_results: list[StudySummary] = Field(default_factory=list)
    selected_uid: str | None = None
    last_tool: ToolName | None = None
    last_error: ToolError | None = None
    broaden_attempts: int = 0
    guardrail_date_range: str | None = None
    requires_selection: bool = False

    def has_uid(self, uid: str) -> bool:
        return any(item.study_instance_uid == uid for item in self.search_results)


class QueryStudiesArgs(BaseModel):
    model_config = ConfigDict(extra="forbid")

    patient_id: str | None = None
    patient_name: str | None = None
    patient_sex: str | None = None
    patient_birth_date: str | None = None
    study_date: str | None = None
    modality_in_study: str | None = Field(
        default=None,
        validation_alias=AliasChoices("modality_in_study", "modality"),
    )
    study_description: str | None = None
    accession_number: str | None = None
    study_instance_uid: str | None = None

    @field_validator(
        "patient_id",
        "patient_name",
        "patient_sex",
        "patient_birth_date",
        "study_date",
        "modality_in_study",
        "study_description",
        "accession_number",
        "study_instance_uid",
    )
    @classmethod
    def normalize_empty(cls, value: str | None) -> str | None:
        if value is None:
            return value
        normalized = value.strip()
        return normalized or None

    @field_validator("patient_sex")
    @classmethod
    def validate_patient_sex(cls, value: str | None) -> str | None:
        if value is None:
            return value
        normalized = value.strip().upper()
        if normalized not in {"F", "M", "O"}:
            raise ValueError("patient_sex must be 'F', 'M', or 'O'")
        return normalized

    @field_validator("study_date")
    @classmethod
    def validate_study_date(cls, value: str | None) -> str | None:
        if value is None:
            return None
        normalized = value.strip()
        if not DATE_RE.match(normalized):
            raise ValueError("study_date must be YYYYMMDD or YYYYMMDD-YYYYMMDD")
        return normalized

    @field_validator("study_instance_uid")
    @classmethod
    def validate_study_uid(cls, value: str | None) -> str | None:
        if value is None:
            return None
        normalized = value.strip()
        if "<" in normalized or ">" in normalized:
            raise ValueError("study_instance_uid must be a real UID")
        if not UID_RE.match(normalized):
            raise ValueError("study_instance_uid must be a dotted numeric UID")
        return normalized

    @field_validator("modality_in_study")
    @classmethod
    def validate_modality(cls, value: str | None) -> str | None:
        if value is None:
            return None
        normalized = unicodedata.normalize("NFKD", value.strip().upper())
        normalized = "".join(ch for ch in normalized if not unicodedata.combining(ch))
        alias_map = {
            "RM": "MR",
            "MRI": "MR",
            "RESSONANCIA": "MR",
            "RESSONANCIA MAGNETICA": "MR",
            "RESSONANCIA MAGNÉTICA": "MR",
            "TC": "CT",
            "TOMOGRAFIA": "CT",
            "USG": "US",
            "ULTRASSOM": "US",
        }
        allowed = {"MR", "CT", "US", "CR", "DX", "SR", "PDF"}
        direct = alias_map.get(normalized, normalized)
        if direct in allowed:
            return direct

        raw = (
            normalized.replace("/", "\\")
            .replace("|", "\\")
            .replace(",", "\\")
            .replace(";", "\\")
        )
        parts = [part for part in re.split(r"[\s\\]+", raw) if part]
        mapped = []
        for part in parts:
            mapped_part = alias_map.get(part, part)
            if mapped_part in allowed and mapped_part not in mapped:
                mapped.append(mapped_part)
        if len(mapped) == 1:
            return mapped[0]
        if len(mapped) > 1:
            return "\\".join(mapped)

        for alias, mapped_value in alias_map.items():
            if " " in alias and alias in normalized and mapped_value in allowed:
                return mapped_value

        raise ValueError(f"modality_in_study inválida: {normalized}")


class QuerySeriesArgs(BaseModel):
    model_config = ConfigDict(extra="forbid")

    study_instance_uid: str
    modality: str | None = None
    series_number: str | None = None
    series_description: str | None = None
    series_instance_uid: str | None = None

    @field_validator(
        "modality",
        "series_number",
        "series_description",
        "series_instance_uid",
        "study_instance_uid",
    )
    @classmethod
    def normalize_empty(cls, value: str | None) -> str | None:
        if value is None:
            return value
        normalized = value.strip()
        return normalized or None

    @field_validator("study_instance_uid")
    @classmethod
    def validate_study_uid(cls, value: str) -> str:
        normalized = value.strip()
        if "<" in normalized or ">" in normalized:
            raise ValueError("study_instance_uid must be a real UID")
        if not UID_RE.match(normalized):
            raise ValueError("study_instance_uid must be a dotted numeric UID")
        return normalized

    @field_validator("series_instance_uid")
    @classmethod
    def validate_series_uid(cls, value: str | None) -> str | None:
        if value is None:
            return None
        normalized = value.strip()
        if "<" in normalized or ">" in normalized:
            raise ValueError("series_instance_uid must be a real UID")
        if not UID_RE.match(normalized):
            raise ValueError("series_instance_uid must be a dotted numeric UID")
        return normalized

    @field_validator("modality")
    @classmethod
    def normalize_series_modality(cls, value: str | None) -> str | None:
        if value is None:
            return None
        normalized = value.strip().upper()
        return normalized or None


class MoveStudyArgs(BaseModel):
    model_config = ConfigDict(extra="forbid")

    destination_node: str
    study_instance_uid: str

    @field_validator("destination_node")
    @classmethod
    def validate_destination(cls, value: str) -> str:
        normalized = value.strip()
        if not normalized:
            raise ValueError("destination_node is required")
        if "<" in normalized or ">" in normalized:
            raise ValueError("destination_node must be explicit")
        return normalized

    @field_validator("study_instance_uid")
    @classmethod
    def validate_move_uid(cls, value: str) -> str:
        normalized = value.strip()
        if "<" in normalized or ">" in normalized:
            raise ValueError("study_instance_uid must be a real UID")
        if not UID_RE.match(normalized):
            raise ValueError("study_instance_uid must be a dotted numeric UID")
        return normalized
