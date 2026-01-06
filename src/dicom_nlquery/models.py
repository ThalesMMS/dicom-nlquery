from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator


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
    patient_sex: str | None = None
    patient_birth_date: str | None = None
    study_date: str | None = None
    modality_in_study: str | None = None
    study_description: str | None = None
    accession_number: str | None = None
    study_instance_uid: str | None = None

    @field_validator(
        "patient_id",
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
