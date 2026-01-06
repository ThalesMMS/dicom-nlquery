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


class NLQueryConfig(BaseModel):
    nodes: dict[str, DicomNodeConfig]
    current_node: str
    calling_aet: str
    llm: LLMConfig
    guardrails: GuardrailsConfig = Field(default_factory=GuardrailsConfig)
    matching: MatchingConfig = Field(default_factory=MatchingConfig)

    @model_validator(mode="after")
    def validate_current_node(self) -> "NLQueryConfig":
        if self.current_node not in self.nodes:
            raise ValueError("current_node must exist in nodes")
        return self


class PatientFilter(BaseModel):
    model_config = ConfigDict(extra="ignore")

    sex: str | None = None
    age_min: int | None = None
    age_max: int | None = None

    @field_validator("sex")
    @classmethod
    def validate_sex(cls, value: str | None) -> str | None:
        if value is None:
            return value
        normalized = value.strip().upper()
        if normalized not in {"F", "M", "O"}:
            raise ValueError("sex must be 'F', 'M', or 'O'")
        return normalized

    @model_validator(mode="after")
    def validate_age_range(self) -> "PatientFilter":
        if self.age_min is not None and self.age_min < 0:
            raise ValueError("age_min must be >= 0")
        if self.age_max is not None and self.age_max < 0:
            raise ValueError("age_max must be >= 0")
        if (
            self.age_min is not None
            and self.age_max is not None
            and self.age_max < self.age_min
        ):
            raise ValueError("age_max must be >= age_min")
        return self


class SeriesRequirement(BaseModel):
    model_config = ConfigDict(extra="ignore")

    name: str
    modality: str | None = None
    within_head: bool
    all_keywords: list[str] | None = None
    any_keywords: list[str] | None = None

    @field_validator("name")
    @classmethod
    def validate_name(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("name cannot be empty")
        return value

    @model_validator(mode="after")
    def validate_match_fields(self) -> "SeriesRequirement":
        if not self.modality and not self.all_keywords and not self.any_keywords:
            raise ValueError("series requirement needs modality or keywords")
        return self


class StudyNarrowing(BaseModel):
    model_config = ConfigDict(extra="ignore")

    modality_in_study: list[str] | None = None
    study_description_keywords: list[str] | None = None


class SearchCriteria(BaseModel):
    model_config = ConfigDict(extra="ignore")

    patient: PatientFilter | None = None
    head_keywords: list[str] | None = None
    required_series: list[SeriesRequirement] | None = None
    study_narrowing: StudyNarrowing | None = None

    @model_validator(mode="after")
    def validate_filters(self) -> "SearchCriteria":
        has_patient = False
        if self.patient is not None:
            has_patient = any(
                [
                    self.patient.sex is not None,
                    self.patient.age_min is not None,
                    self.patient.age_max is not None,
                ]
            )
        has_head = bool(self.head_keywords)
        has_series = bool(self.required_series)
        has_narrowing = False
        if self.study_narrowing is not None:
            has_narrowing = bool(self.study_narrowing.modality_in_study) or bool(
                self.study_narrowing.study_description_keywords
            )
        if not (has_patient or has_head or has_series or has_narrowing):
            raise ValueError("at least one filter must be specified")
        return self


class SearchStats(BaseModel):
    studies_scanned: int
    studies_matched: int
    studies_excluded_no_age: int
    studies_excluded_no_sex: int
    limit_reached: bool
    execution_time_seconds: float
    date_range_applied: str


class SearchResult(BaseModel):
    accession_numbers: list[str]
    stats: SearchStats
