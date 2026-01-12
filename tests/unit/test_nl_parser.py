from __future__ import annotations

from datetime import date
import pytest
from pydantic import ValidationError

import dicom_nlquery.nl_parser as nl_parser
from dicom_nlquery.node_registry import NodeRegistry
from dicom_nlquery.nl_parser import parse_nl_to_criteria


def test_parser_returns_valid_criteria(fake_llm) -> None:
    criteria = parse_nl_to_criteria("women 20 to 40", fake_llm)

    assert criteria.study.patient_sex == "F"


def test_parser_rejects_invalid_schema(fake_llm_invalid) -> None:
    with pytest.raises(ValidationError):
        parse_nl_to_criteria("query", fake_llm_invalid)


def test_parser_ignores_extra_fields(fake_llm_extra) -> None:
    criteria = parse_nl_to_criteria("query", fake_llm_extra)

    assert criteria.study.patient_sex == "F"


def test_parser_strict_removes_inferred_filters() -> None:
    class FakeLLM:
        def chat(self, system_prompt: str, user_prompt: str, **_kwargs) -> str:
            return (
                '{\"study\": {\"patient_sex\": \"O\", '
                '\"modality_in_study\": \"MR\\\\CT\", '
                '\"study_description\": \"fetal MR\"}, '
                '\"series\": null}'
            )

    criteria = parse_nl_to_criteria("fetal MR to RADIANT", FakeLLM(), strict_evidence=True)

    assert criteria.study.patient_sex is None
    assert criteria.study.modality_in_study == "MR"
    assert criteria.study.study_description == "fetal"


@pytest.mark.parametrize(
    "query",
    [
        "MR to RADIANT",
        "MRI to RADIANT",
        "magnetic resonance to RADIANT",
    ],
)
def test_parser_strict_fallbacks_to_query_modality(query: str) -> None:
    class FakeLLM:
        def chat(self, system_prompt: str, user_prompt: str, **_kwargs) -> str:
            return (
                '{\"study\": {\"patient_sex\": \"O\", '
                '\"study_description\": \"RADIANT\"}, \"series\": null}'
            )

    criteria = parse_nl_to_criteria(query, FakeLLM(), strict_evidence=True)

    assert criteria.study.modality_in_study == "MR"


def test_parser_strict_adds_modality_when_missing() -> None:
    class FakeLLM:
        def chat(self, system_prompt: str, user_prompt: str, **_kwargs) -> str:
            return '{\"study\": {\"study_description\": \"cranial\"}, \"series\": null}'

    criteria = parse_nl_to_criteria("MR of cranial", FakeLLM(), strict_evidence=True)

    assert criteria.study.modality_in_study == "MR"


@pytest.mark.parametrize(
    "query,modality,expected",
    [
        (
            "studies from year 2000 until 2022 of CT chest angiograms from node_a to node_b, patients age 20 to 80",
            "CT",
            "*chest*angiograms*",
        ),
        (
            "MR brain perfusion studies from node_a to node_b",
            "MR",
            "*brain*perfusion*",
        ),
        (
            "US abdomen doppler exams for female patients",
            "US",
            "*abdomen*doppler*",
        ),
    ],
)
def test_parser_strict_fallbacks_to_study_description(
    query: str, modality: str, expected: str
) -> None:
    class FakeLLM:
        def chat(self, system_prompt: str, user_prompt: str, **_kwargs) -> str:
            return (
                "{\"study\": {\"modality_in_study\": \""
                + modality
                + "\", \"study_description\": null}, \"series\": null}"
            )

    criteria = parse_nl_to_criteria(query, FakeLLM(), strict_evidence=True)

    assert criteria.study.study_description == expected


@pytest.mark.parametrize(
    "query,llm_description,modality,expected",
    [
        (
            "CT chest angiograms",
            "*CT chest angiogram*",
            "CT",
            "*chest angiogram*",
        ),
        (
            "MRI brain study",
            "*MRI brain study*",
            "MR",
            "*brain study*",
        ),
        (
            "ultrasound abdomen exam",
            "*US abdomen exam*",
            "US",
            "*abdomen exam*",
        ),
    ],
)
def test_parser_strict_preserves_wildcards_after_modality_strip(
    query: str, llm_description: str, modality: str, expected: str
) -> None:
    class FakeLLM:
        def chat(self, system_prompt: str, user_prompt: str, **_kwargs) -> str:
            return (
                f'{{"study": {{"modality_in_study": '
                f'"{modality}", "study_description": "{llm_description}"}}, '
                f'"series": null}}'
            )

    criteria = parse_nl_to_criteria(query, FakeLLM(), strict_evidence=True)

    assert criteria.study.study_description == expected


def test_parser_strict_keeps_patient_name() -> None:
    class FakeLLM:
        def chat(self, system_prompt: str, user_prompt: str, **_kwargs) -> str:
            return '{\"study\": {\"patient_name\": \"TEST PATIENT\"}, \"series\": null}'

    criteria = parse_nl_to_criteria("exams for test patient", FakeLLM(), strict_evidence=True)

    assert criteria.study.patient_name == "TEST PATIENT"


@pytest.mark.parametrize(
    "query",
    [
        "exams for elaine",
        "patients of elaine",
    ],
)
def test_parser_strict_extracts_patient_name_from_query(query: str) -> None:
    class FakeLLM:
        def chat(self, system_prompt: str, user_prompt: str, **_kwargs) -> str:
            return '{\"study\": {}, \"series\": null}'

    criteria = parse_nl_to_criteria(query, FakeLLM(), strict_evidence=True)

    assert criteria.study.patient_name == "ELAINE"


@pytest.mark.parametrize(
    "query",
    [
        "move pelvis exams from ORTHANC to RADIANT",
        "move pelvis, exams from ORTHANC to RADIANT",
        "move exams for pelvis from ORTHANC to RADIANT",
    ],
)
def test_parser_strict_applies_pelvis_keyword(query: str) -> None:
    class FakeLLM:
        def chat(self, system_prompt: str, user_prompt: str, **_kwargs) -> str:
            return '{\"study\": {}, \"series\": {\"series_description\": \"*\"}}'

    criteria = parse_nl_to_criteria(query, FakeLLM(), strict_evidence=True)

    assert criteria.study.study_description == "pelvis"
    assert criteria.series is None


@pytest.mark.parametrize(
    "query",
    [
        "women ages 20 to 40 with cranial MR",
        "women 20-40 with cranial MR",
        "women 20 to 40 years with cranial MR",
    ],
)
def test_parser_strict_converts_age_range(
    monkeypatch: pytest.MonkeyPatch, query: str
) -> None:
    class FakeLLM:
        def chat(self, system_prompt: str, user_prompt: str, **_kwargs) -> str:
            return '{"study": {}, "series": null}'

    class FixedDate(date):
        @classmethod
        def today(cls) -> "FixedDate":
            return cls(2026, 1, 8)

    monkeypatch.setattr(nl_parser, "date", FixedDate)
    criteria = parse_nl_to_criteria(query, FakeLLM(), strict_evidence=True)

    assert criteria.study.patient_birth_date == "19850109-20060108"


@pytest.mark.parametrize(
    "query",
    [
        "studies from year 2000 until 2022 of women",
        "studies between 2000 and 2022 of women",
        "studies from 2000 to 2022 of women",
    ],
)
def test_parser_strict_extracts_study_year_range(
    monkeypatch: pytest.MonkeyPatch, query: str
) -> None:
    class FakeLLM:
        def chat(self, system_prompt: str, user_prompt: str, **_kwargs) -> str:
            return '{"study": {}, "series": null}'

    class FixedDate(date):
        @classmethod
        def today(cls) -> "FixedDate":
            return cls(2026, 1, 8)

    monkeypatch.setattr(nl_parser, "date", FixedDate)
    criteria = parse_nl_to_criteria(query, FakeLLM(), strict_evidence=True)

    assert criteria.study.study_date == "20000101-20221231"


def test_parser_strict_drops_series_without_evidence() -> None:
    class FakeLLM:
        def chat(self, system_prompt: str, user_prompt: str, **_kwargs) -> str:
            return (
                '{"study": {"study_description": "cranial"}, '
                '"series": {"modality": "MR", "series_description": "cranial"}}'
            )

    criteria = parse_nl_to_criteria("cranial MR", FakeLLM(), strict_evidence=True)

    assert criteria.series is None


def test_parser_strict_full_composite_query(monkeypatch: pytest.MonkeyPatch) -> None:
    class FakeLLM:
        def chat(self, system_prompt: str, user_prompt: str, **_kwargs) -> str:
            return (
                '{"study": {"patient_sex": "F", '
                '"patient_birth_date": "20000101-20221231", '
                '"study_date": null, '
                '"modality_in_study": "MR", '
                '"study_description": "*cranial*"}, '
                '"series": {"modality": "MR", "series_description": "*cranial*"}}'
            )

    class FixedDate(date):
        @classmethod
        def today(cls) -> "FixedDate":
            return cls(2026, 1, 8)

    monkeypatch.setattr(nl_parser, "date", FixedDate)
    query = (
        "studies from year 2000 until 2022 of women ages 20 to 40 "
        "with cranial MR from ORTHANC to RADIANT"
    )
    criteria = parse_nl_to_criteria(query, FakeLLM(), strict_evidence=True)

    assert criteria.study.patient_sex == "F"
    assert criteria.study.patient_birth_date == "19850109-20060108"
    assert criteria.study.study_date == "20000101-20221231"
    assert criteria.study.modality_in_study == "MR"
    assert criteria.study.study_description == "*cranial*"
    assert criteria.series is None

    registry = NodeRegistry.from_tool_payload(
        [
            {"name": "orthanc", "ae_title": "ORTHANC"},
            {"name": "radiant", "ae_title": "RADIANT"},
        ]
    )
    matches = registry.match(query)

    assert sorted({match.node_id for match in matches}) == ["orthanc", "radiant"]
    assert {match.source for match in matches} == {"ORTHANC", "RADIANT"}
