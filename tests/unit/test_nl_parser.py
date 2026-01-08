from __future__ import annotations

import pytest
from pydantic import ValidationError

from dicom_nlquery.nl_parser import parse_nl_to_criteria


def test_parser_returns_valid_criteria(fake_llm) -> None:
    criteria = parse_nl_to_criteria("mulheres de 20 a 40", fake_llm)

    assert criteria.study.patient_sex == "F"


def test_parser_rejects_invalid_schema(fake_llm_invalid) -> None:
    with pytest.raises(ValidationError):
        parse_nl_to_criteria("query", fake_llm_invalid)


def test_parser_ignores_extra_fields(fake_llm_extra) -> None:
    criteria = parse_nl_to_criteria("query", fake_llm_extra)

    assert criteria.study.patient_sex == "F"


def test_parser_strict_removes_inferred_filters() -> None:
    class FakeLLM:
        def chat(self, system_prompt: str, user_prompt: str) -> str:
            return (
                '{\"study\": {\"patient_sex\": \"O\", '
                '\"modality_in_study\": \"MR\\\\CT\", '
                '\"study_description\": \"RM fetal\"}, '
                '\"series\": null}'
            )

    criteria = parse_nl_to_criteria("RM fetal para RADIANT", FakeLLM(), strict_evidence=True)

    assert criteria.study.patient_sex is None
    assert criteria.study.modality_in_study == "MR"
    assert criteria.study.study_description == "fetal"


def test_parser_strict_fallbacks_to_query_modality() -> None:
    class FakeLLM:
        def chat(self, system_prompt: str, user_prompt: str) -> str:
            return (
                '{\"study\": {\"patient_sex\": \"O\", '
                '\"study_description\": \"RADIANT\"}, \"series\": null}'
            )

    criteria = parse_nl_to_criteria("RM para RADIANT", FakeLLM(), strict_evidence=True)

    assert criteria.study.modality_in_study == "MR"


def test_parser_strict_adds_modality_when_missing() -> None:
    class FakeLLM:
        def chat(self, system_prompt: str, user_prompt: str) -> str:
            return '{\"study\": {\"study_description\": \"cranio\"}, \"series\": null}'

    criteria = parse_nl_to_criteria("RM de cranio", FakeLLM(), strict_evidence=True)

    assert criteria.study.modality_in_study == "MR"


def test_parser_strict_keeps_patient_name() -> None:
    class FakeLLM:
        def chat(self, system_prompt: str, user_prompt: str) -> str:
            return '{\"study\": {\"patient_name\": \"PACIENTE TESTE\"}, \"series\": null}'

    criteria = parse_nl_to_criteria("exames de paciente teste", FakeLLM(), strict_evidence=True)

    assert criteria.study.patient_name == "PACIENTE TESTE"


def test_parser_strict_extracts_patient_name_from_query() -> None:
    class FakeLLM:
        def chat(self, system_prompt: str, user_prompt: str) -> str:
            return '{\"study\": {}, \"series\": null}'

    criteria = parse_nl_to_criteria("exames de elaine", FakeLLM(), strict_evidence=True)

    assert criteria.study.patient_name == "ELAINE"


def test_parser_strict_applies_pelvis_keyword() -> None:
    class FakeLLM:
        def chat(self, system_prompt: str, user_prompt: str) -> str:
            return '{\"study\": {}, \"series\": {\"series_description\": \"*\"}}'

    criteria = parse_nl_to_criteria(
        "move pelvis exams from ORTHANC to RADIANT",
        FakeLLM(),
        strict_evidence=True,
    )

    assert criteria.study.study_description == "pelvis"
    assert criteria.series is None


def test_parser_strict_applies_pelvis_keyword_with_punctuation() -> None:
    class FakeLLM:
        def chat(self, system_prompt: str, user_prompt: str) -> str:
            return '{\"study\": {}, \"series\": {\"series_description\": \"*\"}}'

    criteria = parse_nl_to_criteria(
        "move pelvis, exams from ORTHANC to RADIANT",
        FakeLLM(),
        strict_evidence=True,
    )

    assert criteria.study.study_description == "pelvis"
    assert criteria.series is None
