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
