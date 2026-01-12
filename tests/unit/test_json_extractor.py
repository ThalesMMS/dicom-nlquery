from __future__ import annotations

import pytest

from dicom_nlquery.nl_parser import extract_json


def test_extract_json_clean() -> None:
    text = '{"patient": {"sex": "F"}}'

    assert extract_json(text) == {"patient": {"sex": "F"}}


def test_extract_json_with_prefix() -> None:
    text = 'Here is the result: {"a": 1}'

    assert extract_json(text) == {"a": 1}


def test_extract_json_yaml_style() -> None:
    text = "{study: {patient_sex: 'F', study_description: cranial}, series: null}"

    assert extract_json(text) == {
        "study": {"patient_sex": "F", "study_description": "cranial"},
        "series": None,
    }


def test_extract_json_bareword_values() -> None:
    text = '{"study": {"modality_in_study": MR}, "series": null}'

    assert extract_json(text) == {"study": {"modality_in_study": "MR"}, "series": None}


def test_extract_json_no_json() -> None:
    with pytest.raises(ValueError):
        extract_json("no json here")


def test_extract_json_malformed() -> None:
    with pytest.raises(ValueError):
        extract_json('{"a": }')
