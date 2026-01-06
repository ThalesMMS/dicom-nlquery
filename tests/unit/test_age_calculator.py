from __future__ import annotations

from dicom_nlquery.matcher import calculate_age


def test_calculate_age_valid() -> None:
    assert calculate_age("19900115", "20200115") == 30


def test_calculate_age_before_birthday() -> None:
    assert calculate_age("19900615", "20200115") == 29


def test_calculate_age_missing_birth() -> None:
    assert calculate_age(None, "20200115") is None
