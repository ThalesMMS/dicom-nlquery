from __future__ import annotations

from dicom_nlquery.matcher import series_matches
from dicom_nlquery.models import SeriesRequirement


def test_series_matches_all_keywords() -> None:
    series = {"SeriesDescription": "AX T1 POS"}
    requirement = SeriesRequirement(
        name="ax pos",
        within_head=False,
        all_keywords=["ax", "pos"],
    )

    assert series_matches(series, requirement) is True


def test_series_matches_all_keywords_partial_fail() -> None:
    series = {"SeriesDescription": "AX T1"}
    requirement = SeriesRequirement(
        name="ax pos",
        within_head=False,
        all_keywords=["ax", "pos"],
    )

    assert series_matches(series, requirement) is False


def test_series_matches_any_keywords() -> None:
    series = {"SeriesDescription": "AX T1"}
    requirement = SeriesRequirement(
        name="t1 or t2",
        within_head=False,
        any_keywords=["t1", "t2"],
    )

    assert series_matches(series, requirement) is True


def test_series_matches_within_head() -> None:
    series = {"SeriesDescription": "AX CRANIO"}
    requirement = SeriesRequirement(
        name="cranio",
        within_head=True,
        all_keywords=["ax"],
    )

    assert series_matches(series, requirement, head_keywords=["cranio"]) is True


def test_series_matches_modality() -> None:
    series = {"SeriesDescription": "AX T1", "Modality": "MR"}
    requirement = SeriesRequirement(
        name="mr",
        within_head=False,
        modality="MR",
        any_keywords=["t1"],
    )

    assert series_matches(series, requirement) is True
