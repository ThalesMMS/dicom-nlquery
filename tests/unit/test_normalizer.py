from __future__ import annotations

from dicom_nlquery.normalizer import normalize


def test_normalize_removes_accents() -> None:
    text = "Cr\u00e2nio \u00c1xial"

    assert normalize(text) == "cranio axial"


def test_normalize_collapses_whitespace() -> None:
    assert normalize("axial   pos") == "axial pos"


def test_normalize_lowercase() -> None:
    assert normalize("AXIAL POS") == "axial pos"
