from __future__ import annotations

from dicom_nlquery.normalizer import normalize


def test_normalize_removes_accents() -> None:
    text = "Caf\u00e9 Axial"

    assert normalize(text) == "cafe axial"


def test_normalize_collapses_whitespace() -> None:
    assert normalize("axial   pos") == "axial pos"


def test_normalize_lowercase() -> None:
    assert normalize("AXIAL POS") == "axial pos"
