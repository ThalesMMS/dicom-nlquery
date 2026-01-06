from __future__ import annotations

import pytest

from dicom_nlquery.dicom_search import execute_search
from dicom_nlquery.nl_parser import parse_nl_to_criteria


@pytest.mark.integration
def test_end_to_end_female_20_40_cranio(orthanc_with_data, fake_llm_female_cranio):
    criteria = parse_nl_to_criteria("mulheres 20 a 40 cranio", fake_llm_female_cranio)
    result = execute_search(
        criteria,
        orthanc_with_data["client"],
        date_range=orthanc_with_data["date_range"],
    )

    assert sorted(result.accession_numbers) == ["ACC001", "ACC002", "ACC004", "ACC005"]


@pytest.mark.integration
def test_end_to_end_no_match(orthanc_with_data, fake_llm_no_match):
    criteria = parse_nl_to_criteria("sem resultados", fake_llm_no_match)
    result = execute_search(
        criteria,
        orthanc_with_data["client"],
        date_range=orthanc_with_data["date_range"],
    )

    assert result.accession_numbers == []
