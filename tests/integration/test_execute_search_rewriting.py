from __future__ import annotations

import pytest

from dicom_nlquery.dicom_search import execute_search
from dicom_nlquery.models import LexiconConfig, SearchPipelineConfig
from dicom_nlquery.nl_parser import parse_nl_to_criteria


@pytest.mark.integration
def test_execute_search_rewriting_feto_vs_fetal(
    orthanc_with_data,
    fake_llm_feto,
    fake_llm_fetal,
):
    pipeline_config = SearchPipelineConfig(max_attempts=6, max_rewrites=6)
    lexicon_config = LexiconConfig(synonyms={"feto": ["fetal", "gestacao"]})

    criteria_feto = parse_nl_to_criteria("rm de feto", fake_llm_feto)
    criteria_fetal = parse_nl_to_criteria("rm fetal", fake_llm_fetal)

    result_feto = execute_search(
        criteria_feto,
        orthanc_with_data["client"],
        date_range=orthanc_with_data["date_range"],
        search_pipeline_config=pipeline_config,
        lexicon_config=lexicon_config,
    )
    result_fetal = execute_search(
        criteria_fetal,
        orthanc_with_data["client"],
        date_range=orthanc_with_data["date_range"],
        search_pipeline_config=pipeline_config,
        lexicon_config=lexicon_config,
    )

    assert "ACC006" in result_feto.accession_numbers
    assert "ACC007" not in result_feto.accession_numbers
    assert set(result_feto.accession_numbers) == set(result_fetal.accession_numbers)
