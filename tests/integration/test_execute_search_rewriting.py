from __future__ import annotations

import pytest

from dicom_nlquery.dicom_search import execute_search
from dicom_nlquery.models import LexiconConfig, SearchPipelineConfig
from dicom_nlquery.nl_parser import parse_nl_to_criteria


EXPECTED_FETAL_ACCESSIONS = {"ACC006"}
NEGATIVE_ACCESSIONS = {"ACC001", "ACC002", "ACC003", "ACC004", "ACC005", "ACC007"}


@pytest.mark.integration
def test_execute_search_rewriting_feto_vs_fetal(
    orthanc_with_data,
    fake_llm_feto,
    fake_llm_fetal,
):
    pipeline_config = SearchPipelineConfig(max_attempts=6, max_rewrites=6)
    lexicon_config = LexiconConfig(synonyms={"fetus": ["fetal", "pregnancy"]})

    criteria_feto = parse_nl_to_criteria("mr for fetus", fake_llm_feto)
    criteria_fetal = parse_nl_to_criteria("fetal mr", fake_llm_fetal)

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

    feto_accessions = set(result_feto.accession_numbers)
    fetal_accessions = set(result_fetal.accession_numbers)

    assert EXPECTED_FETAL_ACCESSIONS.issubset(feto_accessions)
    assert EXPECTED_FETAL_ACCESSIONS.issubset(fetal_accessions)
    assert feto_accessions.issuperset(fetal_accessions) or fetal_accessions.issuperset(
        feto_accessions
    )
    assert not (feto_accessions & NEGATIVE_ACCESSIONS)
    assert not (fetal_accessions & NEGATIVE_ACCESSIONS)
    assert "fetor" not in result_feto.stats.rewrites_tried
    assert "fetor" not in result_fetal.stats.rewrites_tried
