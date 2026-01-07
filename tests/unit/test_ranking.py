from __future__ import annotations

from datetime import date

from dicom_nlquery.models import RankingConfig
from dicom_nlquery.search_pipeline import _rank_accessions


def test_rank_accessions_prefers_recent() -> None:
    today = date.today().strftime("%Y%m%d")
    old = "20100101"
    matches = [
        ({"StudyDescription": "cranio", "StudyDate": old}, "ACC_OLD"),
        ({"StudyDescription": "cranio", "StudyDate": today}, "ACC_NEW"),
    ]
    ranking = RankingConfig(enabled=True, text_match_weight=0.0, recency_weight=1.0)

    ranked = _rank_accessions(matches, "cranio", ranking)

    assert ranked[0] == "ACC_NEW"


def test_rank_accessions_disabled_keeps_order() -> None:
    matches = [
        ({"StudyDescription": "cranio", "StudyDate": "20240101"}, "ACC1"),
        ({"StudyDescription": "cranio", "StudyDate": "20230101"}, "ACC2"),
    ]
    ranking = RankingConfig(enabled=False)

    ranked = _rank_accessions(matches, "cranio", ranking)

    assert ranked == ["ACC1", "ACC2"]
