from __future__ import annotations

from dicom_nlquery.lexicon import Lexicon, normalize_text


def test_lexicon_normalize_text_removes_accents() -> None:
    assert normalize_text("Crânio") == "cranio"


def test_lexicon_expand_and_equivalent() -> None:
    lexicon = Lexicon()
    lexicon.update({"feto": ["fetal", "gestacao"]})

    assert "fetal" in lexicon.expand("feto")
    assert lexicon.equivalent("feto", "fetal")


def test_lexicon_match_text() -> None:
    lexicon = Lexicon()
    lexicon.update({"feto": ["fetal"]})

    assert lexicon.match_text("feto", "RM fetal study")
    assert not lexicon.match_text("feto", "RM cranio study")
    assert not lexicon.match_text("feto", "RM fetor study")


def test_lexicon_expand_text_respects_max_candidates() -> None:
    lexicon = Lexicon()
    lexicon.update({"feto": ["fetal", "gestacao"], "cranio": ["cranial"]})

    candidates = lexicon.expand_text("feto cranio", max_candidates=2)

    assert candidates
    assert len(candidates) <= 2
    assert any("fetal" in item or "gestacao" in item for item in candidates)
