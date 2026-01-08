from __future__ import annotations

from dicom_nlquery.lexicon import Lexicon, normalize_text


def test_lexicon_normalize_text_removes_accents() -> None:
    assert normalize_text("Café") == "cafe"


def test_lexicon_expand_and_equivalent() -> None:
    lexicon = Lexicon()
    lexicon.update({"fetus": ["fetal", "pregnancy"]})

    assert "fetal" in lexicon.expand("fetus")
    assert lexicon.equivalent("fetus", "fetal")


def test_lexicon_match_text() -> None:
    lexicon = Lexicon()
    lexicon.update({"fetus": ["fetal"]})

    assert lexicon.match_text("fetus", "MR fetal study")
    assert not lexicon.match_text("fetus", "MR cranial study")
    assert not lexicon.match_text("fetus", "MR fetor study")


def test_lexicon_expand_text_respects_max_candidates() -> None:
    lexicon = Lexicon()
    lexicon.update({"fetus": ["fetal", "pregnancy"], "cranial": ["skull"]})

    candidates = lexicon.expand_text("fetus cranial", max_candidates=2)

    assert candidates
    assert len(candidates) <= 2
    assert any("fetal" in item or "pregnancy" in item for item in candidates)
